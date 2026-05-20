/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <sys/types.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <utility>

#include <riscv_vector.h> // for Riscv

#include "tflite/kernels/cpu_backend_context.h"
#include "tflite/kernels/cpu_backend_gemm.h"
#include "tflite/kernels/cpu_backend_gemm_params.h"
#include "tflite/kernels/internal/common.h"
#include "tflite/kernels/internal/compatibility.h"
#include "tflite/kernels/internal/cppmath.h"
#include "tflite/kernels/internal/optimized/cpu_check.h"
#include "tflite/kernels/internal/optimized/riscv_tensor_utils_impl.h"

#ifdef USE_RISCV

// aligned_alloc is avail able (via cstdlib/stdlib.h) with C++17/C11.
// (introduced in stdc11 but realized in C++17)
#if __cplusplus >= 201703L && __STDC_VERSION__ >= 201112L
#if !defined(__ANDROID__) || __ANDROID_API__ >= 28
// Neither Apple nor Windows provide aligned_alloc.
#if !defined(__APPLE__) && !defined(_WIN32)
#define TFLITE_USE_STD_ALIGNED_ALLOC
#endif
#endif
#endif

// Note: This is the same as ABSL_HAVE_BUILTIN, but can't include the header.
#ifdef __has_builtin
#define TFLITE_HAS_BUILTIN(x) __has_builtin(x)
#else
#define TFLITE_HAS_BUILTIN(x) 0
#endif

// Note: This is the same as ABSL_PREDICT_FALSE, but can't include the header.
#if TFLITE_HAS_BUILTIN(__builtin_expect) ||                                    \
    (defined(__GNUC__) && !defined(__clang__))
#define TFLITE_UNLIKELY(x) (__builtin_expect(false || (x), false))
#else
#define TFLITE_UNLIKELY(x) (x)
#endif

namespace tflite {
namespace tensor_utils {

void RISCVMatrixBatchVectorMultiplyAccumulate(const float *matrix, int m_rows,
                                              int m_cols, const float *vector,
                                              int n_batch, float *result) {
  for (int b = 0; b < n_batch; b++) {
    float *result_in_batch = result + b * m_rows;
    const float *vector_in_batch = vector + b * m_cols;
    const float *matrix_row = matrix;

    for (int r = 0; r < m_rows; r++) {
      int c = 0;
      int cols_left = m_cols;

      size_t vlmax = __riscv_vsetvlmax_e32m1();

      vfloat32m1_t v_acc = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);

      while (cols_left > 0) {
        size_t vl = __riscv_vsetvl_e32m1(cols_left);

        vfloat32m1_t vector_rvv =
            __riscv_vle32_v_f32m1(vector_in_batch + c, vl);
        vfloat32m1_t matrix_rvv = __riscv_vle32_v_f32m1(matrix_row + c, vl);
        v_acc = __riscv_vfmacc_vv_f32m1(v_acc, matrix_rvv, vector_rvv, vl);
        c += vl;
        cols_left -= vl;
      }

      vfloat32m1_t v_zero = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
      vfloat32m1_t v_res =
          __riscv_vfredusum_vs_f32m1_f32m1(v_acc, v_zero, vlmax);

      *result_in_batch += __riscv_vfmv_f_s_f32m1_f32(v_res);

      matrix_row += m_cols;
      ++result_in_batch;
    }
  }
}

void RISCVMatrixBatchVectorMultiplyAccumulate(
    const int8_t *__restrict__ matrix, const int m_rows, const int m_cols,
    const int8_t *__restrict__ vectors, const float *scaling_factors,
    int n_batch, float *__restrict__ result) {
  for (int batch = 0; batch < n_batch; ++batch, vectors += m_cols) {
    const float batch_scaling_factor = scaling_factors[batch];
    const int8_t *row_ptr = matrix;

    for (int row = 0; row < m_rows; ++row) {
      int c = 0;
      int cols_left = m_cols;

      size_t vlmax_i8 = __riscv_vsetvlmax_e8m1();

      vint16m2_t v_dotprod = __riscv_vmv_v_x_i16m2(0, vlmax_i8);

      while (cols_left > 0) {
        size_t vl = __riscv_vsetvl_e8m1(cols_left);

        vint8m1_t v_mat_i8 = __riscv_vle8_v_i8m1(row_ptr + c, vl);
        vint8m1_t v_vec_i8 = __riscv_vle8_v_i8m1(vectors + c, vl);

        v_dotprod = __riscv_vwmacc_vv_i16m2(v_dotprod, v_mat_i8, v_vec_i8, vl);

        cols_left -= vl;
        c += vl;
      }

      vint16m1_t v_zero = __riscv_vmv_v_x_i16m1(0, __riscv_vsetvlmax_e16m1());

      vint16m1_t v_res =
          __riscv_vredsum_vs_i16m2_i16m1(v_dotprod, v_zero, vlmax_i8);

      int16_t final_dotprod = __riscv_vmv_x_s_i16m1_i16(v_res);

      *result += static_cast<float>(final_dotprod) * batch_scaling_factor;
      ++result;

      row_ptr += m_cols;
    }
  }
}

void RISCVMatrixBatchVectorMultiplyAccumulate(
    const int8_t *__restrict__ matrix, const int m_rows, const int m_cols,
    const int8_t *__restrict__ vectors, const float *scaling_factors,
    int n_batch, float *__restrict__ result, const float *per_channel_scale,
    const int32_t *input_offset, int32_t *scratch, int32_t *row_sums,
    bool *compute_row_sums, CpuBackendContext *context) {
  if (input_offset == nullptr) {
    RISCVMatrixBatchVectorMultiplyAccumulate(matrix, m_rows, m_cols, vectors,
                                             scaling_factors, n_batch, result);
    return;
  }

  for (int batch = 0; batch < n_batch; ++batch, vectors += m_cols) {
    const float batch_scaling_factor = scaling_factors[batch];

    const int32_t batch_offset = input_offset[batch];
    const int8_t *row_ptr = matrix;

    for (int row = 0; row < m_rows; ++row) {
      int c = 0;
      int cols_left = m_cols;

      size_t vlmax_i8 = __riscv_vsetvlmax_e8m1();

      vint32m4_t v_dotprod = __riscv_vmv_v_x_i32m4(0, vlmax_i8);

      while (cols_left > 0) {
        size_t vl = __riscv_vsetvl_e8m1(cols_left);

        vint8m1_t v_mat_i8 = __riscv_vle8_v_i8m1(row_ptr + c, vl);
        vint8m1_t v_vec_i8 = __riscv_vle8_v_i8m1(vectors + c, vl);

        vint16m2_t v_mat_i16 = __riscv_vsext_vf2_i16m2(v_mat_i8, vl);
        vint16m2_t v_vec_i16 = __riscv_vsext_vf2_i16m2(v_vec_i8, vl);

        v_dotprod =
            __riscv_vwmacc_vv_i32m4(v_dotprod, v_mat_i16, v_vec_i16, vl);

        cols_left -= vl;
        c += vl;
      }

      vint32m1_t v_zero = __riscv_vmv_v_x_i32m1(0, __riscv_vsetvlmax_e32m1());

      vint32m1_t v_res =
          __riscv_vredsum_vs_i32m4_i32m1(v_dotprod, v_zero, vlmax_i8);

      int32_t dotprod = __riscv_vmv_x_s_i32m1_i32(v_res);
      dotprod -= row_sums[row] * batch_offset;

      float final_scale = batch_scaling_factor;
      if (per_channel_scale) {
        final_scale *= per_channel_scale[row];
      }

      *result += static_cast<float>(dotprod) * final_scale;
      ++result;

      row_ptr += m_cols;
    }
  }
}

bool RISCVIsZeroVector(const int8_t *vector, int v_size) {
  int c = 0;
  int elements_left = v_size;

  while (elements_left > 0) {
    size_t vl = __riscv_vsetvl_e8m8(elements_left);

    vint8m8_t v_data = __riscv_vle8_v_i8m8(vector + c, vl);
    vbool1_t v_mask = __riscv_vmsne_vx_i8m8_b1(v_data, 0, vl);
    long active_count = __riscv_vcpop_m_b1(v_mask, vl);

    if (active_count > 0) {
      return false;
    }

    elements_left -= vl;
    c += vl;
  }

  return true;
}

bool RISCVIsZeroVector(const float *vector, int v_size) {
  int c = 0;
  int elements_left = v_size;

  while (elements_left > 0) {
    size_t vl = __riscv_vsetvl_e32m8(elements_left);

    vfloat32m8_t v_data = __riscv_vle32_v_f32m8(vector + c, vl);
    vbool4_t v_mask = __riscv_vmfne_vf_f32m8_b4(v_data, 0.0f, vl);
    long active_count = __riscv_vcpop_m_b4(v_mask, vl);

    if (active_count > 0) {
      return false;
    }

    elements_left -= vl;
    c += vl;
  }

  return true;
}

void RISCVSub1Vector(const int16_t *vector, int v_size, int16_t *result) {
  int c = 0;
  int elements_left = v_size;

  while (elements_left > 0) {
    size_t vl = __riscv_vsetvl_e16m8(elements_left);

    vint16m8_t v_data = __riscv_vle16_v_i16m8(vector + c, vl);
    vint16m8_t v_res = __riscv_vrsub_vx_i16m8(v_data, 32767, vl);

    __riscv_vse16_v_i16m8(result + c, v_res, vl);

    elements_left -= vl;
    c += vl;
  }
}

void RISCVSub1Vector(const float *vector, int v_size, float *result) {
  int c = 0;
  int elements_left = v_size;

  while (elements_left > 0) {
    size_t vl = __riscv_vsetvl_e32m8(elements_left);

    vfloat32m8_t v_data = __riscv_vle32_v_f32m8(vector + c, vl);
    vfloat32m8_t v_res = __riscv_vfrsub_vf_f32m8(v_data, 1.0f, vl);

    __riscv_vse32_v_f32m8(result + c, v_res, vl);

    elements_left -= vl;
    c += vl;
  }
}

void RISCVSparseMatrixBatchVectorMultiplyAccumulate1x4(
    const float *__restrict__ matrix, const int32_t *__restrict__ segments,
    const int32_t *__restrict__ indices, int m_rows, int m_cols,
    const float *__restrict__ vector, int n_batch, float *__restrict__ result) {
  const int kBlockSize = 4;

  const float *matrix_ptr = matrix;

  for (int batch = 0; batch < n_batch; batch++) {
    const float *vector_in_batch = vector + batch * m_cols;

    for (int row = 0; row < m_rows; row++) {
      size_t vlmax = __riscv_vsetvlmax_e32m1();
      vfloat32m1_t v_dotprod = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);

      int start_block = segments[row];
      int end_block = segments[row + 1];

      for (int i = start_block; i < end_block; i++) {
        const int block_start_index = indices[i] * kBlockSize;

        const float *vector_block_in_batch_ptr =
            vector_in_batch + block_start_index;

        size_t vl = __riscv_vsetvl_e32m1(kBlockSize);

        vfloat32m1_t v_vec =
            __riscv_vle32_v_f32m1(vector_block_in_batch_ptr, vl);
        vfloat32m1_t v_mat = __riscv_vle32_v_f32m1(matrix_ptr, vl);

        matrix_ptr += kBlockSize;

        v_dotprod = __riscv_vfmacc_vv_f32m1(v_dotprod, v_mat, v_vec, vl);
      }

      float dot_prod_scalar = 0.0f;

      vfloat32m1_t v_res_scalar = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
      v_res_scalar =
          __riscv_vfredusum_vs_f32m1_f32m1(v_dotprod, v_res_scalar, vlmax);
      dot_prod_scalar = __riscv_vfmv_f_s_f32m1_f32(v_res_scalar);

      result[batch * m_rows + row] += dot_prod_scalar;
    }
  }
}

void RISCVSparseMatrixBatchVectorMultiplyAccumulate(
    const float *__restrict__ matrix, const uint8_t *__restrict__ ledger,
    int m_rows, int m_cols, const float *__restrict__ vector, int n_batch,
    float *__restrict__ result) {
  const int kBlockSize = 16;

  const float *matrix_ptr = matrix;

  for (int batch = 0; batch < n_batch; batch++) {
    const float *vector_in_batch = vector + batch * m_cols;
    const uint8_t *ledger_ptr = ledger;

    for (int row = 0; row < m_rows; row++) {
      size_t vlmax = __riscv_vsetvlmax_e32m4();
      vfloat32m4_t v_dotprod = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);

      int num_nonzero_blocks = *ledger_ptr++;
      if (num_nonzero_blocks > 0) {

        for (int i = 0; i < num_nonzero_blocks; i++) {
          uint8_t original_col_block_index = *ledger_ptr++;
          const int block_start_index =
              original_col_block_index * kBlockSize; // kBlockSize = 16

          const float *vector_block_in_batch_ptr =
              vector_in_batch + block_start_index;

          size_t vl = __riscv_vsetvl_e32m4(kBlockSize);
          vfloat32m4_t v_vec =
              __riscv_vle32_v_f32m4(vector_block_in_batch_ptr, vl);

          vfloat32m4_t v_mat = __riscv_vle32_v_f32m4(matrix_ptr, vl);

          matrix_ptr += kBlockSize;

          v_dotprod = __riscv_vfmacc_vv_f32m4(v_dotprod, v_mat, v_vec, vl);
        }
      }
      float dot_prod_scalar = 0.0f;

      size_t vlmax_m1 = __riscv_vsetvlmax_e32m1();
      vfloat32m1_t v_res_scalar = __riscv_vfmv_v_f_f32m1(0.0f, vlmax_m1);

      v_res_scalar =
          __riscv_vfredusum_vs_f32m4_f32m1(v_dotprod, v_res_scalar, vlmax);
      dot_prod_scalar = __riscv_vfmv_f_s_f32m1_f32(v_res_scalar);
      result[batch * m_rows + row] += dot_prod_scalar;
    }
  }
}
void RISCVCwiseMul(const int16_t *input_1, const int16_t *input_2, int n_batch,
                   int n_input, int shift, int16_t *output) {
  int total_elements = n_batch * n_input;

  int index = 0;
  while (index < total_elements) {
    size_t vl = __riscv_vsetvl_e16m4(total_elements - index);

    vint16m4_t va = __riscv_vle16_v_i16m4(input_1 + index, vl);
    vint16m4_t vb = __riscv_vle16_v_i16m4(input_2 + index, vl);
    vint32m8_t v_value = __riscv_vwmul_vv_i32m8(va, vb, vl);
    vint16m4_t v_out =
        __riscv_vnclip_wx_i16m4(v_value, shift, __RISCV_VXRM_RNU, vl);

    __riscv_vse16_v_i16m4(output + index, v_out, vl);

    index += vl;
  }
}

void RISCVCwiseMul(const int16_t *input_1, const int16_t *input_2,
                   int32_t multiplier, int32_t shift, int32_t n_batch,
                   int32_t n_input, int32_t output_zp, int8_t *output) {
  int total_elements = n_batch * n_input;
  int index = 0;
  int left_shift = shift > 0 ? shift : 0;
  int right_shift = shift > 0 ? 0 : -shift;

  while (index < total_elements) {
    size_t vl = __riscv_vsetvl_e16m4(total_elements - index);

    vint16m4_t va = __riscv_vle16_v_i16m4(input_1 + index, vl);
    vint16m4_t vb = __riscv_vle16_v_i16m4(input_2 + index, vl);
    vint32m8_t v_val = __riscv_vwmul_vv_i32m8(va, vb, vl);

    if (left_shift > 0) {
      v_val = __riscv_vsll_vx_i32m8(v_val, left_shift, vl);
    }

    vint32m8_t v_scaled =
        __riscv_vsmul_vx_i32m8(v_val, multiplier, __RISCV_VXRM_RNU, vl);

    if (right_shift > 0) {
      v_scaled =
          __riscv_vssra_vx_i32m8(v_scaled, right_shift, __RISCV_VXRM_RNU, vl);
    }

    vint32m8_t v_zp_added = __riscv_vadd_vx_i32m8(v_scaled, output_zp, vl);

    vint16m4_t v_narrow_16 =
        __riscv_vnclip_wx_i16m4(v_zp_added, 0, __RISCV_VXRM_RNU, vl);
    vint8m2_t v_out =
        __riscv_vnclip_wx_i8m2(v_narrow_16, 0, __RISCV_VXRM_RNU, vl);

    __riscv_vse8_v_i8m2(output + index, v_out, vl);

    index += vl;
  }
}

void RISCVCwiseAdd(const int16_t *input_1, const int16_t *input_2, int n_batch,
                   int n_input, int16_t *output) {
  int total_elements = n_batch * n_input;
  int index = 0;

  while (index < total_elements) {
    size_t vl = __riscv_vsetvl_e16m8(total_elements - index);

    vint16m8_t va = __riscv_vle16_v_i16m8(input_1 + index, vl);
    vint16m8_t vb = __riscv_vle16_v_i16m8(input_2 + index, vl);
    vint16m8_t v_out = __riscv_vsadd_vv_i16m8(va, vb, vl);

    __riscv_vse16_v_i16m8(output + index, v_out, vl);

    index += vl;
  }
}

void RISCVCwiseClipping(float *__restrict__ vector, const int v_size,
                        const float clipping_value) {

  int index = 0;
  float neg_clipping_value = -clipping_value;

  while (index < v_size) {
    size_t vl = __riscv_vsetvl_e32m8(v_size - index);

    vfloat32m8_t v_val = __riscv_vle32_v_f32m8(vector + index, vl);

    v_val = __riscv_vfmin_vf_f32m8(v_val, clipping_value, vl);
    v_val = __riscv_vfmax_vf_f32m8(v_val, neg_clipping_value, vl);

    __riscv_vse32_v_f32m8(vector + index, v_val, vl);

    index += vl;
  }
}

void RISCVCwiseClipping(int16_t *__restrict__ vector, const int v_size,
                        const int16_t clipping_value) {
  int index = 0;
  int16_t neg_clipping_value = -clipping_value;

  while (index < v_size) {
    size_t vl = __riscv_vsetvl_e16m8(v_size - index);

    vint16m8_t v_val = __riscv_vle16_v_i16m8(vector + index, vl);
    v_val = __riscv_vmin_vx_i16m8(v_val, clipping_value, vl);
    v_val = __riscv_vmax_vx_i16m8(v_val, neg_clipping_value, vl);

    __riscv_vse16_v_i16m8(vector + index, v_val, vl);

    index += vl;
  }
}

void RISCVCwiseClipping(int8_t *__restrict__ vector, const int v_size,
                        const int8_t clipping_value) {
  int index = 0;
  int8_t neg_clipping_value = -clipping_value;

  while (index < v_size) {
    size_t vl = __riscv_vsetvl_e8m8(v_size - index);

    vint8m8_t v_val = __riscv_vle8_v_i8m8(vector + index, vl);
    v_val = __riscv_vmin_vx_i8m8(v_val, clipping_value, vl);
    v_val = __riscv_vmax_vx_i8m8(v_val, neg_clipping_value, vl);

    __riscv_vse8_v_i8m8(vector + index, v_val, vl);

    index += vl;
  }
}

void RISCVPortableBatchVectorBatchVectorDotProduct(
    const int16_t *__restrict__ vector1, const int16_t *__restrict__ vector2,
    int v_size, int n_batch, int32_t *__restrict__ result) {
  for (int b = 0; b < n_batch; b++) {

    size_t vlmax = __riscv_vsetvlmax_e32m8();
    vint32m8_t v_acc = __riscv_vmv_v_x_i32m8(0, vlmax);

    int elements_left = v_size;
    const int16_t *ptr1 = vector1;
    const int16_t *ptr2 = vector2;

    while (elements_left > 0) {
      size_t vl = __riscv_vsetvl_e16m4(elements_left);

      vint16m4_t va = __riscv_vle16_v_i16m4(ptr1, vl);
      vint16m4_t vb = __riscv_vle16_v_i16m4(ptr2, vl);
      v_acc = __riscv_vwmacc_vv_i32m8(v_acc, va, vb, vl);

      ptr1 += vl;
      ptr2 += vl;
      elements_left -= vl;
    }

    size_t vl_m1 = __riscv_vsetvlmax_e32m1();
    vint32m1_t v_res_scalar = __riscv_vmv_v_x_i32m1(0, vl_m1);
    v_res_scalar = __riscv_vredsum_vs_i32m8_i32m1(v_acc, v_res_scalar, vlmax);
    result[b] = __riscv_vmv_x_s_i32m1_i32(v_res_scalar);

    vector1 += v_size;
    vector2 += v_size;
  }
}

float RISCVVectorVectorDotProduct(const float *vector1, const float *vector2,
                                  int v_size) {
  int elements_left = v_size;
  size_t vlmax = __riscv_vsetvlmax_e32m8();

  vfloat32m8_t v_acc = __riscv_vfmv_v_f_f32m8(0.0f, vlmax);

  while (elements_left > 0) {
    size_t vl = __riscv_vsetvl_e32m8(elements_left);

    vfloat32m8_t va = __riscv_vle32_v_f32m8(vector1, vl);
    vfloat32m8_t vb = __riscv_vle32_v_f32m8(vector2, vl);
    v_acc = __riscv_vfmacc_vv_f32m8(v_acc, va, vb, vl);

    vector1 += vl;
    vector2 += vl;
    elements_left -= vl;
  }

  size_t vl_m1 = __riscv_vsetvlmax_e32m1();
  vfloat32m1_t v_res_scalar = __riscv_vfmv_v_f_f32m1(0.0f, vl_m1);

  v_res_scalar = __riscv_vfredusum_vs_f32m8_f32m1(v_acc, v_res_scalar, vlmax);

  return __riscv_vfmv_f_s_f32m1_f32(v_res_scalar);
}

void RISCVVectorBatchVectorCwiseProductAccumulate(
    const int16_t *vector, int v_size, const int16_t *batch_vector, int n_batch,
    int32_t multiplier, int shift, int16_t *result) {
  int left_shift = shift > 0 ? shift : 0;
  int right_shift = shift > 0 ? 0 : -shift;

  for (int b = 0; b < n_batch; b++) {

    int elements_left = v_size;
    const int16_t *v_ptr = vector;

    while (elements_left > 0) {
      size_t vl = __riscv_vsetvl_e16m4(elements_left);

      vint16m4_t va = __riscv_vle16_v_i16m4(v_ptr, vl);
      vint16m4_t vb = __riscv_vle16_v_i16m4(batch_vector, vl);
      vint16m4_t v_res_in = __riscv_vle16_v_i16m4(result, vl);

      vint32m8_t v_prod = __riscv_vwmul_vv_i32m8(va, vb, vl);

      if (left_shift > 0) {
        v_prod =
            __riscv_vsll_vx_i32m8(v_prod, left_shift, vl); // 先左移垫高精度
      }
      v_prod = __riscv_vsmul_vx_i32m8(v_prod, multiplier, __RISCV_VXRM_RNU, vl);

      if (right_shift > 0) {
        v_prod = __riscv_vssra_vx_i32m8(v_prod, right_shift, __RISCV_VXRM_RNU,
                                        vl); // 后右移砍尾巴
      }

      vint32m8_t v_acc = __riscv_vwadd_wv_i32m8(v_prod, v_res_in, vl);
      vint16m4_t v_out =
          __riscv_vnclip_wx_i16m4(v_acc, 0, __RISCV_VXRM_RNU, vl);

      __riscv_vse16_v_i16m4(result, v_out, vl);

      v_ptr += vl;
      batch_vector += vl;
      result += vl;
      elements_left -= vl;
    }
  }
}

void RISCVVectorScalarMultiply(const int8_t *vector, const int v_size,
                               const float scale, float *result) {
  int index = 0;

  while (index < v_size) {
    size_t vl = __riscv_vsetvl_e8m2(v_size - index);

    vint8m2_t v_in = __riscv_vle8_v_i8m2(vector + index, vl);
    vint16m4_t v_ext = __riscv_vsext_vf2_i16m4(v_in, vl);

    vfloat32m8_t v_f32 = __riscv_vfwcvt_f_x_v_f32m8(v_ext, vl);
    vfloat32m8_t v_res = __riscv_vfmul_vf_f32m8(v_f32, scale, vl);

    __riscv_vse32_v_f32m8(result + index, v_res, vl);

    index += vl;
  }
}

void RISCVAsymmetricQuantizeFloats(const float *values, const int size,
                                   int8_t *quantized_values,
                                   float *scaling_factor, int32_t *offset) {
  const int32_t kMinScale = -128;
  const int32_t kMaxScale = 127;
  const double qmin_double = kMinScale;
  const double qmax_double = kMaxScale;

  const auto minmax = std::minmax_element(values, values + size);
  const double rmin = static_cast<double>(std::min(0.0f, *minmax.first));
  const double rmax = static_cast<double>(std::max(0.0f, *minmax.second));

  if (rmin == rmax) {
    std::memset(quantized_values, 0, size * sizeof(int8_t));
    *scaling_factor = 1.0f;
    *offset = 0;
    return;
  }

  double scale = (rmax - rmin) / (qmax_double - qmin_double);

  const double zero_point_from_min = qmin_double - rmin / scale;
  const double zero_point_from_max = qmax_double - rmax / scale;
  const double zero_point_from_min_error =
      std::abs(qmin_double) + std::abs(rmin / scale);
  const double zero_point_from_max_error =
      std::abs(qmax_double) + std::abs(rmax / scale);
  const double zero_point_double =
      zero_point_from_min_error < zero_point_from_max_error
          ? zero_point_from_min
          : zero_point_from_max;

  int8_t nudged_zero_point = 0;
  if (zero_point_double <= qmin_double) {
    nudged_zero_point = kMinScale;
  } else if (zero_point_double >= qmax_double) {
    nudged_zero_point = kMaxScale;
  } else {
    nudged_zero_point = static_cast<int8_t>(std::round(zero_point_double));
  }

  *scaling_factor = static_cast<float>(scale);
  *offset = nudged_zero_point;

  float scaling_factor_inv = 1.0f / (*scaling_factor);
  float offset_f = static_cast<float>(*offset);

  int index = 0;

  while (index < size) {
    size_t vl = __riscv_vsetvl_e32m8(size - index);
    vfloat32m8_t v_val = __riscv_vle32_v_f32m8(values + index, vl);

    vfloat32m8_t v_scaled =
        __riscv_vfmul_vf_f32m8(v_val, scaling_factor_inv, vl);
    vfloat32m8_t v_shifted = __riscv_vfadd_vf_f32m8(v_scaled, offset_f, vl);

    vint32m8_t v_i32 = __riscv_vfcvt_x_f_v_i32m8(v_shifted, vl);

    vint16m4_t v_i16 = __riscv_vnclip_wx_i16m4(v_i32, 0, __RISCV_VXRM_RNU, vl);
    vint8m2_t v_i8 = __riscv_vnclip_wx_i8m2(v_i16, 0, __RISCV_VXRM_RNU, vl);

    __riscv_vse8_v_i8m2(quantized_values + index, v_i8, vl);

    index += vl;
  }
}

void RISCVSymmetricQuantizeFloats(const float *values, const int size,
                                  int8_t *quantized_values, float min_value,
                                  float max_value, float *scaling_factor) {
  const int32_t kScale = 127;
  const float range = std::max(std::abs(min_value), std::abs(max_value));

  if (range == 0) {
    std::memset(quantized_values, 0, size * sizeof(int8_t));
    *scaling_factor = 1.0f;
    return;
  }

  *scaling_factor = range / kScale;
  const float scaling_factor_inv = kScale / range;

  int i = 0;
  int n = size;

  while (n > 0) {
    size_t vl = __riscv_vsetvl_e32m8(n);

    vfloat32m8_t v_src = __riscv_vle32_v_f32m8(values + i, vl);
    vfloat32m8_t v_scaled =
        __riscv_vfmul_vf_f32m8(v_src, scaling_factor_inv, vl);

    vint32m8_t v_int32 = __riscv_vfcvt_x_f_v_i32m8(v_scaled, vl);

    vint16m4_t v_i16 =
        __riscv_vnclip_wx_i16m4(v_int32, 0, __RISCV_VXRM_RNU, vl);
    vint8m2_t v_i8 = __riscv_vnclip_wx_i8m2(v_i16, 0, __RISCV_VXRM_RNU, vl);

    __riscv_vse8_v_i8m2(quantized_values + i, v_i8, vl);

    i += vl;
    n -= vl;
  }
}

namespace {
void RISCVFindMinMax(const float *values, const int size, float *min_value,
                     float *max_value) {
  int i = 0;
  int n = size;

  vfloat32m8_t v_max_accumulator = __riscv_vfmv_v_f_f32m8(
      -std::numeric_limits<float>::max(), __riscv_vsetvlmax_e32m8());
  vfloat32m8_t v_min_accumulator = __riscv_vfmv_v_f_f32m8(
      std::numeric_limits<float>::max(), __riscv_vsetvlmax_e32m8());

  while (n > 0) {
    size_t vl = __riscv_vsetvl_e32m8(n);

    vfloat32m8_t v_src = __riscv_vle32_v_f32m8(values + i, vl);
    v_max_accumulator = __riscv_vfmax_vv_f32m8(v_max_accumulator, v_src, vl);
    v_min_accumulator = __riscv_vfmin_vv_f32m8(v_min_accumulator, v_src, vl);

    i += vl;
    n -= vl;
  }

  vfloat32m1_t v_scalar_max =
      __riscv_vfmv_v_f_f32m1(-std::numeric_limits<float>::max(), 1);
  vfloat32m1_t v_scalar_min =
      __riscv_vfmv_v_f_f32m1(std::numeric_limits<float>::max(), 1);

  v_scalar_max = __riscv_vfredmax_vs_f32m8_f32m1(
      v_max_accumulator, v_scalar_max, __riscv_vsetvlmax_e32m4());
  v_scalar_min = __riscv_vfredmin_vs_f32m8_f32m1(
      v_min_accumulator, v_scalar_min, __riscv_vsetvlmax_e32m4());

  *max_value = __riscv_vfmv_f_s_f32m1_f32(v_scalar_max);
  *min_value = __riscv_vfmv_f_s_f32m1_f32(v_scalar_min);
}
} // namespace

void RISCVSymmetricQuantizeFloats(const float *values, const int size,
                                  int8_t *quantized_values, float *min_value,
                                  float *max_value, float *scaling_factor) {

  RISCVFindMinMax(values, size, min_value, max_value);
  RISCVSymmetricQuantizeFloats(values, size, quantized_values, *min_value,
                               *max_value, scaling_factor);
}

} // namespace tensor_utils
} // namespace tflite

#endif // USE_RISCV
