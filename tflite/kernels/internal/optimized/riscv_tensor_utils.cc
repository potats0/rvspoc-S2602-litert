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

namespace {
inline vint32m4_t VectorMultiplyByQuantizedMultiplier(
    vint32m4_t v_x, int32_t quantized_multiplier, int shift, size_t vl) {
  int left_shift = shift > 0 ? shift : 0;
  int right_shift = shift > 0 ? 0 : -shift;

  if (left_shift > 0) {
    v_x = __riscv_vsll_vx_i32m4(v_x, left_shift, vl);
  }

  v_x = __riscv_vsmul_vx_i32m4(v_x, quantized_multiplier, __RISCV_VXRM_RNU, vl);
  if (right_shift > 0) {
    vint32m4_t v_sign = __riscv_vsra_vx_i32m4(v_x, 31, vl);

    int32_t offset = 1 << (right_shift - 1);
    vint32m4_t v_adjusted = __riscv_vadd_vx_i32m4(v_x, offset, vl);
    v_adjusted = __riscv_vadd_vv_i32m4(v_adjusted, v_sign, vl);
    v_x = __riscv_vsra_vx_i32m4(v_adjusted, right_shift, vl);
  }
  return v_x;
}

} // namespace

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
        v_acc = __riscv_vfmacc_vv_f32m1_tu(v_acc, matrix_rvv, vector_rvv, vl);
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

      vint32m4_t v_dotprod = __riscv_vmv_v_x_i32m4(0, vlmax_i8);

      while (cols_left > 0) {
        size_t vl = __riscv_vsetvl_e8m1(cols_left);

        vint8m1_t v_mat_i8 = __riscv_vle8_v_i8m1(row_ptr + c, vl);
        vint8m1_t v_vec_i8 = __riscv_vle8_v_i8m1(vectors + c, vl);

        vint16m2_t v_mat_i16 = __riscv_vsext_vf2_i16m2(v_mat_i8, vl);
        vint16m2_t v_vec_i16 = __riscv_vsext_vf2_i16m2(v_vec_i8, vl);

        v_dotprod =
            __riscv_vwmacc_vv_i32m4_tu(v_dotprod, v_mat_i16, v_vec_i16, vl);

        cols_left -= vl;
        c += vl;
      }

      vint32m1_t v_zero = __riscv_vmv_v_x_i32m1(0, vlmax_i8);
      vint32m1_t v_res =
          __riscv_vredsum_vs_i32m4_i32m1(v_dotprod, v_zero, vlmax_i8);
      int32_t final_dotprod = __riscv_vmv_x_s_i32m1_i32(v_res);

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

  if (!compute_row_sums || *compute_row_sums) {
    RISCVReductionSumVector(matrix, row_sums, m_rows, m_cols);
    if (compute_row_sums) {
      *compute_row_sums = false;
    }
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

void RISCVMatrixBatchVectorMultiplyAccumulate(
    const int8_t *input, const int32_t *bias,
    const int8_t *input_to_gate_weights, int32_t multiplier, int32_t shift,
    int32_t n_batch, int32_t n_input, int32_t n_output, int32_t output_zp,
    int32_t *scratch, int16_t *output, CpuBackendContext *context) {

  for (int batch = 0; batch < n_batch; ++batch) {
    const int8_t *current_input = input + batch * n_input;

    for (int row = 0; row < n_output; ++row) {
      int32_t acc = bias[row];
      const int8_t *current_weights = input_to_gate_weights + row * n_input;

      size_t vlmax = __riscv_vsetvlmax_e32m8();
      vint32m8_t v_acc32 = __riscv_vmv_v_x_i32m8(0, vlmax);

      int col_left = n_input;
      int col = 0;
      while (col_left > 0) {
        size_t vl = __riscv_vsetvl_e8m2(col_left);

        vint8m2_t v_in8 = __riscv_vle8_v_i8m2(&current_input[col], vl);
        vint8m2_t v_w8 = __riscv_vle8_v_i8m2(&current_weights[col], vl);

        vint16m4_t v_in16 = __riscv_vsext_vf2_i16m4(v_in8, vl);
        vint16m4_t v_w16 = __riscv_vsext_vf2_i16m4(v_w8, vl);

        v_acc32 = __riscv_vwmacc_vv_i32m8(v_acc32, v_in16, v_w16, vl);

        col += vl;
        col_left -= vl;
      }

      vint32m1_t v_zero = __riscv_vmv_s_x_i32m1(0, vlmax);
      vint32m1_t v_red = __riscv_vredsum_vs_i32m8_i32m1(v_acc32, v_zero, vlmax);

      scratch[row] = acc + __riscv_vmv_x_s_i32m1_i32(v_red);
    }

    int row_left = n_output;
    int row = 0;
    while (row_left > 0) {
      size_t vl = __riscv_vsetvl_e32m4(row_left);

      vint32m4_t v_acc = __riscv_vle32_v_i32m4(&scratch[row], vl);
      v_acc = VectorMultiplyByQuantizedMultiplier(v_acc, multiplier, shift, vl);

      v_acc = __riscv_vadd_vx_i32m4(v_acc, output_zp, vl);

      int16_t *current_output = &output[batch * n_output + row];
      vint16m2_t v_out_old = __riscv_vle16_v_i16m2(current_output, vl);
      vint32m4_t v_out_old_32 = __riscv_vsext_vf2_i32m4(v_out_old, vl);
      v_acc = __riscv_vadd_vv_i32m4(v_acc, v_out_old_32, vl);

      vint16m2_t v_out_new =
          __riscv_vnclip_wx_i16m2(v_acc, 0, __RISCV_VXRM_RNU, vl);

      __riscv_vse16_v_i16m2(current_output, v_out_new, vl);

      row += vl;
      row_left -= vl;
    }
  }
}

void RISCVMatrixBatchVectorMultiplyAccumulate(
    const int8_t *input, const int32_t *bias,
    const int8_t *input_to_gate_weights, int32_t multiplier, int32_t shift,
    int32_t n_batch, int32_t n_input, int32_t n_output, int32_t output_zp,
    int32_t *scratch, int8_t *output, CpuBackendContext *context) {

  for (int batch = 0; batch < n_batch; ++batch) {
    const int8_t *current_input = input + batch * n_input;

    for (int row = 0; row < n_output; ++row) {
      int32_t acc = bias[row];
      const int8_t *current_weights = input_to_gate_weights + row * n_input;

      size_t vlmax = __riscv_vsetvlmax_e32m8();
      vint32m8_t v_acc32 = __riscv_vmv_v_x_i32m8(0, vlmax);

      int col_left = n_input;
      int col = 0;
      while (col_left > 0) {
        size_t vl = __riscv_vsetvl_e8m2(col_left);

        vint8m2_t v_in8 = __riscv_vle8_v_i8m2(&current_input[col], vl);
        vint8m2_t v_w8 = __riscv_vle8_v_i8m2(&current_weights[col], vl);

        vint16m4_t v_in16 = __riscv_vsext_vf2_i16m4(v_in8, vl);
        vint16m4_t v_w16 = __riscv_vsext_vf2_i16m4(v_w8, vl);

        v_acc32 = __riscv_vwmacc_vv_i32m8(v_acc32, v_in16, v_w16, vl);

        col += vl;
        col_left -= vl;
      }

      vint32m1_t v_zero = __riscv_vmv_s_x_i32m1(0, vlmax);
      vint32m1_t v_red = __riscv_vredsum_vs_i32m8_i32m1(v_acc32, v_zero, vlmax);

      scratch[row] = acc + __riscv_vmv_x_s_i32m1_i32(v_red);
    }

    int row_left = n_output;
    int row = 0;
    while (row_left > 0) {
      size_t vl = __riscv_vsetvl_e32m4(row_left);

      vint32m4_t v_acc = __riscv_vle32_v_i32m4(&scratch[row], vl);

      v_acc = VectorMultiplyByQuantizedMultiplier(v_acc, multiplier, shift, vl);

      v_acc = __riscv_vadd_vx_i32m4(v_acc, output_zp, vl);

      int8_t *current_output = &output[batch * n_output + row];
      vint8m1_t v_out_old = __riscv_vle8_v_i8m1(current_output, vl);
      vint32m4_t v_out_old_32 = __riscv_vsext_vf4_i32m4(v_out_old, vl);

      v_acc = __riscv_vadd_vv_i32m4(v_acc, v_out_old_32, vl);

      vint16m2_t v_acc16 =
          __riscv_vnclip_wx_i16m2(v_acc, 0, __RISCV_VXRM_RNU, vl);
      vint8m1_t v_out_new =
          __riscv_vnclip_wx_i8m1(v_acc16, 0, __RISCV_VXRM_RNU, vl);

      __riscv_vse8_v_i8m1(current_output, v_out_new, vl);

      row += vl;
      row_left -= vl;
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

void RISCVSparseMatrixBatchVectorMultiplyAccumulate1x16(
    const int8_t *__restrict__ matrix, const int32_t *__restrict__ segments,
    const int32_t *__restrict__ indices, int m_rows, int m_cols,
    const int8_t *__restrict__ vector, const int32_t *__restrict__ bias_vector,
    int n_batch, const int32_t input_offset, const int32_t output_multiplier,
    const int32_t output_shift, const int32_t *per_channel_scale,
    const int32_t *per_channel_shift, const int32_t output_offset,
    const int32_t output_activation_min, const int32_t output_activation_max,
    int8_t *__restrict__ result) {
  const int kBlockSize = 16;
  TFLITE_DCHECK_EQ(m_cols % kBlockSize, 0);

  for (int batch = 0; batch < n_batch; ++batch) {
    // matrix_ptr 针对每一个 batch 都会重置，因为稀疏矩阵数据是不变的
    const int8_t *matrix_ptr = matrix;

    for (int row = 0; row < m_rows; ++row) {
      int32_t dot_prod = 0;
      const int8_t *vector_in_batch = vector + batch * m_cols;

      for (int i = segments[row]; i < segments[row + 1]; ++i) {
        const int block_start_index = indices[i] * kBlockSize;
        const int8_t *vector_block_in_batch_ptr =
            vector_in_batch + block_start_index;

        int c = 0;
        while (c < kBlockSize) {
          size_t vl = __riscv_vsetvl_e8m1(kBlockSize - c);

          vint8m1_t vm = __riscv_vle8_v_i8m1(matrix_ptr, vl);
          vint8m1_t vv = __riscv_vle8_v_i8m1(vector_block_in_batch_ptr, vl);

          vint16m2_t vm16 = __riscv_vsext_vf2_i16m2(vm, vl);
          vint16m2_t vv16 = __riscv_vsext_vf2_i16m2(vv, vl);

          // 根据分配律: M * V + M * Offset = M * (V + Offset)
          // 转换为 16-bit 后相加避免了 8-bit 溢出
          vv16 = __riscv_vadd_vx_i16m2(vv16, static_cast<int16_t>(input_offset),
                                       vl);

          // 4. 宽度扩展乘法: 16-bit * 16-bit -> 32-bit (Vector Widening
          // Multiply)
          vint32m4_t v_prod = __riscv_vwmul_vv_i32m4(vm16, vv16, vl);

          // 5. 向量规约求和 (Vector Reduction Sum)
          vint32m1_t v_red_init =
              __riscv_vmv_v_x_i32m1(0, vl); // 初始化归约标量为 0
          vint32m1_t v_red =
              __riscv_vredsum_vs_i32m4_i32m1(v_prod, v_red_init, vl);

          // 将规约得到的寄存器首元素提取到普通标量并累加
          dot_prod += __riscv_vmv_x_s_i32m1_i32(v_red);

          // 指针步进
          matrix_ptr += vl;
          vector_block_in_batch_ptr += vl;
          c += vl;
        }
      }

      const int32_t bias_value = bias_vector != nullptr ? bias_vector[row] : 0;
      dot_prod = MultiplyByQuantizedMultiplier(
          dot_prod + bias_value,
          per_channel_scale ? per_channel_scale[row] : output_multiplier,
          per_channel_shift ? per_channel_shift[row] : output_shift);
      dot_prod += output_offset;

      result[batch * m_rows + row] =
          static_cast<int8_t>(ActivationFunctionWithMinMax(
              dot_prod, output_activation_min, output_activation_max));
    }
  }
}

void RISCVSparseMatrixBatchVectorMultiplyAccumulate(
    const float *__restrict__ matrix, const uint8_t *__restrict__ ledger,
    int m_rows, int m_cols, const float *__restrict__ vector, int n_batch,
    float *__restrict__ result) {
  const int kBlockSize = 16;

  for (int batch = 0; batch < n_batch; batch++) {
    const float *vector_in_batch = vector + batch * m_cols;

    // ✅ 修复1：对于每个 batch，稀疏矩阵和 ledger 的指针必须重置到开头
    const uint8_t *ledger_ptr = ledger;
    const float *matrix_ptr = matrix;

    for (int row = 0; row < m_rows; row++) {
      // ✅ 修复2：因为我们固定按 kBlockSize (16) 处理，统一使用 vl=16
      size_t vl = __riscv_vsetvl_e32m4(kBlockSize);
      vfloat32m4_t v_dotprod = __riscv_vfmv_v_f_f32m4(0.0f, vl);

      int num_nonzero_blocks = *ledger_ptr++;
      if (num_nonzero_blocks > 0) {
        for (int i = 0; i < num_nonzero_blocks; i++) {
          uint8_t original_col_block_index = *ledger_ptr++;
          const int block_start_index = original_col_block_index * kBlockSize;

          const float *vector_block_in_batch_ptr =
              vector_in_batch + block_start_index;

          vfloat32m4_t v_vec =
              __riscv_vle32_v_f32m4(vector_block_in_batch_ptr, vl);
          vfloat32m4_t v_mat = __riscv_vle32_v_f32m4(matrix_ptr, vl);

          matrix_ptr += kBlockSize;

          v_dotprod = __riscv_vfmacc_vv_f32m4(v_dotprod, v_mat, v_vec, vl);
        }
      }

      // ✅ 修复3：规约求和必须使用真实的计算长度 vl，而不是 vlmax
      vfloat32m1_t v_res_scalar = __riscv_vfmv_v_f_f32m1(0.0f, vl);
      v_res_scalar =
          __riscv_vfredusum_vs_f32m4_f32m1(v_dotprod, v_res_scalar, vl);
      float dot_prod_scalar = __riscv_vfmv_f_s_f32m1_f32(v_res_scalar);

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
    vint32m8_t v_shifted =
        __riscv_vssra_vx_i32m8(v_value, shift, __RISCV_VXRM_RNU, vl);
    vint16m4_t v_out = __riscv_vncvt_x_x_w_i16m4(v_shifted, vl);

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

void RISCVBatchVectorBatchVectorDotProduct(const int16_t *__restrict__ vector1,
                                           const int16_t *__restrict__ vector2,
                                           int v_size, int n_batch,
                                           int32_t *__restrict__ result) {
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

      // 这里理论上可以用MultiplyByQuantizedMultiplier
      // 但是类型很难统一，先暂时不提换
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

void RISCVReductionSumVector(const float *input_vector, float *output_vector,
                             int output_size, int reduction_size) {
  for (int o = 0; o < output_size; o++) {
    int r = reduction_size;
    int i = 0;

    vfloat32m8_t v_sum_accumulator =
        __riscv_vfmv_v_f_f32m8(0.0f, __riscv_vsetvlmax_e32m8());

    while (r > 0) {
      size_t vl = __riscv_vsetvl_e32m8(r);
      vfloat32m8_t v_src = __riscv_vle32_v_f32m8(input_vector + i, vl);

      v_sum_accumulator = __riscv_vfadd_vv_f32m8(v_sum_accumulator, v_src, vl);

      i += vl;
      r -= vl;
    }

    vfloat32m1_t v_scalar_sum = __riscv_vfmv_v_f_f32m1(0.0f, 1);

    v_scalar_sum = __riscv_vfredusum_vs_f32m8_f32m1(
        v_sum_accumulator, v_scalar_sum, __riscv_vsetvlmax_e32m8());

    output_vector[o] = __riscv_vfmv_f_s_f32m1_f32(v_scalar_sum);

    input_vector += reduction_size;
  }
}

void RISCVReductionSumVector(const int32_t *input_vector,
                             int32_t *output_vector, int output_size,
                             int reduction_size) {
  for (int o = 0; o < output_size; o++) {
    int r = reduction_size;
    int i = 0;

    vint32m8_t v_sum_accumulator =
        __riscv_vmv_v_x_i32m8(0, __riscv_vsetvlmax_e32m8());

    while (r > 0) {
      size_t vl = __riscv_vsetvl_e32m8(r);
      vint32m8_t v_src = __riscv_vle32_v_i32m8(input_vector + i, vl);

      v_sum_accumulator = __riscv_vadd_vv_i32m8(v_sum_accumulator, v_src, vl);

      i += vl;
      r -= vl;
    }

    vint32m1_t v_scalar_sum = __riscv_vmv_v_x_i32m1(0, 1);
    v_scalar_sum = __riscv_vredsum_vs_i32m8_i32m1(
        v_sum_accumulator, v_scalar_sum, __riscv_vsetvlmax_e32m8());

    output_vector[o] = __riscv_vmv_x_s_i32m1_i32(v_scalar_sum);

    input_vector += reduction_size;
  }
}

void RISCVReductionSumVector(const int8_t *input_vector, int32_t *output_vector,
                             int output_size, int reduction_size) {
  for (int o = 0; o < output_size; o++) {
    int r = reduction_size;
    int i = 0;

    vint32m8_t v_sum_accumulator =
        __riscv_vmv_v_x_i32m8(0, __riscv_vsetvlmax_e32m8());

    while (r > 0) {
      size_t vl = __riscv_vsetvl_e8m2(r);
      vint8m2_t v_src = __riscv_vle8_v_i8m2(input_vector + i, vl);
      vint16m4_t v_src_i16 = __riscv_vsext_vf2_i16m4(v_src, vl);
      v_sum_accumulator =
          __riscv_vwadd_wv_i32m8(v_sum_accumulator, v_src_i16, vl);

      i += vl;
      r -= vl;
    }

    vint32m1_t v_scalar_sum = __riscv_vmv_v_x_i32m1(0, 1);
    v_scalar_sum = __riscv_vredsum_vs_i32m8_i32m1(
        v_sum_accumulator, v_scalar_sum, __riscv_vsetvlmax_e32m8());

    output_vector[o] = __riscv_vmv_x_s_i32m1_i32(v_scalar_sum);

    input_vector += reduction_size;
  }
}

void RISCVMeanStddevNormalization(const float *__restrict__ input_vector,
                                  float *__restrict__ output_vector, int v_size,
                                  int n_batch) {
  constexpr float kNormalizationConstant = 1e-8f;

  for (int batch = 0; batch < n_batch; ++batch) {
    int r = v_size;
    int i = 0;

    vfloat32m8_t v_sum_acc =
        __riscv_vfmv_v_f_f32m8(0.0f, __riscv_vsetvlmax_e32m8());

    while (r > 0) {
      size_t vl = __riscv_vsetvl_e32m8(r);
      vfloat32m8_t v_src = __riscv_vle32_v_f32m8(input_vector + i, vl);

      v_sum_acc = __riscv_vfadd_vv_f32m8(v_sum_acc, v_src, vl);

      i += vl;
      r -= vl;
    }

    vfloat32m1_t v_scalar_sum = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    v_scalar_sum = __riscv_vfredusum_vs_f32m8_f32m1(v_sum_acc, v_scalar_sum,
                                                    __riscv_vsetvlmax_e32m8());

    const float mean = __riscv_vfmv_f_s_f32m1_f32(v_scalar_sum) / v_size;

    r = v_size;
    i = 0;

    vfloat32m8_t v_sq_diff_acc =
        __riscv_vfmv_v_f_f32m8(0.0f, __riscv_vsetvlmax_e32m8());

    while (r > 0) {
      size_t vl = __riscv_vsetvl_e32m8(r);
      vfloat32m8_t v_src = __riscv_vle32_v_f32m8(input_vector + i, vl);

      vfloat32m8_t v_diff = __riscv_vfsub_vf_f32m8(v_src, mean, vl);

      v_sq_diff_acc =
          __riscv_vfmacc_vv_f32m8(v_sq_diff_acc, v_diff, v_diff, vl);

      i += vl;
      r -= vl;
    }

    vfloat32m1_t v_scalar_sq_diff = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    v_scalar_sq_diff = __riscv_vfredusum_vs_f32m8_f32m1(
        v_sq_diff_acc, v_scalar_sq_diff, __riscv_vsetvlmax_e32m8());

    const float variance =
        __riscv_vfmv_f_s_f32m1_f32(v_scalar_sq_diff) / v_size;
    const float stddev_inv =
        1.0f / std::sqrt(variance + kNormalizationConstant);

    r = v_size;
    i = 0;

    while (r > 0) {
      size_t vl = __riscv_vsetvl_e32m8(r);
      vfloat32m8_t v_src = __riscv_vle32_v_f32m8(input_vector + i, vl);

      vfloat32m8_t v_res = __riscv_vfsub_vf_f32m8(v_src, mean, vl);

      v_res = __riscv_vfmul_vf_f32m8(v_res, stddev_inv, vl);

      __riscv_vse32_v_f32m8(output_vector + i, v_res, vl);

      i += vl;
      r -= vl;
    }

    input_vector += v_size;
    output_vector += v_size;
  }
}

void RISCVApplyLayerNorm(const int16_t *input,
                         const int16_t *layer_norm_weights, const int32_t *bias,
                         int32_t layer_norm_scale_a, int32_t layer_norm_scale_b,
                         int32_t variance_limit, int n_batch, int n_input,
                         int16_t *output) {
  static const int kTwoToPower20 = 1 << 20;
  for (int i = 0; i < n_batch; ++i) {
    const int16_t *current_input = input + i * n_input;
    int16_t *current_output = output + i * n_input;

    size_t vl;
    vint64m8_t v_sum = __riscv_vmv_v_x_i64m8(0, __riscv_vsetvlmax_e64m8());
    vint64m8_t v_sum_sq = __riscv_vmv_v_x_i64m8(0, __riscv_vsetvlmax_e64m8());

    for (int j = 0; j < n_input; j += vl) {
      vl = __riscv_vsetvl_e16m2(n_input - j);

      vint16m2_t v_in16 = __riscv_vle16_v_i16m2(&current_input[j], vl);
      vint32m4_t v_in32 = __riscv_vsext_vf2_i32m4(v_in16, vl);

      v_sum = __riscv_vwadd_wv_i64m8(v_sum, v_in32, vl);
      v_sum_sq = __riscv_vwmacc_vv_i64m8(v_sum_sq, v_in32, v_in32, vl);
    }

    vl = __riscv_vsetvlmax_e64m8();
    vint64m1_t v_red_start = __riscv_vmv_s_x_i64m1(0, 1);
    vint64m1_t v_red_sum =
        __riscv_vredsum_vs_i64m8_i64m1(v_sum, v_red_start, vl);
    int64_t sum = __riscv_vmv_x_s_i64m1_i64(v_red_sum);

    vint64m1_t v_red_sum_sq =
        __riscv_vredsum_vs_i64m8_i64m1(v_sum_sq, v_red_start, vl);
    int64_t sum_sq = __riscv_vmv_x_s_i64m1_i64(v_red_sum_sq);

    int32_t mean = static_cast<int32_t>(sum * 1024 / n_input);
    int32_t temp = kTwoToPower20 / n_input;
    int64_t variance = sum_sq * temp - static_cast<int64_t>(mean) * mean;
    int32_t variance2 = static_cast<int32_t>(variance / kTwoToPower20);
    if (variance2 < 1)
      variance2 = variance_limit;

    int32_t stddev_inverse_a;
    int stddev_inverse_b;
    GetInvSqrtQuantizedMultiplierExp(variance2, -1, &stddev_inverse_a,
                                     &stddev_inverse_b);

    for (int j = 0; j < n_input; j += vl) {
      vl = __riscv_vsetvl_e16m2(n_input - j);

      vint16m2_t v_in16 = __riscv_vle16_v_i16m2(&current_input[j], vl);
      vint32m4_t v_val = __riscv_vsext_vf2_i32m4(v_in16, vl);

      // shifted = 1024 * val - mean
      vint32m4_t v_shifted = __riscv_vmul_vx_i32m4(v_val, 1024, vl);
      v_shifted = __riscv_vsub_vx_i32m4(v_shifted, mean, vl);

      // rescaled
      vint32m4_t v_rescaled = VectorMultiplyByQuantizedMultiplier(
          v_shifted, stddev_inverse_a, stddev_inverse_b, vl);

      // val3 = rescaled * layer_norm_weights[j] + bias[j]
      vint16m2_t v_w16 = __riscv_vle16_v_i16m2(&layer_norm_weights[j], vl);
      vint32m4_t v_w32 = __riscv_vsext_vf2_i32m4(v_w16, vl);
      vint32m4_t v_bias = __riscv_vle32_v_i32m4(&bias[j], vl);

      //  int32 * int32 -> int64
      vint64m8_t v_val3 = __riscv_vwmul_vv_i64m8(v_rescaled, v_w32, vl);
      //  int64 + int32 -> int64
      v_val3 = __riscv_vwadd_wv_i64m8(v_val3, v_bias, vl);

      //  (val3 > 0 ? val3 + 512 : val3 - 512) / 1024
      vint64m8_t v_sign = __riscv_vsra_vx_i64m8(v_val3, 63, vl);
      vint64m8_t v_adjusted = __riscv_vadd_vx_i64m8(v_val3, 512, vl);

      // 3. 再加上符号位（正数加了 0，负数加了 -1 以修正右移截断误差）
      v_adjusted = __riscv_vadd_vv_i64m8(v_adjusted, v_sign, vl);

      // 4. 算术右移 10 位，完美等价于 C++ 的有条件除以 1024
      vint64m8_t v_rounded_i64 = __riscv_vsra_vx_i64m8(v_adjusted, 10, vl);

      vint32m4_t v_val4 = __riscv_vncvt_x_x_w_i32m4(v_rounded_i64, vl);
      vint32m4_t v_val5 = VectorMultiplyByQuantizedMultiplier(
          v_val4, layer_norm_scale_a, layer_norm_scale_b + 12, vl);

      vint16m2_t v_out =
          __riscv_vnclip_wx_i16m2(v_val5, 0, __RISCV_VXRM_RNU, vl);
      __riscv_vse16_v_i16m2(&current_output[j], v_out, vl);
    }
  }
}

void RISCVApplyLayerNormFloat(const int16_t *input,
                              const int16_t *layer_norm_weights,
                              int32_t layer_norm_scale_a,
                              int32_t layer_norm_scale_b, const int32_t *bias,
                              int n_batch, int n_input, int16_t *output) {
  // 离线预计算标量常数
  const float layer_norm_scale =
      layer_norm_scale_a *
      std::pow(2.0, static_cast<double>(layer_norm_scale_b - 31));
  const float bias_scale =
      static_cast<float>(std::pow(2.0, -10)) * layer_norm_scale;
  const float re_quant_scale = layer_norm_scale * 4096.0f; // 4096 即 2^12
  const float bias_re_quant_scale = bias_scale * 4096.0f;

  for (int batch = 0; batch < n_batch; ++batch) {
    const int16_t *current_input = input + batch * n_input;
    int16_t *current_output = output + batch * n_input;

    // --- 第一阶段：计算 sum 和 sum_sq ---
    size_t vlmax = __riscv_vsetvlmax_e32m8();
    vfloat32m8_t v_sum_acc = __riscv_vfmv_v_f_f32m8(0.0f, vlmax);
    vfloat32m8_t v_sum_sq_acc = __riscv_vfmv_v_f_f32m8(0.0f, vlmax);

    size_t vl;
    for (int i = 0; i < n_input; i += vl) {
      vl = __riscv_vsetvl_e16m4(n_input - i);

      vint16m4_t v_in16 = __riscv_vle16_v_i16m4(&current_input[i], vl);
      vint32m8_t v_in32 = __riscv_vsext_vf2_i32m8(v_in16, vl);

      // 【修复1】正确的 int32 到 float32 转换指令
      vfloat32m8_t v_f32 = __riscv_vfcvt_f_x_v_f32m8(v_in32, vl);

      v_sum_acc = __riscv_vfadd_vv_f32m8(v_sum_acc, v_f32, vl);
      v_sum_sq_acc = __riscv_vfmacc_vv_f32m8(v_sum_sq_acc, v_f32, v_f32, vl);
    }

    vfloat32m1_t v_zero = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    vfloat32m1_t v_red_sum =
        __riscv_vfredusum_vs_f32m8_f32m1(v_sum_acc, v_zero, vlmax);
    vfloat32m1_t v_red_sum_sq =
        __riscv_vfredusum_vs_f32m8_f32m1(v_sum_sq_acc, v_zero, vlmax);

    float sum = __riscv_vfmv_f_s_f32m1_f32(v_red_sum);
    float sum_sq = __riscv_vfmv_f_s_f32m1_f32(v_red_sum_sq);

    float mean = sum / n_input;
    float variance = sum_sq / n_input - mean * mean;
    float stddev_inv = (variance <= 0.0f) ? (1.0f / std::sqrt(1e-8f))
                                          : (1.0f / std::sqrt(variance));

    // --- 第二阶段：归一化、放射变换与截断输出 ---
    for (int i = 0; i < n_input; i += vl) {
      vl = __riscv_vsetvl_e16m4(n_input - i);

      vint16m4_t v_in16 = __riscv_vle16_v_i16m4(&current_input[i], vl);
      vint32m8_t v_in32 = __riscv_vsext_vf2_i32m8(v_in16, vl);
      vfloat32m8_t v_f32 = __riscv_vfcvt_f_x_v_f32m8(v_in32, vl); // 【修复1】

      vfloat32m8_t v_norm = __riscv_vfsub_vf_f32m8(v_f32, mean, vl);
      v_norm = __riscv_vfmul_vf_f32m8(v_norm, stddev_inv, vl);

      vint16m4_t v_w16 = __riscv_vle16_v_i16m4(&layer_norm_weights[i], vl);
      vint32m8_t v_w32 = __riscv_vsext_vf2_i32m8(v_w16, vl);
      vfloat32m8_t v_w32_f = __riscv_vfcvt_f_x_v_f32m8(v_w32, vl); // 【修复1】

      vfloat32m8_t v_res = __riscv_vfmul_vv_f32m8(v_norm, v_w32_f, vl);
      v_res = __riscv_vfmul_vf_f32m8(v_res, re_quant_scale, vl);

      vint32m8_t v_bias32 = __riscv_vle32_v_i32m8(&bias[i], vl);
      vfloat32m8_t v_bias32_f =
          __riscv_vfcvt_f_x_v_f32m8(v_bias32, vl); // 【修复1】

      v_res =
          __riscv_vfmacc_vf_f32m8(v_res, bias_re_quant_scale, v_bias32_f, vl);

      // 使用 __RISCV_FRM_RMM (Round to Max Magnitude)，完美等价于 C++ 标准库的
      // std::round()
      vint32m8_t v_quant32 =
          __riscv_vfcvt_x_f_v_i32m8_rm(v_res, __RISCV_FRM_RMM, vl);

      vint16m4_t v_out16 =
          __riscv_vnclip_wx_i16m4(v_quant32, 0, __RISCV_VXRM_RNU, vl);

      __riscv_vse16_v_i16m4(&current_output[i], v_out16, vl);
    }
  }
}

void RISCVMatrixScalarMultiplyAccumulate(const int8_t *matrix, int32_t scalar,
                                         int32_t n_row, int32_t n_col,
                                         int32_t *output) {
  for (int i = 0; i < n_row; ++i) {
    int32_t row_sum = 0;
    int remaining_col = n_col;
    const int8_t *current_row = matrix + i * n_col;

    size_t vl;
    for (int j = 0; j < n_col; j += vl) {
      vl = __riscv_vsetvl_e8m2(remaining_col);

      vint8m2_t v_in8 = __riscv_vle8_v_i8m2(current_row, vl);

      vint32m8_t v_in32 = __riscv_vsext_vf4_i32m8(v_in8, vl);

      vint32m1_t v_zero = __riscv_vmv_s_x_i32m1(0, vl);
      vint32m1_t v_sum = __riscv_vredsum_vs_i32m8_i32m1(v_in32, v_zero, vl);

      row_sum += __riscv_vmv_x_s_i32m1_i32(v_sum);

      current_row += vl;
      remaining_col -= vl;
    }

    output[i] += row_sum * scalar;
  }
}

void RISCVSparseMatrixBatchVectorMultiplyAccumulate(
    const int8_t *__restrict__ matrix, const uint8_t *ledger, const int m_rows,
    const int m_cols, const int8_t *__restrict__ vectors,
    const float *scaling_factors, int n_batch, float *__restrict__ result,
    const float *per_channel_scale) {

  static const int kBlockSize = 16;

  for (int batch = 0; batch < n_batch; ++batch, vectors += m_cols) {
    const float batch_scaling_factor = scaling_factors[batch];
    const uint8_t *ledger_ptr = ledger;
    const int8_t *row_ptr = matrix;

    for (int row = 0; row < m_rows; ++row) {
      size_t vlmax = __riscv_vsetvlmax_e32m8();
      vint32m8_t v_acc = __riscv_vmv_v_x_i32m8(0, vlmax);

      int num_nonzero_blocks = *ledger_ptr++;

      for (int i = 0; i < num_nonzero_blocks; i++) {
        const int block_start_index = *ledger_ptr++ * kBlockSize;
        const int8_t *vector_block_ptr = vectors + block_start_index;

        int c = kBlockSize;
        while (c > 0) {
          // e8m2 配置：占用 2 组寄存器读 int8，
          // 这样后续拓宽到 int16(m4) 和 int32(m8) 时才装得下
          size_t vl = __riscv_vsetvl_e8m2(c);

          vint8m2_t v_row = __riscv_vle8_v_i8m2(row_ptr, vl);
          vint8m2_t v_vec = __riscv_vle8_v_i8m2(vector_block_ptr, vl);

          // 第一级拓宽：把 8 位符号扩展成 16 位
          vint16m4_t v_row16 = __riscv_vsext_vf2_i16m4(v_row, vl);
          vint16m4_t v_vec16 = __riscv_vsext_vf2_i16m4(v_vec, vl);

          // 第二级拓宽与乘累加：16位 * 16位 + 32位 -> 32位累加器
          v_acc = __riscv_vwmacc_vv_i32m8(v_acc, v_row16, v_vec16, vl);

          row_ptr += vl;
          vector_block_ptr += vl;
          c -= vl;
        } // while block
      } // for num_nonzero_blocks

      // 2. 循环外水平规约求和 (Reduction)
      vint32m1_t v_zero = __riscv_vmv_s_x_i32m1(0, vlmax);
      vint32m1_t v_red = __riscv_vredsum_vs_i32m8_i32m1(v_acc, v_zero, vlmax);

      int32_t dotprod = __riscv_vmv_x_s_i32m1_i32(v_red);

      // 3. 浮点反量化缩放
      float scaling_factor = batch_scaling_factor;
      if (per_channel_scale) {
        scaling_factor *= per_channel_scale[row];
      }
      result[batch * m_rows + row] += dotprod * scaling_factor;

    } // for row
  } // for batch
}
} // namespace tensor_utils
} // namespace tflite

#endif // USE_RISCV
