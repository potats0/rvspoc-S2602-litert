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

} // namespace tensor_utils
} // namespace tflite

#endif // USE_RISCV
