/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_KERNELS_CPU_BACKEND_RISCV_H_
#define TENSORFLOW_LITE_KERNELS_CPU_BACKEND_RISCV_H_

#include "tflite/kernels/cpu_backend_context.h"
#include "tflite/kernels/cpu_backend_gemm_params.h"
#include "tflite/kernels/internal/common.h"
#include "tflite/kernels/internal/compatibility.h"
#include "tflite/kernels/internal/optimized/cpu_check.h"
#include <cstdint>

namespace tflite {
namespace cpu_backend_gemm {
namespace detail {

#ifdef USE_RISCV
#include <riscv_vector.h>
#define RVV_MR 3

namespace {
template <typename Scalar>
inline std::unique_ptr<Scalar[]> TransposeColToRowMajor(const Scalar *src_data,
                                                        int rows, int cols) {
  std::unique_ptr<Scalar[]> dst_buf(new Scalar[rows * cols]);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      // 读 Col-Major，写 Row-Major
      dst_buf[r * cols + c] = src_data[c * rows + r];
    }
  }
  return dst_buf;
}

template <typename Scalar>
inline std::unique_ptr<Scalar[]> TransposeRowToColMajor(const Scalar *src_data,
                                                        int rows, int cols) {
  std::unique_ptr<Scalar[]> dst_buf(new Scalar[rows * cols]);
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      // 读 Row-Major，写 Col-Major
      dst_buf[c * rows + r] = src_data[r * cols + c];
    }
  }
  return dst_buf;
}
} // namespace

template <QuantizationFlavor Flavor, typename DstScalar> struct RuyQuantizer {
  static_assert(sizeof(DstScalar) < sizeof(int32_t) ||
                    Flavor == QuantizationFlavor::kFloatingPoint,
                "Quantization is only for downcasting int32 to smaller types.");

  static inline int32_t
  Apply(int32_t acc, int row_idx,
        const GemmParams<int32_t, DstScalar, Flavor> &params,
        int32_t dst_zero_point) {
    if constexpr (Flavor == QuantizationFlavor::kFloatingPoint) {
      return acc;
    } else {
      int32_t multiplier = 0;
      int32_t exponent = 0;

      if constexpr (Flavor ==
                    QuantizationFlavor::kIntegerWithUniformMultiplier) {
        multiplier = params.multiplier_fixedpoint;
        exponent = params.multiplier_exponent;
      } else if constexpr (Flavor ==
                           QuantizationFlavor::kIntegerWithPerRowMultiplier) {
        multiplier = params.multiplier_fixedpoint_perchannel[row_idx];
        exponent = params.multiplier_exponent_perchannel[row_idx];
      }

      acc = MultiplyByQuantizedMultiplier(acc, multiplier, exponent);

      acc += dst_zero_point;
      acc = std::max<int32_t>(params.clamp_min, acc);
      acc = std::min<int32_t>(params.clamp_max, acc);

      return acc;
    }
  }
};

template <typename LhsScalar, typename RhsScalar, typename AccumScalar,
          typename DstScalar, QuantizationFlavor quantization_flavor>
struct GemmImplRISCV {
  static void
  Run(const MatrixParams<LhsScalar> &lhs_params, const LhsScalar *lhs_data,
      const MatrixParams<RhsScalar> &rhs_params, const RhsScalar *rhs_data,
      const MatrixParams<DstScalar> &dst_params, DstScalar *dst_data,
      const GemmParams<AccumScalar, DstScalar, quantization_flavor> &params,
      CpuBackendContext *context) {
    // static_assert(sizeof(LhsScalar) == 0, "Unsupported GEMM Types for
    // RISC-V!");
  }
};

template <>
struct GemmImplRISCV<float, float, float, float,
                     QuantizationFlavor::kFloatingPoint> {
  static void Run(const MatrixParams<float> &lhs_params, const float *lhs_data,
                  const MatrixParams<float> &rhs_params, const float *rhs_data,
                  const MatrixParams<float> &dst_params, float *dst_data,
                  const GemmParams<float, float,
                                   QuantizationFlavor::kFloatingPoint> &params,
                  CpuBackendContext *context) {
    std::unique_ptr<float[]> lhs_buf;
    const float *final_lhs_data = lhs_data;

    // 如果左矩阵是列优先，当然实际情况很低
    if (lhs_params.order == Order::kColMajor) {
      lhs_buf = TransposeColToRowMajor<float>(lhs_data, lhs_params.rows,
                                              lhs_params.cols);
      final_lhs_data = lhs_buf.get();
    }

    const float *final_rhs_data = rhs_data;
    std::unique_ptr<float[]> rhs_buf;

    // 右矩阵如果是行优先（默认），就要弄成列优先，cache友好
    if (rhs_params.order == Order::kRowMajor) {
      rhs_buf = TransposeRowToColMajor<float>(rhs_data, rhs_params.rows,
                                              rhs_params.cols);
      final_rhs_data = rhs_buf.get();
    }
    int M = lhs_params.rows;
    int K = lhs_params.cols; // K 也是 rhs_params.rows (内积维度)
    int N = rhs_params.cols;

    // 最外层循环：遍历右矩阵的每一列
    for (int c = 0; c < N; ++c) {
      const float *rhs_col_ptr = final_rhs_data + c * K;
      int r = 0;

      for (; r <= M - RVV_MR; r += RVV_MR) {
        int k_offset = 0;
        int k_left = K;

        size_t vlmax = __riscv_vsetvlmax_e32m4();

        vfloat32m4_t v_acc0 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
        vfloat32m4_t v_acc1 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);
        vfloat32m4_t v_acc2 = __riscv_vfmv_v_f_f32m4(0.0f, vlmax);

        // 提取 4 行左矩阵的起始指针
        const float *lhs_row0 = final_lhs_data + (r + 0) * K;
        const float *lhs_row1 = final_lhs_data + (r + 1) * K;
        const float *lhs_row2 = final_lhs_data + (r + 2) * K;

        while (k_left > 0) {
          size_t vl = __riscv_vsetvl_e32m4(k_left);

          vfloat32m4_t v_rhs_col =
              __riscv_vle32_v_f32m4(rhs_col_ptr + k_offset, vl);

          // 分别读取左矩阵的 4 行数据
          vfloat32m4_t v_lhs0 = __riscv_vle32_v_f32m4(lhs_row0 + k_offset, vl);
          vfloat32m4_t v_lhs1 = __riscv_vle32_v_f32m4(lhs_row1 + k_offset, vl);
          vfloat32m4_t v_lhs2 = __riscv_vle32_v_f32m4(lhs_row2 + k_offset, vl);

          // 复用 v_rhs_col，执行 4 次乘加
          v_acc0 = __riscv_vfmacc_vv_f32m4_tu(v_acc0, v_lhs0, v_rhs_col, vl);
          v_acc1 = __riscv_vfmacc_vv_f32m4_tu(v_acc1, v_lhs1, v_rhs_col, vl);
          v_acc2 = __riscv_vfmacc_vv_f32m4_tu(v_acc2, v_lhs2, v_rhs_col, vl);

          k_offset += vl;
          k_left -= vl;
        }

        // K 维度遍历结束，将 4 个向量寄存器规约求和
        vfloat32m1_t v_zero = __riscv_vfmv_v_f_f32m1(0.0f, vlmax);
        float sum0 = __riscv_vfmv_f_s_f32m1_f32(
            __riscv_vfredusum_vs_f32m4_f32m1(v_acc0, v_zero, vlmax));
        float sum1 = __riscv_vfmv_f_s_f32m1_f32(
            __riscv_vfredusum_vs_f32m4_f32m1(v_acc1, v_zero, vlmax));
        float sum2 = __riscv_vfmv_f_s_f32m1_f32(
            __riscv_vfredusum_vs_f32m4_f32m1(v_acc2, v_zero, vlmax));

        if (params.bias) {
          sum0 += params.bias[r + 0];
          sum1 += params.bias[r + 1];
          sum2 += params.bias[r + 2];
        }
        sum0 = std::max(params.clamp_min, std::min(params.clamp_max, sum0));
        sum1 = std::max(params.clamp_min, std::min(params.clamp_max, sum1));
        sum2 = std::max(params.clamp_min, std::min(params.clamp_max, sum2));

        if (dst_params.order == Order::kColMajor) {
          dst_data[c * M + (r + 0)] = sum0;
          dst_data[c * M + (r + 1)] = sum1;
          dst_data[c * M + (r + 2)] = sum2;
        } else {
          dst_data[(r + 0) * N + c] = sum0;
          dst_data[(r + 1) * N + c] = sum1;
          dst_data[(r + 2) * N + c] = sum2;
        }
      }

      for (; r < M; ++r) {
        int k_offset = 0;
        int k_left = K;
        size_t vlmax = __riscv_vsetvlmax_e32m8();
        vfloat32m8_t v_acc = __riscv_vfmv_v_f_f32m8(0.0f, vlmax);
        const float *lhs_row_tail = final_lhs_data + r * K;

        while (k_left > 0) {
          size_t vl = __riscv_vsetvl_e32m8(k_left);
          vfloat32m8_t v_rhs_col =
              __riscv_vle32_v_f32m8(rhs_col_ptr + k_offset, vl);
          vfloat32m8_t v_lhs_row =
              __riscv_vle32_v_f32m8(lhs_row_tail + k_offset, vl);
          v_acc = __riscv_vfmacc_vv_f32m8_tu(v_acc, v_lhs_row, v_rhs_col, vl);

          k_offset += vl;
          k_left -= vl;
        }

        size_t vlmax_m1 = __riscv_vsetvlmax_e32m1();
        vfloat32m1_t v_zero = __riscv_vfmv_v_f_f32m1(0.0f, vlmax_m1);
        float sum = __riscv_vfmv_f_s_f32m1_f32(
            __riscv_vfredusum_vs_f32m8_f32m1(v_acc, v_zero, vlmax));

        if (params.bias)
          sum += params.bias[r];
        sum = std::max(params.clamp_min, std::min(params.clamp_max, sum));

        if (dst_params.order == Order::kColMajor) {
          dst_data[c * M + r] = sum;
        } else {
          dst_data[r * N + c] = sum;
        }
      }
    }
  }
};

template <typename DstScalar, typename AccumScalar,
          QuantizationFlavor quantization_flavor>
struct GemmImplRISCV<std::int8_t, std::int8_t, AccumScalar, DstScalar,
                     quantization_flavor> {
  static void
  Run(const MatrixParams<int8_t> &lhs_params, const int8_t *lhs_data,
      const MatrixParams<int8_t> &rhs_params, const int8_t *rhs_data,
      const MatrixParams<DstScalar> &dst_params, DstScalar *dst_data,
      const GemmParams<AccumScalar, DstScalar, quantization_flavor> &params,
      CpuBackendContext *context) {
    using Quantizer = RuyQuantizer<quantization_flavor, DstScalar>;

    std::unique_ptr<int8_t[]> lhs_buf;
    const int8_t *final_lhs_data = lhs_data;

    // 如果左矩阵是列优先，当然实际情况很低
    if (lhs_params.order == Order::kColMajor) {
      lhs_buf = TransposeColToRowMajor<int8_t>(lhs_data, lhs_params.rows,
                                               lhs_params.cols);
      final_lhs_data = lhs_buf.get();
    }

    const int8_t *final_rhs_data = rhs_data;
    std::unique_ptr<int8_t[]> rhs_buf;

    // 右矩阵如果是行优先（默认），就要弄成列优先，cache友好
    if (rhs_params.order == Order::kRowMajor) {
      rhs_buf = TransposeRowToColMajor<int8_t>(rhs_data, rhs_params.rows,
                                               rhs_params.cols);
      final_rhs_data = rhs_buf.get();
    }
    int M = lhs_params.rows;
    int K = lhs_params.cols; // K 也是 rhs_params.rows (内积维度)
    int N = rhs_params.cols;

    size_t vlmax = __riscv_vsetvlmax_e8m2();
    // 最外层循环：遍历右矩阵的每一列
    for (int c = 0; c < N; ++c) {
      const int8_t *rhs_col_ptr = final_rhs_data + c * K;
      int r = 0;

      for (; r <= M - RVV_MR; r += RVV_MR) {
        int k_offset = 0;
        int k_left = K;

        vint32m8_t v_acc0 = __riscv_vmv_v_x_i32m8(0, vlmax);
        vint32m8_t v_acc1 = __riscv_vmv_v_x_i32m8(0, vlmax);
        vint32m8_t v_acc2 = __riscv_vmv_v_x_i32m8(0, vlmax);

        const int8_t *lhs_row0 = final_lhs_data + (r + 0) * K;
        const int8_t *lhs_row1 = final_lhs_data + (r + 1) * K;
        const int8_t *lhs_row2 = final_lhs_data + (r + 2) * K;

        while (k_left > 0) {
          size_t vl = __riscv_vsetvl_e8m2(k_left);

          vint8m2_t v_rhs_col = __riscv_vle8_v_i8m2(rhs_col_ptr + k_offset, vl);
          vint16m4_t v_rhs_col_16 = __riscv_vsext_vf2_i16m4(v_rhs_col, vl);
          v_rhs_col_16 = __riscv_vsub_vx_i16m4(
              v_rhs_col_16, static_cast<int16_t>(rhs_params.zero_point), vl);

          // 第 1 行：load -> compute
          vint8m2_t v_lhs_tmp = __riscv_vle8_v_i8m2(lhs_row0 + k_offset, vl);
          vint16m4_t v_lhs_tmp_16 = __riscv_vsext_vf2_i16m4(v_lhs_tmp, vl);
          v_lhs_tmp_16 = __riscv_vsub_vx_i16m4(
              v_lhs_tmp_16, static_cast<int16_t>(lhs_params.zero_point), vl);
          v_acc0 = __riscv_vwmacc_vv_i32m8_tu(v_acc0, v_lhs_tmp_16,
                                              v_rhs_col_16, vl);

          v_lhs_tmp = __riscv_vle8_v_i8m2(lhs_row1 + k_offset, vl);
          v_lhs_tmp_16 = __riscv_vsext_vf2_i16m4(v_lhs_tmp, vl);
          v_lhs_tmp_16 = __riscv_vsub_vx_i16m4(
              v_lhs_tmp_16, static_cast<int16_t>(lhs_params.zero_point), vl);
          v_acc1 = __riscv_vwmacc_vv_i32m8_tu(v_acc1, v_lhs_tmp_16,
                                              v_rhs_col_16, vl);

          // 第 3 行：再次复用 -> compute
          v_lhs_tmp = __riscv_vle8_v_i8m2(lhs_row2 + k_offset, vl);
          v_lhs_tmp_16 = __riscv_vsext_vf2_i16m4(v_lhs_tmp, vl);
          v_lhs_tmp_16 = __riscv_vsub_vx_i16m4(
              v_lhs_tmp_16, static_cast<int16_t>(lhs_params.zero_point), vl);
          v_acc2 = __riscv_vwmacc_vv_i32m8_tu(v_acc2, v_lhs_tmp_16,
                                              v_rhs_col_16, vl);

          k_offset += vl;
          k_left -= vl;
        }

        size_t vlmax_m1 = __riscv_vsetvlmax_e32m1();
        vint32m1_t v_zero = __riscv_vmv_v_x_i32m1(0, vlmax_m1);

        int32_t sum0 = __riscv_vmv_x_s_i32m1_i32(
            __riscv_vredsum_vs_i32m8_i32m1(v_acc0, v_zero, vlmax));
        int32_t sum1 = __riscv_vmv_x_s_i32m1_i32(
            __riscv_vredsum_vs_i32m8_i32m1(v_acc1, v_zero, vlmax));
        int32_t sum2 = __riscv_vmv_x_s_i32m1_i32(
            __riscv_vredsum_vs_i32m8_i32m1(v_acc2, v_zero, vlmax));

        if (params.bias) {
          sum0 += params.bias[r + 0];
          sum1 += params.bias[r + 1];
          sum2 += params.bias[r + 2];
        }

        sum0 = Quantizer::Apply(sum0, r + 0, params, dst_params.zero_point);
        sum1 = Quantizer::Apply(sum1, r + 1, params, dst_params.zero_point);
        sum2 = Quantizer::Apply(sum2, r + 2, params, dst_params.zero_point);

        if (dst_params.order == Order::kColMajor) {
          dst_data[c * M + (r + 0)] = static_cast<DstScalar>(sum0);
          dst_data[c * M + (r + 1)] = static_cast<DstScalar>(sum1);
          dst_data[c * M + (r + 2)] = static_cast<DstScalar>(sum2);
        } else {
          dst_data[(r + 0) * N + c] = static_cast<DstScalar>(sum0);
          dst_data[(r + 1) * N + c] = static_cast<DstScalar>(sum1);
          dst_data[(r + 2) * N + c] = static_cast<DstScalar>(sum2);
        }
      }

      for (; r < M; ++r) {
        int k_offset = 0;
        int k_left = K;
        vint32m8_t v_acc = __riscv_vmv_v_x_i32m8(0, vlmax);
        const int8_t *lhs_row_tail = final_lhs_data + r * K;

        while (k_left > 0) {
          size_t vl = __riscv_vsetvl_e8m2(k_left);
          vint8m2_t v_rhs_col = __riscv_vle8_v_i8m2(rhs_col_ptr + k_offset, vl);
          vint16m4_t v_rhs_col_16 = __riscv_vsext_vf2_i16m4(v_rhs_col, vl);
          v_rhs_col_16 = __riscv_vsub_vx_i16m4(
              v_rhs_col_16, static_cast<int16_t>(rhs_params.zero_point), vl);

          vint8m2_t v_lhs_row =
              __riscv_vle8_v_i8m2(lhs_row_tail + k_offset, vl);
          vint16m4_t v_lhs_row_16 = __riscv_vsext_vf2_i16m4(v_lhs_row, vl);
          v_lhs_row_16 = __riscv_vsub_vx_i16m4(
              v_lhs_row_16, static_cast<int16_t>(lhs_params.zero_point), vl);
          v_acc =
              __riscv_vwmacc_vv_i32m8_tu(v_acc, v_rhs_col_16, v_lhs_row_16, vl);

          k_offset += vl;
          k_left -= vl;
        }

        size_t vlmax_m1 = __riscv_vsetvlmax_e32m1();
        vint32m1_t v_zero = __riscv_vmv_v_x_i32m1(0, vlmax_m1);
        int32_t sum = __riscv_vmv_x_s_i32m1_i32(
            __riscv_vredsum_vs_i32m8_i32m1(v_acc, v_zero, vlmax));

        if (params.bias)
          sum += params.bias[r];

        sum = Quantizer::Apply(sum, r, params, dst_params.zero_point);

        if (dst_params.order == Order::kColMajor) {
          dst_data[c * M + r] = static_cast<DstScalar>(sum);
        } else {
          dst_data[r * N + c] = static_cast<DstScalar>(sum);
        }
      }
    }
  }
};

template <typename DstScalar, typename AccuScalar,
          QuantizationFlavor quantization_flavor>
struct GemmImplRISCV<std::uint8_t, std::uint8_t, AccuScalar, DstScalar,
                     quantization_flavor> {
  static void
  Run(const MatrixParams<uint8_t> &lhs_params, const uint8_t *lhs_data,
      const MatrixParams<uint8_t> &rhs_params, const uint8_t *rhs_data,
      const MatrixParams<DstScalar> &dst_params, DstScalar *dst_data,
      const GemmParams<AccuScalar, DstScalar, quantization_flavor> &params,
      CpuBackendContext *context) {
    using Quantizer = RuyQuantizer<quantization_flavor, DstScalar>;

    std::unique_ptr<uint8_t[]> lhs_buf;
    const uint8_t *final_lhs_data = lhs_data;

    // 如果左矩阵是列优先，当然实际情况很低
    if (lhs_params.order == Order::kColMajor) {
      lhs_buf = TransposeColToRowMajor<uint8_t>(lhs_data, lhs_params.rows,
                                                lhs_params.cols);
      final_lhs_data = lhs_buf.get();
    }

    const uint8_t *final_rhs_data = rhs_data;
    std::unique_ptr<uint8_t[]> rhs_buf;

    // 右矩阵如果是行优先（默认），就要弄成列优先，cache友好
    if (rhs_params.order == Order::kRowMajor) {
      rhs_buf = TransposeRowToColMajor<uint8_t>(rhs_data, rhs_params.rows,
                                                rhs_params.cols);
      final_rhs_data = rhs_buf.get();
    }
    int M = lhs_params.rows;
    int K = lhs_params.cols; // K 也是 rhs_params.rows (内积维度)
    int N = rhs_params.cols;

    size_t vlmax = __riscv_vsetvlmax_e8m2();
    // 最外层循环：遍历右矩阵的每一列
    for (int c = 0; c < N; ++c) {
      const uint8_t *rhs_col_ptr = final_rhs_data + c * K;
      int r = 0;

      for (; r <= M - RVV_MR; r += RVV_MR) {
        int k_offset = 0;
        int k_left = K;

        vint32m8_t v_acc0 = __riscv_vmv_v_x_i32m8(0, vlmax);
        vint32m8_t v_acc1 = __riscv_vmv_v_x_i32m8(0, vlmax);
        vint32m8_t v_acc2 = __riscv_vmv_v_x_i32m8(0, vlmax);

        const uint8_t *lhs_row0 = final_lhs_data + (r + 0) * K;
        const uint8_t *lhs_row1 = final_lhs_data + (r + 1) * K;
        const uint8_t *lhs_row2 = final_lhs_data + (r + 2) * K;

        while (k_left > 0) {
          size_t vl = __riscv_vsetvl_e8m2(k_left);

          vuint8m2_t v_rhs_col =
              __riscv_vle8_v_u8m2(rhs_col_ptr + k_offset, vl);
          vuint16m4_t v_rhs_col_16u = __riscv_vzext_vf2_u16m4(v_rhs_col, vl);
          vint16m4_t v_rhs_col_16 =
              __riscv_vreinterpret_v_u16m4_i16m4(v_rhs_col_16u);
          v_rhs_col_16 = __riscv_vsub_vx_i16m4(
              v_rhs_col_16, static_cast<int16_t>(rhs_params.zero_point), vl);

          // 第 1 行：load -> compute
          vuint8m2_t v_lhs_tmp = __riscv_vle8_v_u8m2(lhs_row0 + k_offset, vl);
          vuint16m4_t v_lhs_tmp_16u = __riscv_vzext_vf2_u16m4(v_lhs_tmp, vl);
          vint16m4_t v_lhs_tmp_16 =
              __riscv_vreinterpret_v_u16m4_i16m4(v_lhs_tmp_16u);
          v_lhs_tmp_16 = __riscv_vsub_vx_i16m4(
              v_lhs_tmp_16, static_cast<int16_t>(lhs_params.zero_point), vl);
          v_acc0 = __riscv_vwmacc_vv_i32m8_tu(v_acc0, v_lhs_tmp_16,
                                              v_rhs_col_16, vl);

          // load second row for gemm
          v_lhs_tmp = __riscv_vle8_v_u8m2(lhs_row1 + k_offset, vl);
          v_lhs_tmp_16u = __riscv_vzext_vf2_u16m4(v_lhs_tmp, vl);
          v_lhs_tmp_16 = __riscv_vreinterpret_v_u16m4_i16m4(v_lhs_tmp_16u);
          v_lhs_tmp_16 = __riscv_vsub_vx_i16m4(
              v_lhs_tmp_16, static_cast<int16_t>(lhs_params.zero_point), vl);
          v_acc1 = __riscv_vwmacc_vv_i32m8_tu(v_acc1, v_lhs_tmp_16,
                                              v_rhs_col_16, vl);

          v_lhs_tmp = __riscv_vle8_v_u8m2(lhs_row2 + k_offset, vl);
          v_lhs_tmp_16u = __riscv_vzext_vf2_u16m4(v_lhs_tmp, vl);
          v_lhs_tmp_16 = __riscv_vreinterpret_v_u16m4_i16m4(v_lhs_tmp_16u);
          v_lhs_tmp_16 = __riscv_vsub_vx_i16m4(
              v_lhs_tmp_16, static_cast<int16_t>(lhs_params.zero_point), vl);
          v_acc2 = __riscv_vwmacc_vv_i32m8_tu(v_acc2, v_lhs_tmp_16,
                                              v_rhs_col_16, vl);

          k_offset += vl;
          k_left -= vl;
        }

        size_t vlmax_m1 = __riscv_vsetvlmax_e32m1();
        vint32m1_t v_zero = __riscv_vmv_v_x_i32m1(0, vlmax_m1);

        int32_t sum0 = __riscv_vmv_x_s_i32m1_i32(
            __riscv_vredsum_vs_i32m8_i32m1(v_acc0, v_zero, vlmax));
        int32_t sum1 = __riscv_vmv_x_s_i32m1_i32(
            __riscv_vredsum_vs_i32m8_i32m1(v_acc1, v_zero, vlmax));
        int32_t sum2 = __riscv_vmv_x_s_i32m1_i32(
            __riscv_vredsum_vs_i32m8_i32m1(v_acc2, v_zero, vlmax));

        if (params.bias) {
          sum0 += params.bias[r + 0];
          sum1 += params.bias[r + 1];
          sum2 += params.bias[r + 2];
        }

        sum0 = Quantizer::Apply(sum0, r + 0, params, dst_params.zero_point);
        sum1 = Quantizer::Apply(sum1, r + 1, params, dst_params.zero_point);
        sum2 = Quantizer::Apply(sum2, r + 2, params, dst_params.zero_point);

        if (dst_params.order == Order::kColMajor) {
          dst_data[c * M + (r + 0)] = static_cast<DstScalar>(sum0);
          dst_data[c * M + (r + 1)] = static_cast<DstScalar>(sum1);
          dst_data[c * M + (r + 2)] = static_cast<DstScalar>(sum2);
        } else {
          dst_data[(r + 0) * N + c] = static_cast<DstScalar>(sum0);
          dst_data[(r + 1) * N + c] = static_cast<DstScalar>(sum1);
          dst_data[(r + 2) * N + c] = static_cast<DstScalar>(sum2);
        }
      }

      for (; r < M; ++r) {
        int k_offset = 0;
        int k_left = K;
        vint32m8_t v_acc = __riscv_vmv_v_x_i32m8(0, vlmax);
        const uint8_t *lhs_row_tail = final_lhs_data + r * K;

        while (k_left > 0) {
          size_t vl = __riscv_vsetvl_e8m2(k_left);
          vuint8m2_t v_rhs_col =
              __riscv_vle8_v_u8m2(rhs_col_ptr + k_offset, vl);
          vuint16m4_t v_rhs_col_16u = __riscv_vzext_vf2_u16m4(v_rhs_col, vl);
          vint16m4_t v_rhs_col_16 =
              __riscv_vreinterpret_v_u16m4_i16m4(v_rhs_col_16u);
          v_rhs_col_16 = __riscv_vsub_vx_i16m4(
              v_rhs_col_16, static_cast<int16_t>(rhs_params.zero_point), vl);

          vuint8m2_t v_lhs_row =
              __riscv_vle8_v_u8m2(lhs_row_tail + k_offset, vl);
          vuint16m4_t v_lhs_row_16u = __riscv_vzext_vf2_u16m4(v_lhs_row, vl);
          vint16m4_t v_lhs_row_16 =
              __riscv_vreinterpret_v_u16m4_i16m4(v_lhs_row_16u);
          v_lhs_row_16 = __riscv_vsub_vx_i16m4(
              v_lhs_row_16, static_cast<int16_t>(lhs_params.zero_point), vl);
          v_acc =
              __riscv_vwmacc_vv_i32m8_tu(v_acc, v_lhs_row_16, v_rhs_col_16, vl);

          k_offset += vl;
          k_left -= vl;
        }

        size_t vlmax_m1 = __riscv_vsetvlmax_e32m1();
        vint32m1_t v_zero = __riscv_vmv_v_x_i32m1(0, vlmax_m1);
        int32_t sum = __riscv_vmv_x_s_i32m1_i32(
            __riscv_vredsum_vs_i32m8_i32m1(v_acc, v_zero, vlmax));

        if (params.bias)
          sum += params.bias[r];

        sum = Quantizer::Apply(sum, r, params, dst_params.zero_point);

        if (dst_params.order == Order::kColMajor) {
          dst_data[c * M + r] = static_cast<DstScalar>(sum);
        } else {
          dst_data[r * N + c] = static_cast<DstScalar>(sum);
        }
      }
    }
  }
};

// template <typename SrcScalar, QuantizationFlavor quantization_flavor>
// struct GemmImplRISCV<SrcScalar, SrcScalar, std::int32_t, std::int8_t,
//                      quantization_flavor>
//     : detail::GemmImplUsingRuy<SrcScalar, SrcScalar, std::int32_t,
//     std::int8_t,
//                                quantization_flavor> {};
//
// template <typename DstScalar, QuantizationFlavor quantization_flavor>
// struct GemmImplRISCV<std::int8_t, std::int8_t, std::int32_t, DstScalar,
//                      quantization_flavor>
//     : detail::GemmImplUsingRuy<std::int8_t, std::int8_t, std::int32_t,
//                                DstScalar, quantization_flavor> {};
//
// template <QuantizationFlavor quantization_flavor>
// struct GemmImplRISCV<std::int8_t, std::int8_t, std::int32_t, std::int8_t,
//                      quantization_flavor>
//     : detail::GemmImplUsingRuy<std::int8_t, std::int8_t, std::int32_t,
//                                std::int8_t, quantization_flavor> {};
#endif
} // namespace detail
} // namespace cpu_backend_gemm
} // namespace tflite

#endif // TENSORFLOW_LITE_KERNELS_CPU_BACKEND_RISCV_H_
