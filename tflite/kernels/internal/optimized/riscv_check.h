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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RISCV_CHECK_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RISCV_CHECK_H_

#if defined(__riscv_vector) 
#define USE_RISCV
#include <riscv_vector.h>  // RVV pragma: export
#endif

// RISCV_OR_PORTABLE(SomeFunc, args) calls RISCVSomeFunc(args) if USE_RISCV is
// defined, PortableSomeFunc(args) otherwise.
#ifdef USE_RISCV
// Always use RISCV code
#define RISCV_OR_PORTABLE(funcname, ...) RISCV##funcname(__VA_ARGS__)
#define PORTABLE_RISCV(funcname, ...) Portable##funcname(__VA_ARGS__)

#else
// No RISCV available: Use Portable code
#define RISCV_OR_PORTABLE(funcname, ...) Portable##funcname(__VA_ARGS__)

#endif  // defined(USE_RISCV)

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_RISCV_CHECK_H_
