# /home/lzb/source/riscv/rvspoc-S2602-litert/third_party/toolchains/riscv64/BUILD

load(":cc_toolchain_config.bzl", "cc_toolchain_config")

package(default_visibility = ["//visibility:public"])


cc_toolchain_config(name = "riscv64_toolchain_config")

cc_toolchain(
    name = "riscv64_toolchain_definition",
    all_files = ":empty",
    compiler_files = ":empty",
    dwp_files = ":empty",
    linker_files = ":empty",
    objcopy_files = ":empty",
    strip_files = ":empty",
    supports_param_files = 0,
    toolchain_config = ":riscv64_toolchain_config",
    toolchain_identifier = "riscv64-toolchain",
)

toolchain(
    name = "riscv64_toolchain",
    exec_compatible_with = [
        "@platforms//cpu:x86_64",
        "@platforms//os:linux",
    ],
    # 注意这里：直接引用标准约束，而不是引用你定义的那个 platform
    target_compatible_with = [
        "@platforms//cpu:riscv64",
        "@platforms//os:linux",
    ],
    toolchain = ":riscv64_toolchain_definition",
    toolchain_type = "@bazel_tools//tools/cpp:toolchain_type",
)

filegroup(name = "empty")