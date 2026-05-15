# third_party/toolchains/riscv64/cc_toolchain_config.bzl
load("@bazel_tools//tools/cpp:cc_toolchain_config_lib.bzl", "feature", "flag_group", "flag_set", "tool_path")

def _impl(ctx):
    tool_paths = [
        tool_path(name = "gcc", path = "/usr/lib/llvm-21/bin/clang"),
        tool_path(name = "ld", path = "/usr/bin/ld.lld"), 
        tool_path(name = "ar", path = "/usr/lib/llvm-21/bin/llvm-ar"),
        tool_path(name = "cpp", path = "/usr/lib/llvm-21/bin/clang-cpp"),
        tool_path(name = "nm", path = "/usr/lib/llvm-21/bin/llvm-nm"),
        tool_path(name = "objcopy", path = "/usr/lib/llvm-21/bin/llvm-objcopy"),
        tool_path(name = "objdump", path = "/usr/lib/llvm-21/bin/llvm-objdump"),
        tool_path(name = "strip", path = "/usr/lib/llvm-21/bin/llvm-strip"),
    ]

    action_configs = []
    features = [
        feature(
            name = "default_flags",
            enabled = True,
            flag_sets = [
                flag_set(
                    actions = ["c-compile", "c++-compile", "c++-link-executable", "c++-link-nodeps-dynamic-library"],
                    flag_groups = [
                        flag_group(
                            flags = [
                                "--target=riscv64-linux-gnu",
                                "-march=rv64gcv",
                                "-mabi=lp64d",
                                "-I/usr/riscv64-linux-gnu/include",
                                "-L/usr/riscv64-linux-gnu/lib",
                                "-L/usr/lib/gcc-cross/riscv64-linux-gnu/15",
                                "-fuse-ld=lld",
                                "--gcc-toolchain=/usr",
                                "-Wl,--sysroot=/",
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ]

    return cc_common.create_cc_toolchain_config_info(
        ctx = ctx,
        tool_paths = tool_paths,
        features = features,
        toolchain_identifier = "riscv64-toolchain",
        cxx_builtin_include_directories = [
            "/usr/lib/llvm-21/lib/clang/21/include",
            "/usr/riscv64-linux-gnu/include",
            "/usr/lib/gcc-cross/riscv64-linux-gnu/15",
            "/usr/include",
        ],
        builtin_sysroot = "/usr/riscv64-linux-gnu",
        target_libc = "glibc",
        compiler = "clang",
        abi_version = "lp64d",
        abi_libc_version = "glibc_2.39", # 根据 sysroot 实际情况
        target_cpu = "riscv64",
        target_system_name = "riscv64-linux-gnu",
        host_system_name = "x86_64-unknown-linux-gnu",
    )

cc_toolchain_config = rule(
    implementation = _impl,
    attrs = {},
    provides = [CcToolchainConfigInfo],
)