"""Build configuration for the tc02 PPU extension."""

import os

from torch.utils.cpp_extension import CUDAExtension

# This backend requires the tc02 FP32 MMA atom and has been tested on ZW-M890P.
PPU_TARGETS = {"tc02": {"acompute": 10500, "arch": "-arch=ppu_15"}}


def ppu_extension(this_dir, get_nvcc_thread_args, is_flag_set):
    target = os.getenv("FLASH_KDA_PPU_TARGET", "tc02").lower()
    if target not in PPU_TARGETS:
        raise ValueError(f"Only FLASH_KDA_PPU_TARGET=tc02 is supported, got {target!r}")
    acompute = PPU_TARGETS[target]["acompute"]
    arch = PPU_TARGETS[target]["arch"]

    # Match FlashAttention/FlashMLA: ACTLIZE supplies cutlass/, cute/ and ppu/.
    # Initialize this submodule before running cudafy, which converts it with
    # the PPU sources. Keep the upstream NVIDIA cutlass submodule for CUDA.
    actlize = os.path.join(this_dir, "actlize")
    ppu_include = os.path.join(actlize, "include")
    for directory in ("ppu", "cute", "cutlass"):
        if not os.path.isdir(os.path.join(ppu_include, directory)):
            raise RuntimeError(
                "Missing ACTLIZE headers. Run git submodule update --init actlize, "
                "then python /path/to/cudafy-for-sail/cudafy.py flashkda . "
                "in the PPU build checkout. See docs/PPU.md."
            )

    hgcc_args = [
        "-O3",
        "-std=c++17",
        "-DNDEBUG",
        "-D_USE_MATH_DEFINES",
        "-Wno-deprecated-declarations",
        # Opposite sign to the CUDA backend, which undefines these. ACTLIZE
        # needs them defined for this backend.
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
        "--ptxas-options=-v,--register-usage-level=10,--warn-on-spills",
        # Preserve the validated PPU backend tuning through cudafy.
        # -ppu-max-vreg-count is measured to be ignored, so it is not here: use
        # __launch_bounds__ to control occupancy.
        "-Xllvm",
        "-ppu-sink-matrix-addr=true",
        "-Xllvm",
        "-ppu-max-alloca-byte-size=320",
        "-Xllvm",
        "-ppu-sink-async-addr=true",
        "-Xllvm",
        "-ppu-sink-load-addr=true",
        "-Xllvm",
        "-ppu-sink-store-addr=true",
        "-Xllvm",
        "-ppu-alloca-half-ldst-simplify=true",
        "-DUSE_PPU",
        "-DUSE_AIU=1",
        f"-DACOMPUTE_VERSION={acompute}",
        arch,
        *get_nvcc_thread_args(),
    ]

    if is_flag_set("FLASH_KDA_KEEP_ASM"):
        keep_dir = os.path.join(this_dir, "build", "keep")
        os.makedirs(keep_dir, exist_ok=True)
        hgcc_args += ["-keep", "-keep-dir", keep_dir]

    # Tuning sweep hook. The pipeline depth, warp split and CTAs/CU trade shared
    # memory against occupancy against accumulator chains, and which point wins is
    # measured, not derived. Every knob has an #ifndef default in
    # csrc/ppu/fwd_launch_utils_ppu.cuh, so a sweep is a rebuild rather than an edit:
    #   FLASH_KDA_PPU_DEFINES="-DFLASH_KDA_PPU_INPUT_STAGES=3"
    extra_defines = os.getenv("FLASH_KDA_PPU_DEFINES", "").split()
    if extra_defines:
        print(f"PPU tuning defines: {' '.join(extra_defines)}")
        hgcc_args += extra_defines

    return CUDAExtension(
        name="flash_kda_C",
        sources=[
            "csrc/flash_kda.cpp",
            "csrc/ppu/torch_api.cpp",
            "csrc/ppu/decode_launch_ppu.cu",
            "csrc/ppu/verify_launch_ppu.cu",
            "csrc/ppu/fwd_launch_ppu.cu",
        ],
        include_dirs=[
            ppu_include,
            os.path.join(this_dir, "csrc"),
        ],
        extra_compile_args={
            # Compile host and device translation units for the same backend.
            "cxx": ["-O3", "-Wno-psabi", "-DUSE_PPU", *extra_defines],
            "nvcc": hgcc_args,
        },
    )
