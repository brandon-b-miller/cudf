# Copyright (c) 2023, NVIDIA CORPORATION.

import glob
import os

from numba import config

ANY_PTX_FILE = os.path.dirname(__file__) + "/../core/udf/shim_60.ptx"


def _setup_numba():
    """
    Configure numba for use with cuDF. This consists of potentially
    putting numba into enhanced compatibility mode based on the user
    driver and runtime versions as well as the version of the CUDA
    Toolkit used to build the PTX files shipped with the user cuDF
    package. It also sets any other config options within numba that
    are desired for cuDF's operation.
    """
    _setup_numba_linker(ANY_PTX_FILE)
    config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


def _get_best_ptx_file(archs, max_compute_capability):
    """
    Determine of the available PTX files which one is
    the most recent up to and including the device compute capability.
    """
    filtered_archs = [x for x in archs if x[0] <= max_compute_capability]
    if filtered_archs:
        return max(filtered_archs, key=lambda x: x[0])
    else:
        return None


def _get_ptx_file(path, prefix):
    if "RAPIDS_NO_INITIALIZE" in os.environ:
        # cc=60 ptx is always built
        cc = int(os.environ.get("STRINGS_UDF_CC", "60"))
    else:
        from numba import cuda

        dev = cuda.get_current_device()

        # Load the highest compute capability file available that is less than
        # the current device's.
        cc = int("".join(str(x) for x in dev.compute_capability))
    files = glob.glob(os.path.join(path, f"{prefix}*.ptx"))
    if len(files) == 0:
        raise RuntimeError(f"Missing PTX files for cc={cc}")
    regular_sms = []

    for f in files:
        file_name = os.path.basename(f)
        sm_number = file_name.rstrip(".ptx").lstrip(prefix)
        if sm_number.endswith("a"):
            processed_sm_number = int(sm_number.rstrip("a"))
            if processed_sm_number == cc:
                return f
        else:
            regular_sms.append((int(sm_number), f))

    regular_result = None

    if regular_sms:
        regular_result = _get_best_ptx_file(regular_sms, cc)

    if regular_result is None:
        raise RuntimeError(
            "This cuDF installation is missing the necessary PTX "
            f"files that are <={cc}."
        )
    else:
        return regular_result[1]


def _setup_numba_linker(path):
    try:
        # By default, ptxcompiler will not be installed with CUDA 12
        # packages. This is ok, because in this situation putting
        # numba in enhanced compatibility mode is not necessary.
        from ptxcompiler.patch import NO_DRIVER, safe_get_versions

        versions = safe_get_versions()
        if versions != NO_DRIVER:
            driver_version, runtime_version = versions
            # Don't check if CEC is necessary in the possible edge
            # case where a user has a CUDA 12 package and ptxcompiler
            # in their environment anyways, perhaps installed separately
            if driver_version < (12, 0):
                ptx_toolkit_version = _get_cuda_version_from_ptx_file(path)
                maybe_patch_numba_linker(
                    driver_version, runtime_version, ptx_toolkit_version
                )
    except ImportError:
        pass


def maybe_patch_numba_linker(
    driver_version, runtime_version, ptx_toolkit_version
):
    # Numba thinks cubinlinker is only needed if the driver is older than
    # the CUDA runtime, but when PTX files are present, it might also need
    # to patch because those PTX files may be compiled by a CUDA version
    # that is newer than the driver as well
    if (driver_version < ptx_toolkit_version) or (
        driver_version < runtime_version
    ):
        config.NUMBA_CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY = 1


def _get_cuda_version_from_ptx_file(path):
    """
    https://docs.nvidia.com/cuda/parallel-thread-execution/
    Each PTX module must begin with a .version
    directive specifying the PTX language version

    example header:
    //
    // Generated by NVIDIA NVVM Compiler
    //
    // Compiler Build ID: CL-31057947
    // Cuda compilation tools, release 11.6, V11.6.124
    // Based on NVVM 7.0.1
    //

    .version 7.6
    .target sm_52
    .address_size 64

    """
    with open(path) as ptx_file:
        for line in ptx_file:
            if line.startswith(".version"):
                ver_line = line
                break
        else:
            raise ValueError("Could not read CUDA version from ptx file.")
    version = ver_line.strip("\n").split(" ")[1]
    # from ptx_docs/release_notes above:
    ver_map = {
        "7.5": (11, 5),
        "7.6": (11, 6),
        "7.7": (11, 7),
        "7.8": (11, 8),
        "8.0": (12, 0),
    }

    cuda_ver = ver_map.get(version)
    if cuda_ver is None:
        raise ValueError(
            f"Could not map PTX version {version} to a CUDA version"
        )

    return cuda_ver