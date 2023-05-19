# Copyright (c) 2023, NVIDIA CORPORATION.

import glob
import math
import os
import subprocess
import sys
import warnings

from numba import config

CC_60_PTX_FILE = os.path.dirname(__file__) + "/../core/udf/shim_60.ptx"
_NO_DRIVER = (math.inf, math.inf)

CMD = """\
from ctypes import c_int, byref
from numba import cuda
dv = c_int(0)
cuda.cudadrv.driver.driver.cuDriverGetVersion(byref(dv))
drv_major = dv.value // 1000
drv_minor = (dv.value - (drv_major * 1000)) // 10
run_major, run_minor = cuda.runtime.get_version()
print(f'{drv_major} {drv_minor} {run_major} {run_minor}')
"""


def _get_versions():
    """
    This function is mostly vendored from ptxcompiler and is used
    to check the system CUDA driver and runtime versions in its absence.
    """
    cp = subprocess.run([sys.executable, "-c", CMD], capture_output=True)
    if cp.returncode:
        return _NO_DRIVER

    versions = [int(s) for s in cp.stdout.strip().split()]
    driver_version = tuple(versions[:2])
    runtime_version = tuple(versions[2:])

    return driver_version, runtime_version


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


def _setup_numba():
    """
    Configure the numba linker for use with cuDF. This consists of
    potentially putting numba into enhanced compatibility mode
    based on the user driver and runtime versions as well as the
    version of the CUDA Toolkit used to build the PTX files shipped
    with the user cuDF package.
    """
    try:
        # By default, ptxcompiler will not be installed with CUDA 12
        # packages. This is ok, because in this situation putting
        # numba in enhanced compatibility mode is not necessary.
        from ptxcompiler.patch import NO_DRIVER, safe_get_versions
    except ModuleNotFoundError:
        versions = _get_versions()
        if versions != _NO_DRIVER:
            driver_version, runtime_version = versions
            if runtime_version > driver_version:
                warnings.warn(
                    f"Using CUDA toolkit version {runtime_version} with CUDA "
                    f"driver version {driver_version} requires minor version "
                    "compatibility, which is not yet supported for CUDA "
                    "driver versions newer than 12.0. It is likely that many "
                    "cuDF operations will not work in this state. Please "
                    f"install CUDA toolkit version {driver_version} to "
                    "continue using cuDF."
                )
    versions = safe_get_versions()
    if versions != NO_DRIVER:
        driver_version, runtime_version = versions
        # Don't check if CEC is necessary in the possible edge
        # case where a user has a CUDA 12 package and ptxcompiler
        # in their environment anyways, perhaps installed separately
        if driver_version < (12, 0):
            ptx_toolkit_version = _get_cuda_version_from_ptx_file(
                CC_60_PTX_FILE
            )
            # Numba thinks cubinlinker is only needed if the driver is older
            # than the CUDA runtime, but when PTX files are present, it might
            # also need to patch because those PTX files may be compiled by
            # a CUDA version that is newer than the driver as well
            if (driver_version < ptx_toolkit_version) or (
                driver_version < runtime_version
            ):
                config.CUDA_ENABLE_MINOR_VERSION_COMPATIBILITY = 1


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


class _CUDFNumbaConfig:
    def __enter__(self):
        self.enter_val = config.CUDA_LOW_OCCUPANCY_WARNINGS
        config.CUDA_LOW_OCCUPANCY_WARNINGS = 0

    def __exit__(self, exc_type, exc_value, traceback):
        config.CUDA_LOW_OCCUPANCY_WARNINGS = self.enter_val
