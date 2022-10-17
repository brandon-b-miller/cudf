# Copyright (c) 2022, NVIDIA CORPORATION.
import glob
import os
import re
import subprocess
import sys

from cubinlinker.patch import _numba_version_ok, get_logger, new_patched_linker
from numba import cuda
from numba.cuda.cudadrv.driver import Linker
from ptxcompiler.patch import CMD

from . import _version

__version__ = _version.get_versions()["version"]

logger = get_logger()

strings_udf_ptx_version = (11, 5)


def _get_appropriate_file(sms, cc):
    filtered_sms = list(filter(lambda x: x[0] <= cc, sms))
    if filtered_sms:
        return max(filtered_sms, key=lambda y: y[0])
    else:
        return None


def maybe_patch_numba_linker(driver_version):
    # Numba thinks cubinlinker is only needed if the driver is older than the ctk
    # but when strings_udf is present, it might also need to patch because the PTX
    # file strings_udf relies on may be newer than the driver as well
    if driver_version < strings_udf_ptx_version:
        logger.debug(
            "Driver version %s.%s needs patching due to strings_udf"
            % driver_version
        )
        if _numba_version_ok:
            logger.debug("Patching Numba Linker")
            Linker.new = new_patched_linker
        else:
            logger.debug("Cannot patch Numba Linker - unsupported version")


# adapted from PTXCompiler
cp = subprocess.run([sys.executable, "-c", CMD], capture_output=True)
# must have a driver to proceed
if cp.returncode == 0:
    versions = [int(s) for s in cp.stdout.strip().split()]
    driver_version = tuple(versions[:2])
    maybe_patch_numba_linker(driver_version)
    # Load the highest compute capability file available that is less than
    # the current device's.
    dev = cuda.get_current_device()
    cc = int("".join(str(x) for x in dev.compute_capability))
    files = glob.glob(os.path.join(os.path.dirname(__file__), "shim_*.ptx"))
    if len(files) == 0:
        raise RuntimeError(
            "This strings_udf installation is missing the necessary PTX "
            "files. Please file an issue reporting this error and how you "
            "installed cudf and strings_udf."
        )

    suffix_a_sm = None
    regular_sms = []

    for f in files:
        file_name = os.path.basename(f)
        sm_number = file_name.rstrip(".ptx").lstrip("shim_")
        if sm_number.endswith("a"):
            processed_sm_number = int(sm_number.rstrip("a"))
            if processed_sm_number == cc:
                suffix_a_sm = (processed_sm_number, f)
        else:
            regular_sms.append((int(sm_number), f))

    regular_result = None

    if regular_sms:
        regular_result = _get_appropriate_file(regular_sms, cc)

    if suffix_a_sm is None and regular_result is None:
        raise RuntimeError(
            "This strings_udf installation is missing the necessary PTX "
            f"files that are <={cc}."
        )
    elif suffix_a_sm is not None:
        ptxpath = suffix_a_sm[1]
    else:
        ptxpath = regular_result[1]
