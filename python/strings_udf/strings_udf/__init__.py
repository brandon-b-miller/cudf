# Copyright (c) 2022, NVIDIA CORPORATION.
from ptxcompiler.patch import patch_needed, CMD
import os
import sys
import subprocess
import re


def versions_compatible(path):
    """
    Example PTX header:

    // Generated by NVIDIA NVVM Compiler
    //
    // Compiler Build ID: CL-30672275
    // Cuda compilation tools, release 11.5, V11.5.119
    // Based on NVVM 7
    """
    # obtain the cuda version used to compile this PTX file
    file = open(path).read()
    major, minor = (
        re.search("Cuda compilation tools, release ([0-9\.]+)", file)
        .group(1)
        .split(".")
    )

    # adapted from patch_needed()
    cp = subprocess.run([sys.executable, "-c", CMD], capture_output=True)
    if cp.returncode:
        # no driver
        return False

    versions = [int(s) for s in cp.stdout.strip().split()]
    driver_version = tuple(versions[:2])

    return driver_version >= (int(major), int(minor)) and not patch_needed()


# ptxpath = os.getenv("CONDA_PREFIX") + "/lib/shim.ptx"
ptxpath = os.path.join(os.path.dirname(__file__), "shim.ptx")
ENABLED = versions_compatible(ptxpath)

from . import _version

__version__ = _version.get_versions()["version"]
