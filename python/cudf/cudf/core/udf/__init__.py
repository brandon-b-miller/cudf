# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""
UDF compilation support.  Typing/lowering registration is loaded from the
MLIR/numba_cuda_mlir backend.
"""

from cudf.core.udf.mlir_backend import (
    groupby_lowering,
    groupby_typing,
    masked_lowering,
    masked_typing,
    strings_lowering,
    strings_typing,
)

# The numba-cuda-mlir compile-time branch prewarms the typing/target contexts
# exactly once (during the first MLIRDispatcher creation, which fires inside
# groupby_typing → utils.py).  All cudf typing/lowering registrations that
# happen in the _register() calls above are therefore "new" to the
# RegistryLoader and won't be seen by the contexts unless we re-install.
# Force a re-install here, after every backend module has finished registering.
from numba_cuda_mlir.extending import refresh_registries

refresh_registries(include_uninitialized_cuda=False)
