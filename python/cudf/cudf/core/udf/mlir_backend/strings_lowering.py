# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typing and lowering for ``mlir_string`` operations.

Currently only ``len`` (UTF-8 character count). Registered with
``numba_cuda_mlir`` at import time via :func:`_register`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numba_cuda_mlir import types
from numba_cuda_mlir._mlir.dialects import arith
from numba_cuda_mlir._mlir.extras import types as T
from numba_cuda_mlir.extending import lowering_registry, typing_registry
from numba_cuda_mlir.numba_cuda.typing.templates import (
    AbstractTemplate,
    Signature,
)
from numba_cuda_mlir.typing import signature as nb_signature

from cudf.core.udf.mlir_backend import string_lowering_impl as _impl
from cudf.core.udf.mlir_backend.strings_typing import (
    MLIRStringType,
    mlir_string,
)

if TYPE_CHECKING:
    from numba_cuda_mlir.mlir_lowering import MLIRLower
    from numba_cuda_mlir.numba_cuda.core.ir import Var


class LenMLIRStringTemplate(AbstractTemplate):
    """``len(mlir_string)`` -> ``int64`` (UTF-8 character count)."""

    key = len

    def generic(
        self, args: tuple[types.Type, ...], kws: dict
    ) -> Signature | None:
        """Resolve ``len(mlir_string) -> int64``.

        Parameters
        ----------
        args : tuple of types.Type
            Positional argument types.
        kws : dict
            Keyword argument types (must be empty).

        Returns
        -------
        Signature or None
            The resolved signature, or ``None`` when ``args`` is not a single
            ``mlir_string``.
        """
        if len(args) == 1 and not kws and isinstance(args[0], MLIRStringType):
            return nb_signature(types.int64, mlir_string)
        return None


def _lower_len(
    builder: MLIRLower, target: Var, args: list[Var], kwargs: list
) -> None:
    """``len(mlir_string)``: count UTF-8 characters, returned as ``int64``."""
    ms_val = builder.load_var(args[0])
    count_i32 = _impl._lower_len(ms_val)
    builder.store_var(target, arith.extui(T.i64(), count_i32))


def _register() -> None:
    """Register ``len`` typing and lowering with ``numba_cuda_mlir``."""
    typing_registry.register_global(len)(LenMLIRStringTemplate)
    lowering_registry.lower(len, mlir_string)(_lower_len)


_register()
