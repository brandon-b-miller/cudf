# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""
cudf-side typing glue for string UDFs.

MLIRStringType and all its method typing live in cudf.core.udf.mlir_backend.string_typing_impl.
This module provides:
  - ManagedStrArrayWrapper (typeof -> CPointer(mlir_string))
  - MLIRStringArgHandler (kernel arg marshalling)
  - NRT_decref sentinel + typing
  - StringLiteral -> mlir_string cast typing
"""

from __future__ import annotations

from numba_cuda_mlir.extending import typing_registry
from numba_cuda_mlir.numba_cuda import types
from numba_cuda_mlir.numba_cuda.extending import typeof_impl
from numba_cuda_mlir.numba_cuda.typing import signature as nb_signature
from numba_cuda_mlir.numba_cuda.typing.templates import AbstractTemplate
from cudf.core.udf.mlir_backend.string_types import MLIRStringType

mlir_string = MLIRStringType()
size_type = types.int32


class ManagedStrArrayWrapper:
    """Wrapper so typeof() returns CPointer(mlir_string) for the output buffer."""

    def __init__(self, buffer):
        self._buffer = buffer

    @property
    def ptr(self):
        return self._buffer.ptr


@typeof_impl.register(ManagedStrArrayWrapper)
def _typeof_mlir_str_array_wrapper(val, c):
    return types.CPointer(mlir_string)


class MLIRStringArgHandler:
    """Converts mlir_string* to raw pointer arguments for kernel launch."""

    def prepare_args(self, ty, val, **kwargs):
        if isinstance(ty, types.CPointer) and isinstance(
            ty.dtype, MLIRStringType
        ):
            return types.uint64, val.ptr
        if isinstance(ty, types.Tuple) and len(ty) >= 1:
            first_ty = ty[0]
            if isinstance(first_ty, types.CPointer) and isinstance(
                first_ty.dtype, MLIRStringType
            ):
                ptr_val = val[0].ptr
                if len(ty) == 2:
                    return ty, (ptr_val, val[1])
                return ty, (ptr_val,)
        return ty, val


mlir_string_arg_handler = MLIRStringArgHandler()


def NRT_decref(st):
    pass


def _register():
    # --- Typing: NRT_decref(mlir_string) -> void ---
    class NRT_decrefTemplate(AbstractTemplate):
        key = NRT_decref

        def generic(self, args, kws):
            if len(args) == 1 and isinstance(args[0], MLIRStringType):
                return nb_signature(types.void, mlir_string)
            return None

    typing_registry.register_global(
        NRT_decref, types.Function(NRT_decrefTemplate)
    )

    # --- Cast: StringLiteral -> mlir_string ---
    class CastStringLiteralToMLIRString(AbstractTemplate):
        key = "cast"

        def generic(self, args, kws):
            if isinstance(args[0], types.StringLiteral):
                return nb_signature(mlir_string, args[0])


_register()
