# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the MLIR backend strings typing module (PR 2).

These exercise the Python-level surface added by
``cudf.core.udf.mlir_backend.strings_typing``:

* Type identity / sizing / hash / repr.
* ``typeof()`` of the host-side wrappers used to marshal string-view
  arrays to kernels.
* ``str_view_arg_handler.prepare_args`` for both the bare ``CPointer``
  case and the ``Tuple(CPointer, mask)`` case used when the column is
  nullable.

Lowering-level kernel tests live in ``test_strings_lowering.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("numba_cuda_mlir")

from numba_cuda_mlir import types  # noqa: E402
from numba_cuda_mlir.numba_cuda.typing.typeof import typeof  # noqa: E402

from cudf.core.udf.mlir_backend.strings_typing import (  # noqa: E402
    ManagedStrArrayWrapper,
    ManagedUDFString,
    NRT_decref,
    StringView,
    StrViewArgHandler,
    StrViewArrayWrapper,
    UDFString,
    managed_udf_string,
    size_type,
    str_view_arg_handler,
    string_view,
)


class _FakeBuffer:
    """Minimal stand-in for a Buffer-like object exposing ``ptr``."""

    def __init__(self, ptr):
        self.ptr = ptr


def test_extensionty_size_constants():
    """ABI sizes used by ``_get_extensionty_size`` short-circuit."""
    assert StringView._extensionty_size == 16
    assert ManagedUDFString._extensionty_size == 24
    # UDFString does not declare _extensionty_size; only the managed
    # wrapper and the view do.
    assert getattr(UDFString, "_extensionty_size", None) is None


def test_string_view_repr_and_return_as():
    assert repr(string_view) == "string_view"
    assert string_view.return_as is managed_udf_string


def test_managed_and_udf_string_repr():
    assert repr(managed_udf_string) == "managed_udf_string"
    assert UDFString().name == "udf_string"


def test_managed_and_udf_string_np_dtype_object():
    assert ManagedUDFString.np_dtype == np.dtype("O")
    assert UDFString.np_dtype == np.dtype("O")


def test_typeof_str_view_array_wrapper():
    """``typeof(StrViewArrayWrapper(...))`` -> ``CPointer(string_view)``."""
    wrapper = StrViewArrayWrapper(_FakeBuffer(0xDEAD_BEEF))
    ty = typeof(wrapper)
    assert isinstance(ty, types.CPointer)
    assert ty.dtype is string_view


def test_typeof_managed_str_array_wrapper():
    """``typeof(ManagedStrArrayWrapper(...))`` -> ``CPointer(managed_udf_string)``."""
    wrapper = ManagedStrArrayWrapper(_FakeBuffer(0x1234_5678))
    ty = typeof(wrapper)
    assert isinstance(ty, types.CPointer)
    assert ty.dtype is managed_udf_string


def test_str_view_arg_handler_cpointer_string_view():
    """Bare ``CPointer(string_view)`` is rewritten to ``(uint64, ptr)``."""
    handler = StrViewArgHandler()
    wrapper = StrViewArrayWrapper(_FakeBuffer(0xCAFE_F00D))
    ty = types.CPointer(string_view)
    new_ty, new_val = handler.prepare_args(ty, wrapper)
    assert new_ty is types.uint64
    assert new_val == 0xCAFE_F00D


def test_str_view_arg_handler_cpointer_managed():
    """``CPointer(managed_udf_string)`` is also unwrapped to ``(uint64, ptr)``."""
    handler = StrViewArgHandler()
    wrapper = ManagedStrArrayWrapper(_FakeBuffer(0xBEEF_FACE))
    ty = types.CPointer(managed_udf_string)
    new_ty, new_val = handler.prepare_args(ty, wrapper)
    assert new_ty is types.uint64
    assert new_val == 0xBEEF_FACE


def test_str_view_arg_handler_cpointer_udf_string():
    """``CPointer(udf_string)`` is also unwrapped (covers UDFString in the union)."""
    handler = StrViewArgHandler()
    wrapper = StrViewArrayWrapper(_FakeBuffer(0xABCD_1234))
    ty = types.CPointer(UDFString())
    new_ty, new_val = handler.prepare_args(ty, wrapper)
    assert new_ty is types.uint64
    assert new_val == 0xABCD_1234


def test_str_view_arg_handler_tuple_with_mask():
    """``Tuple(CPointer(sv), mask)`` -> tuple with the pointer extracted.

    This is the only tuple shape produced in practice (masked string
    columns). The implementation also has a code path for a 1-element
    tuple, but that shape is never produced by ``_masked_array_type_from_col``
    so it isn't covered here.
    """
    handler = StrViewArgHandler()
    wrapper = StrViewArrayWrapper(_FakeBuffer(0xAA))
    mask_value = "<opaque-mask-array>"
    tuple_ty = types.Tuple((types.CPointer(string_view), types.boolean[::1]))
    new_ty, new_val = handler.prepare_args(tuple_ty, (wrapper, mask_value))
    assert new_ty is tuple_ty
    assert new_val == (0xAA, mask_value)


def test_str_view_arg_handler_passthrough_for_non_str_types():
    """Types unrelated to string-view wrappers pass through unchanged."""
    handler = StrViewArgHandler()
    plain_ty = types.float64[::1]
    sentinel = object()
    new_ty, new_val = handler.prepare_args(plain_ty, sentinel)
    assert new_ty is plain_ty
    assert new_val is sentinel


def test_module_exposes_str_view_arg_handler_singleton():
    """The module-level handler is a ``StrViewArgHandler`` instance."""
    assert isinstance(str_view_arg_handler, StrViewArgHandler)


def test_nrt_decref_is_callable_stub():
    """``NRT_decref`` is a Python-callable placeholder; the real lowering
    is registered in ``strings_lowering`` via the typing template."""
    # TODO for Brandon - can we just use the numba-cuda-mlir decref without
    # having to wrap our own now?
    # No-op when called from Python - only meaningful inside a JIT context.
    assert NRT_decref(object()) is None
