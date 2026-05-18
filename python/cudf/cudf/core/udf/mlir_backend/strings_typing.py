# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import operator

import numpy as np
from numba_cuda_mlir import types
from numba_cuda_mlir._mlir import ir as mlir_ir
from numba_cuda_mlir._mlir.dialects import llvm
from numba_cuda_mlir.extending import typing_registry
from numba_cuda_mlir.models import register_model
from numba_cuda_mlir.numba_cuda.datamodel import PrimitiveModel
from numba_cuda_mlir.numba_cuda.extending import typeof_impl
from numba_cuda_mlir.numba_cuda.typing.templates import (
    AbstractTemplate,
    AttributeTemplate,
)
from numba_cuda_mlir.typing import signature as nb_signature

# libcudf size_type
size_type = types.int32


class StringView(types.Type):
    # ABI size (ptr + bytes i32 + length i32) for MLIR-only path; no Numba StructModel.
    _extensionty_size = 16

    def __init__(self):
        super().__init__(name="string_view")

    @property
    def return_as(self):
        return managed_udf_string


class UDFString(types.Type):
    np_dtype: np.dtype[np.object_] = np.dtype("object")

    def __init__(self):
        super().__init__(name="udf_string")

    @property
    def return_as(self):
        return self


class ManagedUDFString(types.Type):
    # ABI size: {void* meminfo (8), udf_string {char* (8), i32 (4), i32 (4)}} = 24 bytes
    _extensionty_size = 24

    np_dtype: np.dtype[np.object_] = np.dtype("object")

    def __init__(self):
        super().__init__(name="managed_udf_string")

    @property
    def return_as(self):
        return self


string_view = StringView()
udf_string = UDFString()
managed_udf_string = ManagedUDFString()


class StrViewArrayWrapper:
    """Wrapper so typeof() returns CPointer(string_view) for kernel marshalling."""

    def __init__(self, buffer):
        self._buffer = buffer

    @property
    def ptr(self):
        return self._buffer.ptr


@typeof_impl.register(StrViewArrayWrapper)
def _typeof_str_view_array_wrapper(val, c):
    return types.CPointer(string_view)


class ManagedStrArrayWrapper:
    """Wrapper so typeof() returns CPointer(managed_udf_string) for the output buffer."""

    def __init__(self, buffer):
        self._buffer = buffer

    @property
    def ptr(self):
        return self._buffer.ptr


@typeof_impl.register(ManagedStrArrayWrapper)
def _typeof_managed_str_array_wrapper(val, c):
    return types.CPointer(managed_udf_string)


class StrViewArgHandler:
    """Converts string_view* to raw pointer arguments for kernel launch."""

    def prepare_args(self, ty, val, **kwargs):
        if isinstance(ty, types.CPointer) and isinstance(
            ty.dtype, (StringView, UDFString, ManagedUDFString)
        ):
            return types.uint64, val.ptr
        if isinstance(ty, types.Tuple) and len(ty) >= 1:
            first_ty = ty[0]
            if isinstance(first_ty, types.CPointer) and isinstance(
                first_ty.dtype, (StringView, UDFString, ManagedUDFString)
            ):
                ptr_val = val[0].ptr
                if len(ty) == 2:
                    return ty, (ptr_val, val[1])
                return ty, (ptr_val,)
        return ty, val


str_view_arg_handler = StrViewArgHandler()


def NRT_decref(st):
    pass


def _get_udf_string_mlir_type():
    """MLIR struct type for cudf::strings::udf::udf_string: {char*, i32 bytes, i32 capacity}."""
    ptr_ty = llvm.PointerType.get()
    i32 = mlir_ir.IntegerType.get_signless(32)
    return llvm.StructType.new_identified("udf_string", [ptr_ty, i32, i32])


def _get_managed_udf_string_mlir_type():
    """MLIR struct type matching C++ managed_udf_string: {void* meminfo, udf_string udf_str}."""
    ptr_ty = llvm.PointerType.get()
    return llvm.StructType.new_identified(
        "managed_udf_string", [ptr_ty, _get_udf_string_mlir_type()]
    )


def _register():
    # --- MLIR data model for ManagedUDFString ---
    # Layout matches C++ managed_udf_string: {void* meminfo, udf_string udf_str}
    # where udf_string is {char* m_data, i32 m_bytes, i32 m_capacity}.
    @register_model(ManagedUDFString)
    class ManagedUDFStringModel(PrimitiveModel):
        def __init__(self, dmm, fe_type):
            be_type = _get_managed_udf_string_mlir_type()
            super().__init__(dmm, fe_type, be_type)

        def has_nrt_meminfo(self):
            return True

        def get_nrt_meminfo(self, value):
            return llvm.extractvalue(llvm.PointerType.get(), value, [0])

    # --- MLIR data model for StringView (layout: data ptr, bytes, length) ---
    @register_model(StringView)
    class StringViewTypeModel(PrimitiveModel):
        def __init__(self, dmm, fe_type):
            ptr_ty = llvm.PointerType.get()
            i32 = mlir_ir.IntegerType.get_signless(32)
            be_type = llvm.StructType.new_identified(
                "string_view", [ptr_ty, i32, i32]
            )
            super().__init__(dmm, fe_type, be_type)

    # --- Typing: len(string_view) -> size_type ---
    class LenStringViewTemplate(AbstractTemplate):
        key = len

        def generic(self, args, kws):
            if len(args) != 1 or kws:
                return None
            if args[0] is string_view:
                return nb_signature(size_type, string_view)
            return None

    typing_registry.register_global(len, types.Function(LenStringViewTemplate))

    # --- Typing: string_view.isupper, .islower, .isalpha, etc. -> boolean ---
    _id_unary_funcs = [
        "isalpha",
        "isalnum",
        "isdecimal",
        "isdigit",
        "isupper",
        "islower",
        "isspace",
        "isnumeric",
        "istitle",
    ]

    def _make_string_view_is_attr(attrname):
        class StringViewIsAttrTemplate(AbstractTemplate):
            key = f"StringView.{attrname}"

            def generic(self, args, kws):
                return nb_signature(types.boolean, recvr=self.this)

        def resolve_attr(self, mod):
            return types.BoundFunction(StringViewIsAttrTemplate, string_view)

        return resolve_attr

    # --- Typing: string_view.upper, .lower -> managed_udf_string ---
    def _make_string_view_transform_attr(attrname):
        class TransformTemplate(AbstractTemplate):
            key = f"StringView.{attrname}"

            def generic(self, args, kws):
                return nb_signature(managed_udf_string, recvr=self.this)

        def resolve_attr(self, mod):
            return types.BoundFunction(TransformTemplate, string_view)

        return resolve_attr

    # --- Typing: string_view.replace(old, new) -> managed_udf_string ---
    class ReplaceTemplate(AbstractTemplate):
        key = "StringView.replace"

        def generic(self, args, kws):
            if len(args) == 2 and not kws:
                return nb_signature(
                    managed_udf_string, args[0], args[1], recvr=self.this
                )

    def _resolve_replace(self, mod):
        return types.BoundFunction(ReplaceTemplate, string_view)

    # --- Typing: string_view.strip/lstrip/rstrip(chars) -> managed_udf_string ---
    def _make_strip_attr(attrname):
        class StripTemplate(AbstractTemplate):
            key = f"StringView.{attrname}"

            def generic(self, args, kws):
                if len(args) == 1 and not kws:
                    return nb_signature(
                        managed_udf_string, args[0], recvr=self.this
                    )

        def resolve_attr(self, mod):
            return types.BoundFunction(StripTemplate, string_view)

        return resolve_attr

    @typing_registry.register_attr
    class StringViewAttrs(AttributeTemplate):
        key = string_view

        def resolve_len(self, mod):
            return size_type

        resolve_replace = _resolve_replace

    for attrname in _id_unary_funcs:
        setattr(
            StringViewAttrs,
            f"resolve_{attrname}",
            _make_string_view_is_attr(attrname),
        )

    for attrname in ("upper", "lower"):
        setattr(
            StringViewAttrs,
            f"resolve_{attrname}",
            _make_string_view_transform_attr(attrname),
        )

    for attrname in ("strip", "lstrip", "rstrip"):
        setattr(
            StringViewAttrs,
            f"resolve_{attrname}",
            _make_strip_attr(attrname),
        )

    # --- Typing: string_view.find, .rfind (-> size_type), .startswith, .endswith (-> boolean) ---
    def _make_string_view_binary_attr(attrname, retty):
        class BinaryAttrTemplate(AbstractTemplate):
            key = f"StringView.{attrname}"

            def generic(self, args, kws):
                if (
                    len(args) == 1
                    and not kws
                    and (
                        args[0] is string_view
                        or isinstance(args[0], types.StringLiteral)
                    )
                ):
                    return nb_signature(retty, args[0], recvr=self.this)

        def resolve_attr(self, mod):
            return types.BoundFunction(BinaryAttrTemplate, string_view)

        return resolve_attr

    for attrname in ("find", "rfind", "count"):
        setattr(
            StringViewAttrs,
            f"resolve_{attrname}",
            _make_string_view_binary_attr(attrname, size_type),
        )
    for attrname in ("startswith", "endswith"):
        setattr(
            StringViewAttrs,
            f"resolve_{attrname}",
            _make_string_view_binary_attr(attrname, types.boolean),
        )

    # --- Typing: operator.contains (substr in str) (string_view, string_view) and with StringLiteral ---
    class ContainsTemplate(AbstractTemplate):
        key = operator.contains

        def generic(self, args, kws):
            if len(args) != 2 or kws:
                return None
            a, b = args[0], args[1]
            if (a is string_view or isinstance(a, types.StringLiteral)) and (
                b is string_view or isinstance(b, types.StringLiteral)
            ):
                return nb_signature(types.boolean, a, b)

    typing_registry.register_global(
        operator.contains, types.Function(ContainsTemplate)
    )

    # --- Typing: NRT_decref(managed_udf_string) -> void ---
    class NRT_decrefTemplate(AbstractTemplate):
        key = NRT_decref

        def generic(self, args, kws):
            if len(args) == 1 and isinstance(args[0], ManagedUDFString):
                return nb_signature(types.void, managed_udf_string)
            return None

    typing_registry.register_global(
        NRT_decref, types.Function(NRT_decrefTemplate)
    )

    # --- Typing: scalar string comparison ops (string_view, string_view) -> boolean ---
    # Masked string cmpops are in mlir_masked_typing.
    _cmpops = (
        operator.eq,
        operator.ne,
        operator.lt,
        operator.le,
        operator.gt,
        operator.ge,
    )

    def _make_cmpop_template(cmpop):
        class CmpOpTemplate(AbstractTemplate):
            key = cmpop

            def generic(self, args, kws):
                if (
                    len(args) == 2
                    and not kws
                    and args[0] is string_view
                    and args[1] is string_view
                ):
                    return nb_signature(
                        types.boolean,
                        string_view,
                        string_view,
                    )

        return CmpOpTemplate

    for op in _cmpops:
        typing_registry.register_global(
            op, types.Function(_make_cmpop_template(op))
        )

    # --- Typing: operator.add (string concatenation) -> managed_udf_string ---
    any_string_ty = (
        StringView,
        UDFString,
        ManagedUDFString,
        types.StringLiteral,
    )

    class AddStringTemplate(AbstractTemplate):
        key = operator.add

        def generic(self, args, kws):
            if (
                len(args) == 2
                and not kws
                and isinstance(args[0], any_string_ty)
                and isinstance(args[1], any_string_ty)
            ):
                return nb_signature(
                    managed_udf_string, string_view, string_view
                )

    typing_registry.register_global(
        operator.add, types.Function(AddStringTemplate)
    )

    # --- ManagedUDFString gets the same attrs as StringView (upper, lower, etc.) ---
    @typing_registry.register_attr
    class ManagedUDFStringAttrs(StringViewAttrs):
        key = managed_udf_string

    # --- Cast: string_view -> managed_udf_string ---
    class CastStringViewToManaged(AbstractTemplate):
        key = "cast"

        def generic(self, args, kws):
            if args[0] is string_view:
                return nb_signature(managed_udf_string, string_view)

    # --- Cast: managed_udf_string -> string_view ---
    class CastManagedToStringView(AbstractTemplate):
        key = "cast"

        def generic(self, args, kws):
            if isinstance(args[0], ManagedUDFString):
                return nb_signature(string_view, managed_udf_string)


_register()
