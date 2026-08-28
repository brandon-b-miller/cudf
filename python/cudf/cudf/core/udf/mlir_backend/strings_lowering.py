# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""
MLIR lowering for scalar mlir_string operations.

All string operations use pure-MLIR helpers from cudf.core.udf.mlir_backend.string_lowering_impl.
No C++ shims are required for string UDF operations.
"""

from __future__ import annotations

import operator

from numba_cuda_mlir._mlir import ir as mlir_ir
from numba_cuda_mlir._mlir.dialects import arith, func, llvm
from numba_cuda_mlir._mlir.extras import types as T
from numba_cuda_mlir.extending import lower_cast, lowering_registry
from cudf.core.udf.mlir_backend.string_lowering_impl import (
    allocate_mlir_string,
    build_mlir_string,
    lower_compare,
    lower_concat,
    lower_contains,
    lower_count,
    lower_endswith,
    lower_find,
    lower_isalnum,
    lower_isalpha,
    lower_isdecimal,
    lower_isdigit,
    lower_islower,
    lower_isnumeric,
    lower_isspace,
    lower_istitle,
    lower_isupper,
    lower_len,
    lower_lower,
    lower_lstrip,
    lower_replace,
    lower_rfind,
    lower_rstrip,
    lower_startswith,
    lower_strip,
    lower_upper,
    mlir_string_to_view,
)
from numba_cuda_mlir.lowering_utilities import (
    DeferredMethodCall,
    get_or_insert_function,
)
from numba_cuda_mlir.numba_cuda import types

from cudf.core.udf.mlir_backend.unicode import (
    get_character_cases_table_ptr,
    get_character_flags_table_ptr,
    get_special_case_mapping_table_ptr,
)
from cudf.core.udf.mlir_backend.strings_typing import (
    MLIRStringType,
    NRT_decref,
    mlir_string,
)

# ---------------------------------------------------------------------------
# Pure-MLIR helpers
# ---------------------------------------------------------------------------


def lower_len_value(builder, str_view):
    """len(mlir_string view) -> i32 via pure MLIR."""
    return lower_len(str_view)


_MLIR_IS_FUNCS = {
    "isupper": lower_isupper,
    "islower": lower_islower,
    "isalpha": lower_isalpha,
    "isalnum": lower_isalnum,
    "isdecimal": lower_isdecimal,
    "isdigit": lower_isdigit,
    "isspace": lower_isspace,
    "isnumeric": lower_isnumeric,
    "istitle": lower_istitle,
}


def _flags_table_ptr_const():
    """Return an MLIR ptr to the character flags table (device pointer)."""
    addr = int(get_character_flags_table_ptr())
    i64 = mlir_ir.IntegerType.get_signless(64)
    if addr >= (1 << 63):
        addr -= 1 << 64
    return llvm.inttoptr(llvm.PointerType.get(), arith.constant(i64, addr))


def lower_is_xyz_value(builder, str_view, mlir_fn):
    """mlir_string.isXyz() -> i1 via pure MLIR."""
    return mlir_fn(str_view, _flags_table_ptr_const())


_CMPOP_PREDICATES = {
    "eq": arith.CmpIPredicate.eq,
    "ne": arith.CmpIPredicate.ne,
    "lt": arith.CmpIPredicate.slt,
    "le": arith.CmpIPredicate.sle,
    "gt": arith.CmpIPredicate.sgt,
    "ge": arith.CmpIPredicate.sge,
}
_CMPOP_NAMES = tuple(_CMPOP_PREDICATES.keys())

_BOOL_STR_STR_OPS = {
    "startswith": lower_startswith,
    "endswith": lower_endswith,
    "contains": lower_contains,
}

_INT_STR_STR_OPS = {
    "find": lower_find,
    "rfind": lower_rfind,
    "count": lower_count,
}


def call_string_cmpop_mlir(lhs, rhs, op_name):
    """Pure MLIR string compare -> i1."""
    cmp_result = lower_compare(lhs, rhs)
    zero = arith.constant(mlir_ir.IntegerType.get_signless(32), 0)
    return arith.cmpi(_CMPOP_PREDICATES[op_name], cmp_result, zero)


def call_string_bool_str_str_mlir(lhs, rhs, op_name):
    """Pure MLIR (startswith/endswith/contains) -> i1."""
    return _BOOL_STR_STR_OPS[op_name](lhs, rhs)


def call_string_int_str_str_mlir(lhs, rhs, op_name):
    """Pure MLIR (find/rfind/count) -> i32."""
    return _INT_STR_STR_OPS[op_name](lhs, rhs)


# ---------------------------------------------------------------------------
# Var-to-view SSA helpers (internal MLIR value, not a Numba type)
# ---------------------------------------------------------------------------


def _literal_to_view(builder, literal_var):
    """Materialize a StringLiteral as a view SSA value {ptr, i32, i32}."""
    literal_ty = builder.get_numba_type(literal_var.name)
    if not isinstance(literal_ty, types.StringLiteral):
        raise TypeError("expects StringLiteral variable")
    literal_value = literal_ty.literal_value
    data_ptr = builder.load_var(literal_var)

    ptr_ty = llvm.PointerType.get()
    i32 = mlir_ir.IntegerType.get_signless(32)
    sv_ty = llvm.StructType.get_literal([ptr_ty, i32, i32])
    bytes_val = len(literal_value.encode("UTF-8"))
    length_val = len(literal_value)
    undef = llvm.UndefOp(sv_ty)
    with_data = llvm.insertvalue(
        container=undef,
        value=llvm.extractvalue(ptr_ty, data_ptr, [0]),
        position=mlir_ir.DenseI64ArrayAttr.get([0]),
    )
    with_bytes = llvm.insertvalue(
        container=with_data,
        value=arith.constant(i32, bytes_val),
        position=mlir_ir.DenseI64ArrayAttr.get([1]),
    )
    return llvm.insertvalue(
        container=with_bytes,
        value=arith.constant(i32, length_val),
        position=mlir_ir.DenseI64ArrayAttr.get([2]),
    )


def _any_var_to_view(builder, var):
    """Convert any string-typed variable to a view SSA value."""
    numba_ty = builder.get_numba_type(var.name)
    if isinstance(numba_ty, types.StringLiteral):
        return _literal_to_view(builder, var)
    val = builder.load_var(var)
    if isinstance(numba_ty, MLIRStringType):
        return mlir_string_to_view(val)
    return val


def _i64_const_from_ptr(addr):
    """Create an i64 constant from a host pointer address (for lookup tables)."""
    i64 = mlir_ir.IntegerType.get_signless(64)
    if addr >= (1 << 63):
        addr -= 1 << 64
    return arith.constant(i64, addr)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def _register():
    lower = lowering_registry.lower
    lower_getattr = lowering_registry.lower_getattr

    # === NRT_decref ===

    def _lower_nrt_decref_mlir_string(builder, target, args, kwargs):
        ms_val = builder.load_var(args[0])
        meminfo = llvm.extractvalue(llvm.PointerType.get(), ms_val, [0])
        nrt_decref_ty = mlir_ir.FunctionType.get([llvm.PointerType.get()], [])
        callee = get_or_insert_function(
            "NRT_decref",
            nrt_decref_ty,
            builder.mlir_gpu_module,
        )
        func.call(result=[], callee=callee.name.value, operands_=[meminfo])

    lower(NRT_decref, mlir_string)(_lower_nrt_decref_mlir_string)

    # === Cast: StringLiteral -> mlir_string ===

    @lower_cast(types.StringLiteral, mlir_string)
    def cast_string_literal_to_mlir_string(
        context, builder, fromty, toty, val
    ):
        """Copy literal string bytes into an NRT-managed mlir_string."""
        literal_value = fromty.literal_value
        nbytes = len(literal_value.encode("UTF-8"))
        i64 = mlir_ir.IntegerType.get_signless(64)
        nbytes_i64 = arith.constant(i64, nbytes)
        data = llvm.extractvalue(llvm.PointerType.get(), val, [0])
        mi, out_data = allocate_mlir_string(
            builder.mlir_gpu_module, nbytes_i64
        )
        llvm.intr_memcpy(out_data, data, nbytes_i64, False)
        return build_mlir_string(mi, out_data, nbytes_i64)

    # === mlir_string methods ===

    def _load_as_view(builder, var):
        """Load a mlir_string var, convert to view SSA value."""
        return mlir_string_to_view(builder.load_var(var))

    def _make_lower_is(mlir_fn):
        def _lower_impl(builder, target, args, kwargs):
            str_view = _load_as_view(builder, args[0])
            result = lower_is_xyz_value(builder, str_view, mlir_fn)
            builder.store_var(target, result)

        def _getattr(context, builder, target, value, attr=None):
            builder.store_var(target, DeferredMethodCall(value, _lower_impl))

        return _getattr

    for attrname, mlir_fn in _MLIR_IS_FUNCS.items():
        lower_getattr(mlir_string, attrname)(_make_lower_is(mlir_fn))

    def _make_lower_binary_int(op_name):
        def _lower_impl(builder, target, args, kwargs):
            lhs = _load_as_view(builder, args[0])
            rhs = _any_var_to_view(builder, args[1])
            result = call_string_int_str_str_mlir(lhs, rhs, op_name)
            builder.store_var(target, result)

        def _getattr(context, builder, target, value, attr=None):
            builder.store_var(target, DeferredMethodCall(value, _lower_impl))

        return _getattr

    for attrname in _INT_STR_STR_OPS:
        lower_getattr(mlir_string, attrname)(_make_lower_binary_int(attrname))

    def _make_lower_binary_bool(op_name):
        def _lower_impl(builder, target, args, kwargs):
            lhs = _load_as_view(builder, args[0])
            rhs = _any_var_to_view(builder, args[1])
            result = call_string_bool_str_str_mlir(lhs, rhs, op_name)
            builder.store_var(target, result)

        def _getattr(context, builder, target, value, attr=None):
            builder.store_var(target, DeferredMethodCall(value, _lower_impl))

        return _getattr

    for attrname in ("startswith", "endswith"):
        lower_getattr(mlir_string, attrname)(_make_lower_binary_bool(attrname))

    def _lower_replace_impl(builder, target, args, kwargs):
        src = _load_as_view(builder, args[0])
        old = _any_var_to_view(builder, args[1])
        new = _any_var_to_view(builder, args[2])
        result = lower_replace(builder.mlir_gpu_module, src, old, new)
        builder.store_var(target, result)

    def _lower_replace_getattr(context, builder, target, value, attr=None):
        builder.store_var(
            target, DeferredMethodCall(value, _lower_replace_impl)
        )

    lower_getattr(mlir_string, "replace")(_lower_replace_getattr)

    def _make_lower_upper_or_lower(mlir_fn):
        def _lower_impl(builder, target, args, kwargs):
            str_view = _any_var_to_view(builder, args[0])
            flags_const = _i64_const_from_ptr(
                int(get_character_flags_table_ptr())
            )
            cases_const = _i64_const_from_ptr(
                int(get_character_cases_table_ptr())
            )
            special_const = _i64_const_from_ptr(
                int(get_special_case_mapping_table_ptr())
            )
            result = mlir_fn(
                builder.mlir_gpu_module,
                str_view,
                flags_const,
                cases_const,
                special_const,
            )
            builder.store_var(target, result)

        def _getattr(context, builder, target, value, attr=None):
            builder.store_var(target, DeferredMethodCall(value, _lower_impl))

        return _getattr

    lower_getattr(mlir_string, "upper")(
        _make_lower_upper_or_lower(lower_upper)
    )
    lower_getattr(mlir_string, "lower")(
        _make_lower_upper_or_lower(lower_lower)
    )

    def _make_lower_strip(mlir_fn):
        def _lower_impl(builder, target, args, kwargs):
            str_view = _any_var_to_view(builder, args[0])
            strip_view = (
                _any_var_to_view(builder, args[1]) if len(args) > 1 else None
            )
            result = mlir_fn(builder.mlir_gpu_module, str_view, strip_view)
            builder.store_var(target, result)

        def _getattr(context, builder, target, value, attr=None):
            builder.store_var(target, DeferredMethodCall(value, _lower_impl))

        return _getattr

    lower_getattr(mlir_string, "strip")(_make_lower_strip(lower_strip))
    lower_getattr(mlir_string, "lstrip")(_make_lower_strip(lower_lstrip))
    lower_getattr(mlir_string, "rstrip")(_make_lower_strip(lower_rstrip))

    def _lower_len(builder, target, args, kwargs):
        str_view = _load_as_view(builder, args[0])
        builder.store_var(target, lower_len_value(builder, str_view))

    lower(len, mlir_string)(_lower_len)

    def _lower_contains(builder, target, args, kwargs):
        lhs = _any_var_to_view(builder, args[0])
        rhs = _any_var_to_view(builder, args[1])
        result = lower_contains(lhs, rhs)
        builder.store_var(target, result)

    lower(operator.contains, mlir_string, mlir_string)(_lower_contains)
    lower(operator.contains, mlir_string, types.StringLiteral)(_lower_contains)
    lower(operator.contains, types.StringLiteral, mlir_string)(_lower_contains)

    def _lower_concat(builder, target, args, kwargs):
        lhs = _any_var_to_view(builder, args[0])
        rhs = _any_var_to_view(builder, args[1])
        result = lower_concat(builder.mlir_gpu_module, lhs, rhs)
        builder.store_var(target, result)

    lower(operator.add, mlir_string, mlir_string)(_lower_concat)
    lower(operator.add, mlir_string, types.StringLiteral)(_lower_concat)
    lower(operator.add, types.StringLiteral, mlir_string)(_lower_concat)

    # === setitem: CPointer(mlir_string)[int] = mlir_string ===
    def _lower_setitem_cpointer_mlir_string(builder, target, args, kwargs):
        from numba_cuda_mlir.lowering_utilities import convert

        ptr, idx, val = [builder.load_var(a) for a in args]
        cs_ty = builder.get_mlir_type(mlir_string)
        ptr_ty = llvm.PointerType.get()
        GEP_DYNAMIC = -2147483648
        idx_i64 = convert(idx, T.i64())
        element_ptr = llvm.getelementptr(
            ptr_ty, ptr, [idx_i64], [GEP_DYNAMIC], cs_ty, None
        )
        llvm.store(value=val, addr=element_ptr)
        meminfo = llvm.extractvalue(ptr_ty, val, [0])
        nrt_incref_ty = mlir_ir.FunctionType.get([ptr_ty], [])
        callee = get_or_insert_function(
            "NRT_incref", nrt_incref_ty, builder.mlir_gpu_module
        )
        func.call(result=[], callee=callee.name.value, operands_=[meminfo])

    lower(
        operator.setitem,
        types.CPointer(mlir_string),
        types.Integer,
        types.Any,
    )(_lower_setitem_cpointer_mlir_string)


_register()
