# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""
MLIR (numba_cuda_mlir) lowering for scalar string_view: len(string_view) -> size_type,
string_view.isupper/.islower/.isalpha/etc. -> boolean, and string cmpops (eq, ne, lt, le, gt, ge): len/is*/casts; actual cmpops registered in mlir_masked_lowering (one impl, all signatures).
Calls shims "len", "pyisupper"/etc., and "eq"/"ne"/etc. Depends on mlir_strings_typing.
"""

from __future__ import annotations

import operator

from numba_cuda_mlir.extending import lower_cast, lowering_registry
from numba_cuda_mlir.lowering_utilities import (
    DeferredMethodCall,
    get_or_insert_function,
)
from numba_cuda_mlir._mlir import ir as mlir_ir
from numba_cuda_mlir._mlir.dialects import arith, func, llvm
from numba_cuda_mlir import types

from cudf._lib.strings_udf import (
    get_character_cases_table_ptr,
    get_character_flags_table_ptr,
    get_special_case_mapping_table_ptr,
)
from cudf.core.udf.mlir_backend.strings_typing import (
    NRT_decref,
    managed_udf_string,
    string_view,
)


def lower_len_string_view_value(builder, sv_val):
    """
    Central impl: given MLIR value of string_view, return MLIR value of length (i32).
    Used by both len(string_view) and len(Masked(string_view)) lowerings.
    """
    sv_ty = builder.get_mlir_type(string_view)
    sv_ptr = builder.alloca(sv_ty)
    llvm.store(value=sv_val, addr=sv_ptr)
    i32 = mlir_ir.IntegerType.get_signless(32)
    retval_ptr = builder.alloca(i32, count=1)
    func_ty = mlir_ir.FunctionType.get(
        inputs=[llvm.PointerType.get(), llvm.PointerType.get()],
        results=[i32],
    )
    callee = get_or_insert_function("len", func_ty, builder.mlir_gpu_module)
    func.call(
        result=[i32],
        callee=callee.name.value,
        operands_=[retval_ptr, sv_ptr],
    )
    return llvm.load(res=i32, addr=retval_ptr)


# Shim: int pyis*(bool* nb_retval, void const* str, std::uintptr_t chars_table)
_SHIM_IS_NAMES = {
    "isupper": "pyisupper",
    "islower": "pyislower",
    "isalpha": "pyisalpha",
    "isalnum": "pyisalnum",
    "isdecimal": "pyisdecimal",
    "isdigit": "pyisdigit",
    "isspace": "pyisspace",
    "isnumeric": "pyisnumeric",
    "istitle": "pyistitle",
}


def lower_string_view_is_xyz_value(builder, sv_val, shim_name):
    """
    Central impl: given MLIR value of string_view and shim name (e.g. "pyisupper"),
    return MLIR value of result (i1 boolean).
    """
    sv_ty = builder.get_mlir_type(string_view)
    sv_ptr = builder.alloca(sv_ty)
    llvm.store(value=sv_val, addr=sv_ptr)
    i1 = mlir_ir.IntegerType.get_signless(1)
    i64 = mlir_ir.IntegerType.get_signless(64)
    retval_ptr = builder.alloca(i1, count=1)
    chars_table_addr = int(get_character_flags_table_ptr())
    # MLIR IntegerAttr for i64 expects a value in signed range
    if chars_table_addr >= (1 << 63):
        chars_table_addr -= 1 << 64
    chars_table_const = arith.constant(i64, chars_table_addr)
    i32 = mlir_ir.IntegerType.get_signless(32)
    func_ty = mlir_ir.FunctionType.get(
        inputs=[llvm.PointerType.get(), llvm.PointerType.get(), i64],
        results=[i32],  # shim returns int (status)
    )
    callee = get_or_insert_function(
        shim_name, func_ty, builder.mlir_gpu_module
    )
    func.call(
        result=[i32],
        callee=callee.name.value,
        operands_=[retval_ptr, sv_ptr, chars_table_const],
    )
    return llvm.load(res=i1, addr=retval_ptr)


def lower_managed_to_sv_value(builder, managed_val):
    """Convert a managed_udf_string MLIR value to a string_view MLIR value.

    Calls the C++ shim that extracts a string_view from the udf_string
    embedded in the managed struct.  Safe to call inline within any
    lowering body -- the managed string remains alive because Numba's
    del insertion operates at the IR-node level, not mid-lowering.
    """
    ptr_ty = llvm.PointerType.get()
    i32 = mlir_ir.IntegerType.get_signless(32)
    managed_ty = builder.get_mlir_type(managed_udf_string)
    sv_ty = builder.get_mlir_type(string_view)

    managed_ptr = builder.alloca(managed_ty)
    llvm.store(value=managed_val, addr=managed_ptr)
    udf_str_ptr = llvm.getelementptr(
        ptr_ty, managed_ptr, [], [0, 1], managed_ty, None
    )

    sv_ptr = builder.alloca(sv_ty)
    retval_ptr = builder.alloca(i32)
    func_ty = mlir_ir.FunctionType.get(
        inputs=[ptr_ty, ptr_ty, ptr_ty], results=[i32]
    )
    callee = get_or_insert_function(
        "string_view_from_udf_string", func_ty, builder.mlir_gpu_module
    )
    func.call(
        result=[i32],
        callee=callee.name.value,
        operands_=[retval_ptr, udf_str_ptr, sv_ptr],
    )
    return llvm.load(res=sv_ty, addr=sv_ptr)


def _literal_to_string_view_value(builder, literal_var):
    """
    Materialize a StringLiteral variable as a string_view struct value (SSA value).
    Used by the cast Literal -> string_view and by literal_to_string_view_ptr.
    """
    literal_ty = builder.get_numba_type(literal_var.name)
    if not isinstance(literal_ty, types.StringLiteral):
        raise TypeError("expects StringLiteral variable")
    literal_value = literal_ty.literal_value
    data_ptr = builder.load_var(literal_var)
    sv_ty = builder.get_mlir_type(string_view)
    i32 = mlir_ir.IntegerType.get_signless(32)
    bytes_val = len(literal_value.encode("UTF-8"))
    length_val = len(literal_value)
    undef = llvm.UndefOp(sv_ty)
    with_data = llvm.insertvalue(
        container=undef,
        value=llvm.extractvalue(llvm.PointerType.get(), data_ptr, [0]),
        position=mlir_ir.DenseI64ArrayAttr.get([0]),
    )
    with_bytes = llvm.insertvalue(
        container=with_data,
        value=arith.constant(i32, bytes_val),
        position=mlir_ir.DenseI64ArrayAttr.get([1]),
    )
    with_length = llvm.insertvalue(
        container=with_bytes,
        value=arith.constant(i32, length_val),
        position=mlir_ir.DenseI64ArrayAttr.get([2]),
    )
    return with_length


def literal_to_string_view_ptr(builder, literal_var):
    """
    Materialize a StringLiteral variable as a string_view struct and return
    a pointer to it (for passing to shims that take void const* string_view).
    """
    sv_val = _literal_to_string_view_value(builder, literal_var)
    sv_ty = builder.get_mlir_type(string_view)
    sv_ptr = builder.alloca(sv_ty)
    llvm.store(value=sv_val, addr=sv_ptr)
    return sv_ptr


# Shim: int eq(bool* nb_retval, void const* str, void const* rhs) etc.
_CMPOP_SHIMS = ("eq", "ne", "lt", "le", "gt", "ge")

# Bool-returning (bool* retval, str, substr): contains, startswith, endswith
_BOOL_STR_STR_SHIMS = ("contains", "startswith", "endswith")

# Int-returning (int* retval, str, substr): find, rfind, count (shim name "pycount")
_INT_STR_STR_SHIMS = (
    ("find", "find"),
    ("rfind", "rfind"),
    ("count", "pycount"),
)


def call_string_cmpop_shim(builder, lhs_sv_ptr, rhs_sv_ptr, shim_name):
    """
    Call shim (eq/ne/lt/le/gt/ge) with two string_view pointers; return i1 result.
    Used by both scalar and Masked(sv) vs Literal lowerings.
    """
    i1 = mlir_ir.IntegerType.get_signless(1)
    i32 = mlir_ir.IntegerType.get_signless(32)
    ptr_ty = llvm.PointerType.get()
    retval_ptr = builder.alloca(i1, count=1)
    func_ty = mlir_ir.FunctionType.get(
        inputs=[ptr_ty, ptr_ty, ptr_ty],
        results=[i32],
    )
    callee = get_or_insert_function(
        shim_name, func_ty, builder.mlir_gpu_module
    )
    func.call(
        result=[i32],
        callee=callee.name.value,
        operands_=[retval_ptr, lhs_sv_ptr, rhs_sv_ptr],
    )
    return llvm.load(res=i1, addr=retval_ptr)


def call_string_bool_str_str_shim(builder, lhs_sv_ptr, rhs_sv_ptr, shim_name):
    """
    Call shim (contains/startswith/endswith) with two string_view pointers; return i1.
    Same C signature as cmpops: int shim(bool* retval, void const* str, void const* substr).
    """
    return call_string_cmpop_shim(builder, lhs_sv_ptr, rhs_sv_ptr, shim_name)


def call_string_int_str_str_shim(builder, lhs_sv_ptr, rhs_sv_ptr, shim_name):
    """
    Call shim (find/rfind) with two string_view pointers; return i32 (size_type).
    """
    i32 = mlir_ir.IntegerType.get_signless(32)
    ptr_ty = llvm.PointerType.get()
    retval_ptr = builder.alloca(i32, count=1)
    func_ty = mlir_ir.FunctionType.get(
        inputs=[ptr_ty, ptr_ty, ptr_ty],
        results=[i32],
    )
    callee = get_or_insert_function(
        shim_name, func_ty, builder.mlir_gpu_module
    )
    func.call(
        result=[i32],
        callee=callee.name.value,
        operands_=[retval_ptr, lhs_sv_ptr, rhs_sv_ptr],
    )
    return llvm.load(res=i32, addr=retval_ptr)


def _get_lhs_rhs_sv_ptrs(builder, lhs_var, rhs_var):
    """Get (lhs_sv_ptr, rhs_sv_ptr) for two args that may be string_view or StringLiteral."""
    sv_ty = builder.get_mlir_type(string_view)
    if isinstance(builder.get_numba_type(lhs_var.name), types.StringLiteral):
        lhs_ptr = literal_to_string_view_ptr(builder, lhs_var)
    else:
        lhs_val = builder.load_var(lhs_var)
        lhs_ptr = builder.alloca(sv_ty)
        llvm.store(value=lhs_val, addr=lhs_ptr)
    if isinstance(builder.get_numba_type(rhs_var.name), types.StringLiteral):
        rhs_ptr = literal_to_string_view_ptr(builder, rhs_var)
    else:
        rhs_val = builder.load_var(rhs_var)
        rhs_ptr = builder.alloca(sv_ty)
        llvm.store(value=rhs_val, addr=rhs_ptr)
    return lhs_ptr, rhs_ptr


def _call_shim_returning_managed(builder, shim_name, input_ptrs):
    """
    Call a C shim that produces a managed_udf_string.

    All string-producing shims follow the convention:
        int shim(void** out_meminfo, void* udf_str_placement, ...)
    The shim placement-news a udf_string into *udf_str_placement*, allocates
    a heap MemInfo+udf_string pair, memcpys the placement result there, and
    writes the MemInfo* to *out_meminfo*.

    Returns an MLIR SSA value of type managed_udf_string.
    """
    ptr_ty = llvm.PointerType.get()
    i32 = mlir_ir.IntegerType.get_signless(32)
    managed_ty = builder.get_mlir_type(managed_udf_string)

    # Stack space for the managed struct (shim writes udf_string into field 1)
    managed_ptr = builder.alloca(managed_ty)
    # Pointer to the embedded udf_string field (field 1 of managed struct)
    udf_str_ptr = llvm.getelementptr(
        ptr_ty, managed_ptr, [], [0, 1], managed_ty, None
    )
    # Stack space for the meminfo return (void**)
    nb_retval_ptr = builder.alloca(ptr_ty)

    # Build function type: (void**, void*, <input_ptrs...>) -> i32
    func_inputs = [ptr_ty, ptr_ty] + [ptr_ty] * len(input_ptrs)
    func_ty = mlir_ir.FunctionType.get(inputs=func_inputs, results=[i32])
    callee = get_or_insert_function(
        shim_name, func_ty, builder.mlir_gpu_module
    )
    func.call(
        result=[i32],
        callee=callee.name.value,
        operands_=[nb_retval_ptr, udf_str_ptr] + list(input_ptrs),
    )

    # Load meminfo from the retval slot
    meminfo = llvm.load(res=ptr_ty, addr=nb_retval_ptr)
    # Load the struct from stack (udf_string fields populated by placement new)
    managed_val = llvm.load(res=managed_ty, addr=managed_ptr)
    # Insert the meminfo pointer into field 0
    result = llvm.insertvalue(
        container=managed_val,
        value=meminfo,
        position=mlir_ir.DenseI64ArrayAttr.get([0]),
    )
    return result


def _sv_var_to_ptr(builder, var):
    """Convert a string_view or StringLiteral variable to a stack pointer."""
    if isinstance(builder.get_numba_type(var.name), types.StringLiteral):
        return literal_to_string_view_ptr(builder, var)
    sv_ty = builder.get_mlir_type(string_view)
    sv_val = builder.load_var(var)
    ptr = builder.alloca(sv_ty)
    llvm.store(value=sv_val, addr=ptr)
    return ptr


def _sv_val_to_ptr(builder, sv_val):
    """Store a string_view SSA value to stack and return the pointer."""
    sv_ty = builder.get_mlir_type(string_view)
    ptr = builder.alloca(sv_ty)
    llvm.store(value=sv_val, addr=ptr)
    return ptr


def _i64_const_from_ptr(addr):
    """Create an i64 constant from a host pointer address (for lookup tables)."""
    i64 = mlir_ir.IntegerType.get_signless(64)
    if addr >= (1 << 63):
        addr -= 1 << 64
    return arith.constant(i64, addr)


def _register():
    lower = lowering_registry.lower
    lower_getattr = lowering_registry.lower_getattr

    def _lower_len_string_view(builder, target, args, kwargs):
        sv_var = args[0]
        sv_val = builder.load_var(sv_var)
        len_result = lower_len_string_view_value(builder, sv_val)
        builder.store_var(target, len_result)

    lower(len, string_view)(_lower_len_string_view)

    # --- string_view.isupper, .islower, .isalpha, etc. -> boolean ---
    for attrname, shim_name in _SHIM_IS_NAMES.items():

        def _make_lower_is(shim_n):
            def _lower_is_impl(builder, target, args, kwargs):
                sv_var = args[0]
                sv_val = builder.load_var(sv_var)
                result = lower_string_view_is_xyz_value(
                    builder, sv_val, shim_n
                )
                builder.store_var(target, result)

            def _getattr(context, builder, target, value, attr=None):
                builder.store_var(
                    target, DeferredMethodCall(value, _lower_is_impl)
                )

            return _getattr

        lower_getattr(string_view, attrname)(_make_lower_is(shim_name))

    # --- string_view.find, .rfind (-> size_type); .startswith, .endswith (-> boolean) ---
    def _make_lower_binary_str_str_int(shim_n):
        def _lower_impl(builder, target, args, kwargs):
            lhs_ptr, rhs_ptr = _get_lhs_rhs_sv_ptrs(builder, args[0], args[1])
            result = call_string_int_str_str_shim(
                builder, lhs_ptr, rhs_ptr, shim_n
            )
            builder.store_var(target, result)

        def _getattr(context, builder, target, value, attr=None):
            builder.store_var(target, DeferredMethodCall(value, _lower_impl))

        return _getattr

    def _make_lower_binary_str_str_bool(shim_n):
        def _lower_impl(builder, target, args, kwargs):
            lhs_ptr, rhs_ptr = _get_lhs_rhs_sv_ptrs(builder, args[0], args[1])
            result = call_string_bool_str_str_shim(
                builder, lhs_ptr, rhs_ptr, shim_n
            )
            builder.store_var(target, result)

        def _getattr(context, builder, target, value, attr=None):
            builder.store_var(target, DeferredMethodCall(value, _lower_impl))

        return _getattr

    for attrname, shim_name in _INT_STR_STR_SHIMS:
        lower_getattr(string_view, attrname)(
            _make_lower_binary_str_str_int(shim_name)
        )
    for attrname in _BOOL_STR_STR_SHIMS:
        lower_getattr(string_view, attrname)(
            _make_lower_binary_str_str_bool(attrname)
        )

    # --- operator.contains (string_view, string_view) and (string_view, StringLiteral), (StringLiteral, string_view) -> boolean ---
    def _lower_contains(builder, target, args, kwargs):
        lhs_ptr, rhs_ptr = _get_lhs_rhs_sv_ptrs(builder, args[0], args[1])
        result = call_string_bool_str_str_shim(
            builder, lhs_ptr, rhs_ptr, "contains"
        )
        builder.store_var(target, result)

    lower(operator.contains, string_view, string_view)(_lower_contains)
    lower(operator.contains, string_view, types.StringLiteral)(_lower_contains)
    lower(operator.contains, types.StringLiteral, string_view)(_lower_contains)

    # --- NRT_decref(managed_udf_string) -> void ---
    # Must call NRT_decref_managed_string (not NRT_decref) so that the
    # dtor function pointer is reset to the current module's udf_str_dtor
    # before decrementing.  Device function pointers are module-local in
    # CUDA, so the dtor stored by the UDF kernel module would be invalid
    # when called from a different module (e.g. the free kernel).
    def _lower_nrt_decref(builder, target, args, kwargs):
        managed_val = builder.load_var(args[0])
        meminfo = llvm.extractvalue(llvm.PointerType.get(), managed_val, [0])
        nrt_decref_ty = mlir_ir.FunctionType.get([llvm.PointerType.get()], [])
        callee = get_or_insert_function(
            "NRT_decref_managed_string",
            nrt_decref_ty,
            builder.mlir_gpu_module,
        )
        func.call(result=[], callee=callee.name.value, operands_=[meminfo])

    lower(NRT_decref, managed_udf_string)(_lower_nrt_decref)

    # String cmpops (all signatures) are registered in mlir_masked_lowering with one unified impl.

    # --- Cast: Literal -> string_view ---
    @lower_cast(types.StringLiteral, string_view)
    def cast_string_literal_to_string_view(
        context, builder, fromty, toty, val
    ):
        sv_ty = builder.get_mlir_type(string_view)
        i32 = mlir_ir.IntegerType.get_signless(32)
        bytes_val = len(fromty.literal_value.encode("UTF-8"))
        length_val = len(fromty.literal_value)
        undef = llvm.UndefOp(sv_ty)
        with_data = llvm.insertvalue(
            container=undef,
            value=val,
            position=mlir_ir.DenseI64ArrayAttr.get([0]),
        )
        with_bytes = llvm.insertvalue(
            container=with_data,
            value=arith.constant(i32, bytes_val),
            position=mlir_ir.DenseI64ArrayAttr.get([1]),
        )
        with_length = llvm.insertvalue(
            container=with_bytes,
            value=arith.constant(i32, length_val),
            position=mlir_ir.DenseI64ArrayAttr.get([2]),
        )
        return with_length

    # --- Cast: string_view -> managed_udf_string ---
    # Shim: int udf_string_from_string_view(void** out_meminfo, void const* sv, void* udf_str)
    @lower_cast(string_view, managed_udf_string)
    def cast_string_view_to_managed(context, builder, fromty, toty, val):
        ptr_ty = llvm.PointerType.get()
        i32 = mlir_ir.IntegerType.get_signless(32)
        managed_ty = builder.get_mlir_type(managed_udf_string)

        sv_ptr = _sv_val_to_ptr(builder, val)

        managed_ptr = builder.alloca(managed_ty)
        udf_str_ptr = llvm.getelementptr(
            ptr_ty, managed_ptr, [], [0, 1], managed_ty, None
        )
        nb_retval_ptr = builder.alloca(ptr_ty)

        func_ty = mlir_ir.FunctionType.get(
            inputs=[ptr_ty, ptr_ty, ptr_ty], results=[i32]
        )
        callee = get_or_insert_function(
            "udf_string_from_string_view", func_ty, builder.mlir_gpu_module
        )
        func.call(
            result=[i32],
            callee=callee.name.value,
            operands_=[nb_retval_ptr, sv_ptr, udf_str_ptr],
        )
        meminfo = llvm.load(res=ptr_ty, addr=nb_retval_ptr)
        managed_val = llvm.load(res=managed_ty, addr=managed_ptr)
        return llvm.insertvalue(
            container=managed_val,
            value=meminfo,
            position=mlir_ir.DenseI64ArrayAttr.get([0]),
        )

    # --- Cast: managed_udf_string -> string_view ---
    # Shim: int string_view_from_udf_string(int* nb_retval, void const* udf_str, void* sv)
    @lower_cast(managed_udf_string, string_view)
    def cast_managed_to_string_view(context, builder, fromty, toty, val):
        return lower_managed_to_sv_value(builder, val)

    # --- operator.add (concat): string_view + string_view -> managed_udf_string ---
    def _lower_concat(builder, target, args, kwargs):
        lhs_ptr = _sv_var_to_ptr(builder, args[0])
        rhs_ptr = _sv_var_to_ptr(builder, args[1])
        result = _call_shim_returning_managed(
            builder, "concat", [lhs_ptr, rhs_ptr]
        )
        builder.store_var(target, result)

    lower(operator.add, string_view, string_view)(_lower_concat)
    lower(operator.add, string_view, types.StringLiteral)(_lower_concat)
    lower(operator.add, types.StringLiteral, string_view)(_lower_concat)

    # --- StringView.replace(old, new) -> managed_udf_string ---
    # Shim: int replace(void** out, void* udf_str, void* src, void* old, void* new)
    def _lower_replace_impl(builder, target, args, kwargs):
        src_ptr = _sv_var_to_ptr(builder, args[0])
        old_ptr = _sv_var_to_ptr(builder, args[1])
        new_ptr = _sv_var_to_ptr(builder, args[2])
        result = _call_shim_returning_managed(
            builder, "replace", [src_ptr, old_ptr, new_ptr]
        )
        builder.store_var(target, result)

    def _lower_replace_getattr(context, builder, target, value, attr=None):
        builder.store_var(
            target, DeferredMethodCall(value, _lower_replace_impl)
        )

    lower_getattr(string_view, "replace")(_lower_replace_getattr)

    # --- StringView.upper / StringView.lower -> managed_udf_string ---
    # Shim: int upper(void** out, void* udf_str, void const* sv, uintptr_t flags, uintptr_t cases, uintptr_t special)
    def _make_lower_upper_or_lower(shim_name):
        def _lower_impl(builder, target, args, kwargs):
            sv_ptr = _sv_var_to_ptr(builder, args[0])
            i64 = mlir_ir.IntegerType.get_signless(64)
            ptr_ty = llvm.PointerType.get()
            i32 = mlir_ir.IntegerType.get_signless(32)
            managed_ty = builder.get_mlir_type(managed_udf_string)

            flags_const = _i64_const_from_ptr(
                int(get_character_flags_table_ptr())
            )
            cases_const = _i64_const_from_ptr(
                int(get_character_cases_table_ptr())
            )
            special_const = _i64_const_from_ptr(
                int(get_special_case_mapping_table_ptr())
            )

            managed_ptr = builder.alloca(managed_ty)
            udf_str_ptr = llvm.getelementptr(
                ptr_ty, managed_ptr, [], [0, 1], managed_ty, None
            )
            nb_retval_ptr = builder.alloca(ptr_ty)

            func_ty = mlir_ir.FunctionType.get(
                inputs=[ptr_ty, ptr_ty, ptr_ty, i64, i64, i64],
                results=[i32],
            )
            callee = get_or_insert_function(
                shim_name, func_ty, builder.mlir_gpu_module
            )
            func.call(
                result=[i32],
                callee=callee.name.value,
                operands_=[
                    nb_retval_ptr,
                    udf_str_ptr,
                    sv_ptr,
                    flags_const,
                    cases_const,
                    special_const,
                ],
            )

            meminfo = llvm.load(res=ptr_ty, addr=nb_retval_ptr)
            managed_val = llvm.load(res=managed_ty, addr=managed_ptr)
            result = llvm.insertvalue(
                container=managed_val,
                value=meminfo,
                position=mlir_ir.DenseI64ArrayAttr.get([0]),
            )
            builder.store_var(target, result)

        def _getattr(context, builder, target, value, attr=None):
            builder.store_var(target, DeferredMethodCall(value, _lower_impl))

        return _getattr

    lower_getattr(string_view, "upper")(_make_lower_upper_or_lower("upper"))
    lower_getattr(string_view, "lower")(_make_lower_upper_or_lower("lower"))

    # --- StringView.strip / lstrip / rstrip -> managed_udf_string ---
    # Shim: int strip(void** out, void* udf_str, void const* to_strip, void const* strip_char)
    def _make_lower_strip(shim_name):
        def _lower_impl(builder, target, args, kwargs):
            src_ptr = _sv_var_to_ptr(builder, args[0])
            chars_ptr = _sv_var_to_ptr(builder, args[1])
            result = _call_shim_returning_managed(
                builder, shim_name, [src_ptr, chars_ptr]
            )
            builder.store_var(target, result)

        def _getattr(context, builder, target, value, attr=None):
            builder.store_var(target, DeferredMethodCall(value, _lower_impl))

        return _getattr

    lower_getattr(string_view, "strip")(_make_lower_strip("strip"))
    lower_getattr(string_view, "lstrip")(_make_lower_strip("lstrip"))
    lower_getattr(string_view, "rstrip")(_make_lower_strip("rstrip"))

    # --- managed_udf_string methods ---
    # Each method converts managed → string_view internally, then dispatches
    # to the same string_view lowering helpers.  The conversion happens within
    # the lowering body so the managed string stays alive (del can't fire
    # mid-lowering).

    def _managed_sv_from_var(builder, var):
        """Load a managed_udf_string var, convert to string_view value."""
        return lower_managed_to_sv_value(builder, builder.load_var(var))

    def _make_lower_managed_is(shim_n):
        def _lower_impl(builder, target, args, kwargs):
            sv_val = _managed_sv_from_var(builder, args[0])
            result = lower_string_view_is_xyz_value(builder, sv_val, shim_n)
            builder.store_var(target, result)

        def _getattr(context, builder, target, value, attr=None):
            builder.store_var(target, DeferredMethodCall(value, _lower_impl))

        return _getattr

    for attrname, shim_name in _SHIM_IS_NAMES.items():
        lower_getattr(managed_udf_string, attrname)(
            _make_lower_managed_is(shim_name)
        )

    def _make_lower_managed_binary_str_str_int(shim_n):
        def _lower_impl(builder, target, args, kwargs):
            lhs_ptr = _sv_val_to_ptr(builder, _managed_sv_from_var(builder, args[0]))
            rhs_ptr = _sv_var_to_ptr(builder, args[1])
            result = call_string_int_str_str_shim(builder, lhs_ptr, rhs_ptr, shim_n)
            builder.store_var(target, result)

        def _getattr(context, builder, target, value, attr=None):
            builder.store_var(target, DeferredMethodCall(value, _lower_impl))

        return _getattr

    for attrname, shim_name in _INT_STR_STR_SHIMS:
        lower_getattr(managed_udf_string, attrname)(
            _make_lower_managed_binary_str_str_int(shim_name)
        )

    def _make_lower_managed_binary_str_str_bool(shim_n):
        def _lower_impl(builder, target, args, kwargs):
            lhs_ptr = _sv_val_to_ptr(builder, _managed_sv_from_var(builder, args[0]))
            rhs_ptr = _sv_var_to_ptr(builder, args[1])
            result = call_string_bool_str_str_shim(builder, lhs_ptr, rhs_ptr, shim_n)
            builder.store_var(target, result)

        def _getattr(context, builder, target, value, attr=None):
            builder.store_var(target, DeferredMethodCall(value, _lower_impl))

        return _getattr

    for attrname in _BOOL_STR_STR_SHIMS:
        lower_getattr(managed_udf_string, attrname)(
            _make_lower_managed_binary_str_str_bool(attrname)
        )

    def _lower_managed_replace_impl(builder, target, args, kwargs):
        src_ptr = _sv_val_to_ptr(builder, _managed_sv_from_var(builder, args[0]))
        old_ptr = _sv_var_to_ptr(builder, args[1])
        new_ptr = _sv_var_to_ptr(builder, args[2])
        result = _call_shim_returning_managed(
            builder, "replace", [src_ptr, old_ptr, new_ptr]
        )
        builder.store_var(target, result)

    def _lower_managed_replace_getattr(context, builder, target, value, attr=None):
        builder.store_var(
            target, DeferredMethodCall(value, _lower_managed_replace_impl)
        )

    lower_getattr(managed_udf_string, "replace")(_lower_managed_replace_getattr)

    def _make_lower_managed_upper_or_lower(shim_name):
        def _lower_impl(builder, target, args, kwargs):
            sv_val = _managed_sv_from_var(builder, args[0])
            sv_ptr = _sv_val_to_ptr(builder, sv_val)
            i64 = mlir_ir.IntegerType.get_signless(64)
            ptr_ty = llvm.PointerType.get()
            i32 = mlir_ir.IntegerType.get_signless(32)
            managed_ty = builder.get_mlir_type(managed_udf_string)

            flags_const = _i64_const_from_ptr(int(get_character_flags_table_ptr()))
            cases_const = _i64_const_from_ptr(int(get_character_cases_table_ptr()))
            special_const = _i64_const_from_ptr(int(get_special_case_mapping_table_ptr()))

            managed_ptr = builder.alloca(managed_ty)
            udf_str_ptr = llvm.getelementptr(
                ptr_ty, managed_ptr, [], [0, 1], managed_ty, None
            )
            nb_retval_ptr = builder.alloca(ptr_ty)

            func_ty = mlir_ir.FunctionType.get(
                inputs=[ptr_ty, ptr_ty, ptr_ty, i64, i64, i64],
                results=[i32],
            )
            callee = get_or_insert_function(
                shim_name, func_ty, builder.mlir_gpu_module
            )
            func.call(
                result=[i32],
                callee=callee.name.value,
                operands_=[
                    nb_retval_ptr, udf_str_ptr, sv_ptr,
                    flags_const, cases_const, special_const,
                ],
            )

            meminfo = llvm.load(res=ptr_ty, addr=nb_retval_ptr)
            managed_val = llvm.load(res=managed_ty, addr=managed_ptr)
            result = llvm.insertvalue(
                container=managed_val,
                value=meminfo,
                position=mlir_ir.DenseI64ArrayAttr.get([0]),
            )
            builder.store_var(target, result)

        def _getattr(context, builder, target, value, attr=None):
            builder.store_var(target, DeferredMethodCall(value, _lower_impl))

        return _getattr

    lower_getattr(managed_udf_string, "upper")(
        _make_lower_managed_upper_or_lower("upper")
    )
    lower_getattr(managed_udf_string, "lower")(
        _make_lower_managed_upper_or_lower("lower")
    )

    def _make_lower_managed_strip(shim_name):
        def _lower_impl(builder, target, args, kwargs):
            src_ptr = _sv_val_to_ptr(builder, _managed_sv_from_var(builder, args[0]))
            chars_ptr = _sv_var_to_ptr(builder, args[1])
            result = _call_shim_returning_managed(
                builder, shim_name, [src_ptr, chars_ptr]
            )
            builder.store_var(target, result)

        def _getattr(context, builder, target, value, attr=None):
            builder.store_var(target, DeferredMethodCall(value, _lower_impl))

        return _getattr

    lower_getattr(managed_udf_string, "strip")(_make_lower_managed_strip("strip"))
    lower_getattr(managed_udf_string, "lstrip")(_make_lower_managed_strip("lstrip"))
    lower_getattr(managed_udf_string, "rstrip")(_make_lower_managed_strip("rstrip"))

    # --- setitem: CPointer(managed_udf_string)[int] = managed_udf_string ---
    # Stores the struct into the output array AND increfs the meminfo,
    # because storing creates a new reference.
    def _lower_setitem_cpointer_managed(builder, target, args, kwargs):
        from numba_cuda_mlir.lowering_utilities import convert
        from numba_cuda_mlir._mlir.extras import types as T

        ptr, idx, val = [builder.load_var(a) for a in args]
        managed_ty = builder.get_mlir_type(managed_udf_string)
        ptr_ty = llvm.PointerType.get()
        GEP_DYNAMIC = -2147483648
        idx_i64 = convert(idx, T.i64())
        element_ptr = llvm.getelementptr(
            ptr_ty, ptr, [idx_i64], [GEP_DYNAMIC], managed_ty, None
        )
        llvm.store(value=val, addr=element_ptr)
        # Incref: storing into the output array creates a new reference
        meminfo = llvm.extractvalue(ptr_ty, val, [0])
        nrt_incref_ty = mlir_ir.FunctionType.get([ptr_ty], [])
        callee = get_or_insert_function(
            "NRT_incref", nrt_incref_ty, builder.mlir_gpu_module
        )
        func.call(result=[], callee=callee.name.value, operands_=[meminfo])

    lower(
        operator.setitem,
        types.CPointer(managed_udf_string),
        types.Integer,
        types.Any,
    )(_lower_setitem_cpointer_managed)


_register()
