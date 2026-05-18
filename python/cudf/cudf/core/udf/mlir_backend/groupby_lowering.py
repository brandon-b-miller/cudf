# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

from numba_cuda_mlir import types
from numba_cuda_mlir._mlir import ir as mlir_ir
from numba_cuda_mlir._mlir.dialects import arith, llvm, memref
from numba_cuda_mlir._mlir.extras import types as T
from numba_cuda_mlir.extending import lowering_registry
from numba_cuda_mlir.lowering_utilities import DeferredMethodCall, convert
from numba_cuda_mlir.numba_cuda.core import ir as numba_ir

from cudf.core.udf.mlir_backend.groupby_typing import (
    Group,
    GroupType,
    _group_slot,
    call_block_functions,
    group_size_type,
    index_default_type,
)

lower = lowering_registry.lower
lower_getattr = lowering_registry.lower_getattr


def _make_temp_var(builder, base_var, name_suffix, numba_type):
    """Create a temp var and register its type. Caller must store_var(temp, value) after."""
    scope = getattr(base_var, "scope", None)
    loc = getattr(base_var, "loc", None)
    name = f"$group_{base_var.name}_{name_suffix}"
    temp = numba_ir.Var(scope=scope, name=name, loc=loc)
    builder.fndesc.typemap[temp.name] = numba_type
    return temp


# --- _group_slot: allocate record and store pointer (like _row_slot) ---
def _lower_group_slot_impl(builder, target, group_type):
    """Allocate record_size bytes and store the pointer in target (Record = ptr to bytes)."""
    record_size = group_type.size
    i8 = builder.get_mlir_type(types.uint8)
    ptr = builder.alloca(i8, count=record_size)
    builder.store_var(target, ptr)


def register_group_slot_lowering(dataframe_group_type):
    """Register lowering for _group_slot() for this kernel's group type."""
    lower(_group_slot, *())(
        lambda builder, target, args, kwargs: _lower_group_slot_impl(
            builder, target, dataframe_group_type
        )
    )


def _get_base_ptr_from_array(builder, array_val, elem_numba_type):
    """
    Get LLVM pointer to the first element of the (possibly sliced) memref.
    Uses extract_strided_metadata so slices (e.g. input_col[offset[b]:offset[b+1]])
    get the correct data pointer, not the whole column's base.
    """
    if not isinstance(array_val.type, mlir_ir.MemRefType):
        return array_val
    md = memref.extract_strided_metadata(array_val)
    base_memref = md[0]
    offset = md[1]
    base_idx = memref.extract_aligned_pointer_as_index(base_memref)
    base_i64 = arith.index_cast(T.i64(), base_idx)
    elem_mlir = mlir_ir.MemRefType(array_val.type).element_type
    if isinstance(elem_mlir, mlir_ir.IntegerType):
        elem_bytes = elem_mlir.width // 8
    elif isinstance(elem_mlir, mlir_ir.FloatType):
        elem_bytes = elem_mlir.width // 8
    else:
        elem_bytes = 8
    offset_i64 = convert(offset, T.i64())
    offset_bytes = arith.muli(offset_i64, arith.constant(T.i64(), elem_bytes))
    data_i64 = arith.addi(base_i64, offset_bytes)
    return llvm.inttoptr(res=llvm.PointerType.get(), arg=data_i64)


# --- Group constructor: build struct (data_ptr, size, index_ptr) and store in target ---
@lower(Group, types.Array, group_size_type, types.Array)
def group_constructor(builder, target, args, kwargs):
    target_type = builder.get_numba_type(target.name)
    struct_ty = builder.get_mlir_type(target_type)
    data_array = builder.load_var(args[0])
    size_val = builder.load_var(args[1])
    index_array = builder.load_var(args[2])
    grp_type = target_type
    data_ptr = _get_base_ptr_from_array(
        builder, data_array, grp_type.group_scalar_type
    )
    index_ptr = _get_base_ptr_from_array(
        builder, index_array, grp_type.index_type
    )
    size_mlir = convert(size_val, builder.get_mlir_type(types.int64))
    undef = llvm.UndefOp(struct_ty)
    with_data = llvm.insertvalue(
        container=undef,
        value=data_ptr,
        position=mlir_ir.DenseI64ArrayAttr.get([0]),
    )
    with_size = llvm.insertvalue(
        container=with_data,
        value=size_mlir,
        position=mlir_ir.DenseI64ArrayAttr.get([1]),
    )
    with_index = llvm.insertvalue(
        container=with_size,
        value=index_ptr,
        position=mlir_ir.DenseI64ArrayAttr.get([2]),
    )
    builder.store_var(target, with_index)


# --- GroupType: struct field getattr (group_data=0, index=2) ---
def _lower_group_type_field(
    context, builder, target, value, attr, field_index, field_numba_type_fn
):
    struct_val = builder.load_var(value)
    grp_type = builder.get_numba_type(value.name)
    field_numba_type = field_numba_type_fn(grp_type)
    field_mlir_ty = builder.get_mlir_type(field_numba_type)
    field_val = llvm.extractvalue(
        res=field_mlir_ty,
        container=struct_val,
        position=mlir_ir.DenseI64ArrayAttr.get([field_index]),
    )
    field_val = convert(field_val, field_mlir_ty)
    builder.store_var(target, field_val)


@lower_getattr(GroupType, "group_data")
def _lower_group_type_group_data(context, builder, target, value, attr=None):
    _lower_group_type_field(
        context,
        builder,
        target,
        value,
        "group_data",
        0,
        lambda g: g.group_data_type,
    )


@lower_getattr(GroupType, "index")
def _lower_group_type_index(context, builder, target, value, attr=None):
    _lower_group_type_field(
        context,
        builder,
        target,
        value,
        "index",
        2,
        lambda g: g.group_index_type,
    )


# --- GroupType: unary reductions (sum, max, min, mean, var, std) via Block* shim ---
def _lower_group_reduction_shim(builder, target, args, kwargs, funcname):
    grp_var = args[0]
    grp_type = builder.get_numba_type(grp_var.name)
    retty = builder.get_numba_type(target.name)
    type_key = (retty, grp_type.group_scalar_type)
    funcs = call_block_functions[funcname]
    extfn = funcs[type_key]

    struct_val = builder.load_var(grp_var)
    data_ptr = llvm.extractvalue(
        res=builder.get_mlir_type(grp_type.group_data_type),
        container=struct_val,
        position=mlir_ir.DenseI64ArrayAttr.get([0]),
    )
    size_val = llvm.extractvalue(
        res=builder.get_mlir_type(types.int64),
        container=struct_val,
        position=mlir_ir.DenseI64ArrayAttr.get([1]),
    )
    data_var = _make_temp_var(
        builder, grp_var, "data", grp_type.group_data_type
    )
    size_var = _make_temp_var(builder, grp_var, "size", group_size_type)
    builder.store_var(data_var, data_ptr)
    builder.store_var(size_var, size_val)
    builder.lower_call_external_function(
        target, extfn, [data_var, size_var], {}
    )


def _make_reduction_getattr(funcname):
    def _lower(builder, target, args, kwargs):
        return _lower_group_reduction_shim(
            builder, target, args, kwargs, funcname
        )

    def _getattr(context, builder, target, value, attr=None):
        builder.store_var(target, DeferredMethodCall(value, _lower))

    return _getattr


for _method in ("sum", "max", "min", "mean", "var", "std"):
    lower_getattr(GroupType, _method)(_make_reduction_getattr(_method))


# --- GroupType.size / GroupType.count: return struct field 1 (size) ---
def _lower_group_size_or_count(builder, target, args, kwargs):
    struct_val = builder.load_var(args[0])
    size_val = llvm.extractvalue(
        res=builder.get_mlir_type(types.int64),
        container=struct_val,
        position=mlir_ir.DenseI64ArrayAttr.get([1]),
    )
    size_val = convert(
        size_val,
        builder.get_mlir_type(builder.get_numba_type(target.name)),
    )
    builder.store_var(target, size_val)


def _make_method_getattr_lower(lowering_fn):
    def _getattr(context, builder, target, value, attr=None):
        builder.store_var(target, DeferredMethodCall(value, lowering_fn))

    return _getattr


lower_getattr(GroupType, "size")(
    _make_method_getattr_lower(_lower_group_size_or_count)
)
lower_getattr(GroupType, "count")(
    _make_method_getattr_lower(_lower_group_size_or_count)
)


# --- GroupType.idxmax / GroupType.idxmin: extract data, index, size; call BlockIdx* ---
def _lower_group_idx_shim(builder, target, args, kwargs, funcname):
    grp_var = args[0]
    grp_type = builder.get_numba_type(grp_var.name)
    type_key = (index_default_type, grp_type.group_scalar_type)
    funcs = call_block_functions[funcname]
    extfn = funcs[type_key]

    struct_val = builder.load_var(grp_var)
    data_ptr = llvm.extractvalue(
        res=builder.get_mlir_type(grp_type.group_data_type),
        container=struct_val,
        position=mlir_ir.DenseI64ArrayAttr.get([0]),
    )
    size_val = llvm.extractvalue(
        res=builder.get_mlir_type(types.int64),
        container=struct_val,
        position=mlir_ir.DenseI64ArrayAttr.get([1]),
    )
    index_ptr = llvm.extractvalue(
        res=builder.get_mlir_type(grp_type.group_index_type),
        container=struct_val,
        position=mlir_ir.DenseI64ArrayAttr.get([2]),
    )
    data_var = _make_temp_var(
        builder, grp_var, "data", grp_type.group_data_type
    )
    index_var = _make_temp_var(
        builder, grp_var, "index", grp_type.group_index_type
    )
    size_var = _make_temp_var(builder, grp_var, "size", group_size_type)
    builder.store_var(data_var, data_ptr)
    builder.store_var(index_var, index_ptr)
    builder.store_var(size_var, size_val)
    builder.lower_call_external_function(
        target, extfn, [data_var, index_var, size_var], {}
    )


def _make_idx_getattr(funcname):
    def _lower(builder, target, args, kwargs):
        return _lower_group_idx_shim(builder, target, args, kwargs, funcname)

    return _make_method_getattr_lower(_lower)


lower_getattr(GroupType, "idxmax")(_make_idx_getattr("idxmax"))
lower_getattr(GroupType, "idxmin")(_make_idx_getattr("idxmin"))


# --- GroupType.corr: two groups, call BlockCorr_* ---
def _lower_group_corr_shim(builder, target, args, kwargs):
    lhs_var, rhs_var = args[0], args[1]
    lhs_type = builder.get_numba_type(lhs_var.name)
    rhs_type = builder.get_numba_type(rhs_var.name)
    type_key = (
        types.float64,
        lhs_type.group_scalar_type,
        rhs_type.group_scalar_type,
    )
    funcs = call_block_functions["corr"]
    extfn = funcs[type_key]

    def _extract_data_size(builder, grp_var):
        grp_type = builder.get_numba_type(grp_var.name)
        struct_val = builder.load_var(grp_var)
        data_ptr = llvm.extractvalue(
            res=builder.get_mlir_type(grp_type.group_data_type),
            container=struct_val,
            position=mlir_ir.DenseI64ArrayAttr.get([0]),
        )
        size_val = llvm.extractvalue(
            res=builder.get_mlir_type(types.int64),
            container=struct_val,
            position=mlir_ir.DenseI64ArrayAttr.get([1]),
        )
        data_var = _make_temp_var(
            builder, grp_var, "data", grp_type.group_data_type
        )
        size_var = _make_temp_var(builder, grp_var, "size", group_size_type)
        builder.store_var(data_var, data_ptr)
        builder.store_var(size_var, size_val)
        return data_var, size_var

    lhs_data, lhs_size = _extract_data_size(builder, lhs_var)
    rhs_data, rhs_size = _extract_data_size(builder, rhs_var)
    builder.lower_call_external_function(
        target, extfn, [lhs_data, rhs_data, lhs_size], {}
    )


lower_getattr(GroupType, "corr")(
    _make_method_getattr_lower(_lower_group_corr_shim)
)
