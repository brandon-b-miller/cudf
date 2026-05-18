# SPDX-FileCopyrightText: Copyright (c) 2020-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""
MLIR (numba_cuda_mlir) typing for GroupBy JIT apply: Group, GroupType, GroupByJITDataFrame,
reduction attributes (max, min, sum, mean, var, std, size, count, idxmax, idxmin, corr),
and _group_slot. Importing this module registers typing with numba_cuda_mlir.
"""

from __future__ import annotations

from numba_cuda_mlir import cuda as numba_cuda_mlir_cuda, models, types
from numba_cuda_mlir.extending import typing_registry
from numba_cuda_mlir.models import PrimitiveModel, register_model
from numba_cuda_mlir.typing import signature as nb_signature
from numba_cuda_mlir._mlir.dialects import llvm
from numba_cuda_mlir.numba_cuda.core.errors import TypingError
from numba_cuda_mlir.numba_cuda.typing.templates import AbstractTemplate, AttributeTemplate
from numba_cuda_mlir.numba_cuda.np import numpy_support

from cudf.core.udf._ops import arith_ops, comparison_ops, unary_ops
from cudf.core.udf.utils import Row, UDFError

# GroupType is a struct: (group_data ptr, size i64, index ptr) = 24 bytes on 64-bit
GROUP_TYPE_SIZE = 24
index_default_type = types.int64
group_size_type = types.int64
SUPPORTED_GROUPBY_NUMBA_TYPES = [
    types.int32,
    types.int64,
    types.float32,
    types.float64,
]
SUPPORTED_GROUPBY_NUMPY_TYPES = [
    numpy_support.as_dtype(dt) for dt in SUPPORTED_GROUPBY_NUMBA_TYPES
]

_UDF_DOC_URL = (
    "https://docs.rapids.ai/api/cudf/stable/user_guide/guide-to-udfs/"
)


class Group:
    """
    A piece of python code whose purpose is to be replaced
    during compilation. After being registered to GroupType,
    serves as a handle for instantiating GroupType objects
    in python code and accessing their attributes.
    """

    pass


class GroupType(types.Type):
    """
    Numba extension type carrying metadata associated with a single
    GroupBy group. This metadata is passed to the device function.
    """

    # ABI size (ptr + i64 + ptr) for MLIR-only path; no Numba StructModel.
    _extensionty_size = GROUP_TYPE_SIZE

    def __init__(self, group_scalar_type, index_type=index_default_type):
        if (
            group_scalar_type not in SUPPORTED_GROUPBY_NUMBA_TYPES
            and not isinstance(group_scalar_type, types.Poison)
        ):
            # A frame containing an column with an unsupported dtype
            # is calling groupby apply. Construct a GroupType with
            # a poisoned type so we can later error if this group is
            # used in the UDF body
            group_scalar_type = types.Poison(group_scalar_type)
        self.group_scalar_type = group_scalar_type
        self.index_type = index_type
        self.group_data_type = types.CPointer(group_scalar_type)
        self.group_size_type = group_size_type
        self.group_index_type = types.CPointer(index_type)
        super().__init__(
            name=f"Group({self.group_scalar_type}, {self.index_type})"
        )


class GroupByJITDataFrame(Row):
    """Row type for groupby apply UDF. Raises TypingError for missing keys so
    can_be_jitted can fall back to non-JIT (same as numba-cuda NumbaKeyError).
    """

    def typeof(self, key):
        if key not in self.fields:
            raise TypingError(
                f"Column {key!r} does not exist in group. "
                f"Valid columns: {list(self.fields.keys())}"
            )
        return super().typeof(key)

    def offset(self, key):
        if key not in self.fields:
            raise TypingError(
                f"Column {key!r} does not exist in group. "
                f"Valid columns: {list(self.fields.keys())}"
            )
        return super().offset(key)


register_model(GroupByJITDataFrame)(models.RecordModel)


# --- MLIR data model for GroupType (struct: ptr, i64, ptr) ---
@register_model(GroupType)
class GroupTypeModel(PrimitiveModel):
    def __init__(self, dmm, fe_type):
        ptr_ty = llvm.PointerType.get()
        i64_ty = dmm.lookup(types.int64).get_value_type()
        be_type = llvm.StructType.new_identified(
            fe_type.name, [ptr_ty, i64_ty, ptr_ty]
        )
        super().__init__(dmm, fe_type, be_type)


# --- Typing: Group(data, size, index) -> GroupType ---
@typing_registry.register_global(Group)
class GroupConstructorTemplate(AbstractTemplate):
    key = Group

    def generic(self, args, kws):
        if len(args) != 3 or kws:
            return None
        group_data, size, index = args
        if (
            isinstance(group_data, types.Array)
            and isinstance(size, types.Integer)
            and isinstance(index, types.Array)
        ):
            return nb_signature(
                GroupType(group_data.dtype, index.dtype),
                group_data,
                size,
                index,
            )
        return None


# --- Typing: GroupType attributes (max, min, sum, mean, var, std, size, count, idxmax, idxmin, corr) ---
def _make_unary_reduction_attr(funcname, retty_fn=None):
    """retty_fn(GroupType) -> return type, or None to use group_scalar_type."""

    class Attr(AbstractTemplate):
        key = f"GroupType.{funcname}"

        def generic(self, args, kws):
            if len(args) != 0 or kws:
                return None
            if not isinstance(self.this, GroupType):
                return None
            if isinstance(self.this.group_scalar_type, types.Poison):
                raise UDFError(
                    f"Use of a column of {self.this.group_scalar_type.ty} detected "
                    "within UDAF body. Only columns of the following dtypes "
                    "may be used through the GroupBy.apply() JIT engine: "
                    f"{[str(x) for x in SUPPORTED_GROUPBY_NUMPY_TYPES]}"
                )
            retty = (
                retty_fn(self.this)
                if retty_fn
                else self.this.group_scalar_type
            )
            fname = self.key.split(".")[-1]
            funcs = call_block_functions.get(fname.lower(), {})
            if (retty, self.this.group_scalar_type) not in funcs:
                dtype_err = str(self.this.group_scalar_type)
                raise UDFError(
                    f"Series.{fname}() is not supported for "
                    f"({dtype_err}) within JIT GroupBy apply. To see "
                    f"what's available, visit {_UDF_DOC_URL}"
                )
            return nb_signature(retty, recvr=self.this)

    def resolve_attr(self, mod):
        return types.BoundFunction(
            Attr, GroupType(mod.group_scalar_type, mod.index_type)
        )

    return resolve_attr


# Return type for each (reduction, group_scalar_type); must match shim.cu make_definition.
def _reduction_return_type(funcname, group_scalar_type):
    if funcname == "max" or funcname == "min":
        return group_scalar_type
    if funcname == "sum":
        if group_scalar_type in (types.int32, types.int64):
            return types.int64
        return group_scalar_type
    if funcname in ("mean", "var", "std"):
        if group_scalar_type in (types.int32, types.int64):
            return types.float64
        return group_scalar_type
    return group_scalar_type


def _make_reduction_retty_fn(funcname):
    return lambda g: _reduction_return_type(funcname, g.group_scalar_type)


@typing_registry.register_attr
class GroupTypeAttr(AttributeTemplate):
    key = GroupType

    resolve_max = _make_unary_reduction_attr(
        "max", retty_fn=_make_reduction_retty_fn("max")
    )
    resolve_min = _make_unary_reduction_attr(
        "min", retty_fn=_make_reduction_retty_fn("min")
    )
    resolve_sum = _make_unary_reduction_attr(
        "sum", retty_fn=_make_reduction_retty_fn("sum")
    )
    resolve_mean = _make_unary_reduction_attr(
        "mean", retty_fn=_make_reduction_retty_fn("mean")
    )
    resolve_var = _make_unary_reduction_attr(
        "var", retty_fn=_make_reduction_retty_fn("var")
    )
    resolve_std = _make_unary_reduction_attr(
        "std", retty_fn=_make_reduction_retty_fn("std")
    )
    resolve_size = _make_unary_reduction_attr(
        "size", retty_fn=lambda _: group_size_type
    )
    resolve_count = _make_unary_reduction_attr(
        "count", retty_fn=lambda _: types.int64
    )

    def resolve_idxmax(self, mod):
        return types.BoundFunction(
            _GroupIdxMaxTemplate,
            GroupType(mod.group_scalar_type, mod.index_type),
        )

    def resolve_idxmin(self, mod):
        return types.BoundFunction(
            _GroupIdxMinTemplate,
            GroupType(mod.group_scalar_type, mod.index_type),
        )

    def resolve_corr(self, mod):
        return types.BoundFunction(
            _GroupCorrTemplate,
            GroupType(mod.group_scalar_type, mod.index_type),
        )

    def resolve_group_data(self, mod):
        return mod.group_data_type

    def resolve_index(self, mod):
        return mod.group_index_type


class _GroupIdxMaxTemplate(AbstractTemplate):
    key = "GroupType.idxmax"

    def generic(self, args, kws):
        return nb_signature(self.this.index_type, recvr=self.this)


class _GroupIdxMinTemplate(AbstractTemplate):
    key = "GroupType.idxmin"

    def generic(self, args, kws):
        return nb_signature(self.this.index_type, recvr=self.this)


class _GroupCorrTemplate(AbstractTemplate):
    key = "GroupType.corr"

    def generic(self, args, kws):
        if len(args) != 1 or kws:
            return None
        if not isinstance(self.this, GroupType) or not isinstance(
            args[0], GroupType
        ):
            return None
        # Check for poisoned (unsupported column) types; match GroupAttrBase.
        for arg in (self.this, args[0]):
            if isinstance(arg.group_scalar_type, types.Poison):
                raise UDFError(
                    f"Use of a column of {arg.group_scalar_type.ty} detected "
                    "within UDAF body. Only columns of the following dtypes "
                    "may be used through the GroupBy.apply() JIT engine: "
                    f"{[str(x) for x in SUPPORTED_GROUPBY_NUMPY_TYPES]}"
                )
        # corr is only supported for certain (lhs, rhs) type pairs (e.g. int in shim).
        corr_funcs = call_block_functions.get("corr", {})
        key = (
            types.float64,
            self.this.group_scalar_type,
            args[0].group_scalar_type,
        )
        if key not in corr_funcs:
            dtype_err = ", ".join(
                [str(g.group_scalar_type) for g in (self.this, args[0])]
            )
            raise UDFError(
                "Series.corr(Series) is not supported for "
                f"({dtype_err}) within JIT GroupBy apply. To see "
                f"what's available, visit {_UDF_DOC_URL}"
            )
        return nb_signature(types.float64, recvr=self.this, *args)


# --- GroupType binary ops (for group + group, etc.) ---
class GroupOpBase(AbstractTemplate):
    def make_error_string(self, args):
        fname = self.key.__name__
        sr_err = ", ".join(["Series" for _ in range(len(args))])
        return (
            f"{fname}({sr_err}) is not supported by JIT GroupBy "
            f"apply. Supported features are listed at: {_UDF_DOC_URL}"
        )

    def generic(self, args, kws):
        if not all(isinstance(arg, GroupType) for arg in args):
            return None
        for arg in args:
            if isinstance(arg.group_scalar_type, types.Poison):
                raise UDFError(
                    f"Use of a column of {arg.group_scalar_type.ty} detected "
                    "within UDF body. Only columns of the following dtypes "
                    "may be used through the GroupBy.apply() JIT engine: "
                    f"{[str(x) for x in SUPPORTED_GROUPBY_NUMPY_TYPES]}"
                )
        # No global op (add, eq, etc.) is supported for GroupType; only
        # attribute-based reductions (group.sum(), group.mean(), etc.) are.
        raise UDFError(self.make_error_string(args))


for op in arith_ops + comparison_ops + unary_ops:
    typing_registry.register_global(op)(GroupOpBase)


# --- DataFrame attribute: only column names (GroupType) are valid ---
@typing_registry.register_attr
class GroupByJITDataFrameAttr(AttributeTemplate):
    key = GroupByJITDataFrame

    def generic_resolve(self, typ, attr):
        if attr in typ.fields:
            return typ.typeof(attr)
        raise UDFError(
            f"JIT GroupBy.apply() does not support DataFrame.{attr}(). "
            "Only column names (e.g. group['col']) are valid."
        )


# --- _group_slot: device-only slot for group record (like _row_slot) ---
def _group_slot():
    """Device-only: return a slot for the group record. Do not call from Python."""
    raise NotImplementedError("_group_slot is only for JIT")


_group_slot_cases: list = []


class GroupSlotTemplate(AbstractTemplate):
    key = _group_slot

    def generic(self, args, kws):
        if len(args) != 0 or kws:
            return None
        for group_type in _group_slot_cases:
            return nb_signature(group_type)
        return None


def register_group_slot(dataframe_group_type):
    """Register typing/lowering for _group_slot() for this kernel's group type."""
    global _group_slot_cases
    _group_slot_cases = [dataframe_group_type]


def _register_group_slot_typing():
    typing_registry.register_global(
        _group_slot, types.Function(GroupSlotTemplate)
    )


_register_group_slot_typing()


# --- Declare Block* device functions from shim.cu (same as legacy groupby_typing) ---
def _shim_ty_name(ty):
    if ty == types.int32:
        return "int32"
    if ty == types.int64:
        return "int64"
    if ty == types.float32:
        return "float32"
    if ty == types.float64:
        return "float64"
    raise ValueError(ty)


call_block_functions: dict = {}
# Shim is linked once in ApplyKernelBase.compile_kernel_string (udf_kernel_base); do not add link= here
# or we get multiple definition errors when linking the kernel.


def _register_block_unary(funcname, inputty, retty):
    name = f"Block{funcname}_{_shim_ty_name(inputty)}"
    ext = numba_cuda_mlir_cuda.declare_device(
        name, retty(types.CPointer(inputty), group_size_type)
    )
    call_block_functions.setdefault(funcname.lower(), {})
    call_block_functions[funcname.lower()][(retty, inputty)] = ext


def _register_block_idx(funcname, inputty):
    name = f"Block{funcname}_{_shim_ty_name(inputty)}"
    ext = numba_cuda_mlir_cuda.declare_device(
        name,
        types.int64(
            types.CPointer(inputty),
            types.CPointer(index_default_type),
            group_size_type,
        ),
    )
    call_block_functions.setdefault(funcname.lower(), {})
    call_block_functions[funcname.lower()][(index_default_type, inputty)] = ext


def _register_block_corr(lty, rty):
    name = f"BlockCorr_{_shim_ty_name(lty)}_{_shim_ty_name(rty)}"
    ext = numba_cuda_mlir_cuda.declare_device(
        name,
        types.float64(
            types.CPointer(lty), types.CPointer(rty), group_size_type
        ),
    )
    call_block_functions.setdefault("corr", {})
    call_block_functions["corr"][(types.float64, lty, rty)] = ext


for ty in SUPPORTED_GROUPBY_NUMBA_TYPES:
    _register_block_unary("Max", ty, ty)
    _register_block_unary("Min", ty, ty)
    _register_block_idx("IdxMax", ty)
    _register_block_idx("IdxMin", ty)
    if ty in {
        types.int8,
        types.int16,
        types.int32,
        types.int64,
        types.uint8,
        types.uint16,
        types.uint32,
        types.uint64,
    }:
        _register_block_corr(ty, ty)

_register_block_unary("Sum", types.int32, types.int64)
_register_block_unary("Sum", types.int64, types.int64)
_register_block_unary("Sum", types.float32, types.float32)
_register_block_unary("Sum", types.float64, types.float64)

_register_block_unary("Mean", types.int32, types.float64)
_register_block_unary("Mean", types.int64, types.float64)
_register_block_unary("Mean", types.float32, types.float32)
_register_block_unary("Mean", types.float64, types.float64)

_register_block_unary("Std", types.int32, types.float64)
_register_block_unary("Std", types.int64, types.float64)
_register_block_unary("Std", types.float32, types.float32)
_register_block_unary("Std", types.float64, types.float64)

_register_block_unary("Var", types.int32, types.float64)
_register_block_unary("Var", types.int64, types.float64)
_register_block_unary("Var", types.float32, types.float32)
_register_block_unary("Var", types.float64, types.float64)
