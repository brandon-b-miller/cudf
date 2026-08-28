# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""
MLIR lowering for MaskedType: Masked constructor, .value/.valid,
pack_return, len(Masked(mlir_string)), and full numeric arithmetic (binary
arith/bitwise/comparison, unary, Masked+NA, scalar+Masked, is_, truth, casts).
Importing this module registers lowerings with numba_cuda_mlir.

Note: The numba-cuda masked_lowering uses context.compile_internal(builder,
lambda x, y: op(x, y), sig, (m1.value, m2.value)) so Numba generates the
scalar op IR. In the numba_cuda_mlir MLIR path, masked **binary** ops still apply
Python operators to MLIR values where supported; masked **unary** ops
delegate to the same registered scalar lowerings as the main pipeline
(`builder.get_registered_builder`), e.g. ``math.sin`` → ``math`` dialect.
``operator.invert`` on ``MaskedType`` uses a dedicated lowering (``arith.xori`` with
an all-ones mask via ``arith.constant(..., -1)`` for integer payloads).
"""

from __future__ import annotations

import operator

import numpy as np
from numba_cuda_mlir._mlir import ir as mlir_ir
from numba_cuda_mlir._mlir.dialects import (
    arith,
    linalg,
    llvm,
    scf,
    tensor,
)
from numba_cuda_mlir._mlir.extras import types as T
from numba_cuda_mlir.extending import lower_cast, lowering_registry
from numba_cuda_mlir.lowering_utilities import (
    DeferredMethodCall,
    bool_of,
    coerce_numpy_scalars_for_binary_op,
    concretize_tuple_to_tensor,
    convert,
    equal,
    false,
    float_of,
    int_of,
    try_extract_constant,
)
from numba_cuda_mlir.numba_cuda import types, typing as nb_typing
from numba_cuda_mlir.numba_cuda.core import ir as numba_ir
from numba_cuda_mlir.numba_cuda.types.misc import unliteral

from cudf.core.udf._ops import (
    arith_ops,
    bitwise_ops,
    comparison_ops,
    unary_ops,
)
from cudf.core.udf.api import Masked, pack_return
from cudf.core.udf.mlir_backend.masked_typing import (
    MaskedType,
    NAType,
    na_type,
)
from cudf.core.udf.mlir_backend.strings_typing import (
    MLIRStringType,
    mlir_string,
    size_type,
)


def _ipowi_expand(base, exp, ty):
    """
    Expand integer power (base ** exp) using scf.while so we avoid math.ipowi,
    which has no LLVM translation in the GPU codegen pipeline.
    For exp < 0 we return 0 (Python int**neg gives float; for int result use 0).
    """
    one = arith.constant(ty, 1)
    zero = arith.constant(ty, 0)
    exp_lt_0 = arith.cmpi(arith.CmpIPredicate.slt, exp, zero)
    while_op = scf.WhileOp([ty, ty], [one, exp])
    while_op.regions[0].blocks.append(ty, ty)
    before_block = while_op.regions[0].blocks[0]
    while_op.regions[1].blocks.append(ty, ty)
    after_block = while_op.regions[1].blocks[0]
    with mlir_ir.InsertionPoint(before_block):
        exp_arg = before_block.arguments[1]
        zero_c = arith.constant(ty, 0)
        exp_gt_0 = arith.cmpi(arith.CmpIPredicate.sgt, exp_arg, zero_c)
        scf.condition(exp_gt_0, list(before_block.arguments))
    with mlir_ir.InsertionPoint(after_block):
        result_arg, exp_arg = (
            after_block.arguments[0],
            after_block.arguments[1],
        )
        new_result = arith.muli(result_arg, base)
        one_c = arith.constant(ty, 1)
        new_exp = arith.subi(exp_arg, one_c)
        scf.yield_([new_result, new_exp])
    loop_result = while_op.results[0]
    return arith.select(exp_lt_0, zero, loop_result)


def _extract_masked_value_valid(struct_val, value_mlir_ty, valid_ty):
    v = llvm.extractvalue(
        res=value_mlir_ty,
        container=struct_val,
        position=mlir_ir.DenseI64ArrayAttr.get([0]),
    )
    valid = llvm.extractvalue(
        res=valid_ty,
        container=struct_val,
        position=mlir_ir.DenseI64ArrayAttr.get([1]),
    )
    return v, valid


def _make_temp_var(builder, base_var, name_suffix, numba_type):
    """Synthetic IR var + typemap entry for reusing numba_cuda_mlir scalar lowerings."""
    scope = getattr(base_var, "scope", None)
    loc = getattr(base_var, "loc", None)
    name = f"$masked_uop_{base_var.name}_{name_suffix}"
    temp = numba_ir.Var(scope=scope, name=name, loc=loc)
    builder.fndesc.typemap[temp.name] = numba_type
    return temp


def _pack_masked_result(builder, target_type, result_value, result_valid):
    struct_ty = builder.get_mlir_type(target_type)
    undef = llvm.UndefOp(struct_ty)
    with_value = llvm.insertvalue(
        container=undef,
        value=result_value,
        position=mlir_ir.DenseI64ArrayAttr.get([0]),
    )
    with_valid = llvm.insertvalue(
        container=with_value,
        value=result_valid,
        position=mlir_ir.DenseI64ArrayAttr.get([1]),
    )
    return with_valid


def _register():
    lower_getattr = lowering_registry.lower_getattr
    lower_getattr_generic = lowering_registry.lower_getattr_generic
    lower = lowering_registry.lower

    # --- Lowering: Masked(value, valid) ---
    def _lower_masked_constructor(builder, target, args, kwargs):
        target_type = builder.get_numba_type(target.name)
        mlir_struct_ty = builder.get_mlir_type(target_type)
        val_var, valid_var = args
        value_mlir_ty = builder.get_mlir_type(target_type.value_type)
        value_mlir = convert(builder.load_var(val_var), value_mlir_ty)
        valid_mlir_ty = builder.get_mlir_type(types.boolean)
        valid_mlir = convert(builder.load_var(valid_var), valid_mlir_ty)
        if builder.nrt.type_has_nrt_meminfo(target_type.value_type):
            builder.incref(target_type.value_type, value_mlir)
        undef = llvm.UndefOp(mlir_struct_ty)
        with_value = llvm.insertvalue(
            container=undef,
            value=value_mlir,
            position=mlir_ir.DenseI64ArrayAttr.get([0]),
        )
        with_valid = llvm.insertvalue(
            container=with_value,
            value=valid_mlir,
            position=mlir_ir.DenseI64ArrayAttr.get([1]),
        )
        builder.store_var(target, with_valid)

    # --- Cast: Masked(mlir_string) -> mlir_string ---
    @lower_cast(MaskedType(mlir_string), mlir_string)
    def cast_masked_mlir_string_to_mlir_string(
        context, builder, fromty, toty, val
    ):
        ms_ty = builder.get_mlir_type(mlir_string)
        result = llvm.extractvalue(
            res=ms_ty,
            container=val,
            position=mlir_ir.DenseI64ArrayAttr.get([0]),
        )
        builder.incref(toty, result)
        return result

    # Ternary branches like `f(x) if x is not NA else NA` unify to Masked; Numba
    # inserts cast(NA) -> Masked. Opaque NA has no MLIR value — build invalid mask.
    @lower_cast(na_type, MaskedType)
    def cast_na_to_masked(context, builder, fromty, toty, val):
        value_mlir_ty = builder.get_mlir_type(toty.value_type)
        undef_val = llvm.UndefOp(value_mlir_ty)
        valid_zero = arith.constant(
            result=builder.get_mlir_type(types.boolean), value=0
        )
        return _pack_masked_result(builder, toty, undef_val, valid_zero)

    def _cast_scalar_to_masked(context, builder, fromty, toty, val):
        """
        Branches like ``return 5`` / ``return 2.5``: Numba unifies to Masked and
        inserts cast(scalar) -> Masked with valid True (see test_apply_return_literal).
        """
        if not isinstance(toty, MaskedType):
            raise TypeError(
                f"cast to Masked expected MaskedType destination, got {toty!r}"
            )
        if isinstance(toty.value_type, MLIRStringType):
            raise NotImplementedError(
                "implicit cast from numeric scalar to Masked(mlir_string)"
            )
        value_mlir_ty = builder.get_mlir_type(toty.value_type)
        scalar_val = convert(val, value_mlir_ty)
        valid_one = arith.constant(
            result=builder.get_mlir_type(types.boolean), value=1
        )
        return _pack_masked_result(builder, toty, scalar_val, valid_one)

    # Integer / float / bool scalars (including int64, float64, …) -> Masked(...)
    for _scalar_cls in (types.Integer, types.Float, types.Boolean):
        lower_cast(_scalar_cls, MaskedType)(_cast_scalar_to_masked)

    # mlir_string -> Masked(mlir_string) (return value from UDF into Masked wrapper)
    @lower_cast(MLIRStringType, MaskedType)
    def _cast_mlir_string_to_masked(context, builder, fromty, toty, val):
        builder.incref(fromty, val)
        valid_one = arith.constant(
            result=builder.get_mlir_type(types.boolean), value=1
        )
        return _pack_masked_result(builder, toty, val, valid_one)

    @lower_cast(MaskedType, MaskedType)
    def cast_masked_to_masked(context, builder, fromty, toty, val):
        """
        Unification (e.g. ``return 5.5`` vs ``return x + y`` → ``Masked(float64)``):
        promote the payload; keep ``valid`` (same as CUDA ``cast_masked_to_masked``).
        """
        if fromty.value_type == toty.value_type:
            return val
        st = llvm.StructType(val.type)
        m_val, m_valid = _extract_masked_value_valid(
            val, st.body[0], st.body[1]
        )
        value_mlir_ty = builder.get_mlir_type(toty.value_type)
        casted = convert(m_val, value_mlir_ty)
        return _pack_masked_result(builder, toty, casted, m_valid)

    # --- Lowering: getattr .value / .valid ---
    @lower_getattr_generic(MaskedType)
    def lower_masked_getattr(context, builder, target, value, attr):
        value_type = builder.get_numba_type(value.name)
        struct_value = builder.load_var(value)
        if attr == "value":
            field_index = 0
        elif attr == "valid":
            field_index = 1
        else:
            raise AttributeError(f"MaskedType has no attribute {attr!r}")
        struct_ty = llvm.StructType(struct_value.type)
        field_mlir_ty = struct_ty.body[field_index]
        field_value = llvm.extractvalue(
            res=field_mlir_ty,
            container=struct_value,
            position=mlir_ir.DenseI64ArrayAttr.get([field_index]),
        )
        target_mlir_ty = builder.get_mlir_type(
            builder.get_numba_type(target.name)
        )
        field_value = convert(field_value, target_mlir_ty)
        if attr == "value" and builder.nrt.type_has_nrt_meminfo(
            value_type.value_type
        ):
            target_numba_ty = builder.get_numba_type(target.name)
            builder.incref(target_numba_ty, field_value)
        builder.store_var(target, field_value)

    def _masked_binary_needs_datetimelike_delegate(op, ty1, ty2):
        """datetime64 / timedelta64 need numba_cuda_mlir ``lowering.datetime`` (unit scaling), not raw i64 add."""
        if op not in (operator.add, operator.sub):
            return False
        return isinstance(
            ty1, (types.NPDatetime, types.NPTimedelta)
        ) or isinstance(ty2, (types.NPDatetime, types.NPTimedelta))

    def _apply_masked_datetimelike_binary(
        builder,
        target,
        target_type,
        v1,
        v2,
        result_valid,
        op,
        ty1,
        ty2,
        ref_var,
    ):
        """Delegate to numba_cuda_mlir ``datetime_add`` / ``datetime_sub`` (same as scalar MLIR binops)."""
        ret_ty = target_type.value_type
        nb_sig = nb_typing.signature(ret_ty, ty1, ty2)
        cg = builder.get_registered_builder(op, nb_sig)
        if cg is None:
            raise NotImplementedError(
                f"No MLIR lowering for masked {op!r} with {ty1}, {ty2}; "
                f"signature {nb_sig}"
            )
        v1_mlir = builder.get_mlir_type(ty1)
        v2_mlir = builder.get_mlir_type(ty2)
        in1 = _make_temp_var(builder, ref_var, "mdt_l", ty1)
        in2 = _make_temp_var(builder, ref_var, "mdt_r", ty2)
        outv = _make_temp_var(builder, ref_var, "mdt_o", ret_ty)
        builder.store_var(in1, convert(v1, v1_mlir))
        builder.store_var(in2, convert(v2, v2_mlir))
        cg(builder, outv, [in1, in2], ())
        result_val = builder.load_var(outv)
        result_mlir_ty = builder.get_mlir_type(ret_ty)
        result_val = convert(result_val, result_mlir_ty)
        packed = _pack_masked_result(
            builder, target_type, result_val, result_valid
        )
        builder.store_var(target, packed)

    # --- Shared: apply binary op to two scalar values and pack result with valid ---
    def _apply_masked_binary_op(
        builder,
        target,
        target_type,
        v1,
        v2,
        result_valid,
        op,
        *,
        inner_ty1=None,
        inner_ty2=None,
        ref_var=None,
    ):
        """Apply op(v1, v2), convert result to target type, pack with result_valid, store."""
        if (
            inner_ty1 is not None
            and inner_ty2 is not None
            and ref_var is not None
            and _masked_binary_needs_datetimelike_delegate(
                op, inner_ty1, inner_ty2
            )
        ):
            _apply_masked_datetimelike_binary(
                builder,
                target,
                target_type,
                v1,
                v2,
                result_valid,
                op,
                inner_ty1,
                inner_ty2,
                ref_var,
            )
            return

        target_value_mlir_ty = builder.get_mlir_type(target_type.value_type)
        v1, v2 = coerce_numpy_scalars_for_binary_op(v1, v2)
        is_cmp = op in comparison_ops
        operand_ty = v1.type if is_cmp else target_value_mlir_ty
        v1 = convert(v1, operand_ty)
        v2 = convert(v2, operand_ty)
        result_val = convert(op(v1, v2), target_value_mlir_ty)
        packed = _pack_masked_result(
            builder, target_type, result_val, result_valid
        )
        builder.store_var(target, packed)

    # --- Generic binary: Masked <op> Masked ---
    def _make_lower_masked_binary(op):
        def _lower(builder, target, args, kwargs):
            target_type = builder.get_numba_type(target.name)
            m1 = builder.load_var(args[0])
            m2 = builder.load_var(args[1])
            st1 = llvm.StructType(m1.type)
            st2 = llvm.StructType(m2.type)
            v1, valid1 = _extract_masked_value_valid(
                m1, st1.body[0], st1.body[1]
            )
            v2, valid2 = _extract_masked_value_valid(
                m2, st2.body[0], st2.body[1]
            )
            result_valid = arith.andi(valid1, valid2)
            ty1 = builder.get_numba_type(args[0].name).value_type
            ty2 = builder.get_numba_type(args[1].name).value_type
            _apply_masked_binary_op(
                builder,
                target,
                target_type,
                v1,
                v2,
                result_valid,
                op,
                inner_ty1=ty1,
                inner_ty2=ty2,
                ref_var=args[0],
            )

        return _lower

    def _scalar_value_from_var(
        builder, s_var, m_var, m_val, masked_value_mlir_ty
    ):
        """
        Get the scalar value for the Masked-vs-scalar path.
        Prefer literal constant when s_var is typed as Literal so we never use
        the masked operand as the scalar (row['a'] < 1 must not become row['a'] < row['a']).
        """
        s_ty = builder.get_numba_type(s_var.name)
        if isinstance(s_ty, types.Literal):
            from numba_cuda_mlir.numba_cuda.types.misc import unliteral

            py_val = s_ty.literal_value
            base_ty = unliteral(s_ty)
            mlir_ty = builder.get_mlir_type(base_ty)
            if hasattr(mlir_ty, "width") and mlir_ty.width == 1:  # i1
                py_val = 1 if py_val else 0
            elif isinstance(py_val, (bool, np.bool_)):
                py_val = 1 if py_val else 0
            return arith.constant(mlir_ty, py_val)
        s_raw = builder.load_var(s_var)
        if str(s_raw.type) == "!llvm.ptr":
            return llvm.load(res=masked_value_mlir_ty, addr=s_raw)
        if getattr(s_raw.type, "body", None) and len(s_raw.type.body) >= 2:
            if getattr(s_var, "name", None) == getattr(m_var, "name", None):
                raise RuntimeError(
                    "Masked vs scalar: scalar variable is the same as masked; "
                    "cannot extract scalar (e.g. row['a'] < 1 became row['a'] < row['a'])."
                )
            st_s = llvm.StructType(s_raw.type)
            return llvm.extractvalue(
                res=st_s.body[0],
                container=s_raw,
                position=mlir_ir.DenseI64ArrayAttr.get([0]),
            )
        return s_raw

    # --- Generic binary: Masked <op> scalar, scalar <op> Masked ---
    def _make_lower_masked_binary_scalar(op, masked_first):
        def _lower(builder, target, args, kwargs):
            target_type = builder.get_numba_type(target.name)
            m_var, s_var = (
                (args[0], args[1]) if masked_first else (args[1], args[0])
            )
            m = builder.load_var(m_var)
            st = llvm.StructType(m.type)
            m_val, m_valid = _extract_masked_value_valid(
                m, st.body[0], st.body[1]
            )
            s_val = _scalar_value_from_var(
                builder, s_var, m_var, m_val, st.body[0]
            )
            m_inner_ty = builder.get_numba_type(m_var.name).value_type
            s_ty = builder.get_numba_type(s_var.name)
            s_inner_ty = (
                unliteral(s_ty) if isinstance(s_ty, types.Literal) else s_ty
            )
            if masked_first:
                _apply_masked_binary_op(
                    builder,
                    target,
                    target_type,
                    m_val,
                    s_val,
                    m_valid,
                    op,
                    inner_ty1=m_inner_ty,
                    inner_ty2=s_inner_ty,
                    ref_var=m_var,
                )
            else:
                _apply_masked_binary_op(
                    builder,
                    target,
                    target_type,
                    s_val,
                    m_val,
                    m_valid,
                    op,
                    inner_ty1=s_inner_ty,
                    inner_ty2=m_inner_ty,
                    ref_var=m_var,
                )

        return _lower

    # --- Binary with NA: result is invalid ---
    def _lower_masked_binary_null(builder, target, args, kwargs):
        target_type = builder.get_numba_type(target.name)
        struct_ty = builder.get_mlir_type(target_type)
        value_mlir_ty = builder.get_mlir_type(target_type.value_type)
        undef_val = llvm.UndefOp(value_mlir_ty)
        valid_zero = arith.constant(
            result=builder.get_mlir_type(types.boolean), value=0
        )
        packed = _pack_masked_result(
            builder, target_type, undef_val, valid_zero
        )
        builder.store_var(target, packed)

    # --- Generic unary: <op> Masked ---
    # Delegate to numba_cuda_mlir scalar lowerings (math.sin -> math dialect, etc.) via
    # get_registered_builder; same signature shape as lower_unary_expr_assign.
    def _make_lower_masked_unary(op):
        def _lower(builder, target, args, kwargs):
            target_type = builder.get_numba_type(target.name)
            result_inner_ty = target_type.value_type
            masked_ty = builder.get_numba_type(args[0].name)
            operand_inner_ty = masked_ty.value_type

            m = builder.load_var(args[0])
            st = llvm.StructType(m.type)
            m_val, m_valid = _extract_masked_value_valid(
                m, st.body[0], st.body[1]
            )
            operand_mlir_ty = builder.get_mlir_type(operand_inner_ty)
            m_val = convert(m_val, operand_mlir_ty)

            sig = result_inner_ty(operand_inner_ty)
            cg = builder.get_registered_builder(op, sig)
            if cg is not None:
                # Same operand (e.g. x) can appear in multiple unary calls in one
                # expression (sin(x) + lgamma(x)); suffix by op so typemap keys stay unique.
                op_tag = getattr(op, "__name__", "op")
                op_var = _make_temp_var(
                    builder, args[0], f"{op_tag}_in", operand_inner_ty
                )
                out_var = _make_temp_var(
                    builder, args[0], f"{op_tag}_out", result_inner_ty
                )
                builder.store_var(op_var, m_val)
                cg(builder, out_var, [op_var], [])
                result_val = builder.load_var(out_var)
            else:
                raise NotImplementedError(
                    "No MLIR lowering for unary "
                    f"{getattr(op, '__name__', op)!r} on {operand_inner_ty} "
                    f"(masked inner type); signature {sig}"
                )

            result_mlir_ty = builder.get_mlir_type(result_inner_ty)
            result_val = convert(result_val, result_mlir_ty)
            packed = _pack_masked_result(
                builder, target_type, result_val, m_valid
            )
            builder.store_var(target, packed)

        return _lower

    # --- operator.invert (bitwise ~) on Masked integers ---
    # Separate from generic unary: no numba_cuda_mlir scalar @lower for invert; must use
    # arith.xori.  All-ones mask is ``constant(..., -1)`` (two's complement), not
    # ``(1 << width) - 1`` — the latter is 2**64-1 for i64 and breaks
    # IntegerAttr (signed int64 range) with std::bad_cast.
    def _lower_masked_invert(builder, target, args, kwargs):
        target_type = builder.get_numba_type(target.name)
        result_inner_ty = target_type.value_type
        masked_ty = builder.get_numba_type(args[0].name)
        operand_inner_ty = masked_ty.value_type

        if not isinstance(operand_inner_ty, types.Integer):
            raise NotImplementedError(
                f"operator.invert on Masked is only supported for integer "
                f"payloads, not {operand_inner_ty}"
            )

        m = builder.load_var(args[0])
        st = llvm.StructType(m.type)
        m_val, m_valid = _extract_masked_value_valid(m, st.body[0], st.body[1])
        operand_mlir_ty = builder.get_mlir_type(operand_inner_ty)
        m_val = convert(m_val, operand_mlir_ty)

        if not isinstance(m_val.type, mlir_ir.IntegerType):
            raise NotImplementedError(
                f"operator.invert expected integer MLIR type, got {m_val.type}"
            )
        mask = arith.constant(result=m_val.type, value=-1)
        result_val = arith.xori(m_val, mask)

        result_mlir_ty = builder.get_mlir_type(result_inner_ty)
        result_val = convert(result_val, result_mlir_ty)
        packed = _pack_masked_result(builder, target_type, result_val, m_valid)
        builder.store_var(target, packed)

    # --- pack_return ---
    def _lower_pack_return_masked(builder, target, args, kwargs):
        val = builder.load_var(args[0])
        target_type = builder.get_numba_type(target.name)
        if isinstance(
            target_type, MaskedType
        ) and builder.nrt.type_has_nrt_meminfo(target_type.value_type):
            builder.incref(target_type, val)
        builder.store_var(target, val)

    def _lower_pack_return_scalar(builder, target, args, kwargs):
        target_type = builder.get_numba_type(target.name)
        mlir_struct_ty = builder.get_mlir_type(target_type)
        value_mlir_ty = builder.get_mlir_type(target_type.value_type)
        scalar_val = convert(builder.load_var(args[0]), value_mlir_ty)
        valid_one = arith.constant(
            result=builder.get_mlir_type(types.boolean), value=1
        )
        packed = _pack_masked_result(
            builder, target_type, scalar_val, valid_one
        )
        builder.store_var(target, packed)

    lower(Masked, types.Any, types.boolean)(_lower_masked_constructor)
    # Row apply may pass a literal bool (e.g. True) as the valid flag, typed as Literal[bool](True)
    lower(Masked, types.Any, types.Literal)(_lower_masked_constructor)

    for binary_op in arith_ops + bitwise_ops + comparison_ops:
        lower(binary_op, MaskedType, MaskedType)(
            _make_lower_masked_binary(binary_op)
        )
        lower(binary_op, MaskedType, types.Number)(
            _make_lower_masked_binary_scalar(binary_op, True)
        )
        lower(binary_op, types.Number, MaskedType)(
            _make_lower_masked_binary_scalar(binary_op, False)
        )
        lower(binary_op, MaskedType, types.Boolean)(
            _make_lower_masked_binary_scalar(binary_op, True)
        )
        lower(binary_op, types.Boolean, MaskedType)(
            _make_lower_masked_binary_scalar(binary_op, False)
        )
        lower(binary_op, MaskedType, NAType)(_lower_masked_binary_null)
        lower(binary_op, NAType, MaskedType)(_lower_masked_binary_null)

    for unary_op in unary_ops:
        if unary_op is operator.invert:
            continue
        lower(unary_op, MaskedType)(_make_lower_masked_unary(unary_op))
    lower(abs, MaskedType)(_make_lower_masked_unary(abs))
    lower(operator.invert, MaskedType)(_lower_masked_invert)

    # --- operator.is_ (Masked is NA) -> not valid ---
    def _lower_masked_is_na_masked_first(builder, target, args, kwargs):
        m = builder.load_var(args[0])
        st = llvm.StructType(m.type)
        _, valid = _extract_masked_value_valid(m, st.body[0], st.body[1])
        one = arith.constant(valid.type, 1)
        result = arith.xori(valid, one)
        builder.store_var(target, result)

    def _lower_masked_is_na_na_first(builder, target, args, kwargs):
        m = builder.load_var(args[1])
        st = llvm.StructType(m.type)
        _, valid = _extract_masked_value_valid(m, st.body[0], st.body[1])
        one = arith.constant(valid.type, 1)
        result = arith.xori(valid, one)
        builder.store_var(target, result)

    lower(operator.is_, MaskedType, NAType)(_lower_masked_is_na_masked_first)
    lower(operator.is_, NAType, MaskedType)(_lower_masked_is_na_na_first)

    # --- operator.is_not (Masked is not NA) = not (is_); store valid bit ---
    def _lower_masked_is_not_na_masked_first(builder, target, args, kwargs):
        m = builder.load_var(args[0])
        st = llvm.StructType(m.type)
        _, valid = _extract_masked_value_valid(m, st.body[0], st.body[1])
        builder.store_var(target, valid)

    def _lower_masked_is_not_na_na_first(builder, target, args, kwargs):
        m = builder.load_var(args[1])
        st = llvm.StructType(m.type)
        _, valid = _extract_masked_value_valid(m, st.body[0], st.body[1])
        builder.store_var(target, valid)

    lower(operator.is_not, MaskedType, NAType)(
        _lower_masked_is_not_na_masked_first
    )
    lower(operator.is_not, NAType, MaskedType)(
        _lower_masked_is_not_na_na_first
    )

    # --- operator.truth / bool(Masked): valid ? bool(payload) : False (matches CUDA) ---
    def _lower_masked_truth(builder, target, args, kwargs):
        m = builder.load_var(args[0])
        st = llvm.StructType(m.type)
        m_val, m_valid = _extract_masked_value_valid(m, st.body[0], st.body[1])
        bool_mlir_ty = builder.get_mlir_type(types.boolean)
        payload_as_bool = bool_of(convert(m_val, bool_mlir_ty))
        result = arith.select(m_valid, payload_as_bool, false())
        builder.store_var(target, result)

    lower(operator.truth, MaskedType)(_lower_masked_truth)
    lower(bool, MaskedType)(_lower_masked_truth)

    # --- float(Masked), int(Masked) ---
    def _make_lower_masked_cast(to_ty):
        def _lower(builder, target, args, kwargs):
            target_type = builder.get_numba_type(target.name)
            target_value_mlir_ty = builder.get_mlir_type(
                target_type.value_type
            )
            m = builder.load_var(args[0])
            st = llvm.StructType(m.type)
            m_val, m_valid = _extract_masked_value_valid(
                m, st.body[0], st.body[1]
            )
            casted = builder.mlir_convert(m_val, target_value_mlir_ty)
            packed = _pack_masked_result(builder, target_type, casted, m_valid)
            builder.store_var(target, packed)

        return _lower

    lower(float, MaskedType)(_make_lower_masked_cast(types.float64))
    lower(int, MaskedType)(_make_lower_masked_cast(types.int64))

    lower(pack_return, MaskedType)(_lower_pack_return_masked)
    for scalar_ty in (
        types.Integer,
        types.int8,
        types.int16,
        types.int32,
        types.int64,
        types.uint8,
        types.uint16,
        types.uint32,
        types.uint64,
        types.float32,
        types.float64,
        types.boolean,
    ):
        lower(pack_return, scalar_ty)(_lower_pack_return_scalar)

    # --- len(Masked(mlir_string)) -> Masked(size_type) ---
    from cudf.core.udf.mlir_backend.strings_lowering import (
        lower_len_value,
    )

    def _lower_len_masked_mlir_string(builder, target, args, kwargs):
        masked_var = args[0]
        masked_val = builder.load_var(masked_var)
        ms_ty = builder.get_mlir_type(mlir_string)
        valid_ty = builder.get_mlir_type(types.boolean)
        ms_val = llvm.extractvalue(
            res=ms_ty,
            container=masked_val,
            position=mlir_ir.DenseI64ArrayAttr.get([0]),
        )
        valid_val = llvm.extractvalue(
            res=valid_ty,
            container=masked_val,
            position=mlir_ir.DenseI64ArrayAttr.get([1]),
        )
        from cudf.core.udf.mlir_backend.string_lowering_impl import (
            mlir_string_to_view as _ms_to_view,
        )

        str_view = _ms_to_view(ms_val)
        len_result = lower_len_value(builder, str_view)
        result_masked_ty = builder.get_mlir_type(MaskedType(size_type))
        undef = llvm.UndefOp(result_masked_ty)
        with_len = llvm.insertvalue(
            container=undef,
            value=len_result,
            position=mlir_ir.DenseI64ArrayAttr.get([0]),
        )
        with_valid = llvm.insertvalue(
            container=with_len,
            value=valid_val,
            position=mlir_ir.DenseI64ArrayAttr.get([1]),
        )
        builder.store_var(target, with_valid)

    lower(len, MaskedType(mlir_string))(_lower_len_masked_mlir_string)

    # --- Masked(mlir_string).isupper, .islower, etc. -> Masked(boolean) ---
    from cudf.core.udf.mlir_backend.strings_lowering import (
        _MLIR_IS_FUNCS,
        _flags_table_ptr_const,
    )

    def _make_lower_masked_is(mlir_fn):
        def _lower_impl(builder, target, args, kwargs):
            valid_ty = builder.get_mlir_type(types.boolean)
            true_val = arith.constant(valid_ty, 1)
            str_view, valid_val = _unified_get_view_and_valid(
                builder, args[0], valid_ty, true_val
            )
            result_val = mlir_fn(str_view, _flags_table_ptr_const())
            target_type = MaskedType(types.boolean)
            packed = _pack_masked_result(
                builder, target_type, result_val, valid_val
            )
            builder.store_var(target, packed)
            _decref_masked_source(builder, args[0])

        def _getattr(context, builder, target, value, attr=None):
            _incref_masked_source(builder, value)
            builder.store_var(target, DeferredMethodCall(value, _lower_impl))

        return _getattr

    for attrname, mlir_fn in _MLIR_IS_FUNCS.items():
        lower_getattr(MaskedType(mlir_string), attrname)(
            _make_lower_masked_is(mlir_fn)
        )

    # --- String cmpops and find/rfind/contains/startswith/endswith: unified pure-MLIR impl ---
    from cudf.core.udf.mlir_backend.string_lowering_impl import (
        mlir_string_to_view as _ms_to_view_for_unified,
    )

    from cudf.core.udf.mlir_backend.strings_lowering import (
        _CMPOP_NAMES,
        _INT_STR_STR_OPS,
        _literal_to_view,
        call_string_bool_str_str_mlir,
        call_string_cmpop_mlir,
        call_string_int_str_str_mlir,
    )

    def _unified_get_view_and_valid(builder, arg_var, valid_ty, true_val):
        """Extract (view SSA value, valid bit) from a string-typed variable.

        Handles MaskedType(mlir_string), bare mlir_string, and StringLiteral.
        """
        arg_ty = builder.get_numba_type(arg_var.name)
        if isinstance(arg_ty, types.StringLiteral):
            str_view = _literal_to_view(builder, arg_var)
            return str_view, true_val
        if isinstance(arg_ty, MaskedType) and isinstance(
            arg_ty.value_type, MLIRStringType
        ):
            masked_val = builder.load_var(arg_var)
            ms_ty = builder.get_mlir_type(mlir_string)
            ms_val = llvm.extractvalue(
                res=ms_ty,
                container=masked_val,
                position=mlir_ir.DenseI64ArrayAttr.get([0]),
            )
            valid_val = llvm.extractvalue(
                res=valid_ty,
                container=masked_val,
                position=mlir_ir.DenseI64ArrayAttr.get([1]),
            )
            str_view = _ms_to_view_for_unified(ms_val)
            return str_view, valid_val
        if isinstance(arg_ty, MLIRStringType):
            ms_val = builder.load_var(arg_var)
            str_view = _ms_to_view_for_unified(ms_val)
            return str_view, true_val
        raise TypeError(f"unexpected string arg type: {arg_ty}")

    def _make_string_cmpop_unified(op_name):
        def _lower(builder, target, args, kwargs):
            valid_ty = builder.get_mlir_type(types.boolean)
            true_val = arith.constant(valid_ty, 1)

            def get_view_and_valid(arg_var):
                return _unified_get_view_and_valid(
                    builder, arg_var, valid_ty, true_val
                )

            lhs_val, valid_l = get_view_and_valid(args[0])
            rhs_val, valid_r = get_view_and_valid(args[1])
            cmp_result = call_string_cmpop_mlir(lhs_val, rhs_val, op_name)

            target_type = builder.get_numba_type(target.name)
            if isinstance(target_type, MaskedType):
                result_valid = arith.andi(valid_l, valid_r)
                packed = _pack_masked_result(
                    builder, target_type, cmp_result, result_valid
                )
                builder.store_var(target, packed)
            else:
                builder.store_var(target, cmp_result)

        return _lower

    _string_cmpop_sigs = [
        (mlir_string, mlir_string),
        (mlir_string, types.StringLiteral),
        (types.StringLiteral, mlir_string),
        (MaskedType(mlir_string), MaskedType(mlir_string)),
        (MaskedType(mlir_string), types.StringLiteral),
        (types.StringLiteral, MaskedType(mlir_string)),
        (MaskedType(mlir_string), mlir_string),
        (mlir_string, MaskedType(mlir_string)),
    ]
    for op_name in _CMPOP_NAMES:
        op = getattr(operator, op_name)
        impl = _make_string_cmpop_unified(op_name)
        for sig in _string_cmpop_sigs:
            lower(op, sig[0], sig[1])(impl)

    # --- String find/rfind/count (-> i32 or Masked(i32)); pure MLIR ---
    def _make_string_int_str_str_unified(op_name):
        def _lower(builder, target, args, kwargs):
            valid_ty = builder.get_mlir_type(types.boolean)
            true_val = arith.constant(valid_ty, 1)

            def get_view_and_valid(arg_var):
                return _unified_get_view_and_valid(
                    builder, arg_var, valid_ty, true_val
                )

            lhs_val, valid_l = get_view_and_valid(args[0])
            rhs_val, valid_r = get_view_and_valid(args[1])
            int_result = call_string_int_str_str_mlir(
                lhs_val, rhs_val, op_name
            )

            target_type = builder.get_numba_type(target.name)
            if isinstance(target_type, MaskedType):
                result_valid = arith.andi(valid_l, valid_r)
                packed = _pack_masked_result(
                    builder, target_type, int_result, result_valid
                )
                builder.store_var(target, packed)
            else:
                builder.store_var(target, int_result)

        return _lower

    # --- String contains/startswith/endswith (-> bool or Masked(bool)); pure MLIR ---
    def _make_string_bool_str_str_unified(op_name):
        def _lower(builder, target, args, kwargs):
            valid_ty = builder.get_mlir_type(types.boolean)
            true_val = arith.constant(valid_ty, 1)

            def get_view_and_valid(arg_var):
                return _unified_get_view_and_valid(
                    builder, arg_var, valid_ty, true_val
                )

            lhs_val, valid_l = get_view_and_valid(args[0])
            rhs_val, valid_r = get_view_and_valid(args[1])
            bool_result = call_string_bool_str_str_mlir(
                lhs_val, rhs_val, op_name
            )

            target_type = builder.get_numba_type(target.name)
            if isinstance(target_type, MaskedType):
                result_valid = arith.andi(valid_l, valid_r)
                packed = _pack_masked_result(
                    builder, target_type, bool_result, result_valid
                )
                builder.store_var(target, packed)
            else:
                builder.store_var(target, bool_result)

        return _lower

    def _make_masked_binary_str_str_getattr(impl_fn):
        def _wrap_with_release(builder, target, args, kwargs):
            impl_fn(builder, target, args, kwargs)
            _decref_masked_source(builder, args[0])

        def _getattr(context, builder, target, value, attr=None):
            _incref_masked_source(builder, value)
            builder.store_var(
                target, DeferredMethodCall(value, _wrap_with_release)
            )

        return _getattr

    for attrname in _INT_STR_STR_OPS:
        impl = _make_string_int_str_str_unified(attrname)
        lower_getattr(MaskedType(mlir_string), attrname)(
            _make_masked_binary_str_str_getattr(impl)
        )
    for attrname in ("startswith", "endswith"):
        impl = _make_string_bool_str_str_unified(attrname)
        lower_getattr(MaskedType(mlir_string), attrname)(
            _make_masked_binary_str_str_getattr(impl)
        )

    # --- operator.contains: ``masked_scalar in (literal_tuple)`` / UniTuple -> Masked(bool) ---
    def _const_mlir_for_membership(py_const, mlir_ty):
        if isinstance(py_const, float):
            return float_of(py_const, mlir_ty)
        if isinstance(py_const, bool):
            return int_of(int(py_const), mlir_ty)
        return int_of(py_const, mlir_ty)

    def _lower_masked_literal_tuple_contains(builder, target, args, kwargs):
        tup = builder.load_var(args[0])
        m = builder.load_var(args[1])
        st = llvm.StructType(m.type)
        m_val, m_valid = _extract_masked_value_valid(m, st.body[0], st.body[1])

        constant_values = []
        for x in tup:
            cv = try_extract_constant(x)
            if cv is None:
                raise NotImplementedError(
                    "Masked membership in tuple is only implemented for "
                    f"constant tuple elements, got {x!r}"
                )
            constant_values.append(cv)

        result = false()
        for const_val in constant_values:
            c = _const_mlir_for_membership(const_val, m_val.type)
            m_v, c_v = coerce_numpy_scalars_for_binary_op(m_val, c)
            result = arith.ori(result, equal(m_v, c_v))

        bool_mlir_ty = builder.get_mlir_type(types.boolean)
        undef_bool = llvm.UndefOp(bool_mlir_ty)
        final_bool = arith.select(m_valid, result, undef_bool)
        target_type = builder.get_numba_type(target.name)
        packed = _pack_masked_result(builder, target_type, final_bool, m_valid)
        builder.store_var(target, packed)

    def _lower_masked_unittuple_contains(builder, target, args, kwargs):
        tup = builder.load_var(args[0])
        if not isinstance(tup, tuple):
            raise NotImplementedError(
                f"UniTuple contains expects a lowered tuple, got {type(tup)}"
            )
        tup_t = concretize_tuple_to_tensor(tup)

        m = builder.load_var(args[1])
        st = llvm.StructType(m.type)
        m_val, m_valid = _extract_masked_value_valid(m, st.body[0], st.body[1])
        elem_ty = tup_t.type.element_type
        m_cmp = convert(m_val, elem_ty)

        def body(_op, element: mlir_ir.Value, accumulator: mlir_ir.Value):
            found = equal(element, m_cmp)
            found = arith.ori(found, accumulator)
            linalg.yield_([found])

        result_type = mlir_ir.RankedTensorType.get((), T.bool())
        init = tensor.splat(result_type, false(), [])
        dims_attr = mlir_ir.DenseI64ArrayAttr.get([0])
        reduce_op = linalg.ReduceOp(
            result=[result_type],
            inputs=[tup_t],
            inits=[init],
            dimensions=dims_attr,
        )
        block = reduce_op.combiner.blocks.append(
            tup_t.type.element_type, result_type.element_type
        )
        with mlir_ir.InsertionPoint(block):
            body(reduce_op, *block.arguments)
        combined = bool_of(tensor.extract(reduce_op.results[0], []))

        bool_mlir_ty = builder.get_mlir_type(types.boolean)
        undef_bool = llvm.UndefOp(bool_mlir_ty)
        final_bool = arith.select(m_valid, combined, undef_bool)
        target_type = builder.get_numba_type(target.name)
        packed = _pack_masked_result(builder, target_type, final_bool, m_valid)
        builder.store_var(target, packed)

    lower(operator.contains, types.Tuple, MaskedType)(
        _lower_masked_literal_tuple_contains
    )
    lower(operator.contains, types.UniTuple, MaskedType)(
        _lower_masked_unittuple_contains
    )

    # operator.contains: register all signatures so (substr in str) works for view/Literal/Masked(view)/Masked(cs).
    _contains_impl = _make_string_bool_str_str_unified("contains")
    for sig in _string_cmpop_sigs:
        lower(operator.contains, sig[0], sig[1])(_contains_impl)

    # --- Masked string-producing operations ---
    # These extract the view + valid from the Masked input,
    # call numba_cuda_mlir's MLIR string helpers, and pack the result
    # as Masked(mlir_string).
    #
    # DeferredMethodCall + NRT: when .upper()/.lower()/etc. is called on a
    # Masked(mlir_string), the getattr creates a DeferredMethodCall.  NRT
    # may del the source Masked before the deferred method executes its reads.
    # To prevent use-after-free, getattr increfs the source's inner meminfo,
    # and the deferred method decrefs it after reads complete.  Same pattern
    # as CooperativeArray (see testing-notes.md).

    def _incref_masked_source(builder, value_var):
        """Incref the inner mlir_string meminfo of a Masked source variable."""
        arg_ty = builder.get_numba_type(value_var.name)
        if isinstance(arg_ty, MaskedType) and builder.nrt.type_has_nrt_meminfo(
            arg_ty.value_type
        ):
            masked_val = builder.load_var(value_var)
            builder.incref(arg_ty, masked_val)

    def _decref_masked_source(builder, arg_var):
        """Decref the inner mlir_string meminfo after the deferred method is done reading."""
        arg_ty = builder.get_numba_type(arg_var.name)
        if isinstance(arg_ty, MaskedType) and builder.nrt.type_has_nrt_meminfo(
            arg_ty.value_type
        ):
            masked_val = builder.load_var(arg_var)
            builder.decref(arg_ty, masked_val)

    from cudf.core.udf.mlir_backend.string_lowering_impl import (
        lower_concat,
        lower_lower,
        lower_lstrip,
        lower_replace,
        lower_rstrip,
        lower_strip,
        lower_upper,
        mlir_string_to_view,
    )

    from cudf.core.udf.mlir_backend.strings_lowering import (
        _i64_const_from_ptr,
    )

    def _get_view_and_valid_from_masked(builder, arg_var):
        """Extract (view SSA value, valid bit) from a masked or scalar variable.

        Handles MaskedType(mlir_string), bare mlir_string, and StringLiteral.
        """
        valid_ty = builder.get_mlir_type(types.boolean)
        arg_ty = builder.get_numba_type(arg_var.name)
        if isinstance(arg_ty, types.StringLiteral):
            str_view = _literal_to_view(builder, arg_var)
            return str_view, arith.constant(valid_ty, 1)
        if isinstance(arg_ty, MaskedType) and isinstance(
            arg_ty.value_type, MLIRStringType
        ):
            masked_val = builder.load_var(arg_var)
            ms_ty = builder.get_mlir_type(mlir_string)
            ms_val = llvm.extractvalue(
                res=ms_ty,
                container=masked_val,
                position=mlir_ir.DenseI64ArrayAttr.get([0]),
            )
            valid_val = llvm.extractvalue(
                res=valid_ty,
                container=masked_val,
                position=mlir_ir.DenseI64ArrayAttr.get([1]),
            )
            str_view = mlir_string_to_view(ms_val)
            return str_view, valid_val
        if isinstance(arg_ty, MLIRStringType):
            ms_val = builder.load_var(arg_var)
            str_view = mlir_string_to_view(ms_val)
            return str_view, arith.constant(valid_ty, 1)
        raise TypeError(f"unexpected string arg type: {arg_ty}")

    # --- operator.add on Masked(mlir_string) -> Masked(mlir_string) ---
    def _make_masked_concat_unified():
        def _lower(builder, target, args, kwargs):
            lhs_val, valid_l = _get_view_and_valid_from_masked(
                builder, args[0]
            )
            rhs_val, valid_r = _get_view_and_valid_from_masked(
                builder, args[1]
            )
            result = lower_concat(builder.mlir_gpu_module, lhs_val, rhs_val)
            result_valid = arith.andi(valid_l, valid_r)
            target_type = builder.get_numba_type(target.name)
            packed = _pack_masked_result(
                builder, target_type, result, result_valid
            )
            builder.store_var(target, packed)

        return _lower

    _masked_concat_impl = _make_masked_concat_unified()
    _masked_add_sigs = [
        (MaskedType(mlir_string), MaskedType(mlir_string)),
        (MaskedType(mlir_string), types.StringLiteral),
        (types.StringLiteral, MaskedType(mlir_string)),
        (MaskedType(mlir_string), mlir_string),
        (mlir_string, MaskedType(mlir_string)),
    ]
    for sig in _masked_add_sigs:
        lower(operator.add, sig[0], sig[1])(_masked_concat_impl)

    # --- operator.getitem: Masked(mlir_string)[int] / [slice] ---
    from cudf.core.udf.mlir_backend.string_lowering_impl import (
        _zext_i32_to_i64,
        string_getitem_int_core,
        string_getitem_slice_core,
        view_extract_data,
        view_extract_nbytes,
    )

    def _lower_masked_getitem_int(builder, target, args, kwargs):
        str_view, valid_val = _get_view_and_valid_from_masked(builder, args[0])
        idx_val = builder.load_var(args[1])
        idx_ty = builder.get_numba_type(args[1].name)
        if isinstance(idx_ty, MaskedType):
            idx_struct_ty = llvm.StructType(idx_val.type)
            idx_int = llvm.extractvalue(idx_struct_ty.body[0], idx_val, [0])
            idx_valid = llvm.extractvalue(idx_struct_ty.body[1], idx_val, [1])
            valid_val = arith.andi(valid_val, idx_valid)
        else:
            idx_int = idx_val
        idx_i64 = builder.mlir_convert(idx_int, T.i64())

        data = view_extract_data(str_view)
        nbytes = _zext_i32_to_i64(view_extract_nbytes(str_view))
        result = string_getitem_int_core(
            builder.mlir_gpu_module, data, nbytes, idx_i64
        )
        target_type = builder.get_numba_type(target.name)
        packed = _pack_masked_result(builder, target_type, result, valid_val)
        builder.store_var(target, packed)

    for int_ty in (types.int32, types.int64, types.intp):
        lower(operator.getitem, MaskedType(mlir_string), int_ty)(
            _lower_masked_getitem_int
        )
        lower(operator.getitem, MaskedType(mlir_string), MaskedType(int_ty))(
            _lower_masked_getitem_int
        )

    def _lower_masked_getitem_slice(builder, target, args, kwargs):
        str_view, valid_val = _get_view_and_valid_from_masked(builder, args[0])
        slice_val = builder.load_var(args[1])
        data = view_extract_data(str_view)
        nbytes = _zext_i32_to_i64(view_extract_nbytes(str_view))
        result = string_getitem_slice_core(
            builder.mlir_gpu_module, data, nbytes, slice_val
        )
        target_type = builder.get_numba_type(target.name)
        packed = _pack_masked_result(builder, target_type, result, valid_val)
        builder.store_var(target, packed)

    lower(operator.getitem, MaskedType(mlir_string), types.slice2_type)(
        _lower_masked_getitem_slice
    )

    # --- Masked(mlir_string).upper / .lower -> Masked(mlir_string) ---
    from cudf.core.udf.mlir_backend.unicode import (
        get_character_cases_table_ptr,
        get_character_flags_table_ptr,
        get_special_case_mapping_table_ptr,
    )

    def _make_masked_upper_or_lower(mlir_fn):
        def _lower_impl(builder, target, args, kwargs):
            str_view, valid_val = _get_view_and_valid_from_masked(
                builder, args[0]
            )
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
            target_type = builder.get_numba_type(target.name)
            packed = _pack_masked_result(
                builder, target_type, result, valid_val
            )
            builder.store_var(target, packed)
            _decref_masked_source(builder, args[0])

        def _getattr(context, builder, target, value, attr=None):
            _incref_masked_source(builder, value)
            builder.store_var(target, DeferredMethodCall(value, _lower_impl))

        return _getattr

    lower_getattr(MaskedType(mlir_string), "upper")(
        _make_masked_upper_or_lower(lower_upper)
    )
    lower_getattr(MaskedType(mlir_string), "lower")(
        _make_masked_upper_or_lower(lower_lower)
    )

    # --- Masked(mlir_string).replace(old, new) -> Masked(mlir_string) ---
    def _lower_masked_replace_impl(builder, target, args, kwargs):
        str_view, valid_src = _get_view_and_valid_from_masked(builder, args[0])
        old_val, valid_old = _get_view_and_valid_from_masked(builder, args[1])
        new_val, valid_new = _get_view_and_valid_from_masked(builder, args[2])
        result = lower_replace(
            builder.mlir_gpu_module, str_view, old_val, new_val
        )
        result_valid = arith.andi(arith.andi(valid_src, valid_old), valid_new)
        target_type = builder.get_numba_type(target.name)
        packed = _pack_masked_result(
            builder, target_type, result, result_valid
        )
        builder.store_var(target, packed)
        _decref_masked_source(builder, args[0])

    def _masked_replace_getattr(context, builder, target, value, attr=None):
        _incref_masked_source(builder, value)
        builder.store_var(
            target,
            DeferredMethodCall(value, _lower_masked_replace_impl),
        )

    lower_getattr(MaskedType(mlir_string), "replace")(_masked_replace_getattr)

    # --- Masked(mlir_string).strip / lstrip / rstrip -> Masked(mlir_string) ---
    def _make_masked_strip(mlir_fn):
        def _lower_impl(builder, target, args, kwargs):
            str_view, valid_src = _get_view_and_valid_from_masked(
                builder, args[0]
            )
            strip_view, valid_chars = _get_view_and_valid_from_masked(
                builder, args[1]
            )
            result = mlir_fn(builder.mlir_gpu_module, str_view, strip_view)
            result_valid = arith.andi(valid_src, valid_chars)
            target_type = builder.get_numba_type(target.name)
            packed = _pack_masked_result(
                builder, target_type, result, result_valid
            )
            builder.store_var(target, packed)
            _decref_masked_source(builder, args[0])

        def _getattr(context, builder, target, value, attr=None):
            _incref_masked_source(builder, value)
            builder.store_var(target, DeferredMethodCall(value, _lower_impl))

        return _getattr

    lower_getattr(MaskedType(mlir_string), "strip")(
        _make_masked_strip(lower_strip)
    )
    lower_getattr(MaskedType(mlir_string), "lstrip")(
        _make_masked_strip(lower_lstrip)
    )
    lower_getattr(MaskedType(mlir_string), "rstrip")(
        _make_masked_strip(lower_rstrip)
    )

    # --- Masked(mlir_string).nbytes -> Masked(int64) ---
    from cudf.core.udf.mlir_backend.string_lowering_impl import (
        ms_extract_data,
        ms_extract_nbytes,
    )

    def _lower_masked_nbytes(context, builder, target, value, attr=None):
        struct_value = builder.load_var(value)
        ms_val = llvm.extractvalue(
            llvm.StructType(struct_value.type).body[0], struct_value, [0]
        )
        valid = llvm.extractvalue(
            llvm.StructType(struct_value.type).body[1], struct_value, [1]
        )
        nbytes_val = ms_extract_nbytes(ms_val)
        target_type = builder.get_numba_type(target.name)
        packed = _pack_masked_result(builder, target_type, nbytes_val, valid)
        builder.store_var(target, packed)

    lower_getattr(MaskedType(mlir_string), "nbytes")(_lower_masked_nbytes)

    # --- Masked(mlir_string).data_ptr -> Masked(int64) ---
    def _lower_masked_data_ptr(context, builder, target, value, attr=None):
        struct_value = builder.load_var(value)
        ms_val = llvm.extractvalue(
            llvm.StructType(struct_value.type).body[0], struct_value, [0]
        )
        valid = llvm.extractvalue(
            llvm.StructType(struct_value.type).body[1], struct_value, [1]
        )
        data_ptr = ms_extract_data(ms_val)
        data_i64 = llvm.ptrtoint(T.i64(), data_ptr)
        target_type = builder.get_numba_type(target.name)
        packed = _pack_masked_result(builder, target_type, data_i64, valid)
        builder.store_var(target, packed)

    lower_getattr(MaskedType(mlir_string), "data_ptr")(_lower_masked_data_ptr)

    # --- mlir_string_from_ptr(Masked(int64), Masked(int64)) -> Masked(mlir_string) ---
    from cudf.core.udf.mlir_backend.string_lowering_impl import build_mlir_string
    from cudf.core.udf.mlir_backend.string_typing_impl import (
        mlir_string_from_ptr as _from_ptr_stub,
    )

    def _lower_masked_from_ptr(builder, target, args, kwargs):
        data_masked = builder.load_var(args[0])
        nbytes_masked = builder.load_var(args[1])
        data_struct_ty = llvm.StructType(data_masked.type)
        data_i64 = llvm.extractvalue(data_struct_ty.body[0], data_masked, [0])
        data_valid = llvm.extractvalue(
            data_struct_ty.body[1], data_masked, [1]
        )
        nbytes_struct_ty = llvm.StructType(nbytes_masked.type)
        nbytes_i64 = llvm.extractvalue(
            nbytes_struct_ty.body[0], nbytes_masked, [0]
        )
        nbytes_valid = llvm.extractvalue(
            nbytes_struct_ty.body[1], nbytes_masked, [1]
        )

        data_ptr = llvm.inttoptr(llvm.PointerType.get(), data_i64)
        null_ptr = llvm.ZeroOp(llvm.PointerType.get())
        ms_val = build_mlir_string(null_ptr, data_ptr, nbytes_i64)
        result_valid = arith.andi(data_valid, nbytes_valid)

        target_type = builder.get_numba_type(target.name)
        packed = _pack_masked_result(
            builder, target_type, ms_val, result_valid
        )
        builder.store_var(target, packed)

    lower(
        _from_ptr_stub,
        MaskedType(types.int64),
        MaskedType(types.int64),
    )(_lower_masked_from_ptr)


_register()
