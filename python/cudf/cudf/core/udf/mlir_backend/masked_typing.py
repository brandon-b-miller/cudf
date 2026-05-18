# SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0
"""
MLIR (numba_cuda_mlir) typing for MaskedType: data model, Masked constructor,
.value/.valid, pack_return, len(Masked(string_view)), and full numeric
arithmetic (binary arith/bitwise/comparison, unary, Masked+NA, scalar+Masked).
String support: only len. All types defined here; no import from masked_typing.
Importing this module registers typing with numba_cuda_mlir.
"""

from __future__ import annotations

import operator

from numba_cuda_mlir import models, types
from numba_cuda_mlir._mlir.dialects import llvm
from numba_cuda_mlir.extending import typing_registry
from numba_cuda_mlir.models import PrimitiveModel, register_model
from numba_cuda_mlir.numba_cuda.extending import typeof_impl
from numba_cuda_mlir.numba_cuda.types.misc import unliteral
from numba_cuda_mlir.numba_cuda.typing.templates import (
    AbstractTemplate,
    AttributeTemplate,
    ConcreteTemplate,
)
from numba_cuda_mlir.typing import signature as nb_signature

from cudf.core.missing import NA
from cudf.core.udf._ops import (
    arith_ops,
    bitwise_ops,
    comparison_ops,
    unary_ops,
)
from cudf.core.udf.api import Masked, pack_return
from cudf.core.udf.mlir_backend.strings_typing import (
    managed_udf_string,
    size_type,
    string_view,
)
from cudf.core.udf.utils import SUPPORTED_MASKED_TYPES, _supported_masked_types


# --- MaskedType and NAType: defined here for MLIR path (no masked_typing import) ---
class MaskedType(types.Type):
    """
    A Numba type (value_type, valid) for the MLIR path. Parameterized by value type.
    """

    def __init__(self, value):
        if isinstance(value, types.Literal):
            value = unliteral(value)
        self.value_type = (
            value
            if isinstance(value, SUPPORTED_MASKED_TYPES)
            else types.Poison(value)
        )
        super().__init__(name=f"Masked({self.value_type})")

    def __hash__(self):
        return hash(repr(self))

    def unify(self, context, other):
        if isinstance(other, NAType):
            return self
        if isinstance(other, MaskedType):
            unified = context.unify_pairs(self.value_type, other.value_type)
            return MaskedType(unified) if unified is not None else None
        unified = context.unify_pairs(self.value_type, other)
        if unified is None:
            return None
        return MaskedType(unified)


class NAType(types.Type):
    """Type for cudf.NA in the MLIR path; ops like Masked + NA, x is NA."""

    def __init__(self):
        super().__init__(name="NA")

    def unify(self, context, other):
        if isinstance(other, MaskedType):
            return None
        if isinstance(other, NAType):
            return self
        return MaskedType(other)


na_type = NAType()


@typeof_impl.register(type(NA))
def _typeof_na(val, c):
    return na_type


register_model(NAType)(models.OpaqueModel)


def _register():
    # --- MLIR data model for MaskedType ---
    @register_model(MaskedType)
    class MaskedTypeModel(PrimitiveModel):
        def __init__(self, dmm, fe_type):
            value_mlir = dmm.lookup(fe_type.value_type).get_value_type()
            valid_mlir = dmm.lookup(types.boolean).get_value_type()
            be_type = llvm.StructType.new_identified(
                fe_type.name, [value_mlir, valid_mlir]
            )
            super().__init__(dmm, fe_type, be_type)

    # --- Typing: Masked(value, valid) ---
    class MaskedConstructor(ConcreteTemplate):
        key = Masked
        cases = [
            nb_signature(MaskedType(t), t, types.boolean)
            for t in _supported_masked_types
        ]

    typing_registry.register_global(Masked, types.Function(MaskedConstructor))

    # --- Typing: len(Masked(string_view)) -> Masked(size_type) ---
    class LenMaskedStringViewTemplate(AbstractTemplate):
        key = len

        def generic(self, args, kws):
            if len(args) != 1 or kws:
                return None
            if (
                isinstance(args[0], MaskedType)
                and args[0].value_type is string_view
            ):
                return nb_signature(MaskedType(size_type), args[0])
            return None

    typing_registry.register_global(
        len, types.Function(LenMaskedStringViewTemplate)
    )

    # --- Typing: .value and .valid ---
    @typing_registry.register_attr
    class MaskedTypeAttrs(AttributeTemplate):
        key = MaskedType

        def generic_resolve(self, typ, attr):
            if attr == "value":
                return typ.value_type
            if attr == "valid":
                return types.boolean
            return None

    # --- Typing: Masked(string_view).isupper, .islower, etc. -> Masked(boolean) ---
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

    def _make_masked_sv_is_attr(attrname):
        class MaskedStringViewIsAttrTemplate(AbstractTemplate):
            key = f"MaskedType.{attrname}"

            def generic(self, args, kws):
                return nb_signature(MaskedType(types.boolean), recvr=self.this)

        def resolve_attr(self, mod):
            return types.BoundFunction(MaskedStringViewIsAttrTemplate, mod)

        return resolve_attr

    @typing_registry.register_attr
    class MaskedStringViewAttrs(AttributeTemplate):
        key = MaskedType(string_view)

        def resolve_value(self, mod):
            return string_view

        def resolve_valid(self, mod):
            return types.boolean

    for attrname in _id_unary_funcs:
        setattr(
            MaskedStringViewAttrs,
            f"resolve_{attrname}",
            _make_masked_sv_is_attr(attrname),
        )

    # --- Typing: Masked(string_view).find / .rfind (-> Masked(size_type)), .startswith / .endswith (-> Masked(boolean)) ---
    def _make_masked_sv_binary_attr(attrname, retty):
        class MaskedStringViewBinaryAttrTemplate(AbstractTemplate):
            key = f"StringView.{attrname}"

            def generic(self, args, kws):
                if (
                    len(args) == 1
                    and not kws
                    and _is_valid_string_arg(self.this)
                    and _is_valid_string_arg(args[0])
                ):
                    return nb_signature(
                        MaskedType(retty),
                        args[0],
                        recvr=self.this,
                    )

        def resolve_attr(self, mod):
            return types.BoundFunction(
                MaskedStringViewBinaryAttrTemplate, MaskedType(string_view)
            )

        return resolve_attr

    for attrname in ("find", "rfind", "count"):
        setattr(
            MaskedStringViewAttrs,
            f"resolve_{attrname}",
            _make_masked_sv_binary_attr(attrname, size_type),
        )
    for attrname in ("startswith", "endswith"):
        setattr(
            MaskedStringViewAttrs,
            f"resolve_{attrname}",
            _make_masked_sv_binary_attr(attrname, types.boolean),
        )

    # --- Typing: Masked(string_view).upper / .lower -> Masked(managed_udf_string) ---
    def _make_masked_sv_transform_attr(attrname):
        class MaskedStringViewTransformTemplate(AbstractTemplate):
            key = f"MaskedType.{attrname}"

            def generic(self, args, kws):
                return nb_signature(
                    MaskedType(managed_udf_string), recvr=self.this
                )

        def resolve_attr(self, mod):
            return types.BoundFunction(MaskedStringViewTransformTemplate, mod)

        return resolve_attr

    for attrname in ("upper", "lower"):
        setattr(
            MaskedStringViewAttrs,
            f"resolve_{attrname}",
            _make_masked_sv_transform_attr(attrname),
        )

    # --- Typing: Masked(string_view).replace(old, new) -> Masked(managed_udf_string) ---
    class MaskedReplaceTemplate(AbstractTemplate):
        key = "MaskedType.replace"

        def generic(self, args, kws):
            if len(args) == 2 and not kws:
                return nb_signature(
                    MaskedType(managed_udf_string),
                    args[0],
                    args[1],
                    recvr=self.this,
                )

    def _resolve_masked_replace(self, mod):
        return types.BoundFunction(
            MaskedReplaceTemplate, MaskedType(string_view)
        )

    setattr(
        MaskedStringViewAttrs,
        "resolve_replace",
        _resolve_masked_replace,
    )

    # --- Typing: Masked(string_view).strip/lstrip/rstrip -> Masked(managed_udf_string) ---
    def _make_masked_strip_attr(attrname):
        class MaskedStripTemplate(AbstractTemplate):
            key = f"MaskedType.{attrname}"

            def generic(self, args, kws):
                if len(args) == 1 and not kws:
                    return nb_signature(
                        MaskedType(managed_udf_string),
                        args[0],
                        recvr=self.this,
                    )

        def resolve_attr(self, mod):
            return types.BoundFunction(
                MaskedStripTemplate, MaskedType(string_view)
            )

        return resolve_attr

    for attrname in ("strip", "lstrip", "rstrip"):
        setattr(
            MaskedStringViewAttrs,
            f"resolve_{attrname}",
            _make_masked_strip_attr(attrname),
        )

    # --- Typing: operator.add on Masked(string_view) -> Masked(managed_udf_string) ---
    class MaskedStringAddTemplate(AbstractTemplate):
        key = operator.add

        def generic(self, args, kws):
            if _is_valid_string_arg(args[0]) and _is_valid_string_arg(args[1]):
                return nb_signature(
                    MaskedType(managed_udf_string),
                    MaskedType(string_view),
                    MaskedType(string_view),
                )

    typing_registry.register_global(
        operator.add, types.Function(MaskedStringAddTemplate)
    )

    # --- Typing: pack_return ---
    @typing_registry.register_global(pack_return)
    class PackReturnTemplate(AbstractTemplate):
        def generic(self, args, kws):
            if isinstance(args[0], MaskedType):
                return nb_signature(args[0], args[0])
            if isinstance(args[0], (types.Number, types.Boolean)):
                return nb_signature(MaskedType(args[0]), args[0])
            return None

    # --- Binary ops: Masked <op> Masked, Masked <op> scalar, scalar <op> Masked ---
    class MaskedScalarArithOp(AbstractTemplate):
        def generic(self, args, kws):
            if isinstance(args[0], MaskedType) and isinstance(
                args[1], MaskedType
            ):
                return_type = self.context.resolve_function_type(
                    self.key, (args[0].value_type, args[1].value_type), kws
                ).return_type
                return nb_signature(MaskedType(return_type), args[0], args[1])
            return None

    class MaskedScalarScalarOp(AbstractTemplate):
        def generic(self, args, kws):
            # Masked <op> scalar (including Literal so row['a'] == 1 matches)
            if isinstance(args[0], MaskedType) and isinstance(
                args[1], (types.Number, types.Boolean)
            ):
                return_type = self.context.resolve_function_type(
                    self.key, (args[0].value_type, args[1]), kws
                ).return_type
                return nb_signature(MaskedType(return_type), args[0], args[1])
            if isinstance(args[0], MaskedType) and isinstance(
                args[1], types.Literal
            ):
                from numba_cuda_mlir.numba_cuda.types.misc import unliteral

                scalar_ty = unliteral(args[1])
                return_type = self.context.resolve_function_type(
                    self.key, (args[0].value_type, scalar_ty), kws
                ).return_type
                return nb_signature(MaskedType(return_type), args[0], args[1])
            # scalar <op> Masked
            if isinstance(
                args[0], (types.Number, types.Boolean)
            ) and isinstance(args[1], MaskedType):
                return_type = self.context.resolve_function_type(
                    self.key, (args[0], args[1].value_type), kws
                ).return_type
                return nb_signature(MaskedType(return_type), args[0], args[1])
            if isinstance(args[0], types.Literal) and isinstance(
                args[1], MaskedType
            ):
                from numba_cuda_mlir.numba_cuda.types.misc import unliteral

                scalar_ty = unliteral(args[0])
                return_type = self.context.resolve_function_type(
                    self.key, (scalar_ty, args[1].value_type), kws
                ).return_type
                return nb_signature(MaskedType(return_type), args[0], args[1])
            return None

    class MaskedScalarNullOp(AbstractTemplate):
        def generic(self, args, kws):
            if isinstance(args[0], MaskedType) and isinstance(args[1], NAType):
                return nb_signature(args[0], args[0], na_type)
            if isinstance(args[0], NAType) and isinstance(args[1], MaskedType):
                return nb_signature(args[1], na_type, args[1])
            return None

    for binary_op in arith_ops + bitwise_ops + comparison_ops:
        typing_registry.register_global(binary_op)(MaskedScalarArithOp)
        typing_registry.register_global(binary_op)(MaskedScalarNullOp)
        typing_registry.register_global(binary_op)(MaskedScalarScalarOp)

    # --- Unary ops: <op> Masked ---
    class MaskedScalarUnaryOp(AbstractTemplate):
        def generic(self, args, kws):
            if len(args) == 1 and isinstance(args[0], MaskedType):
                return_type = self.context.resolve_function_type(
                    self.key, (args[0].value_type,), kws
                ).return_type
                return nb_signature(MaskedType(return_type), args[0])
            return None

    for unary_op in unary_ops:
        typing_registry.register_global(unary_op)(MaskedScalarUnaryOp)

    # --- operator.is_ (Masked is NA) ---
    @typing_registry.register_global(operator.is_)
    class MaskedScalarIsNull(AbstractTemplate):
        def generic(self, args, kws):
            if isinstance(args[0], MaskedType) and isinstance(args[1], NAType):
                return nb_signature(types.boolean, args[0], na_type)
            if isinstance(args[0], NAType) and isinstance(args[1], MaskedType):
                return nb_signature(types.boolean, na_type, args[1])
            return None

    # --- operator.is_not (Masked is not NA) ---
    @typing_registry.register_global(operator.is_not)
    class MaskedScalarIsNotNull(AbstractTemplate):
        def generic(self, args, kws):
            if isinstance(args[0], MaskedType) and isinstance(args[1], NAType):
                return nb_signature(types.boolean, args[0], na_type)
            if isinstance(args[0], NAType) and isinstance(args[1], MaskedType):
                return nb_signature(types.boolean, na_type, args[1])
            return None

    # --- operator.truth / bool (if Masked) ---
    @typing_registry.register_global(operator.truth)
    @typing_registry.register_global(bool)
    class MaskedScalarTruth(AbstractTemplate):
        def generic(self, args, kws):
            if isinstance(args[0], MaskedType):
                return nb_signature(types.boolean, MaskedType(types.boolean))
            return None

    # --- float(Masked), int(Masked) ---
    @typing_registry.register_global(float)
    class MaskedScalarFloatCast(AbstractTemplate):
        def generic(self, args, kws):
            if isinstance(args[0], MaskedType):
                return nb_signature(MaskedType(types.float64), args[0])
            return None

    @typing_registry.register_global(int)
    class MaskedScalarIntCast(AbstractTemplate):
        def generic(self, args, kws):
            if isinstance(args[0], MaskedType):
                return nb_signature(MaskedType(types.int64), args[0])
            return None

    # --- abs(Masked) ---
    @typing_registry.register_global(abs)
    class MaskedScalarAbsoluteValue(AbstractTemplate):
        def generic(self, args, kws):
            if isinstance(args[0], MaskedType):
                if args[0].value_type is string_view:
                    return None
                return_type = self.context.resolve_function_type(
                    self.key, (args[0].value_type,), kws
                ).return_type
                return nb_signature(MaskedType(return_type), args[0])
            return None

    def _is_valid_string_arg(ty):
        return (
            (isinstance(ty, MaskedType) and ty.value_type is string_view)
            or isinstance(ty, types.StringLiteral)
            or isinstance(ty, types.UnicodeType)
        )

    def _is_masked_numeric_membership_item(ty):
        """RHS of ``x in (a, b, ...)`` when x is Masked (not string_view)."""
        return (
            isinstance(ty, MaskedType)
            and ty.value_type is not string_view
            and not isinstance(ty.value_type, types.Poison)
        )

    class MaskedNumericSequenceContainsTemplate(AbstractTemplate):
        """``value in (literal_tuple)`` / ``value in unittuple`` with Masked scalar."""

        key = operator.contains

        def generic(self, args, kws):
            if len(args) != 2:
                return None
            container, item = args
            if not _is_masked_numeric_membership_item(item):
                return None
            if isinstance(container, types.Tuple) and all(
                isinstance(x, types.Literal) for x in container.types
            ):
                return nb_signature(MaskedType(types.boolean), container, item)
            if isinstance(container, types.UniTuple):
                return nb_signature(MaskedType(types.boolean), container, item)
            return None

    class MaskedStringViewCmpOp(AbstractTemplate):
        """
        Return the boolean result of cmpop between two strings.
        Typing is the same for every comparison operator, so reuse for all.
        """

        def generic(self, args, kws):
            if _is_valid_string_arg(args[0]) and _is_valid_string_arg(args[1]):
                return nb_signature(
                    MaskedType(types.boolean),
                    MaskedType(string_view),
                    MaskedType(string_view),
                )

    def _make_masked_string_cmpop_template(cmpop):
        class CmpOpTemplate(MaskedStringViewCmpOp):
            key = cmpop

        return CmpOpTemplate

    for op in comparison_ops:
        typing_registry.register_global(
            op, types.Function(_make_masked_string_cmpop_template(op))
        )

    # --- operator.contains (substr in str) with Masked/Literal string args -> Masked(boolean) ---
    class MaskedStringViewContainsTemplate(AbstractTemplate):
        key = operator.contains

        def generic(self, args, kws):
            if _is_valid_string_arg(args[0]) and _is_valid_string_arg(args[1]):
                return nb_signature(
                    MaskedType(types.boolean),
                    MaskedType(string_view),
                    MaskedType(string_view),
                )

    typing_registry.register_global(
        operator.contains,
        types.Function(
            [
                MaskedNumericSequenceContainsTemplate,
                MaskedStringViewContainsTemplate,
            ]
        ),
    )


_register()
