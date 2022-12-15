# Copyright (c) 2022, NVIDIA CORPORATION.

import operator

import llvmlite.binding as ll
import numpy as np
from numba import types
from numba.core.datamodel import default_manager
from numba.core.extending import models, register_model
from numba.core.typing import signature as nb_signature
from numba.core.typing.templates import AbstractTemplate, AttributeTemplate
from numba.cuda.cudadecl import registry as cuda_decl_registry
from numba.cuda.cudadrv import nvvm

data_layout = nvvm.data_layout

# libcudf size_type
size_type = types.int32

# workaround for numba < 0.56
if isinstance(data_layout, dict):
    data_layout = data_layout[64]
target_data = ll.create_target_data(data_layout)


# String object definitions
class UDFString(types.Type):

    np_dtype = np.dtype("object")

    def __init__(self):
        super().__init__(name="udf_string")
        llty = default_manager[self].get_value_type()
        self.size_bytes = llty.get_abi_size(target_data)

    @property
    def return_type(self):
        return self


class StringView(types.Type):
    np_dtype = np.dtype("object")

    def __init__(self):
        super().__init__(name="string_view")
        llty = default_manager[self].get_value_type()
        self.size_bytes = llty.get_abi_size(target_data)

    @property
    def return_type(self):
        return UDFString()


@register_model(StringView)
class stringview_model(models.StructModel):
    # from string_view.hpp:
    _members = (
        # const char* _data{}
        # Pointer to device memory contain char array for this string
        ("data", types.CPointer(types.char)),
        # size_type _bytes{};
        # Number of bytes in _data for this string
        ("bytes", size_type),
        # mutable size_type _length{};
        # Number of characters in this string (computed)
        ("length", size_type),
    )

    def __init__(self, dmm, fe_type):
        super().__init__(dmm, fe_type, self._members)


@register_model(UDFString)
class udf_string_model(models.StructModel):
    # from udf_string.hpp:
    # private:
    #   char* m_data{};
    #   cudf::size_type m_bytes{};
    #   cudf::size_type m_size{};

    _members = (
        ("m_data", types.CPointer(types.char)),
        ("m_bytes", size_type),
        ("m_size", size_type),
    )

    def __init__(self, dmm, fe_type):
        super().__init__(dmm, fe_type, self._members)


any_string_ty = (StringView, UDFString, types.StringLiteral)
string_view = StringView()
udf_string = UDFString()


class StrViewArgHandler:
    """
    As part of Numba's preprocessing step, incoming function arguments are
    modified based on the associated type for that argument that was used
    to JIT the kernel. However it only knows how to handle built in array
    types natively. With string UDFs, the jitted type is string_view*,
    which numba does not know how to handle.

    This class converts string_view* to raw pointer arguments, which Numba
    knows how to use.

    See numba.cuda.compiler._prepare_args for details.
    """

    def prepare_args(self, ty, val, **kwargs):
        if isinstance(ty, types.CPointer) and isinstance(
            ty.dtype, (StringView, UDFString)
        ):
            return types.uint64, val.ptr
        else:
            return ty, val


str_view_arg_handler = StrViewArgHandler()


# String functions
@cuda_decl_registry.register_global(len)
class StringLength(AbstractTemplate):
    """
    provide the length of a cudf::string_view like struct
    """

    def generic(self, args, kws):
        if isinstance(args[0], any_string_ty) and len(args) == 1:
            # length:
            # string_view -> int32
            # udf_string -> int32
            # literal -> int32
            return nb_signature(size_type, args[0])


def register_stringview_binaryop(op, retty):
    """
    Helper function wrapping numba's low level extension API. Provides
    the boilerplate needed to associate a signature with a function or
    operator expecting a string.
    """

    class StringViewBinaryOp(AbstractTemplate):
        def generic(self, args, kws):
            if isinstance(args[0], any_string_ty) and isinstance(
                args[1], any_string_ty
            ):
                return nb_signature(retty, string_view, string_view)

    cuda_decl_registry.register_global(op)(StringViewBinaryOp)


def create_binary_attrs(attrname, retty):
    """
    Helper function wrapping numba's low level extension API. Provides
    the boilerplate needed to register a binary function of two string
    objects as an attribute of one, e.g. `string.func(other)`.
    """

    class StringViewBinaryAttr(AbstractTemplate):
        key = f"StringView.{attrname}"

        def generic(self, args, kws):
            return nb_signature(retty, string_view, recvr=self.this)

    class UDFStringBinaryAttr(StringViewBinaryAttr):
        key = f"UDFString.{attrname}"

    def string_view_attr(self, mod):
        return types.BoundFunction(StringViewBinaryAttr, string_view)

    def udf_string_attr(self, mod):
        return types.BoundFunction(UDFStringBinaryAttr, string_view)

    return string_view_attr, udf_string_attr


def create_identifier_attrs(attrname, retty):
    """
    Helper function wrapping numba's low level extension API. Provides
    the boilerplate needed to register a unary function of a string
    object as an attribute, e.g. `string.func()`.
    """

    class StringViewIdentifierAttr(AbstractTemplate):
        key = f"StringView.{attrname}"

        def generic(self, args, kws):
            return nb_signature(retty, recvr=self.this)

    class UDFStringIdentifierAttr(StringViewIdentifierAttr):
        key = f"UDFString.{attrname}"

    def string_view_attr(self, mod):
        return types.BoundFunction(StringViewIdentifierAttr, string_view)

    def udf_string_attr(self, mod):
        return types.BoundFunction(UDFStringIdentifierAttr, string_view)

    return string_view_attr, udf_string_attr


class StringViewCount(AbstractTemplate):
    key = "StringView.count"

    def generic(self, args, kws):
        return nb_signature(size_type, string_view, recvr=self.this)


class StringViewReplace(AbstractTemplate):
    key = "StringView.replace"

    def generic(self, args, kws):
        return nb_signature(
            udf_string, string_view, string_view, recvr=self.this
        )


@cuda_decl_registry.register_attr
class StringViewAttrs(AttributeTemplate):
    key = string_view

    def resolve_count(self, mod):
        return types.BoundFunction(StringViewCount, string_view)

    def resolve_replace(self, mod):
        return types.BoundFunction(StringViewReplace, string_view)


class UDFStringCount(StringViewCount):
    key = "UDFString.count"


class UDFStringReplace(StringViewReplace):
    key = "UDFString.replace"


@cuda_decl_registry.register_attr
class UDFStringAttrs(AttributeTemplate):
    key = udf_string

    def resolve_count(self, mod):
        return types.BoundFunction(UDFStringCount, string_view)

    def resolve_replace(self, mod):
        return types.BoundFunction(UDFStringReplace, string_view)


# Build attributes for `MaskedType(string_view)`
bool_binary_funcs = ["startswith", "endswith"]
int_binary_funcs = ["find", "rfind"]
id_unary_funcs = [
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
string_unary_funcs = ["upper", "lower"]
string_return_attrs = ["strip", "lstrip", "rstrip"]

for func in bool_binary_funcs:
    string_view_attr, udf_string_attr = create_binary_attrs(
        func, types.boolean
    )
    setattr(StringViewAttrs, f"resolve_{func}", string_view_attr)
    setattr(UDFStringAttrs, f"resolve_{func}", udf_string_attr)


for func in string_return_attrs:
    string_view_attr, udf_string_attr = create_binary_attrs(func, udf_string)
    setattr(StringViewAttrs, f"resolve_{func}", string_view_attr)
    setattr(UDFStringAttrs, f"resolve_{func}", udf_string_attr)


for func in int_binary_funcs:
    string_view_attr, udf_string_attr = create_binary_attrs(func, size_type)
    setattr(StringViewAttrs, f"resolve_{func}", string_view_attr)
    setattr(UDFStringAttrs, f"resolve_{func}", udf_string_attr)


for func in id_unary_funcs:
    string_view_attr, udf_string_attr = create_identifier_attrs(
        func, types.boolean
    )
    setattr(StringViewAttrs, f"resolve_{func}", string_view_attr)
    setattr(UDFStringAttrs, f"resolve_{func}", udf_string_attr)

for func in string_unary_funcs:
    string_view_attr, udf_string_attr = create_identifier_attrs(
        func, udf_string
    )
    setattr(StringViewAttrs, f"resolve_{func}", string_view_attr)
    setattr(UDFStringAttrs, f"resolve_{func}", udf_string_attr)


cuda_decl_registry.register_attr(StringViewAttrs)
cuda_decl_registry.register_attr(UDFStringAttrs)

register_stringview_binaryop(operator.eq, types.boolean)
register_stringview_binaryop(operator.ne, types.boolean)
register_stringview_binaryop(operator.lt, types.boolean)
register_stringview_binaryop(operator.gt, types.boolean)
register_stringview_binaryop(operator.le, types.boolean)
register_stringview_binaryop(operator.ge, types.boolean)

# st in other
register_stringview_binaryop(operator.contains, types.boolean)

# st + other
register_stringview_binaryop(operator.add, udf_string)
