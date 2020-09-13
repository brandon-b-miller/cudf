# Copyright (c) 2020, NVIDIA CORPORATION.

import numpy as np
import pandas as pd

from libc.stdint cimport (
    int8_t,
    int16_t,
    int32_t,
    int64_t,
    uint8_t,
    uint16_t,
    uint32_t,
    uint64_t,
)
from libcpp.memory cimport unique_ptr
from libcpp cimport bool
from libc.stdint cimport uintptr_t

import cudf
from cudf._lib.types import cudf_to_np_types, duration_unit_map
from cudf._lib.types import datetime_unit_map
from cudf._lib.types cimport underlying_type_t_type_id
from cudf._lib.move cimport move

from cudf._lib.cpp.wrappers.timestamps cimport (
    timestamp_s,
    timestamp_ms,
    timestamp_us,
    timestamp_ns
)
from cudf._lib.cpp.wrappers.durations cimport(
    duration_s,
    duration_ms,
    duration_us,
    duration_ns
)
from cudf._lib.cpp.scalar.scalar cimport (
    scalar,
    numeric_scalar,
    timestamp_scalar,
    duration_scalar,
    string_scalar
)
cimport cudf._lib.cpp.types as libcudf_types
from cudf.utils.dtypes import to_cudf_compatible_scalar
cdef class Scalar:

    def __init__(self, value, dtype=None):
        """
        cudf.Scalar: Type representing a scalar value on the device

        Parameters
        ----------
        value : scalar
            An object of scalar type, i.e., one for which
            `np.isscalar()` returns `True`. Can also be `None`,
            to represent a "null" scalar. In this case,
            dtype *must* be provided.
        dtype : dtype
            A NumPy dtype.
        """

        value = to_cudf_compatible_scalar(value, dtype=dtype)

        valid = value is not None

        if dtype is None:
            if value is None:
                raise TypeError(
                    "dtype required when constructing a null scalar"
                )
            else:
                dtype = value.dtype

        dtype = cudf.dtype(dtype)

        if isinstance(dtype, cudf.StringDtype):
            _set_string_from_np_string(self.c_value, value, valid)
        elif isinstance(dtype, (cudf.Number, cudf.BooleanDtype)):
            _set_numeric_from_np_scalar(self.c_value, value, dtype, valid)
        elif isinstance(dtype, cudf.Datetime):
            _set_datetime64_from_np_scalar(
                self.c_value, value, dtype, valid
            )
        elif isinstance(dtype, cudf.Timedelta):
            _set_timedelta64_from_np_scalar(
                self.c_value, value, dtype, valid
            )
        else:
            raise ValueError(
                f"Cannot convert value of type "
                f"{type(value).__name__} to cudf scalar"
            )

    def __eq__(self, other):
        if isinstance(other, Scalar):
            other = other.value
        return self.value == other

    @property
    def dtype(self):
        """
        The NumPy dtype corresponding to the data type of the underlying
        device scalar.
        """
        cdef libcudf_types.data_type cdtype = self.c_value.get()[0].type()
        return cudf_to_np_types[<underlying_type_t_type_id>(cdtype.id())]

    @property
    def value(self):
        """
        Returns a host copy of the underlying device scalar.
        """
        if cudf.api.types.is_string_dtype(self.dtype):
            return _get_py_string_from_string(self.c_value)
        elif cudf.api.types.is_numerical_dtype(self.dtype):
            return _get_np_scalar_from_numeric(self.c_value)
        elif cudf.api.types.is_datetime64_dtype(self.dtype):
            return _get_np_scalar_from_timestamp64(self.c_value)
        elif cudf.api.types.is_timedelta64_dtype(self.dtype):
            return _get_np_scalar_from_timedelta64(self.c_value)
        else:
            raise ValueError(
                "Could not convert cudf::scalar to a Python value"
            )

    cpdef bool is_valid(self):
        """
        Returns if the Scalar is valid or not(i.e., <NA>).
        """
        return self.c_value.get()[0].is_valid()

    def __repr__(self):
        if self.value is None:
            return f"Scalar(<NA>, {self.dtype.__repr__()})"
        else:
            return f"Scalar({self.value.__repr__()})"

    @staticmethod
    cdef Scalar from_unique_ptr(unique_ptr[scalar] ptr):
        """
        Construct a Scalar object from a unique_ptr<cudf::scalar>.
        """
        cdef Scalar s = Scalar.__new__(Scalar)
        s.c_value = move(ptr)
        return s
    
    @property
    def ptr(self):
        return _get_ptr_from_scalar_any(self.c_value)

    def __int__(self):
        return int(self.value)

    def __float__(self):
        return float(self.value)

    def __add__(self, other):
        return self._scalar_binop(other, '__add__')

    def __sub__(self, other):
        return self._scalar_binop(other, '__sub__')

    def __mul__(self, other):
        return self._scalar_binop(other, '__mul__')

    def __div__(self, other):
        return self._scalar_binop(other, '__div__')

    def __mod__(self, other):
        return self._scalar_binop(other, '__mod__')

    def __divmod__(self, other):
        return self._scalar_binop(other, '__divmod__')

    def __and__(self, other):
        return self._scalar_binop(other, '__and__')

    def __xor__(self, other):
        return self._scalar_binop(other, '__or__')

    def __gt__(self, other):
        return self._scalar_binop(other, '__gt__').value
    
    def __lt__(self, other):
        return self._scalar_binop(other, '__gt__').value

    def __ge__(self, other):
        return self._scalar_binop(other, '__ge__').value

    def __le__(self, other):
        return self._scalar_binop(other, '__le__').value

    def _binop_result_dtype_or_error(self, other):

        if (self.dtype.kind == 'O' and other.dtype.kind != 'O') or (self.dtype.kind != 'O' and other.dtype.kind == 'O'):
            wrong_dtype = self.dtype if self.dtype.kind != 'O' else other.dtype
            raise TypeError(f"Can only concatenate string (not {wrong_dtype}) to string")


        return cudf.api.types.find_common_type([
            self.dtype, other.dtype
        ])

    def _scalar_binop(self, other, op):
        other = to_cudf_compatible_scalar(other)

        if op in  ['__eq__', '__lt__', '__gt__', '__le__', '__ge__']:
            out_dtype = cudf.BooleanDtype()
        else: 
            out_dtype = self._binop_result_dtype_or_error(other)
        valid = self.is_valid() and (isinstance(other, np.generic) or other.is_valid())
        if not valid:
            return Scalar(None, dtype=out_dtype)
        else:
            result = self._dispatch_scalar_binop(other, op)
            return Scalar(result, dtype=out_dtype)

    def _dispatch_scalar_binop(self, other, op):
        if isinstance(other, Scalar):
            other = other.value
        return getattr(self.value, op)(other)

cdef _set_string_from_np_string(unique_ptr[scalar]& s, value, bool valid=True):
    value = value if valid else ""
    s.reset(new string_scalar(value.encode(), valid))


cdef _set_numeric_from_np_scalar(unique_ptr[scalar]& s,
                                 object value,
                                 object dtype,
                                 bool valid=True):
    value = value if valid else 0
    if dtype == "int8":
        s.reset(new numeric_scalar[int8_t](value, valid))
    elif dtype == "int16":
        s.reset(new numeric_scalar[int16_t](value, valid))
    elif dtype == "int32":
        s.reset(new numeric_scalar[int32_t](value, valid))
    elif dtype == "int64":
        s.reset(new numeric_scalar[int64_t](value, valid))
    elif dtype == "uint8":
        s.reset(new numeric_scalar[uint8_t](value, valid))
    elif dtype == "uint16":
        s.reset(new numeric_scalar[uint16_t](value, valid))
    elif dtype == "uint32":
        s.reset(new numeric_scalar[uint32_t](value, valid))
    elif dtype == "uint64":
        s.reset(new numeric_scalar[uint64_t](value, valid))
    elif dtype == "float32":
        s.reset(new numeric_scalar[float](value, valid))
    elif dtype == "float64":
        s.reset(new numeric_scalar[double](value, valid))
    elif dtype == "bool":
        s.reset(new numeric_scalar[bool](<bool>value, valid))
    else:
        raise ValueError(f"dtype not supported: {dtype}")


cdef _set_datetime64_from_np_scalar(unique_ptr[scalar]& s,
                                    object value,
                                    object dtype,
                                    bool valid=True):

    value = value if valid else 0

    if dtype == "datetime64[s]":
        s.reset(
            new timestamp_scalar[timestamp_s](<int64_t>np.int64(value), valid)
        )
    elif dtype == "datetime64[ms]":
        s.reset(
            new timestamp_scalar[timestamp_ms](<int64_t>np.int64(value), valid)
        )
    elif dtype == "datetime64[us]":
        s.reset(
            new timestamp_scalar[timestamp_us](<int64_t>np.int64(value), valid)
        )
    elif dtype == "datetime64[ns]":
        s.reset(
            new timestamp_scalar[timestamp_ns](<int64_t>np.int64(value), valid)
        )
    else:
        raise ValueError(f"dtype not supported: {dtype}")

cdef _set_timedelta64_from_np_scalar(unique_ptr[scalar]& s,
                                     object value,
                                     object dtype,
                                     bool valid=True):

    value = value if valid else 0

    if dtype == "timedelta64[s]":
        s.reset(
            new duration_scalar[duration_s](<int64_t>np.int64(value), valid)
        )
    elif dtype == "timedelta64[ms]":
        s.reset(
            new duration_scalar[duration_ms](<int64_t>np.int64(value), valid)
        )
    elif dtype == "timedelta64[us]":
        s.reset(
            new duration_scalar[duration_us](<int64_t>np.int64(value), valid)
        )
    elif dtype == "timedelta64[ns]":
        s.reset(
            new duration_scalar[duration_ns](<int64_t>np.int64(value), valid)
        )
    else:
        raise ValueError(f"dtype not supported: {dtype}")

cdef _get_py_string_from_string(unique_ptr[scalar]& s):
    if not s.get()[0].is_valid():
        return None
    return (<string_scalar*>s.get())[0].to_string().decode()


cdef _get_np_scalar_from_numeric(unique_ptr[scalar]& s):
    cdef scalar* s_ptr = s.get()
    if not s_ptr[0].is_valid():
        return None

    cdef libcudf_types.data_type cdtype = s_ptr[0].type()

    if cdtype.id() == libcudf_types.INT8:
        return np.int8((<numeric_scalar[int8_t]*>s_ptr)[0].value())
    elif cdtype.id() == libcudf_types.INT16:
        return np.int16((<numeric_scalar[int16_t]*>s_ptr)[0].value())
    elif cdtype.id() == libcudf_types.INT32:
        return np.int32((<numeric_scalar[int32_t]*>s_ptr)[0].value())
    elif cdtype.id() == libcudf_types.INT64:
        return np.int64((<numeric_scalar[int64_t]*>s_ptr)[0].value())
    elif cdtype.id() == libcudf_types.UINT8:
        return np.uint8((<numeric_scalar[uint8_t]*>s_ptr)[0].value())
    elif cdtype.id() == libcudf_types.UINT16:
        return np.uint16((<numeric_scalar[uint16_t]*>s_ptr)[0].value())
    elif cdtype.id() == libcudf_types.UINT32:
        return np.uint32((<numeric_scalar[uint32_t]*>s_ptr)[0].value())
    elif cdtype.id() == libcudf_types.UINT64:
        return np.uint64((<numeric_scalar[uint64_t]*>s_ptr)[0].value())
    elif cdtype.id() == libcudf_types.FLOAT32:
        return np.float32((<numeric_scalar[float]*>s_ptr)[0].value())
    elif cdtype.id() == libcudf_types.FLOAT64:
        return np.float64((<numeric_scalar[double]*>s_ptr)[0].value())
    elif cdtype.id() == libcudf_types.BOOL8:
        return np.bool_((<numeric_scalar[bool]*>s_ptr)[0].value())
    else:
        raise ValueError("Could not convert cudf::scalar to numpy scalar")


cdef _get_np_scalar_from_timestamp64(unique_ptr[scalar]& s):

    cdef scalar* s_ptr = s.get()

    if not s_ptr[0].is_valid():
        return None

    cdef libcudf_types.data_type cdtype = s_ptr[0].type()

    if cdtype.id() == libcudf_types.TIMESTAMP_SECONDS:
        return np.datetime64(
            (
                <timestamp_scalar[timestamp_ms]*> s_ptr
            )[0].ticks_since_epoch_64(),
            datetime_unit_map[<underlying_type_t_type_id>(cdtype.id())]
        )
    elif cdtype.id() == libcudf_types.TIMESTAMP_MILLISECONDS:
        return np.datetime64(
            (
                <timestamp_scalar[timestamp_ms]*> s_ptr
            )[0].ticks_since_epoch_64(),
            datetime_unit_map[<underlying_type_t_type_id>(cdtype.id())]
        )
    elif cdtype.id() == libcudf_types.TIMESTAMP_MICROSECONDS:
        return np.datetime64(
            (
                <timestamp_scalar[timestamp_ms]*> s_ptr
            )[0].ticks_since_epoch_64(),
            datetime_unit_map[<underlying_type_t_type_id>(cdtype.id())]
        )
    elif cdtype.id() == libcudf_types.TIMESTAMP_NANOSECONDS:
        return np.datetime64(
            (
                <timestamp_scalar[timestamp_ms]*> s_ptr
            )[0].ticks_since_epoch_64(),
            datetime_unit_map[<underlying_type_t_type_id>(cdtype.id())]
        )
    else:
        raise ValueError("Could not convert cudf::scalar to numpy scalar")


cdef _get_np_scalar_from_timedelta64(unique_ptr[scalar]& s):

    cdef scalar* s_ptr = s.get()

    if not s_ptr[0].is_valid():
        return None

    cdef libcudf_types.data_type cdtype = s_ptr[0].type()

    if cdtype.id() == libcudf_types.DURATION_SECONDS:
        return np.timedelta64(
            (
                <duration_scalar[duration_s]*> s_ptr
            )[0].ticks(),
            duration_unit_map[<underlying_type_t_type_id>(cdtype.id())]
        )
    elif cdtype.id() == libcudf_types.DURATION_MILLISECONDS:
        return np.timedelta64(
            (
                <duration_scalar[duration_ms]*> s_ptr
            )[0].ticks(),
            duration_unit_map[<underlying_type_t_type_id>(cdtype.id())]
        )
    elif cdtype.id() == libcudf_types.DURATION_MICROSECONDS:
        return np.timedelta64(
            (
                <duration_scalar[duration_us]*> s_ptr
            )[0].ticks(),
            duration_unit_map[<underlying_type_t_type_id>(cdtype.id())]
        )
    elif cdtype.id() == libcudf_types.DURATION_NANOSECONDS:
        return np.timedelta64(
            (
                <duration_scalar[duration_ns]*> s_ptr
            )[0].ticks(),
            duration_unit_map[<underlying_type_t_type_id>(cdtype.id())]
        )
    else:
        raise ValueError("Could not convert cudf::scalar to numpy scalar")


def as_scalar(val, dtype=None):
    dtype = cudf.dtype(dtype)
    if isinstance(val, Scalar):
        if (dtype is None or dtype == val.dtype):
            return val
        else:
            return Scalar(val.value, dtype)
    else:
        return Scalar(value=val, dtype=dtype)

cdef _get_ptr_from_scalar_any(unique_ptr[scalar]& s):
    cdef scalar* s_ptr = s.get()
    if not s_ptr[0].is_valid():
        return None

    cdef libcudf_types.data_type cdtype = s_ptr[0].type()

    if cdtype.id() == libcudf_types.INT8:
        return int(
            <uintptr_t>(<numeric_scalar[int8_t]*>s_ptr)[0].data()
        )
    elif cdtype.id() == libcudf_types.INT16:
        return int(
            <uintptr_t>(<numeric_scalar[int16_t]*>s_ptr)[0].data()
        )
    elif cdtype.id() == libcudf_types.INT32:
        return int(
            <uintptr_t>(<numeric_scalar[int32_t]*>s_ptr)[0].data()
        )
    elif cdtype.id() == libcudf_types.INT64:
        return int(
            <uintptr_t>(<numeric_scalar[int64_t]*>s_ptr)[0].data()
        )
    elif cdtype.id() == libcudf_types.UINT8:
        return int(
            <uintptr_t>(<numeric_scalar[uint8_t]*>s_ptr)[0].data()
        )
    elif cdtype.id() == libcudf_types.UINT16:
        return int(
            <uintptr_t>(<numeric_scalar[uint16_t]*>s_ptr)[0].data()
        )
    elif cdtype.id() == libcudf_types.UINT32:
        return int(
            <uintptr_t>(<numeric_scalar[uint32_t]*>s_ptr)[0].data()
        )
    elif cdtype.id() == libcudf_types.UINT64:
        return int(
            <uintptr_t>(<numeric_scalar[uint64_t]*>s_ptr)[0].data()
        )
    elif cdtype.id() == libcudf_types.FLOAT32:
        return int(
            <uintptr_t>(<numeric_scalar[float]*>s_ptr)[0].data()
        )
    elif cdtype.id() == libcudf_types.FLOAT64:
        return int(
            <uintptr_t>(<numeric_scalar[double]*>s_ptr)[0].data()
        )
    elif cdtype.id() == libcudf_types.BOOL8:
        return int(
            <uintptr_t>(<numeric_scalar[bool]*>s_ptr)[0].data()
        )
    elif cdtype.id() == libcudf_types.TIMESTAMP_NANOSECONDS:
        return int(
            <uintptr_t>(<timestamp_scalar[timestamp_ns]*>s_ptr)[0].data()
        ) 
    elif cdtype.id() == libcudf_types.TIMESTAMP_MICROSECONDS:
        return int(
            <uintptr_t>(<timestamp_scalar[timestamp_us]*>s_ptr)[0].data()
        )
    elif cdtype.id() == libcudf_types.TIMESTAMP_MILLISECONDS:
        return int(
            <uintptr_t>(<timestamp_scalar[timestamp_ms]*>s_ptr)[0].data()
        )
    elif cdtype.id() == libcudf_types.TIMESTAMP_SECONDS:
        return int(
            <uintptr_t>(<timestamp_scalar[timestamp_s]*>s_ptr)[0].data()
        )
    elif cdtype.id() == libcudf_types.DURATION_NANOSECONDS:
        return int(
            <uintptr_t>(<duration_scalar[duration_ns]*>s_ptr)[0].data()
        ) 
    elif cdtype.id() == libcudf_types.DURATION_MICROSECONDS:
        return int(
            <uintptr_t>(<duration_scalar[duration_us]*>s_ptr)[0].data()
        )
    elif cdtype.id() == libcudf_types.DURATION_MILLISECONDS:
        return int(
            <uintptr_t>(<duration_scalar[duration_ms]*>s_ptr)[0].data()
        )
    elif cdtype.id() == libcudf_types.DURATION_SECONDS:
        return int(
            <uintptr_t>(<duration_scalar[duration_s]*>s_ptr)[0].data()
        )  
    else:
        raise ValueError('Could not get pointer from cudf::scalar')
