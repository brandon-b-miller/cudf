import datetime as dt
import inspect
import numbers
from collections import namedtuple
from collections.abc import Sequence

import cupy as cp
import numpy as np
import pandas as pd
import pyarrow as pa
from pandas.core.dtypes.common import infer_dtype_from_object

import cudf
from cudf._lib.scalar import Scalar
from cudf.api.types import is_categorical_dtype

_NA_REP = "<NA>"
_np_pa_dtypes = {
    np.float64: pa.float64(),
    np.float32: pa.float32(),
    np.int64: pa.int64(),
    np.longlong: pa.int64(),
    np.int32: pa.int32(),
    np.int16: pa.int16(),
    np.int8: pa.int8(),
    np.bool_: pa.int8(),
    np.uint64: pa.uint64(),
    np.uint32: pa.uint32(),
    np.uint16: pa.uint16(),
    np.uint8: pa.uint8(),
    np.datetime64: pa.date64(),
    np.object_: pa.string(),
    np.str_: pa.string(),
}

cudf_dtypes_to_pandas_dtypes = {
    np.dtype("uint8"): pd.UInt8Dtype(),
    np.dtype("uint16"): pd.UInt16Dtype(),
    np.dtype("uint32"): pd.UInt32Dtype(),
    np.dtype("uint64"): pd.UInt64Dtype(),
    np.dtype("int8"): pd.Int8Dtype(),
    np.dtype("int16"): pd.Int16Dtype(),
    np.dtype("int32"): pd.Int32Dtype(),
    np.dtype("int64"): pd.Int64Dtype(),
    np.dtype("bool_"): pd.BooleanDtype(),
    np.dtype("object"): pd.StringDtype(),
}

SIGNED_INTEGER_TYPES = {"int8", "int16", "int32", "int64"}
UNSIGNED_TYPES = {"uint8", "uint16", "uint32", "uint64"}
INTEGER_TYPES = SIGNED_INTEGER_TYPES | UNSIGNED_TYPES
FLOAT_TYPES = {"float32", "float64"}
SIGNED_TYPES = SIGNED_INTEGER_TYPES | FLOAT_TYPES
NUMERIC_TYPES = SIGNED_TYPES | UNSIGNED_TYPES
DATETIME_TYPES = {
    "datetime64[s]",
    "datetime64[ms]",
    "datetime64[us]",
    "datetime64[ns]",
}
TIMEDELTA_TYPES = {
    "timedelta64[s]",
    "timedelta64[ms]",
    "timedelta64[us]",
    "timedelta64[ns]",
}
OTHER_TYPES = {"bool", "category", "str"}
ALL_TYPES = NUMERIC_TYPES | DATETIME_TYPES | TIMEDELTA_TYPES | OTHER_TYPES
NEW_NUMERIC_TYPES = {
    "Int8",
    "Int16",
    "Int32",
    "Int64",
    "UInt8",
    "UInt16",
    "UInt32",
    "UInt64",
    "Float32",
    "Float64",
}


def np_to_pa_dtype(dtype):
    """Util to convert numpy dtype to PyArrow dtype.
    """
    if isinstance(dtype, cudf.Generic):
        return dtype.pa_type
    # special case when dtype is np.datetime64
    if dtype.kind == "M":
        time_unit, _ = np.datetime_data(dtype)
        if time_unit in ("s", "ms", "us", "ns"):
            # return a pa.Timestamp of the appropriate unit
            return pa.timestamp(time_unit)
        # default is int64_t UNIX ms
        return pa.date64()
    elif dtype.kind == "m":
        time_unit, _ = np.datetime_data(dtype)
        if time_unit in ("s", "ms", "us", "ns"):
            # return a pa.Duration of the appropriate unit
            return pa.duration(time_unit)
        # default fallback unit is ns
        return pa.duration("ns")
    return _np_pa_dtypes[np.dtype(dtype).type]


def get_numeric_type_info(dtype):
    _TypeMinMax = namedtuple("_TypeMinMax", "min,max")
    if dtype.kind in {"i", "u"}:
        info = np.iinfo(dtype)
        return _TypeMinMax(info.min, info.max)
    elif dtype.kind == "f":
        return _TypeMinMax(dtype.type("-inf"), dtype.type("+inf"))
    else:
        raise TypeError(dtype)


def numeric_normalize_types(*args):
    """Cast all args to a common type using numpy promotion logic
    """
    dtype = np.result_type(*[a.dtype.numpy_dtype for a in args])
    return [a.astype(dtype) for a in args]

def is_datetime_dtype(obj):
    if obj is None:
        return False
    if not hasattr(obj, "str"):
        return False
    return "M8" in obj.str

def is_scalar(val):
    return (
        val is None
        or isinstance(val, Scalar)
        or isinstance(val, str)
        or isinstance(val, numbers.Number)
        or np.isscalar(val)
        or (isinstance(val, (np.ndarray, cp.ndarray)) and val.ndim == 0)
        or isinstance(val, pd.Timestamp)
        or (isinstance(val, pd.Categorical) and len(val) == 1)
        or (isinstance(val, pd.Timedelta))
        or (isinstance(val, pd.Timestamp))
        or (isinstance(val, dt.datetime))
        or (isinstance(val, dt.timedelta))
    )


def to_cudf_compatible_scalar(val, dtype=None):
    """
    Converts the value `val` to a numpy/Pandas scalar,
    optionally casting to `dtype`.

    If `val` is None, returns None.
    """
    if val is None or isinstance(val, cudf._lib.scalar.Scalar):
        return val

    if not is_scalar(val):
        raise ValueError(
            f"Cannot convert value of type {type(val).__name__} "
            " to cudf scalar"
        )

    if isinstance(val, (np.ndarray, cp.ndarray)) and val.ndim == 0:
        val = val.item()

    if ((dtype is None) and isinstance(val, str)) or cudf.api.types.is_string_dtype(dtype):
        dtype = "str"

    if isinstance(val, dt.datetime):
        val = np.datetime64(val)
    elif isinstance(val, dt.timedelta):
        val = np.timedelta64(val)
    elif isinstance(val, pd.Timestamp):
        val = val.to_datetime64()
    elif isinstance(val, pd.Timedelta):
        val = val.to_timedelta64()

    val = pd.api.types.pandas_dtype(type(val)).type(val)

    if dtype is not None:
        if isinstance(dtype, cudf.Generic):
            dtype = dtype.numpy_dtype
        val = val.astype(dtype)

    if val.dtype.type is np.datetime64:
        time_unit, _ = np.datetime_data(val.dtype)
        if time_unit in ("D", "W", "M", "Y"):
            val = val.astype("datetime64[s]")
    elif val.dtype.type is np.timedelta64:
        time_unit, _ = np.datetime_data(val.dtype)
        if time_unit in ("D", "W", "M", "Y"):
            val = val.astype("timedelta64[ns]")

    return val


def is_list_like(obj):
    """
    This function checks if the given `obj`
    is a list-like (list, tuple, Series...)
    type or not.

    Parameters
    ----------
    obj : object of any type which needs to be validated.

    Returns
    -------
    Boolean: True or False depending on whether the
    input `obj` is like-like or not.
    """

    return isinstance(obj, (Sequence, np.ndarray)) and not isinstance(
        obj, (str, bytes)
    )


def is_column_like(obj):
    """
    This function checks if the given `obj`
    is a column-like (Series, Index...)
    type or not.

    Parameters
    ----------
    obj : object of any type which needs to be validated.

    Returns
    -------
    Boolean: True or False depending on whether the
    input `obj` is column-like or not.
    """
    return (
        isinstance(
            obj,
            (
                cudf.core.column.ColumnBase,
                cudf.Series,
                cudf.Index,
                pd.Series,
                pd.Index,
            ),
        )
        or (
            hasattr(obj, "__cuda_array_interface__")
            and len(obj.__cuda_array_interface__["shape"]) == 1
        )
        or (
            hasattr(obj, "__array_interface__")
            and len(obj.__array_interface__["shape"]) == 1
        )
    )


def can_convert_to_column(obj):
    """
    This function checks if the given `obj`
    can be used to create a column or not.

    Parameters
    ----------
    obj : object of any type which needs to be validated.

    Returns
    -------
    Boolean: True or False depending on whether the
    input `obj` is column-compatible or not.
    """
    return is_column_like(obj) or is_list_like(obj)


def min_scalar_type(a, min_size=8):
    return min_signed_type(a, min_size=min_size)


def min_signed_type(x, min_size=8):
    """
    Return the smallest *signed* integer dtype
    that can represent the integer ``x``
    """
    for int_dtype in np.sctypes["int"]:
        if (np.dtype(int_dtype).itemsize * 8) >= min_size:
            if np.iinfo(int_dtype).min <= x <= np.iinfo(int_dtype).max:
                return int_dtype
    # resort to using `int64` and let numpy raise appropriate exception:
    return np.int64(x).dtype


def min_unsigned_type(x, min_size=8):
    """
    Return the smallest *unsigned* integer dtype
    that can represent the integer ``x``
    """
    for int_dtype in np.sctypes["uint"]:
        if (np.dtype(int_dtype).itemsize * 8) >= min_size:
            if 0 <= x <= np.iinfo(int_dtype).max:
                return int_dtype
    # resort to using `uint64` and let numpy raise appropriate exception:
    return np.uint64(x).dtype


def min_column_type(x, expected_type):
    """
    Return the smallest dtype which can represent all
    elements of the `NumericalColumn` `x`
    If the column is not a subtype of `np.signedinteger` or `np.floating`
    returns the same dtype as the dtype of `x` without modification
    """

    expected_type = cudf.dtype(expected_type)
    if not isinstance(x, cudf.core.column.NumericalColumn):
        raise TypeError("Argument x must be of type column.NumericalColumn")
    if x.valid_count == 0:
        return x.dtype

    if isinstance(x.dtype, cudf.Floating):
        max_bound_dtype = np.min_scalar_type(x.max().value)
        min_bound_dtype = np.min_scalar_type(x.min().value)
        result_type = np.promote_types(max_bound_dtype, min_bound_dtype)
        if result_type == np.dtype("float16"):
            # cuDF does not support float16 dtype
            result_type = np.dtype("float32")
        return cudf.dtype(result_type)

    if isinstance(expected_type, cudf.Integer):
        max_bound_dtype = np.min_scalar_type(x.max().value)
        min_bound_dtype = np.min_scalar_type(x.min().value)
        result = np.promote_types(max_bound_dtype, min_bound_dtype)
        return cudf.dtype(result)

    return x.dtype


def check_cast_unsupported_dtype(dtype):

    if isinstance(dtype, cudf.Generic):
        return dtype.numpy_dtype

    if is_categorical_dtype(dtype):
        return dtype

    if isinstance(dtype, pd.core.arrays.numpy_.PandasDtype):
        dtype = dtype.numpy_dtype
    else:
        dtype = np.dtype(dtype)

    if cudf.dtype(dtype) in cudf._lib.types.np_to_cudf_types:
        return dtype

    if dtype == np.dtype("float16"):
        return np.dtype("float32")

    raise NotImplementedError(
        "Cannot cast {0} dtype, as it is not supported by CuDF.".format(dtype)
    )


def is_mixed_with_object_dtype(lhs, rhs):
    return (lhs.dtype == "object" and rhs.dtype != "object") or (
        rhs.dtype == "object" and lhs.dtype != "object"
    )


def get_time_unit(obj):
    if isinstance(
        obj,
        (
            cudf.core.column.datetime.DatetimeColumn,
            cudf.core.column.timedelta.TimeDeltaColumn,
        ),
    ):
        return obj.time_unit
    elif isinstance(obj, cudf.Generic):
        return obj._time_unit
    elif isinstance(obj.dtype, cudf.Generic):
        return obj.dtype._time_unit

    time_unit, _ = np.datetime_data(obj.dtype)

    return time_unit
