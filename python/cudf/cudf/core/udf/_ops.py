import operator

from cudf.core.dtypes import CategoricalDtype
from cudf.utils.dtypes import BOOL_TYPES, NUMERIC_TYPES

arith_ops = [
    operator.add,
    operator.sub,
    operator.mul,
    operator.truediv,
    operator.floordiv,
    operator.mod,
    operator.pow,
]

comparison_ops = [
    operator.eq,
    operator.ne,
    operator.lt,
    operator.le,
    operator.gt,
    operator.ge,
]


# currently only numeric types are supported.
SUPPORTED_TYPES = NUMERIC_TYPES | BOOL_TYPES


def _is_supported_type(dtype):
    # category dtype isn't hashable
    if isinstance(dtype, CategoricalDtype):
        return False
    return str(dtype) in SUPPORTED_TYPES
