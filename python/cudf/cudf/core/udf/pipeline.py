import cupy
import numpy as np
from numba import cuda
from numba.np import numpy_support
from numba.types import Tuple, boolean, void
from nvtx import annotate

import cudf
from cudf._lib.transform import bools_to_mask
from cudf.core.udf.classes import Masked
from cudf.core.udf.typing import MaskedType
from cudf.utils import cudautils

libcudf_bitmask_type = numpy_support.from_dtype(np.dtype("int32"))
mask_bitsize = np.dtype("int32").itemsize * 8


@annotate("NUMBA JIT", color="green", domain="cudf_python")
def compile_masked_udf(func, dtypes):
    """
    Generate an inlineable PTX function that will be injected into
    a variadic kernel inside libcudf

    assume all input types are `MaskedType(input_col.dtype)` and then
    compile the requestied PTX function as a function over those types
    """
    to_compiler_sig = tuple(
        MaskedType(arg)
        for arg in (numpy_support.from_dtype(np_type) for np_type in dtypes)
    )
    # Get the inlineable PTX function
    ptx, output_type = cudautils.compile_udf(func, to_compiler_sig)

    if not isinstance(output_type, MaskedType):
        numba_output_type = numpy_support.from_dtype(np.dtype(output_type))
    else:
        numba_output_type = output_type

    return numba_output_type, ptx


def nulludf(func):
    """
    Mimic pandas API:

    def f(x, y):
        return x + y
    df.apply(lambda row: f(row['x'], row['y']))

    in this scheme, `row` is actually the whole dataframe
    `DataFrame` sends `self` in as `row` and subsequently
    we end up calling `f` on the resulting columns since
    the dataframe is dict-like
    """

    def wrapper(*args):
        from cudf import DataFrame

        # This probably creates copies but is fine for now
        to_udf_table = DataFrame(
            {idx: arg for idx, arg in zip(range(len(args)), args)}
        )
        # Frame._apply
        return to_udf_table._apply(func)

    return wrapper


def masked_arrty_from_np_type(dtype):
    """
    Return a type representing a tuple of arrays,
    the first element an array of the numba type
    corresponding to `dtype`, and the second an
    array of bools represe
    """
    nb_scalar_ty = numpy_support.from_dtype(dtype)
    return Tuple((nb_scalar_ty[::1], libcudf_bitmask_type[::1]))


@cuda.jit(device=True)
def mask_get(mask, pos):
    return (mask[pos // mask_bitsize] >> (pos % mask_bitsize)) & 1


def _define_function(df, scalar_return=False):

    start = "def _kernel(retval, "

    sig = ", ".join(["input_col_" + str(i) for i in range(len(df.columns))])
    start += sig + "):\n"

    start += "\ti = cuda.grid(1)\n"
    start += "\tret_data_arr, ret_mask_arr = retval\n"

    fargs = []
    for i, col in enumerate(df._data.values()):
        ii = str(i)
        start += "\td_" + ii + "," + "m_" + ii + "=input_col_" + ii + "\n"
        arg = "masked_" + ii
        if col.mask is not None:
            start += (
                "\t"
                + arg
                + "="
                + "Masked("
                + "d_"
                + ii
                + "[i]"
                + ","
                + "mask_get(m_"
                + ii
                + ","
                + "i)"
                + ")\n"
            )
        else:
            start += (
                "\t"
                + arg
                + "="
                + "Masked("
                + "d_"
                + ii
                + "[i]"
                + ","
                + "True"
                + ")\n"
            )
        fargs.append(arg)

    fargs = "(" + ",".join(fargs) + ")\n"
    start += "\tret = f_" + fargs + "\n"
    if scalar_return:
        start += "\tret_data_arr[i] = ret\n"
        start += "\tret_mask_arr[i] = True\n"
    else:
        start += "\tret_data_arr[i] = ret.value\n"
        start += "\tret_mask_arr[i] = ret.valid\n"

    return start


def udf_pipeline(df, f):
    numba_return_type, _ = compile_masked_udf(f, df.dtypes)
    _is_scalar_return = not isinstance(numba_return_type, MaskedType)
    scalar_return_type = (
        numba_return_type
        if _is_scalar_return
        else numba_return_type.value_type
    )
    return_type = Tuple((scalar_return_type[::1], boolean[::1]))
    sig = void(
        return_type, *[masked_arrty_from_np_type(dtype) for dtype in df.dtypes]
    )

    f_ = cuda.jit(device=True)(f)
    # Set f_launch into the global namespace
    lcl = {}
    exec(
        _define_function(df, scalar_return=_is_scalar_return),
        {"f_": f_, "cuda": cuda, "Masked": Masked, "mask_get": mask_get},
        lcl,
    )
    _kernel = lcl["_kernel"]
    # compile
    kernel = cuda.jit(sig)(_kernel)
    ans_col = cupy.empty(
        len(df), dtype=numpy_support.as_dtype(scalar_return_type)
    )
    ans_mask = cudf.core.column.column_empty(len(df), dtype="bool")
    launch_args = [(ans_col, ans_mask)]
    for col in df.columns:
        data = df[col]._column.data
        mask = df[col]._column.mask
        if mask is None:
            mask = cudf.core.buffer.Buffer()
        launch_args.append((data, mask))

    kernel[1, len(df)](*launch_args)
    result = cudf.Series(ans_col).set_mask(bools_to_mask(ans_mask))
    return result
