# Copyright (c) 2024, NVIDIA CORPORATION.

from cython.operator cimport dereference
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move

from cudf._lib.cpp.column.column cimport column
from cudf._lib.cpp.filling cimport (
    fill as cpp_fill,
    fill_in_place as cpp_fill_in_place,
    repeat as cpp_repeat,
    sequence as cpp_sequence,
)
from cudf._lib.cpp.table.table cimport table
from cudf._lib.cpp.types cimport size_type

from .column cimport Column
from .scalar cimport Scalar
from .table cimport Table


cpdef Column fill(
    object destination,
    size_type begin,
    size_type end,
    object value,
):

    """Fill destination column from begin to end with value.
    ``destination ``must be a
    :py:class:`~cudf._lib.pylibcudf.column.Column`. ``value`` must be a
    :py:class:`~cudf._lib.pylibcudf.scalar.Scalar`.
    For details, see :cpp:func:`fill`.
    Parameters
    ----------
    destination : Column
        The column to be filled
    begin : size_type
        The index to begin filling from.
    end : size_type
        The index at which to stop filling.
    Returns
    -------
    pylibcudf.Column
        The result of the filling operation
    """

    cdef unique_ptr[column] result
    with nogil:
        result = move(
            cpp_fill(
                (<Column> destination).view(),
                begin,
                end,
                dereference((<Scalar> value).c_obj)
            )
        )
    return Column.from_libcudf(move(result))

cpdef void fill_in_place(
    object destination,
    size_type begin,
    size_type end,
    object value,
):

    """Fill destination column in place from begin to end with value.
    ``destination ``must be a
    :py:class:`~cudf._lib.pylibcudf.column.Column`. ``value`` must be a
    :py:class:`~cudf._lib.pylibcudf.scalar.Scalar`.
    For details, see :cpp:func:`fill_in_place`.
    Parameters
    ----------
    destination : Column
        The column to be filled
    begin : size_type
        The index to begin filling from.
    end : size_type
        The index at which to stop filling.
    """

    with nogil:
        cpp_fill_in_place(
            (<Column> destination).mutable_view(),
            begin,
            end,
            dereference((<Scalar> value).c_obj)
        )

cpdef Column sequence(size_type size, object init, object step):
    """Create a sequence column of size `size` with initial value `init` and
    step `step`.
    Parameters
    ----------
    size : int
        The size of the sequence
    init : Scalar
        The initial value of the sequence
    step : Scalar
        The step of the sequence
    Returns
    -------
    pylibcudf.Column
        The result of the sequence operation
    """

    cdef unique_ptr[column] result
    cdef size_type c_size = size
    with nogil:
        result = move(
            cpp_sequence(
                c_size,
                dereference((<Scalar> init).c_obj),
                dereference((<Scalar> step).c_obj),
            )
        )
    return Column.from_libcudf(move(result))


cpdef Table repeat(
    Table input_table,
    ColumnOrSize count
):
    """Repeat rows of a Table either ``count`` times
    or as specified by an integral column. If ``count``
    is a column, the number of repetitions of each row
    is defined by the value at the corresponding index
    of ``count``.

    For details, see :cpp:func:`repeat`.

    Parameters
    ----------
    input_table : Table
        The table to be repeated
    count : Union[Column, size_type]
        Integer value to repeat each row by or
        Non-nullable column of an integral type
    Returns
    -------
    pylibcudf.Table
        The result of the repeat operation
    """

    cdef unique_ptr[table] result

    if count is Column:
        with nogil:
            result = move(
                cpp_repeat(
                    input_table.view(),
                    count.view()
                )
            )
    if count is size_type:
        with nogil:
            result = move(
                cpp_repeat(
                    input_table.view(),
                    count
                )
            )
    return Table.from_libcudf(move(result))
