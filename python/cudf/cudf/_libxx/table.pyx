# Copyright (c) 2020, NVIDIA CORPORATION.

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

import itertools

import numpy as np

from cython.operator cimport dereference
from libc.stdint cimport uintptr_t

from cudf._libxx.lib cimport *
from cudf._libxx.column cimport Column
from cudf.utils.utils import OrderedColumnDict
from collections import OrderedDict

cdef class Table:
    def __init__(self, data=None, index=None):
        """
        Table: A collection of Column objects with an optional index.

        Parameters
        ----------
        data : OrderedColumnDict
            An OrderedColumnDict mapping column names to Columns
        index : Table
            A Table representing the (optional) index columns.
        """
        if data is None:
            data = {}
        self._data = OrderedColumnDict(data)
        self._index = index

    @property
    def _num_columns(self):
        return len(self._data)

    @property
    def _num_rows(self):
        if self._index is not None:
            if len(self._index._data) == 0:
                return 0
            return self._index._num_rows
        return len(next(iter(self._data.values())))

    @property
    def _column_names(self):
        return tuple(self._data.keys())

    @property
    def _index_names(self):
        return None if self._index is None else tuple(
            self._index._data.keys())

    @property
    def _columns(self):
        """
        Return a list of Column objects backing this dataframe
        """
        return tuple(self._data.values())

    @staticmethod
    cdef Table from_unique_ptr(
        unique_ptr[table] c_tbl,
        object column_names,
        object index_names=[],
        object index_pos=None
    ):
        """
        Construct a Table from a unique_ptr to a cudf::table.
        Parameters
        ----------
        c_tbl : unique_ptr[cudf::table]
        index_names : iterable
        column_names : iterable
        index_pos : iterable, column indices corresponding to column_names
        """

        cdef vector[unique_ptr[column]] columns
        columns = c_tbl.get()[0].release()

        cdef vector[unique_ptr[column]].iterator it = columns.begin()

        num_cols_input = 0
        num_cols_input += len(column_names) if column_names is not None else 0
        num_cols_input += len(index_names) if index_names is not None else 0

        cols_py = OrderedDict()
        for i in range(num_cols_input):
            cols_py[i] =(Column.from_unique_ptr(
                move(dereference(it))
            ))
            it += 1

        index = None
        if index_names not in [None, []]:
            index_columns = []
            if index_pos is None:
                index_pos = range(len(index_names))
            for idx in index_pos:
                index_columns.append(cols_py.pop(idx))
            index = Table(OrderedColumnDict(zip(index_names, index_columns)))

        data = OrderedColumnDict(zip(column_names, cols_py.values()))

        return Table(data=data, index=index)

    @staticmethod
    cdef Table from_table_view(
        table_view tv,
        object owner,
        object column_names,
        object index_names=None
    ):
        """
        Given a ``cudf::table_view``, constructs a ``cudf.Table`` from it,
        along with referencing an ``owner`` Python object that owns the memory
        lifetime. If ``owner`` is a ``cudf.Table``, we reach inside of it and
        reach inside of each ``cudf.Column`` to make the owner of each newly
        created ``Buffer`` underneath the ``cudf.Column`` objects of the
        created ``cudf.Table`` the respective ``Buffer`` from the relevant
        ``cudf.Column`` of the ``owner`` ``cudf.Table``.
        """
        cdef size_type column_idx = 0
        table_owner = isinstance(owner, Table)

        # First construct the index, if any
        index = None
        if index_names is not None:
            index_columns = []
            for _ in index_names:
                column_owner = owner
                if table_owner:
                    column_owner = table_owner._columns[column_idx]
                index_columns.append(
                    Column.from_column_view(
                        tv.column(column_idx),
                        column_owner
                    )
                )
                column_idx += 1
            index = Table(OrderedColumnDict(zip(index_names, index_columns)))

        # Construct the data OrderedColumnDict
        data_columns = []
        for _ in column_names:
            column_owner = owner
            if table_owner:
                column_owner = table_owner._columns[column_idx]
            data_columns.append(
                Column.from_column_view(tv.column(column_idx), column_owner)
            )
            column_idx += 1
        data = OrderedColumnDict(zip(column_names, data_columns))

        return Table(data=data, index=index)

    cdef table_view view(self) except *:
        """
        Return a cudf::table_view of all columns (including index columns)
        of this Table.
        """
        if self._index is None:
            return _make_table_view(
                self._data.values()
            )
        return _make_table_view(
            itertools.chain(
                self._index._data.values(),
                self._data.values(),
            )
        )

    cdef mutable_table_view mutable_view(self) except *:
        """
        Return a cudf::mutable_table_view of all columns
        (including index columns) of this Table.
        """
        if self._index is None:
            return _make_mutable_table_view(
                self._data.values()
            )
        return _make_mutable_table_view(
            itertools.chain(
                self._index._data.values(),
                self._data.values(),
            )
        )

    cdef table_view data_view(self) except *:
        """
        Return a cudf::table_view of just the data columns
        of this Table.
        """
        return _make_table_view(
            self._data.values()
        )

    cdef mutable_table_view mutable_data_view(self) except *:
        """
        Return a cudf::mutable_table_view of just the data columns
        of this Table.
        """
        return _make_mutable_table_view(
            self._data.values()
        )

    cdef table_view index_view(self) except *:
        """
        Return a cudf::table_view of just the index columns
        of this Table.
        """
        if self._index is None:
            raise ValueError("Cannot get index_view of a Table "
                             "that has no index")
        return _make_table_view(
            self._index.values()
        )

    cdef mutable_table_view mutable_index_view(self) except *:
        """
        Return a cudf::mutable_table_view of just the index columns
        of this Table.
        """
        if self._index is None:
            raise ValueError("Cannot get mutable_index_view of a Table "
                             "that has no index")
        return _make_mutable_table_view(
            self._index._data.values()
        )


cdef table_view _make_table_view(columns) except*:
    """
    Helper function to create a cudf::table_view from
    a list of Columns
    """
    cdef vector[column_view] column_views

    cdef Column col
    for col in columns:
        column_views.push_back(col.view())

    return table_view(column_views)

cdef mutable_table_view _make_mutable_table_view(columns) except*:
    """
    Helper function to create a cudf::mutable_table_view from
    a list of Columns
    """
    cdef vector[mutable_column_view] mutable_column_views

    cdef Column col
    for col in columns:
        mutable_column_views.push_back(col.mutable_view())

    return mutable_table_view(mutable_column_views)
