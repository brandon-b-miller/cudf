# Copyright (c) 2018, NVIDIA CORPORATION.

from itertools import product

import numpy as np
import pandas as pd
import pytest

from cudf.core import DataFrame, Series
from cudf.core.column import NumericalColumn
from cudf.tests.utils import assert_eq

sort_nelem_args = [2, 257]
sort_dtype_args = [np.int32, np.int64, np.float32, np.float64]
sort_slice_args = [slice(1, None), slice(None, -1), slice(1, -1)]


@pytest.mark.parametrize(
    "nelem,dtype", list(product(sort_nelem_args, sort_dtype_args))
)
def test_dataframe_sort_values(nelem, dtype):
    np.random.seed(0)
    df = DataFrame()
    df["a"] = aa = (100 * np.random.random(nelem)).astype(dtype)
    df["b"] = bb = (100 * np.random.random(nelem)).astype(dtype)
    sorted_df = df.sort_values(by="a")
    # Check
    sorted_index = np.argsort(aa, kind="mergesort")
    np.testing.assert_array_equal(sorted_df.index.values, sorted_index)
    np.testing.assert_array_equal(sorted_df["a"], aa[sorted_index])
    np.testing.assert_array_equal(sorted_df["b"], bb[sorted_index])


@pytest.mark.parametrize(
    "nelem,sliceobj", list(product([10, 100], sort_slice_args))
)
def test_dataframe_sort_values_sliced(nelem, sliceobj):
    np.random.seed(0)
    df = pd.DataFrame()
    df["a"] = np.random.random(nelem)

    expect = df[sliceobj]["a"].sort_values()
    gdf = DataFrame.from_pandas(df)
    got = gdf[sliceobj]["a"].sort_values()
    assert (got.to_pandas() == expect).all()


@pytest.mark.parametrize(
    "nelem,dtype,asc",
    list(product(sort_nelem_args, sort_dtype_args, [True, False])),
)
def test_series_argsort(nelem, dtype, asc):
    np.random.seed(0)
    sr = Series((100 * np.random.random(nelem)).astype(dtype))
    res = sr.argsort(ascending=asc)

    if asc:
        expected = np.argsort(sr.to_array(), kind="mergesort")
    else:
        expected = np.argsort(-sr.to_array(), kind="mergesort")
    np.testing.assert_array_equal(expected, res.to_array())


@pytest.mark.parametrize(
    "nelem,asc", list(product(sort_nelem_args, [True, False]))
)
def test_series_sort_index(nelem, asc):
    np.random.seed(0)
    sr = Series((100 * np.random.random(nelem)))
    orig = sr.to_array()
    got = sr.sort_values().sort_index(ascending=asc).to_array()
    if not asc:
        # Reverse the array for descending sort
        got = got[::-1]
    np.testing.assert_array_equal(orig, got)


def test_series_nlargest():
    """Indirectly tests Series.sort_values()
    """
    sr = Series([0, 1, 1, 2, 2, 2, 3, 3])
    got = sr.nlargest(3)  # default keep='first'
    assert list(got) == [3, 3, 2]
    assert list(got.index.values) == [6, 7, 3]

    got = sr.nlargest(3, keep="last")
    assert list(got) == [3, 3, 2]
    assert list(got.index.values) == [7, 6, 5]

    with pytest.raises(ValueError) as raises:
        sr.nlargest(3, keep="what")
    assert raises.match('keep must be either "first", "last"')


def test_series_nsmallest():
    """Indirectly tests Series.sort_values()
    """
    sr = Series([0, 1, 1, 2, 2, 2, 3, 3])
    got = sr.nsmallest(3)  # default keep='first'
    assert list(got) == [0, 1, 1]
    assert list(got.index.values) == [0, 1, 2]

    got = sr.nsmallest(3, keep="last")
    assert list(got) == [0, 1, 1]
    assert list(got.index.values) == [0, 2, 1]

    with pytest.raises(ValueError) as raises:
        sr.nsmallest(3, keep="what")
    assert raises.match('keep must be either "first", "last"')


@pytest.mark.parametrize("nelem,n", [(1, 1), (100, 100), (10, 5), (100, 10)])
def test_dataframe_nlargest(nelem, n):
    np.random.seed(0)
    df = DataFrame()
    df["a"] = aa = np.random.random(nelem)
    df["b"] = bb = np.random.random(nelem)
    res = df.nlargest(n, "a")

    # Check
    inds = np.argsort(aa)
    np.testing.assert_array_equal(res["a"].to_array(), aa[inds][-n:][::-1])
    np.testing.assert_array_equal(res["b"].to_array(), bb[inds][-n:][::-1])
    np.testing.assert_array_equal(res.index.values, inds[-n:][::-1])


@pytest.mark.parametrize("nelem,n", [(10, 5), (100, 10)])
def test_dataframe_nsmallest(nelem, n):
    np.random.seed(0)
    df = DataFrame()
    df["a"] = aa = np.random.random(nelem)
    df["b"] = bb = np.random.random(nelem)
    res = df.nsmallest(n, "a")

    # Check
    inds = np.argsort(-aa)
    np.testing.assert_array_equal(res["a"].to_array(), aa[inds][-n:][::-1])
    np.testing.assert_array_equal(res["b"].to_array(), bb[inds][-n:][::-1])
    np.testing.assert_array_equal(res.index.values, inds[-n:][::-1])


@pytest.mark.parametrize(
    "counts,sliceobj", list(product([(10, 5), (100, 10)], sort_slice_args))
)
def test_dataframe_nlargest_sliced(counts, sliceobj):
    nelem, n = counts
    np.random.seed(0)
    df = pd.DataFrame()
    df["a"] = np.random.random(nelem)
    df["b"] = np.random.random(nelem)

    expect = df[sliceobj].nlargest(n, "a")
    gdf = DataFrame.from_pandas(df)
    got = gdf[sliceobj].nlargest(n, "a")
    assert (got.to_pandas() == expect).all().all()


@pytest.mark.parametrize(
    "counts,sliceobj", list(product([(10, 5), (100, 10)], sort_slice_args))
)
def test_dataframe_nsmallest_sliced(counts, sliceobj):
    nelem, n = counts
    np.random.seed(0)
    df = pd.DataFrame()
    df["a"] = np.random.random(nelem)
    df["b"] = np.random.random(nelem)

    expect = df[sliceobj].nsmallest(n, "a")
    gdf = DataFrame.from_pandas(df)
    got = gdf[sliceobj].nsmallest(n, "a")
    assert (got.to_pandas() == expect).all().all()


@pytest.mark.parametrize("num_cols", [1, 2, 3, 5])
@pytest.mark.parametrize("num_rows", [0, 1, 2, 1000])
@pytest.mark.parametrize(
    "dtype",
    [
        "int8",
        "int16",
        "int32",
        "int64",
        "float32",
        "float64",
        "datetime64[ms]",
    ],
)
@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("na_position", ["first", "last"])
def test_dataframe_multi_column(
    num_cols, num_rows, dtype, ascending, na_position
):

    from string import ascii_lowercase

    np.random.seed(0)
    by = list(ascii_lowercase[:num_cols])
    pdf = pd.DataFrame()

    for i in range(5):
        colname = ascii_lowercase[i]
        data = np.random.randint(0, 26, num_rows).astype(dtype)
        pdf[colname] = data

    gdf = DataFrame.from_pandas(pdf)

    got = gdf.sort_values(by, ascending=ascending, na_position=na_position)
    expect = pdf.sort_values(by, ascending=ascending, na_position=na_position)

    assert_eq(
        got[by].reset_index(drop=True), expect[by].reset_index(drop=True)
    )


@pytest.mark.parametrize("num_cols", [1, 2, 3])
@pytest.mark.parametrize("num_rows", [0, 1, 2, 3, 5])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
@pytest.mark.parametrize("nulls", ["some", "all"])
@pytest.mark.parametrize("ascending", [True, False])
@pytest.mark.parametrize("na_position", ["first", "last"])
def test_dataframe_multi_column_nulls(
    num_cols, num_rows, dtype, nulls, ascending, na_position
):

    from string import ascii_lowercase

    np.random.seed(0)
    by = list(ascii_lowercase[:num_cols])
    pdf = pd.DataFrame()

    for i in range(3):
        colname = ascii_lowercase[i]
        data = np.random.randint(0, 26, num_rows).astype(dtype)
        if nulls == "some":
            idx = np.array([], dtype="int64")
            if num_rows > 0:
                idx = np.random.choice(
                    num_rows, size=int(num_rows / 4), replace=False
                )
            data[idx] = np.nan
        elif nulls == "all":
            data[:] = np.nan
        pdf[colname] = data

    gdf = DataFrame.from_pandas(pdf)

    got = gdf.sort_values(by, ascending=ascending, na_position=na_position)
    expect = pdf.sort_values(by, ascending=ascending, na_position=na_position)

    assert_eq(
        got[by].reset_index(drop=True), expect[by].reset_index(drop=True)
    )


@pytest.mark.parametrize("nelem", [1, 100])
def test_series_nlargest_nelem(nelem):
    np.random.seed(0)
    elems = np.random.random(nelem)
    gds = Series(elems).nlargest(nelem)
    pds = pd.Series(elems).nlargest(nelem)

    assert (pds == gds.to_pandas()).all().all()


@pytest.mark.parametrize("map_size", [1, 2, 8])
@pytest.mark.parametrize("nelem", [1, 10, 100])
@pytest.mark.parametrize("keep", [True, False])
def test_dataframe_scatter_by_map(map_size, nelem, keep):

    strlist = ["dog", "cat", "fish", "bird", "pig", "fox", "cow", "goat"]
    np.random.seed(0)
    df = DataFrame()
    df["a"] = np.random.choice(strlist[:map_size], nelem)
    df["b"] = np.random.uniform(low=0, high=map_size, size=nelem)
    df["c"] = np.random.randint(map_size, size=nelem)
    df["d"] = df["a"]._column.as_categorical_column(np.int32)

    def _check_scatter_by_map(dfs, col):
        assert len(dfs) == map_size
        nrows = 0
        # print(col._column)
        name = col.name
        for i, df in enumerate(dfs):
            nrows += len(df)
            if len(df) > 0:
                # Make sure the column types were preserved
                assert isinstance(df[name]._column, type(col._column))
            sr = df[name].astype(np.int32)
            assert sr.nunique() <= 1
            if sr.nunique() == 1:
                if isinstance(df[name]._column, NumericalColumn):
                    assert sr[0] == i
        assert nrows == nelem

    _check_scatter_by_map(
        df.scatter_by_map("a", map_size, keep_index=keep), df["a"]
    )
    _check_scatter_by_map(
        df.scatter_by_map("b", map_size, keep_index=keep), df["b"]
    )
    _check_scatter_by_map(
        df.scatter_by_map("c", map_size, keep_index=keep), df["c"]
    )
    _check_scatter_by_map(
        df.scatter_by_map("d", map_size, keep_index=keep), df["d"]
    )

    if map_size == 2 and nelem == 100:
        df.scatter_by_map("a")  # Auto-detect map_size
        with pytest.raises(ValueError):
            df.scatter_by_map("a", 1)  # Bad map_size

    # Test GenericIndex
    df2 = df.set_index("c")
    generic_result = df2.scatter_by_map("b", map_size, keep_index=keep)
    _check_scatter_by_map(generic_result, df2["b"])
    if keep:
        for frame in generic_result:
            isinstance(frame.index, type(df2.index))

    # Test MultiIndex
    df2 = df.set_index(["a", "c"])
    multiindex_result = df2.scatter_by_map("b", map_size, keep_index=keep)
    _check_scatter_by_map(multiindex_result, df2["b"])
    if keep:
        for frame in multiindex_result:
            isinstance(frame.index, type(df2.index))
