# Copyright (c) 2022, NVIDIA CORPORATION.

import operator

import numpy as np
import pytest

from dask import dataframe as dd

from .utils import _make_random_frame, _make_random_frame_float

_binops = [
    operator.add,
    operator.sub,
    operator.mul,
    operator.truediv,
    operator.floordiv,
    operator.mod,
    operator.pow,
    operator.eq,
    operator.ne,
    operator.gt,
    operator.ge,
    operator.lt,
    operator.le,
]


@pytest.mark.parametrize("binop", _binops)
def test_series_binops_integer(binop):
    np.random.seed(0)
    size = 1000
    lhs_df, lhs_gdf = _make_random_frame(size)
    rhs_df, rhs_gdf = _make_random_frame(size)
    got = binop(lhs_gdf.x, rhs_gdf.y)
    exp = binop(lhs_df.x, rhs_df.y)
    dd.assert_eq(got, exp)


@pytest.mark.parametrize("binop", _binops)
def test_series_binops_float(binop):
    np.random.seed(0)
    size = 1000
    lhs_df, lhs_gdf = _make_random_frame_float(size)
    rhs_df, rhs_gdf = _make_random_frame_float(size)
    got = binop(lhs_gdf.x, rhs_gdf.y)
    exp = binop(lhs_df.x, rhs_df.y)
    dd.assert_eq(got, exp)


@pytest.mark.parametrize("operator", _binops)
def test_df_series_bind_ops(operator):
    np.random.seed(0)
    size = 1000
    lhs_df, lhs_gdf = _make_random_frame_float(size)
    rhs = np.random.rand()

    for col in lhs_gdf.columns:
        got = getattr(lhs_gdf[col], operator.__name__)(rhs)
        exp = getattr(lhs_df[col], operator.__name__)(rhs)
        dd.assert_eq(got, exp)

    if operator.__name__ not in ["eq", "ne", "lt", "gt", "le", "ge"]:
        got = getattr(lhs_gdf, operator.__name__)(rhs)
        exp = getattr(lhs_df, operator.__name__)(rhs)

        dd.assert_eq(got, exp)
