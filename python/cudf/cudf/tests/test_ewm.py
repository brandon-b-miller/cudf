# Copyright (c) 2022-2024, NVIDIA CORPORATION.
import pytest

import cudf
from cudf.testing import assert_eq


@pytest.mark.parametrize(
    "data",
    [
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [5.0, cudf.NA, 3.0, cudf.NA, 8.5],
        [5.0, cudf.NA, 3.0, cudf.NA, cudf.NA, 4.5],
        [5.0, cudf.NA, 3.0, 4.0, cudf.NA, 5.0],
    ],
)
@pytest.mark.parametrize(
    "params",
    [
        {"com": 0.1},
        {"com": 0.5},
        {"span": 1.5},
        {"span": 2.5},
        {"halflife": 0.5},
        {"halflife": 1.5},
        {"alpha": 0.1},
        {"alpha": 0.5},
    ],
)
@pytest.mark.parametrize("adjust", [True, False])
@pytest.mark.parametrize("method", [
    "mean", 
    pytest.param("var", marks=pytest.mark.xfail(reason="Not yet supported")), 
    pytest.param("std", marks=pytest.mark.xfail(reason="Not yet supported")),
    pytest.param("cov", marks=pytest.mark.xfail(reason="Not yet supported")),
    pytest.param("corr", marks=pytest.mark.xfail(reason="Not yet supported")),
])
def test_ewma(data, params, method, adjust):
    """
    The most basic test asserts that we obtain
    the same numerical values as pandas for various
    sets of keyword arguemnts that effect the raw
    coefficients of the formula
    """
    params["adjust"] = adjust

    gsr = cudf.Series(data, dtype="float64")
    psr = gsr.to_pandas()

    expect = getattr(psr.ewm(**params), method)()
    got = getattr(gsr.ewm(**params), method)()

    assert_eq(expect, got)

def test_ewm_error_cases():
    data = cudf.Series([1.0, 2.0, 3.0, 4.0, 5.0], dtype="float64")
    with pytest.raises(NotImplementedError):
        data.ewm(min_periods=1)
    with pytest.raises(NotImplementedError):
        data.ewm(ignore_na=True)
    with pytest.raises(NotImplementedError):
        data.ewm(axis=1)
    with pytest.raises(NotImplementedError):
        data.ewm(times=[1, 2, 3, 4, 5])
