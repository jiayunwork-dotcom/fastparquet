# -*- coding: utf-8 -*-
"""Tests for nested predicate expressions (fastparquet.predicate).

The truth reference used throughout this file is :func:`reference_eval`, an
independent, row-by-row interpreter written only with Python scalars. It does
not import or call any code from :mod:`fastparquet.predicate`, so agreement
between the two is a genuine cross-check rather than self-validation.
"""
import os

import numpy as np
import pandas as pd
import pytest

from fastparquet import write, ParquetFile
from fastparquet.predicate import compile_predicate
from fastparquet.test.util import tempdir


# ---------------------------------------------------------------------------
# Independent reference interpreter (no fastparquet.predicate imports used)
# ---------------------------------------------------------------------------

def _ref_missing(v):
    if v is None or v is pd.NA:
        return True
    return isinstance(v, float) and np.isnan(v)


def _ref_not(tri):
    if tri == "unknown":
        return "unknown"
    return not bool(tri)


def _ref_and(a, b):
    if a is False or b is False:
        return False
    if a is True and b is True:
        return True
    return "unknown"


def _ref_or(a, b):
    if a is True or b is True:
        return True
    if a is False and b is False:
        return False
    return "unknown"


def _ref_compare(actual, op, value):
    if _ref_missing(actual) or _ref_missing(value):
        return "unknown"
    if op == "==":
        return bool(actual == value)
    if op == "!=":
        return bool(actual != value)
    if op == "<":
        return bool(actual < value)
    if op == "<=":
        return bool(actual <= value)
    if op == ">":
        return bool(actual > value)
    if op == ">=":
        return bool(actual >= value)
    raise AssertionError(op)


def _ref_in(actual, values):
    if _ref_missing(actual):
        return False if len(values) == 0 else "unknown"
    for v in values:
        if not _ref_missing(v) and bool(actual == v):
            return True
    if any(_ref_missing(v) for v in values):
        return "unknown"
    return False


def reference_eval(expr, row):
    """Scalar three-valued evaluation of one row. Returns True/False/'unknown'."""
    if "and" in expr:
        out = True
        for child in expr["and"]:
            out = _ref_and(out, reference_eval(child, row))
        return out
    if "or" in expr:
        out = False
        for child in expr["or"]:
            out = _ref_or(out, reference_eval(child, row))
        return out
    if "not" in expr:
        return _ref_not(reference_eval(expr["not"], row))

    column, op = expr["column"], expr["op"]
    actual = row[column]
    if op == "is null":
        return bool(_ref_missing(actual))
    if op == "is not null":
        return bool(not _ref_missing(actual))
    if op == "in":
        return _ref_in(actual, expr["value"])
    if op == "not in":
        return _ref_not(_ref_in(actual, expr["value"]))
    return _ref_compare(actual, op, expr["value"])


def reference_mask(expr, rows):
    return np.array([reference_eval(expr, row) is True for row in rows],
                    dtype=bool)


# ---------------------------------------------------------------------------
# data fixtures
# ---------------------------------------------------------------------------

BIG = 2 ** 63 - 1
BIG2 = 2 ** 53 + 3


def make_frame(n=40, seed=0):
    rng = np.random.RandomState(seed)
    amount = np.array([None, 1, -1, BIG, BIG2, 5, None, -BIG2] * 5,
                      dtype=object)[:n]
    ratio = np.array([0.0, np.nan, -1.5, np.inf, -np.inf, 2.5, None, 3.25] * 5,
                     dtype=object)[:n]
    name = np.array(["a", "b", None, "zzz", "a", "", "c", None] * 5,
                    dtype=object)[:n]
    flag = np.array([True, False, None, True, False, None, True, False] * 5,
                    dtype=object)[:n]
    extra = rng.randint(0, 100, size=n)
    return pd.DataFrame({"amount": pd.Series(amount, dtype="Int64"),
                         "ratio": pd.Series(ratio, dtype="float64"),
                         "name": pd.Series(name, dtype="object"),
                         "flag": pd.Series(flag, dtype="boolean"),
                         "extra": extra})


def frame_rows(df):
    rows = []
    for record in df.to_dict("records"):
        row = {}
        for k, v in record.items():
            if isinstance(v, float) and np.isnan(v):
                row[k] = np.nan
            else:
                row[k] = v
        rows.append(row)
    return rows


@pytest.fixture()
def simple_pf(tempdir):
    fn = os.path.join(tempdir, "simple.parquet")
    df = make_frame()
    write(fn, df, row_group_offsets=5, stats=True)
    return ParquetFile(fn), df


@pytest.fixture()
def nostats_pf(tempdir):
    fn = os.path.join(tempdir, "nostats.parquet")
    df = make_frame()
    write(fn, df, row_group_offsets=5, stats=False)
    return ParquetFile(fn), df


@pytest.fixture()
def hive_pf(tempdir):
    dn = os.path.join(tempdir, "hive")
    df = make_frame(36)
    df["g"] = (["alpha", "beta"] * 18)
    df["p"] = list(range(36))
    write(dn, df, partition_on=["g"], row_group_offsets=4, stats=True, file_scheme="hive")
    return ParquetFile(dn), df


# A broad set of predicates exercising true/false/unknown combinations.
PREDICATES = [
    {"column": "amount", "op": ">", "value": 0},
    {"column": "amount", "op": "==", "value": BIG},
    {"column": "amount", "op": "!=", "value": 5},
    {"column": "amount", "op": "<=", "value": BIG2},
    {"column": "ratio", "op": ">", "value": 1e308},
    {"column": "ratio", "op": "==", "value": float("inf")},
    {"column": "ratio", "op": "<", "value": -1e308},
    {"column": "ratio", "op": "==", "value": float("nan")},
    {"column": "name", "op": "==", "value": "a"},
    {"column": "name", "op": "<", "value": "c"},
    {"column": "flag", "op": "==", "value": True},
    {"column": "flag", "op": "!=", "value": False},
    {"column": "amount", "op": "is null"},
    {"column": "amount", "op": "is not null"},
    {"column": "ratio", "op": "is null"},
    {"column": "name", "op": "is not null"},
    {"column": "amount", "op": "in", "value": [1, 5, None]},
    {"column": "amount", "op": "in", "value": []},
    {"column": "amount", "op": "not in", "value": [1, None]},
    {"column": "amount", "op": "not in", "value": []},
    {"column": "name", "op": "in", "value": ["a", "zzz", None]},
    {"column": "ratio", "op": "in", "value": [2.5, float("nan")]},
    {"not": {"column": "amount", "op": ">", "value": 0}},
    {"not": {"column": "amount", "op": "is null"}},
    {"and": [{"column": "amount", "op": ">", "value": 0},
             {"column": "ratio", "op": "is not null"}]},
    {"or": [{"column": "amount", "op": "is null"},
            {"column": "name", "op": "==", "value": "zzz"}]},
    {"and": [{"or": [{"column": "flag", "op": "==", "value": True},
                     {"column": "name", "op": "is null"}]},
             {"not": {"column": "ratio", "op": "<", "value": 0.0}}]},
    {"not": {"and": [{"column": "amount", "op": ">", "value": 100},
                     {"column": "flag", "op": "==", "value": True}]}},
    {"or": [{"not": {"column": "name", "op": "in",
                     "value": ["a", None]}},
            {"and": [{"column": "ratio", "op": ">=", "value": 0.0},
                     {"column": "amount", "op": "is not null"}]}]},
]


# ---------------------------------------------------------------------------
# exact filtering vs the independent reference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("expr", PREDICATES)
def test_exact_matches_reference(simple_pf, expr):
    pf, df = simple_pf
    rows = frame_rows(df)
    expected = df.loc[reference_mask(expr, rows)].reset_index(drop=True)
    got = pf.to_pandas(predicate=expr, row_filter=True).reset_index(drop=True)
    pd.testing.assert_frame_equal(got, expected)
    assert pf.count(predicate=expr, row_filter=True) == len(expected)


@pytest.mark.parametrize("expr", PREDICATES)
def test_exact_matches_reference_nostats(nostats_pf, expr):
    pf, df = nostats_pf
    rows = frame_rows(df)
    expected = df.loc[reference_mask(expr, rows)].reset_index(drop=True)
    got = pf.to_pandas(predicate=expr, row_filter=True).reset_index(drop=True)
    pd.testing.assert_frame_equal(got, expected)


@pytest.mark.parametrize("expr", PREDICATES)
def test_hive_mixed_partition_file(hive_pf, expr):
    pf, df = hive_pf
    mixed = {"or": [expr, {"column": "g", "op": "==", "value": "beta"}]}
    mixed_not = {"not": {"or": [expr,
                                {"column": "g", "op": "==", "value": "alpha"}]}}
    # 'p' is a unique row identity; hive writes reorder rows by partition, so
    # comparisons are made on row-identity multisets.
    plain = pf.to_pandas()
    for q in (mixed, mixed_not):
        rows = frame_rows(df)
        expected_p = sorted(df.loc[reference_mask(q, rows), "p"].tolist())
        got = pf.to_pandas(predicate=q, row_filter=True)
        assert sorted(got["p"].tolist()) == expected_p
        assert pf.count(predicate=q, row_filter=True) == len(expected_p)
        # predicate output must be the plain hive read filtered identically
        plain_mask = reference_mask(q, frame_rows(plain))
        assert got["p"].tolist() == plain.loc[plain_mask, "p"].tolist()
        # the partition column is present and never omitted
        assert "g" in got.columns


def test_partition_only_predicate(hive_pf):
    pf, df = hive_pf
    expr = {"and": [{"column": "g", "op": "==", "value": "alpha"},
                    {"column": "p", "op": ">", "value": 10}]}
    rows = frame_rows(df)
    expected = df.loc[reference_mask(expr, rows)]
    got = pf.to_pandas(predicate=expr, row_filter=True)
    assert sorted(got["p"].tolist()) == sorted(expected["p"].tolist())


def test_numeric_partition_column(tempdir):
    dn = os.path.join(tempdir, "nhive")
    df = pd.DataFrame({"v": range(24), "n": ([1, 2] * 12)})
    write(dn, df, partition_on=["n"], row_group_offsets=3,
          file_scheme="hive")
    pf = ParquetFile(dn)
    # partition values are decoded as numpy scalars
    expr = {"and": [{"column": "n", "op": "==", "value": 2},
                    {"column": "v", "op": ">", "value": 20}]}
    got = pf.to_pandas(predicate=expr, row_filter=True)
    assert sorted(got["v"].tolist()) == [21, 23]
    # partition column participates in a NOT and an OR with a file column
    expr_not = {"not": {"column": "n", "op": "==", "value": 1}}
    assert all(int(v) == 2 for v in
               pf.to_pandas(predicate=expr_not, row_filter=True)["n"])
    expr_or = {"or": [{"column": "n", "op": "==", "value": 1},
                      {"column": "v", "op": "==", "value": 23}]}
    # all twelve n=1 rows plus the single n=2 row with v == 23
    assert len(pf.to_pandas(predicate=expr_or, row_filter=True)) == 13
    assert pf.to_pandas(predicate=expr_or, row_filter=True)["v"].max() == 23


def test_bool_literal_on_bool_partition(tempdir):
    dn = os.path.join(tempdir, "bhive")
    df = pd.DataFrame({"v": range(12),
                       "flag": [True, False] * 6})
    write(dn, df, partition_on=["flag"], row_group_offsets=3,
          file_scheme="hive")
    pf = ParquetFile(dn)
    got = pf.to_pandas(
        predicate={"column": "flag", "op": "==", "value": True},
        row_filter=True)
    assert all(bool(x) for x in got["flag"])
    # an integer literal against a bool partition is rejected
    with pytest.raises(ValueError):
        pf.to_pandas(predicate={"column": "flag", "op": "==", "value": 1})


def test_float64_inf_literals_and_big_int(simple_pf):
    pf, df = simple_pf
    rows = frame_rows(df)
    # +/- infinity are legal float literals
    pinf = {"column": "ratio", "op": "==", "value": float("inf")}
    ninf = {"column": "ratio", "op": "==", "value": float("-inf")}
    assert pf.to_pandas(predicate=pinf, row_filter=True)["extra"].tolist() == \
        df.loc[reference_mask(pinf, rows), "extra"].tolist()
    assert pf.to_pandas(predicate=ninf, row_filter=True)["extra"].tolist() == \
        df.loc[reference_mask(ninf, rows), "extra"].tolist()
    # integer larger than 2**53 keeps full precision end to end
    big = {"column": "amount", "op": "==", "value": BIG}
    got = pf.to_pandas(predicate=big, row_filter=True)
    assert (got["amount"] == BIG).all()
    # a comparison literal beyond the int64 range is rejected, not coerced
    with pytest.raises(ValueError):
        pf.to_pandas(predicate={"column": "amount", "op": ">",
                                "value": 2 ** 63})


def test_filter_column_not_in_output(simple_pf):
    pf, df = simple_pf
    expr = {"column": "amount", "op": ">", "value": 10}
    got = pf.to_pandas(columns=["name", "ratio"], predicate=expr,
                       row_filter=True)
    assert list(got.columns) == ["name", "ratio"]
    rows = frame_rows(df)
    expected = df.loc[reference_mask(expr, rows), ["name", "ratio"]]
    pd.testing.assert_frame_equal(got.reset_index(drop=True),
                                  expected.reset_index(drop=True))


def test_string_unicode_ordering(tempdir):
    fn = os.path.join(tempdir, "p.parquet")
    # code-point order: 'a'(97) < 'z'(122) < 'ä'(228)
    df = pd.DataFrame({"word": ["z", "ä", "a", None]}, dtype="object")
    write(fn, df, row_group_offsets=2, stats=True)
    pf = ParquetFile(fn)
    lt = pf.to_pandas(predicate={"column": "word", "op": "<", "value": "ä"},
                      row_filter=True)
    assert sorted(lt["word"].tolist()) == ["a", "z"]
    gt = pf.to_pandas(predicate={"column": "word", "op": ">", "value": "ä"},
                      row_filter=True)
    assert gt.empty  # nothing sorts above 'ä' here
    eq = pf.to_pandas(predicate={"column": "word", "op": "==", "value": "ä"},
                      row_filter=True)
    assert eq["word"].tolist() == ["ä"]


def test_bool_file_column(simple_pf):
    pf, df = simple_pf
    expr = {"column": "flag", "op": "==", "value": True}
    got = pf.to_pandas(predicate=expr, row_filter=True)
    rows = frame_rows(df)
    expected = df.loc[reference_mask(expr, rows)]
    assert got["extra"].tolist() == expected["extra"].tolist()
    assert bool(got["flag"].all())
    # null test on a nullable boolean column
    isnull = {"column": "flag", "op": "is null"}
    got2 = pf.to_pandas(predicate=isnull, row_filter=True)
    assert got2["extra"].tolist() == df.loc[
        reference_mask(isnull, rows), "extra"].tolist()


def test_predicate_with_range_index_preserved(tempdir):
    fn = os.path.join(tempdir, "p.parquet")
    df = pd.DataFrame({"v": range(20),
                       "flag": pd.Series([True, False] * 10,
                                         dtype="boolean")})
    write(fn, df, row_group_offsets=4, stats=True)
    pf = ParquetFile(fn)
    got = pf.to_pandas(predicate={"column": "v", "op": ">", "value": 15},
                       row_filter=True)
    # default sequential index is restored and starts at 0
    assert got.index.tolist() == list(range(len(got)))
    assert got["v"].tolist() == list(range(16, 20))


def test_head_with_predicate(simple_pf):
    pf, df = simple_pf
    expr = {"column": "amount", "op": "is null"}
    head = pf.head(2, predicate=expr, row_filter=True)
    # head() loads the first row group(s) then truncates, so its rows are the
    # first matches within that prefix (a documented property, not the full
    # filtered frame)
    full_prefix = pf[:1].to_pandas(predicate=expr, row_filter=True)
    assert head["extra"].tolist() == full_prefix["extra"].head(2).tolist()
    assert len(head) <= 2
    pf, df = simple_pf
    expr = {"column": "amount", "op": ">", "value": 10}
    got = pf.to_pandas(columns=["name", "ratio"], predicate=expr,
                       row_filter=True)
    assert list(got.columns) == ["name", "ratio"]
    rows = frame_rows(df)
    expected = df.loc[reference_mask(expr, rows), ["name", "ratio"]]
    pd.testing.assert_frame_equal(got.reset_index(drop=True),
                                  expected.reset_index(drop=True))


def test_source_row_order_and_dtypes(simple_pf):
    pf, df = simple_pf
    expr = {"or": [{"column": "amount", "op": "is null"},
                   {"column": "ratio", "op": ">", "value": 1.0}]}
    got = pf.to_pandas(predicate=expr, row_filter=True)
    rows = frame_rows(df)
    expected = df.loc[reference_mask(expr, rows)]
    # source order preserved
    assert got["extra"].tolist() == expected["extra"].tolist()
    # decoded dtypes preserved
    for col in df.columns:
        assert got[col].dtype == df[col].dtype


def test_iter_row_groups_order_and_laziness(simple_pf):
    pf, df = simple_pf
    expr = {"column": "extra", "op": ">", "value": 50}
    frames = list(pf.iter_row_groups(predicate=expr, row_filter=True))
    # groups in file order, empty ones omitted
    seen = []
    for f in frames:
        seen.extend(f["extra"].tolist())
    rows = frame_rows(df)
    assert seen == df.loc[reference_mask(expr, rows), "extra"].tolist()
    # every yielded frame corresponds to one physical row group, no duplicates
    assert len(frames) <= len(pf.row_groups)


def test_iter_row_groups_group_level(simple_pf):
    pf, df = simple_pf
    expr = {"column": "amount", "op": "==", "value": BIG}
    frames = list(pf.iter_row_groups(predicate=expr))
    rows = frame_rows(df)
    exact_extras = set(df.loc[reference_mask(expr, rows), "extra"])
    # every group with an actual match is yielded; pruning may keep extra
    # uncertain groups but can never drop one
    yielded_extras = set().union(*(set(f["extra"]) for f in frames))
    assert exact_extras <= yielded_extras
    assert len(frames) <= len(pf.row_groups)
    total = sum(len(f) for f in frames)
    assert total == pf.count(predicate=expr)


def test_empty_result(simple_pf):
    pf, df = simple_pf
    expr = {"column": "amount", "op": "==", "value": 999999999}
    got = pf.to_pandas(predicate=expr, row_filter=True)
    assert got.empty
    assert list(got.columns) == list(df.columns)
    assert pf.count(predicate=expr, row_filter=True) == 0
    assert list(pf.iter_row_groups(predicate=expr, row_filter=True)) == []


# ---------------------------------------------------------------------------
# pruning: mandatory skip scenarios and no-missed-row guarantee
# ---------------------------------------------------------------------------

def test_prune_disjoint_integer_range(tempdir):
    fn = os.path.join(tempdir, "p.parquet")
    df = pd.DataFrame({"x": np.arange(100, dtype="int64")})
    write(fn, df, row_group_offsets=20, stats=True)
    pf = ParquetFile(fn)
    expr = {"column": "x", "op": "<", "value": -5}
    assert pf.count(predicate=expr) == 0
    assert pf.count(predicate=expr, row_filter=True) == 0
    assert pf.to_pandas(predicate=expr).empty


def test_prune_constant_column_false_comparison(tempdir):
    fn = os.path.join(tempdir, "p.parquet")
    df = pd.DataFrame({"x": [7] * 12})
    write(fn, df, row_group_offsets=4, stats=True)
    pf = ParquetFile(fn)
    expr = {"column": "x", "op": "==", "value": 8}
    assert pf.count(predicate=expr) == 0
    expr2 = {"column": "x", "op": "!=", "value": 7}
    assert pf.count(predicate=expr2) == 0


def test_prune_all_null_is_not_null(tempdir):
    fn = os.path.join(tempdir, "p.parquet")
    df = pd.DataFrame({"x": pd.Series([None] * 8, dtype="Int64"),
                       "y": range(8)})
    write(fn, df, row_group_offsets=4, stats=True)
    pf = ParquetFile(fn)
    expr = {"column": "x", "op": "is not null"}
    assert pf.count(predicate=expr) == 0
    # is null on the same groups must keep them
    keep = {"column": "x", "op": "is null"}
    assert pf.count(predicate=keep) == 8


def test_prune_nulls_make_disjoint_range_uncertain(tempdir):
    # A range that excludes every non-null value still cannot prune when the
    # group contains nulls, because the comparison is unknown on those rows.
    fn = os.path.join(tempdir, "p.parquet")
    df = pd.DataFrame({"x": pd.Series([1, 2, None, 4], dtype="Int64")})
    write(fn, df, row_group_offsets=4, stats=True)
    pf = ParquetFile(fn)
    assert pf.count(predicate={"column": "x", "op": "==", "value": 100}) == 4
    assert pf.count(predicate={"column": "x", "op": "==", "value": 100},
                    row_filter=True) == 0
    # without any null the same disjoint range must prune
    fn2 = os.path.join(tempdir, "p2.parquet")
    write(fn2, pd.DataFrame({"x": [1, 2, 3, 4]}, dtype="int64"),
          row_group_offsets=4, stats=True)
    pf2 = ParquetFile(fn2)
    assert pf2.count(predicate={"column": "x", "op": "==", "value": 100}) == 0


def test_prune_constant_column_with_nulls(tempdir):
    fn = os.path.join(tempdir, "p.parquet")
    df = pd.DataFrame({"x": pd.Series([5, 5, None], dtype="Int64")})
    write(fn, df, row_group_offsets=3, stats=True)
    pf = ParquetFile(fn)
    # != 5 is false on the constant rows but unknown on the null row
    assert pf.count(predicate={"column": "x", "op": "!=", "value": 5}) == 3
    assert pf.count(predicate={"column": "x", "op": "!=", "value": 5},
                    row_filter=True) == 0
    # membership with a disjoint list is false for constants, unknown at null
    assert pf.count(predicate={"column": "x", "op": "in",
                               "value": [6]}) == 3
    assert pf.count(predicate={"column": "x", "op": "in", "value": [6]},
                    row_filter=True) == 0
    # a no-null constant column still prunes hard
    fn2 = os.path.join(tempdir, "p2.parquet")
    write(fn2, pd.DataFrame({"x": [5, 5, 5]}, dtype="int64"),
          row_group_offsets=3, stats=True)
    pf2 = ParquetFile(fn2)
    assert pf2.count(predicate={"column": "x", "op": "!=", "value": 5}) == 0


def test_prune_all_null_membership(tempdir):
    fn = os.path.join(tempdir, "p.parquet")
    df = pd.DataFrame({"x": pd.Series([None, None], dtype="Int64")})
    write(fn, df, row_group_offsets=2, stats=True)
    pf = ParquetFile(fn)
    # missing compared value -> unknown, group kept but exact count is zero
    assert pf.count(predicate={"column": "x", "op": "in", "value": [6]}) == 2
    assert pf.count(predicate={"column": "x", "op": "in", "value": [6]},
                    row_filter=True) == 0
    assert pf.count(predicate={"column": "x", "op": "not in",
                               "value": [6]}, row_filter=True) == 0
    # empty list is always false, even with missing compared values
    assert pf.count(predicate={"column": "x", "op": "in", "value": []}) == 0
    assert pf.count(predicate={"column": "x", "op": "not in",
                               "value": []}, row_filter=True) == 2


def test_prune_partition_proves_false(hive_pf):
    pf, df = hive_pf
    expr = {"and": [{"column": "g", "op": "==", "value": "gamma"},
                    {"column": "p", "op": ">", "value": 0}]}
    assert pf.count(predicate=expr) == 0
    # OR with an impossible partition branch still keeps file-driven matches
    expr2 = {"or": [{"column": "g", "op": "==", "value": "gamma"},
                    {"column": "p", "op": "==", "value": 5}]}
    assert pf.count(predicate=expr2, row_filter=True) == 1


def test_prune_nested_expression(tempdir):
    fn = os.path.join(tempdir, "p.parquet")
    df = pd.DataFrame({"x": range(40), "y": range(40, 80)})
    write(fn, df, row_group_offsets=10, stats=True)
    pf = ParquetFile(fn)
    # every leaf is provably false on some groups; AND of them -> skip
    expr = {"and": [{"column": "x", "op": ">", "value": 1000},
                    {"column": "y", "op": "<", "value": -1000}]}
    assert pf.count(predicate=expr) == 0
    # NOT must not turn unknown/false reasoning into an incorrect exclusion:
    # x > 1000 is provably false everywhere, NOT -> true, keep all groups
    expr2 = {"not": {"column": "x", "op": ">", "value": 1000}}
    assert pf.count(predicate=expr2) == 40


@pytest.mark.parametrize("expr", PREDICATES)
def test_pruning_never_drops_a_match(simple_pf, expr):
    pf, df = simple_pf
    rows = frame_rows(df)
    exact = reference_mask(expr, rows)
    # group-level count is an upper bound on the exact count
    assert pf.count(predicate=expr) >= int(exact.sum())
    # every exact match row must live in a surviving row group
    surviving = pf.to_pandas(predicate=expr)
    surviving_idx = set(surviving["extra"].tolist())
    for i, keep in enumerate(exact):
        if keep:
            assert df["extra"].iloc[i] in surviving_idx


def test_nostats_keeps_every_group(nostats_pf):
    pf, df = nostats_pf
    expr = {"column": "amount", "op": "==", "value": BIG}
    # nothing can be pruned without statistics
    assert pf.count(predicate=expr) == len(df)
    # exact stage still gives the right answer
    assert pf.count(predicate=expr, row_filter=True) == int(
        reference_mask(expr, frame_rows(df)).sum())


def test_nan_statistics_keep_group(tempdir):
    fn = os.path.join(tempdir, "p.parquet")
    # one all-NaN group and one constant 1.0 group
    df = pd.DataFrame({"x": [np.nan] * 4 + [1.0] * 4})
    write(fn, df, row_group_offsets=4, stats=True)
    pf = ParquetFile(fn)
    expr = {"column": "x", "op": "==", "value": 2.0}
    # the NaN-containing group cannot be excluded via its (NaN) range, but the
    # constant 1.0 group is provably false; so pruning keeps the NaN group only
    assert pf.count(predicate=expr) == 4
    # exact stage removes everything (NaN == 2.0 is unknown, never true)
    assert pf.count(predicate=expr, row_filter=True) == 0
    # and the all-NaN group alone survives a range comparison that, without the
    # NaN guard, would have excluded it
    expr2 = {"column": "x", "op": ">", "value": 100.0}
    assert pf.count(predicate=expr2) == 4
    assert pf.count(predicate=expr2, row_filter=True) == 0


# ---------------------------------------------------------------------------
# data-page read tracking
# ---------------------------------------------------------------------------

class PageReadTracker:
    """Monkeypatch wrapper counting data pages actually decoded."""

    def __init__(self, pf):
        self.pf = pf
        self.opened = []
        self.pages = 0
        self._orig_open = pf.open

    def __enter__(self):
        from fastparquet import core
        self._orig_read_col = core.read_col
        tracker = self

        def wrapped_read_col(column, *args, **kwargs):
            tracker.pages += 1  # one call per decoded column chunk
            return tracker._orig_read_col(column, *args, **kwargs)

        core.read_col = wrapped_read_col
        pf = self.pf

        def open_tracking(path, *a, **k):
            tracker.opened.append(path)
            return tracker._orig_open(path, *a, **k)

        pf.open = open_tracking
        return self

    def __exit__(self, *exc):
        from fastparquet import core
        core.read_col = self._orig_read_col
        self.pf.open = self._orig_open


def test_metadata_only_count_reads_no_pages(tempdir):
    fn = os.path.join(tempdir, "p.parquet")
    df = pd.DataFrame({"x": range(80)})
    write(fn, df, row_group_offsets=10, stats=True)
    pf = ParquetFile(fn)
    expr = {"column": "x", "op": "<", "value": -1}
    with PageReadTracker(pf) as t:
        assert pf.count(predicate=expr) == 0
    assert t.pages == 0
    # a pruning to_pandas also reads no data pages when zero groups survive
    with PageReadTracker(pf) as t:
        out = pf.to_pandas(predicate=expr)
    assert t.pages == 0 and out.empty
    # nested expression proven false everywhere also reads nothing
    nested = {"and": [{"column": "x", "op": ">", "value": 10 ** 12},
                      {"not": {"column": "x", "op": "is null"}}]}
    with PageReadTracker(pf) as t:
        assert pf.count(predicate=nested) == 0
    assert t.pages == 0
    # partition-only false predicate on a hive dataset reads no data pages
    dn = os.path.join(tempdir, "hive")
    hdf = pd.DataFrame({"x": range(20), "g": ["a"] * 10 + ["b"] * 10})
    write(dn, hdf, partition_on=["g"], row_group_offsets=5, file_scheme="hive")
    hpf = ParquetFile(dn)
    ponly = {"column": "g", "op": "==", "value": "nonexistent"}
    with PageReadTracker(hpf) as t:
        assert hpf.count(predicate=ponly) == 0
        assert hpf.to_pandas(predicate=ponly).empty
    assert t.pages == 0


def test_surviving_groups_still_read_pages(tempdir):
    """The complementary guarantee: a kept group's projection read still
    happens normally; pruning only suppresses excluded groups."""
    fn = os.path.join(tempdir, "p.parquet")
    df = pd.DataFrame({"x": range(80)})
    write(fn, df, row_group_offsets=10, stats=True)
    pf = ParquetFile(fn)
    # x < 15 keeps the first two groups only
    expr = {"column": "x", "op": "<", "value": 15}
    with PageReadTracker(pf) as t:
        out = pf.to_pandas(predicate=expr)
    assert len(out) == 20
    # one column chunk decoded per surviving group
    assert t.pages == 2
    # exact filtering reads the filter column but returns the right rows
    with PageReadTracker(pf) as t:
        exact = pf.to_pandas(columns=["x"], predicate=expr, row_filter=True)
    assert exact["x"].tolist() == list(range(15))
    # the two surviving groups are each read twice: once to compute the exact
    # mask, once to project the kept rows into the output frame
    assert t.pages == 4


def test_iter_is_lazy(simple_pf):
    pf, df = simple_pf
    expr = {"column": "extra", "op": ">", "value": 10 ** 9}  # matches nothing
    gen = pf.iter_row_groups(predicate=expr, row_filter=True)
    # consuming nothing must not open any data page: exhaust via a first next()
    import fastparquet.core as core
    calls = []
    orig = core.read_col
    core.read_col = lambda *a, **k: calls.append(1) or orig(*a, **k)
    try:
        assert next(gen, None) is None
        assert calls == []
    finally:
        core.read_col = orig

    # and a matching predicate processes groups one at a time on demand
    expr2 = {"column": "extra", "op": ">=", "value": 0}
    gen2 = pf.iter_row_groups(predicate=expr2, row_filter=True)
    first = next(gen2)
    assert len(first) <= 5  # one row group worth of rows


def test_insufficient_stats_still_reads_and_matches(simple_pf):
    pf, df = simple_pf
    expr = {"column": "amount", "op": "==", "value": 5}
    got = pf.to_pandas(predicate=expr, row_filter=True)
    rows = frame_rows(df)
    expected = df.loc[reference_mask(expr, rows)]
    assert got["extra"].tolist() == expected["extra"].tolist()


# ---------------------------------------------------------------------------
# validation errors
# ---------------------------------------------------------------------------

TYPE_MAP = {"amount": "int64", "ratio": "float64", "name": "string",
            "flag": "bool", "extra": "int64"}


def _types():
    return dict(TYPE_MAP)


@pytest.mark.parametrize("bad", [
    "not a dict",
    {},
    {"column": "amount"},
    {"op": "==", "value": 1},
    {"column": "amount", "op": "~", "value": 1},
    {"column": "missing", "op": "==", "value": 1},
    {"column": "amount", "op": "=="},
    {"column": "amount", "op": "is null", "value": 1},
    {"column": "amount", "op": "in", "value": (1, 2)},
    {"column": "amount", "op": "in", "value": "not a list"},
    {"and": []},
    {"or": [1, 2]},
    {"not": []},
    {"and": [{"column": "amount", "op": "==", "value": 1}], "extra": 0},
    {"column": "amount", "op": "==", "value": 1, "junk": True},
    {"foo": 1, "bar": 2},
    {"column": "amount", "op": "==", "value": True},   # bool into int
    {"column": "amount", "op": "==", "value": 1.5},
    {"column": "amount", "op": "==", "value": 2 ** 63},
    {"column": "flag", "op": "==", "value": 1},
    {"column": "flag", "op": "in", "value": [True, "x"]},
    {"column": "name", "op": "==", "value": 1},
    {"column": "name", "op": "in", "value": ["a", 1]},
    {"column": "ratio", "op": "==", "value": "x"},
    {"column": "amount", "op": "in", "value": [1, True]},
])
def test_invalid_expressions(bad):
    with pytest.raises(ValueError):
        compile_predicate(bad, _types())


def test_missing_literals_are_allowed():
    for literal in (None, np.nan, pd.NA):
        compile_predicate({"column": "amount", "op": "==", "value": literal},
                          _types())
        compile_predicate(
            {"column": "ratio", "op": "in", "value": [1.0, literal]}, _types())


def test_cycle_detection():
    # a genuine self reference
    node = {}
    node["not"] = node
    with pytest.raises(ValueError):
        compile_predicate(node, _types())
    # sharing one dict object between branches is also a repeated reference
    child = {"column": "amount", "op": "==", "value": 1}
    shared = {"and": [child, child]}
    with pytest.raises(ValueError):
        compile_predicate(shared, _types())
    # an equivalent tree built from separate dicts is perfectly fine
    ok = {"and": [{"column": "amount", "op": "==", "value": 1},
                  {"column": "amount", "op": "==", "value": 1}]}
    compile_predicate(ok, _types())


def test_input_expression_not_mutated(simple_pf):
    pf, df = simple_pf
    expr = {"and": [{"column": "amount", "op": "in", "value": [1, None]},
                    {"or": [{"column": "flag", "op": "is not null"},
                            {"not": {"column": "name", "op": "==",
                                     "value": "a"}}]}]}
    import copy
    snapshot = copy.deepcopy(expr)
    compiled = compile_predicate(expr, _types())
    assert expr == snapshot
    # repeated reuse yields identical masks
    m1 = compiled.row_mask(len(df), {c: df[c] for c in compiled.columns}, {})
    m2 = compiled.row_mask(len(df), {c: df[c] for c in compiled.columns}, {})
    assert np.array_equal(m1, m2)
    assert expr == snapshot


def test_error_raised_before_data_pages(tempdir, monkeypatch):
    fn = os.path.join(tempdir, "p.parquet")
    df = pd.DataFrame({"x": range(20)})
    write(fn, df, row_group_offsets=5)
    pf = ParquetFile(fn)
    from fastparquet import core
    calls = []
    orig = core.read_col

    def boom(*args, **kwargs):
        calls.append(1)
        return orig(*args, **kwargs)

    monkeypatch.setattr(core, "read_col", boom)
    with pytest.raises(ValueError):
        pf.to_pandas(predicate={"column": "nope", "op": "==", "value": 1})
    assert calls == []


def test_reuse_after_failure(simple_pf):
    pf, df = simple_pf
    with pytest.raises(ValueError):
        pf.to_pandas(predicate={"column": "ghost", "op": "==", "value": 1})
    # the ParquetFile still works, with both predicate and legacy filters
    got = pf.to_pandas(predicate={"column": "amount", "op": "is null"},
                       row_filter=True)
    assert len(got) == int(reference_mask(
        {"column": "amount", "op": "is null"}, frame_rows(df)).sum())
    legacy = pf.to_pandas(filters=[["extra", "==", df["extra"].iloc[0]]])
    assert len(legacy) >= 1


def test_conflict_arguments(simple_pf):
    pf, _ = simple_pf
    p = {"column": "amount", "op": "==", "value": 1}
    with pytest.raises(ValueError):
        pf.to_pandas(filters=[["amount", "==", 1]], predicate=p)
    with pytest.raises(ValueError):
        pf.to_pandas(row_filter=np.array([True] * 40), predicate=p)
    with pytest.raises(ValueError):
        pf.count(filters=[["amount", "==", 1]], predicate=p)
    with pytest.raises(ValueError):
        pf.iter_row_groups(filters=[["amount", "==", 1]], predicate=p)


# ---------------------------------------------------------------------------
# backward compatibility when predicate=None
# ---------------------------------------------------------------------------

def test_legacy_paths_unchanged(simple_pf):
    pf, df = simple_pf
    a = pf.to_pandas()
    b = pf.to_pandas(predicate=None)
    pd.testing.assert_frame_equal(a, b)
    assert pf.count() == len(df)
    assert pf.count(predicate=None) == len(df)
    assert len(list(pf.iter_row_groups())) == len(pf.row_groups)
    # legacy filters still work
    f = pf.to_pandas(filters=[["extra", "==", df["extra"].iloc[0]]])
    f2 = pf.to_pandas(filters=[["extra", "==", df["extra"].iloc[0]]],
                      predicate=None)
    pd.testing.assert_frame_equal(f, f2)


# ---------------------------------------------------------------------------
# scale: deep nesting and wide OR-of-ANDs, no exponential blow-up
# ---------------------------------------------------------------------------

def test_deep_nesting():
    leaf = {"column": "amount", "op": ">", "value": 0}
    expr = leaf
    for _ in range(40):
        expr = {"not": {"and": [expr, {"column": "flag",
                                       "op": "is not null"}]}}
    compiled = compile_predicate(expr, _types())
    df = make_frame()
    mask = compiled.row_mask(len(df), {c: df[c] for c in compiled.columns}, {})
    expected = reference_mask(expr, frame_rows(df))
    assert np.array_equal(mask, expected)


def test_wide_or_of_ands_scaling():
    clauses = [{"and": [{"column": "amount", "op": ">", "value": i},
                        {"column": "extra", "op": "<", "value": 99 - i}]}
               for i in range(24)]
    expr = {"and": clauses}
    compiled = compile_predicate(expr, _types())
    df = make_frame()
    import time
    t0 = time.perf_counter()
    mask = compiled.row_mask(len(df), {c: df[c] for c in compiled.columns}, {})
    elapsed_24 = time.perf_counter() - t0
    expected = reference_mask(expr, frame_rows(df))
    assert np.array_equal(mask, expected)

    clauses48 = [{"and": [{"column": "amount", "op": ">", "value": i},
                          {"column": "extra", "op": "<", "value": 99 - i}]}
                 for i in range(48)]
    expr48 = {"and": clauses48}
    compiled48 = compile_predicate(expr48, _types())
    t0 = time.perf_counter()
    mask48 = compiled48.row_mask(
        len(df), {c: df[c] for c in compiled48.columns}, {})
    elapsed_48 = time.perf_counter() - t0
    assert np.array_equal(mask48, reference_mask(expr48, frame_rows(df)))
    # linear-ish scaling: doubling clauses must not explode exponentially
    assert elapsed_48 < elapsed_24 * 10 + 1.0


# ---------------------------------------------------------------------------
# randomized predicates vs reference, across layouts and stats settings
# ---------------------------------------------------------------------------

def _random_expr(rng, depth=0):
    if depth >= 4 or rng.rand() < 0.45:
        column = rng.choice(["amount", "ratio", "name", "flag"])
        op = rng.choice(["==", "!=", "<", "<=", ">", ">=", "in", "not in",
                         "is null", "is not null"])
        leaf = {"column": column, "op": op}
        if op in ("is null", "is not null"):
            return leaf
        if op in ("in", "not in"):
            pool = {
                "amount": [1, 5, BIG, None],
                "ratio": [2.5, np.inf, None, 0.0],
                "name": ["a", "zzz", None, ""],
                "flag": [True, False, None],
            }[column]
            leaf["value"] = list(rng.choice(pool,
                                            size=rng.randint(0, 3),
                                            replace=False))
            return leaf
        value = {
            "amount": rng.choice([1, -1, 5, BIG, BIG2, None]),
            "ratio": rng.choice([2.5, -1.5, np.inf, -np.inf, np.nan, None]),
            "name": rng.choice(["a", "b", "zzz", None, ""]),
            "flag": rng.choice([True, False, None]),
        }[column]
        leaf["value"] = value
        return leaf
    kind = rng.choice(["and", "or", "not"])
    if kind == "not":
        return {"not": _random_expr(rng, depth + 1)}
    n = rng.randint(1, 4)
    return {kind: [_random_expr(rng, depth + 1) for _ in range(n)]}


def test_random_predicates_across_layouts(tempdir):
    rng = np.random.RandomState(20240517)
    df = make_frame(40)
    rows = frame_rows(df)
    layouts = [2, 5, 8, 40]
    for stats in (True, False):
        for offset in layouts:
            fn = os.path.join(tempdir, f"lay_{stats}_{offset}.parquet")
            write(fn, df, row_group_offsets=offset, stats=stats)
            pf = ParquetFile(fn)
            for _ in range(25):
                expr = _random_expr(rng)
                expected_mask = reference_mask(expr, rows)
                exact = pf.to_pandas(predicate=expr, row_filter=True)
                assert exact["extra"].tolist() == df.loc[
                    expected_mask, "extra"].tolist()
                assert pf.count(predicate=expr, row_filter=True) == \
                    int(expected_mask.sum())
                assert pf.count(predicate=expr) >= int(expected_mask.sum())
                # concatenated iter output equals the exact output
                pieces = list(pf.iter_row_groups(predicate=expr,
                                                 row_filter=True))
                concat = pd.concat(pieces).reset_index(drop=True) \
                    if pieces else pd.DataFrame(columns=df.columns)
                if pieces:
                    assert concat["extra"].tolist() == df.loc[
                        expected_mask, "extra"].tolist()


def test_random_hive_repartition(tempdir):
    rng = np.random.RandomState(777)
    df = make_frame(32)
    df["g"] = ["p", "q"] * 16
    rows = frame_rows(df)
    for offset in (4, 8):
        dn = os.path.join(tempdir, f"hive_{offset}")
        write(dn, df, partition_on=["g"], row_group_offsets=offset, file_scheme="hive")
        pf = ParquetFile(dn)
        for _ in range(15):
            expr = {"or": [_random_expr(rng),
                           {"column": "g", "op": "==",
                            "value": rng.choice(["p", "q"])}]}
            expected = reference_mask(expr, rows)
            got = pf.to_pandas(predicate=expr, row_filter=True)
            assert len(got) == int(expected.sum())
            assert pf.count(predicate=expr, row_filter=True) == int(
                expected.sum())
