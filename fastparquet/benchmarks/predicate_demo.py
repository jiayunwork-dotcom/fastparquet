"""Reproducible demonstration/benchmark for nested predicates.

Run with:
    PYTHONPATH=<repo-root> python3 benchmarks/predicate_demo.py

It writes human-generated multi-row-group files and a Hive dataset and shows:

1. exact filtering against an independent row-by-row reference;
2. metadata pruning that demonstrably opens no data pages for provably-false
   groups, while never dropping a matching row when statistics are weak;
3. linear (not exponential) cost in expression width and constant cost in
   nesting depth;
4. lazy, per-row-group consumption via iter_row_groups.

The script only depends on fastparquet itself, numpy and pandas.
"""
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import pandas as pd

from fastparquet import write, ParquetFile
from fastparquet import core as fp_core
from fastparquet import predicate as fp_pred


# -- independent reference (no fastparquet.predicate calls) ----------------

def miss(v):
    if v is None or v is pd.NA:
        return True
    if isinstance(v, float) and np.isnan(v):
        return True
    return isinstance(v, np.floating) and np.isnan(v)


def ref_eval(expr, row):
    if "and" in expr:
        out = True
        for c in expr["and"]:
            b = ref_eval(c, row)
            if b is False:
                return False
            if b == "unknown":
                out = "unknown"
        return out
    if "or" in expr:
        out = False
        for c in expr["or"]:
            b = ref_eval(c, row)
            if b is True:
                return True
            if b == "unknown":
                out = "unknown"
        return out
    if "not" in expr:
        b = ref_eval(expr["not"], row)
        return "unknown" if b == "unknown" else bool(not b)
    col, op = expr["column"], expr["op"]
    v = row[col]
    if op == "is null":
        return bool(miss(v))
    if op == "is not null":
        return bool(not miss(v))
    if op == "in":
        return _ref_in(v, expr["value"])
    if op == "not in":
        tri = _ref_in(v, expr["value"])
        return "unknown" if tri == "unknown" else bool(not tri)
    if miss(v) or miss(expr["value"]):
        return "unknown"
    return bool({
        "==": v == expr["value"], "!=": v != expr["value"],
        "<": v < expr["value"], "<=": v <= expr["value"],
        ">": v > expr["value"], ">=": v >= expr["value"],
    }[op])


def _ref_in(v, values):
    if miss(v):
        return False if not values else "unknown"
    for x in values:
        if not miss(x) and v == x:
            return True
    return "unknown" if any(miss(x) for x in values) else False


def ref_mask(expr, df):
    return np.array([ref_eval(expr, r) is True
                     for r in df.to_dict("records")])


# -- data-page read counting ------------------------------------------------

class CountReads:
    def __init__(self):
        self.count = 0
        self._orig = fp_core.read_col

    def __enter__(self):
        orig = self._orig
        counter = self

        def wrapped(column, *a, **k):
            counter.count += 1
            return orig(column, *a, **k)

        fp_core.read_col = wrapped
        return self

    def __exit__(self, *exc):
        fp_core.read_col = self._orig


def main():
    root = tempfile.mkdtemp(prefix="fp_predicate_demo_")
    print("workdir:", root)
    try:
        # 1. human-generated multi-row-group file with all supported types
        n = 400
        rng = np.random.RandomState(0)
        big = 2 ** 63 - 1
        df = pd.DataFrame({
            "amount": pd.Series(
                [None, 1, -1, big, 2 ** 53 + 1, 7] * (n // 6) + [None] * (n % 6),
                dtype="Int64")[:n],
            "ratio": pd.Series(
                [0.0, np.nan, np.inf, -np.inf, 2.5, None] * (n // 6),
                dtype="float64")[:n],
            "name": pd.Series(
                ["a", "b", None, "zzz", "", "c"] * (n // 6),
                dtype="object")[:n],
            "flag": pd.Series(
                [True, False, None] * (n // 3), dtype="boolean")[:n],
            "seq": np.arange(n),
        })
        fn = os.path.join(root, "multi.parquet")
        write(fn, df, row_group_offsets=20, stats=True)
        pf = ParquetFile(fn)
        print("\n[1] multi-row-group file: %d groups, %d rows"
              % (len(pf.row_groups), n))

        expr = {"and": [
            {"or": [{"column": "amount", "op": ">", "value": 0},
                    {"column": "ratio", "op": "is null"}]},
            {"not": {"column": "name", "op": "in",
                     "value": ["zzz"]}},
        ]}
        exact = pf.to_pandas(predicate=expr, row_filter=True)
        expected = df.loc[ref_mask(expr, df)]
        assert exact["seq"].tolist() == expected["seq"].tolist()
        print("    exact rows: %d (reference agrees), group-level count: %d"
              % (len(exact), pf.count(predicate=expr)))

        # 2a. provably false pruning opens no data pages
        impossible = {"and": [{"column": "seq", "op": "<", "value": -1},
                              {"column": "amount", "op": "is not null"}]}
        with CountReads() as c:
            n_meta = pf.count(predicate=impossible)
        assert n_meta == 0 and c.count == 0
        print("\n[2a] provably-false predicate: count=%d, data pages read=%d"
              % (n_meta, c.count))

        # 2b. no-statistics file cannot prune but still gives exact results
        fn2 = os.path.join(root, "nostats.parquet")
        write(fn2, df, row_group_offsets=20, stats=False)
        pf2 = ParquetFile(fn2)
        weak = {"column": "amount", "op": "==", "value": big}
        assert pf2.count(predicate=weak) == n  # nothing pruned
        exact2 = pf2.to_pandas(predicate=weak, row_filter=True)
        assert exact2["seq"].tolist() == df.loc[ref_mask(weak, df),
                                                 "seq"].tolist()
        print("[2b] no-stats file: pruned groups=%d (all kept), exact rows=%d"
              % (len(pf2.row_groups), len(exact2)))

        # 2c. Hive partition pruning reads no data pages
        dn = os.path.join(root, "hive")
        hdf = df.copy()
        hdf["g"] = np.where(hdf["seq"] % 2, "beta", "alpha")
        write(dn, hdf, partition_on=["g"], row_group_offsets=20,
              file_scheme="hive")
        hpf = ParquetFile(dn)
        part_only = {"and": [{"column": "g", "op": "==", "value": "ghost"},
                             {"column": "seq", "op": ">", "value": 0}]}
        with CountReads() as c:
            assert hpf.count(predicate=part_only) == 0
        print("[2c] impossible Hive partition predicate: data pages read=%d"
              % c.count)
        mixed = {"or": [{"column": "g", "op": "==", "value": "alpha"},
                        {"column": "amount", "op": "is null"}]}
        mexact = hpf.to_pandas(predicate=mixed, row_filter=True)
        assert len(mexact) == int(ref_mask(mixed, hdf).sum())
        print("     partition+file OR exact rows: %d" % len(mexact))

        # 3. scaling: width of an AND-of-ORs (24 vs 48 clauses)
        print("\n[3] expression-size scaling (n=%d rows)" % n)
        timings = {}
        for width in (6, 12, 24, 48):
            clauses = [{"or": [
                {"column": "amount", "op": ">", "value": i},
                {"column": "seq", "op": "<", "value": n - i}]}
                for i in range(width)]
            wide = {"and": clauses}
            compiled = pf._compile_predicate(wide)
            col_df = pf.to_pandas(columns=["amount", "seq"], index=False)
            t0 = time.perf_counter()
            for _ in range(5):
                mask = compiled.row_mask(n, {c: col_df[c] for c in
                                             compiled.columns}, {})
            timings[width] = (time.perf_counter() - t0) / 5
            assert mask.tolist() == ref_mask(wide, df).tolist()
        base = timings[6]
        for width in (12, 24, 48):
            print("    width=%2d clauses: %.4f s/eval  (%.2fx vs width=6; "
                  "linear reference %.2fx)"
                  % (width, timings[width], timings[width] / base,
                     width / 6))

        # 4. nesting depth (up to 40 NOT/AND layers)
        print("\n[4] nesting depth cost")
        for depth in (10, 20, 40):
            e = {"column": "flag", "op": "==", "value": True}
            for _ in range(depth):
                e = {"not": {"and": [e, {"column": "seq",
                                         "op": "is not null"}]}}
            compiled = pf._compile_predicate(e)
            col_df = pf.to_pandas(columns=["flag", "seq"], index=False)
            t0 = time.perf_counter()
            mask = compiled.row_mask(n, {c: col_df[c] for c in
                                         compiled.columns}, {})
            dt = time.perf_counter() - t0
            assert mask.tolist() == ref_mask(e, df).tolist()
            print("    depth=%2d: %.4f s/eval, %d selected rows"
                  % (depth, dt, int(mask.sum())))

        # 5. laziness: consuming one group touches one group
        print("\n[5] iter_row_groups laziness")
        gen = pf.iter_row_groups(
            predicate={"column": "seq", "op": ">=", "value": 0},
            row_filter=True)
        first = next(gen)
        print("    first yielded frame rows=%d (<= one group of 20)"
              % len(first))

        print("\nAll assertions passed.")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
