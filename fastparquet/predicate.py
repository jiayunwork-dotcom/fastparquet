"""Nested predicate expressions with SQL three-valued logic.

This module implements the ``predicate`` keyword accepted by
:meth:`fastparquet.ParquetFile.to_pandas`,
:meth:`fastparquet.ParquetFile.iter_row_groups` and
:meth:`fastparquet.ParquetFile.count`.

A predicate is a (possibly nested) ``dict`` tree:

* comparison leaf: ``{"column": "amount", "op": ">", "value": 0}`` with
  operators ``==``, ``!=``, ``<``, ``<=``, ``>``, ``>=``, ``in`` and
  ``not in`` (the latter two take a ``list`` value);
* null-test leaf: ``{"column": "x", "op": "is null"}`` or
  ``{"column": "x", "op": "is not null"}`` (no ``value`` key);
* logic nodes: ``{"and": [children, ...]}``, ``{"or": [children, ...]}``
  (non-empty lists) and ``{"not": child}``.

Missing values are ``None``, ``float('nan')`` and :data:`pandas.NA`.
Comparisons against a missing operand evaluate to *unknown*; ``AND``,
``OR`` and ``NOT`` follow SQL three-valued logic and only rows where the
whole predicate evaluates to *true* are selected.

The same compiled tree is used for both read stages, so the two stages are
guaranteed to agree:

* :func:`row_group_can_be_false` only uses row-group metadata (statistics and
  Hive partition values) and returns ``True`` unless the metadata proves the
  predicate can never be true for any row in the group -- i.e. it never
  removes a row that the exact stage could keep;
* :func:`row_mask` evaluates the predicate on decoded columns exactly.
"""
import math

import numpy as np
import pandas as pd

from fastparquet import parquet_thrift
from fastparquet import converted_types
from fastparquet import encoding
from fastparquet.util import ensure_bytes, val_to_num

# tri-state outcomes, ordered so that bool(True)/bool(False) keep working
TRUE = 1
FALSE = 0
UNKNOWN = -1

_COMPARISONS = {"==", "!=", "<", "<=", ">", ">="}
_MEMBERSHIPS = {"in", "not in"}
_NULL_TESTS = {"is null", "is not null"}
_LEAF_OPS = _COMPARISONS | _MEMBERSHIPS | _NULL_TESTS
_LOGIC_KEYS = {"and", "or", "not"}

# supported physical column categories
T_INT64 = "int64"
T_FLOAT64 = "float64"
T_BOOL = "bool"
T_STRING = "string"

_INT64_MIN = -(2 ** 63)
_INT64_MAX = 2 ** 63 - 1


class PredicateError(ValueError):
    """Raised for malformed predicates or incompatible literal values."""


# ---------------------------------------------------------------------------
# literal / missing-value handling
# ---------------------------------------------------------------------------

def is_missing(value):
    """Return True for the unified set of missing scalar literals."""
    if value is None or value is pd.NA:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    # numpy float scalars (np.float64('nan')) are python float subclasses in
    # most builds, but guard the generic case explicitly
    if isinstance(value, np.floating) and np.isnan(value):
        return True
    return False


def _describe_position(position):
    return "at predicate expression %s" % position


def _check_signed_int64(value, column, position):
    # bool is a subclass of int and must be rejected for integer columns
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise PredicateError(
            "column %r %s requires a signed 64-bit integer literal, got %r "
            "(%s)" % (column, _describe_position(position), value,
                      type(value).__name__))
    if not (_INT64_MIN <= int(value) <= _INT64_MAX):
        raise PredicateError(
            "column %r %s got integer %r outside of the signed 64-bit range"
            % (column, _describe_position(position), value))


def _check_literal(value, column_type, column, position):
    """Validate one non-missing comparison literal against a column type."""
    if is_missing(value):
        return
    if column_type == T_INT64:
        _check_signed_int64(value, column, position)
    elif column_type == T_FLOAT64:
        if isinstance(value, bool) or not isinstance(
                value, (int, float, np.integer, np.floating)):
            raise PredicateError(
                "column %r %s requires an int or float literal, got %r (%s)"
                % (column, _describe_position(position), value,
                   type(value).__name__))
        if isinstance(value, (int, np.integer)) and not isinstance(
                value, bool):
            if not (_INT64_MIN <= int(value) <= _INT64_MAX):
                raise PredicateError(
                    "column %r %s got integer %r outside of the signed "
                    "64-bit range" % (column, _describe_position(position),
                                      value))
    elif column_type == T_BOOL:
        if not isinstance(value, (bool, np.bool_)):
            raise PredicateError(
                "column %r %s requires a bool literal, got %r (%s)"
                % (column, _describe_position(position), value,
                   type(value).__name__))
    else:  # T_STRING
        if not isinstance(value, str):
            raise PredicateError(
                "column %r %s requires a str literal, got %r (%s)"
                % (column, _describe_position(position), value,
                   type(value).__name__))


# ---------------------------------------------------------------------------
# expression parsing / validation
# ---------------------------------------------------------------------------

def compile_predicate(expr, column_types):
    """Validate ``expr`` and turn it into a reusable compiled tree.

    Parameters
    ----------
    expr: dict
        User supplied predicate tree. The input is never mutated.
    column_types: dict
        Mapping of every usable column name (file columns and Hive partition
        columns alike) to one of :data:`T_INT64`, :data:`T_FLOAT64`,
        :data:`T_BOOL`, :data:`T_STRING`.

    Returns
    -------
    CompiledPredicate
    """
    if not isinstance(expr, dict):
        raise PredicateError(
            "predicate must be a dict, got %s" % type(expr).__name__)
    tree, columns = _parse(expr, "$", column_types, set(), set())
    return CompiledPredicate(tree, frozenset(columns), expr)


def _parse(node, position, column_types, seen_ids, columns):
    if not isinstance(node, dict):
        raise PredicateError(
            "expected a dict expression node %s, got %r"
            % (_describe_position(position), node))
    if id(node) in seen_ids:
        raise PredicateError(
            "predicate contains a repeated (cyclic) reference %s"
            % _describe_position(position))
    seen_ids.add(id(node))

    keys = set(node)
    if not keys:
        raise PredicateError("empty predicate node %s"
                             % _describe_position(position))
    logic_present = keys & _LOGIC_KEYS
    if logic_present:
        if "column" in keys or len(logic_present) > 1:
            raise PredicateError(
                "predicate node %s must contain exactly one logic key, got %s"
                % (_describe_position(position), sorted(keys)))
        key = next(iter(logic_present))
        extra = keys - _LOGIC_KEYS
        if extra:
            raise PredicateError(
                "unexpected key(s) %s in %r logic node %s"
                % (sorted(extra), key, _describe_position(position)))
    elif "column" in keys:
        key = "column"
    else:
        raise PredicateError(
            "unknown predicate key(s) %s %s; expected one of 'column', "
            "'and', 'or', 'not'"
            % (sorted(keys), _describe_position(position)))

    if key in _LOGIC_KEYS:
        if key == "not":
            child = node[key]
            child_tree, columns = _parse(child, position + ".not",
                                         column_types, seen_ids, columns)
            tree = ("not", child_tree)
        else:
            children = node[key]
            if not isinstance(children, list) or not children:
                raise PredicateError(
                    "%r predicate %s needs a non-empty list of sub-expressions"
                    % (key, _describe_position(position)))
            parsed = []
            for i, child in enumerate(children):
                child_tree, columns = _parse(
                    child, "%s.%s[%d]" % (position, key, i), column_types,
                    seen_ids, columns)
                parsed.append(child_tree)
            tree = (key, tuple(parsed))
        # nodes stay marked for the whole walk: a dict reachable via two paths
        # is a repeated/cyclic reference and must be rejected
        return tree, columns

    # comparison / null-test leaf
    extra = keys - {"column", "op", "value"}
    if extra:
        raise PredicateError(
            "unexpected key(s) %s in leaf %s"
            % (sorted(extra), _describe_position(position)))
    if len(keys) < 2:
        raise PredicateError(
            "leaf %s must contain both 'column' and 'op'"
            % _describe_position(position))
    column = node.get("column")
    if not isinstance(column, str) or not column:
        raise PredicateError(
            "leaf %s needs a non-empty string 'column' name, got %r"
            % (_describe_position(position), column))
    try:
        column_type = column_types[column]
    except KeyError:
        raise PredicateError(
            "predicate references unknown column %r %s"
            % (column, _describe_position(position)))
    op = node.get("op")
    if op not in _LEAF_OPS:
        raise PredicateError(
            "unknown operator %r for column %r %s; supported operators are %s"
            % (op, column, _describe_position(position),
               sorted(_LEAF_OPS)))
    columns.add(column)

    if op in _NULL_TESTS:
        if "value" in node:
            raise PredicateError(
                "null test %r on column %r %s must not carry a 'value'"
                % (op, column, _describe_position(position)))
        tree = ("null", column, op, column_type)
    else:
        if "value" not in node:
            raise PredicateError(
                "comparison %r on column %r %s requires a 'value'"
                % (op, column, _describe_position(position)))
        value = node["value"]
        if op in _MEMBERSHIPS:
            if not isinstance(value, list):
                raise PredicateError(
                    "operator %r on column %r %s requires a list value, got "
                    "%s" % (op, column, _describe_position(position),
                            type(value).__name__))
            checked = []
            for j, element in enumerate(value):
                _check_literal(element, column_type, column,
                               "%s.value[%d]" % (position, j))
                # store missing-ness explicitly so NaN identity does not
                # depend on the container type
                checked.append((is_missing(element),
                                None if is_missing(element) else element))
            tree = ("member", column, op, column_type, tuple(checked))
        else:
            _check_literal(value, column_type, column, position + ".value")
            tree = ("cmp", column, op, column_type,
                    is_missing(value),
                    None if is_missing(value) else value)
    return tree, columns


class CompiledPredicate:
    """Immutable, reusable predicate compiled from a user ``dict``."""

    def __init__(self, tree, columns, original):
        self.tree = tree
        self.columns = columns
        # keep a reference so callers re-using the same dict always observe the
        # identical behaviour; the source itself is never modified
        self.original = original

    def file_columns(self, partition_columns):
        return [c for c in self.columns if c not in partition_columns]

    # -- exact, row-wise evaluation --------------------------------------

    def row_mask(self, nrows, column_values, partition_values):
        """Boolean mask of rows for which the predicate evaluates to *true*.

        Parameters
        ----------
        nrows: int
            Number of rows in the row group.
        column_values: dict
            Decoded pandas/numpy series for the referenced file columns.
        partition_values: dict
            Scalar Hive partition values for the row group; these are broadcast
            to length ``nrows`` here.
        """
        broadcast = {}
        for column, value in partition_values.items():
            broadcast[column] = _partition_series(value, nrows)
        tri = self._eval_exact(self.tree, nrows, column_values, broadcast)
        return tri == TRUE

    def _eval_exact(self, node, nrows, column_values, broadcast):
        kind = node[0]
        if kind == "and":
            result = np.ones(nrows, dtype=np.int8)  # all TRUE initially
            for child in node[1]:
                result = _tri_and(result, self._eval_exact(
                    child, nrows, column_values, broadcast))
            return result
        if kind == "or":
            result = np.zeros(nrows, dtype=np.int8)  # all FALSE initially
            for child in node[1]:
                result = _tri_or(result, self._eval_exact(
                    child, nrows, column_values, broadcast))
            return result
        if kind == "not":
            return _tri_not(self._eval_exact(
                node[1], nrows, column_values, broadcast))
        if kind == "null":
            _, column, op, column_type = node
            missing = _series_isna(broadcast[column] if column in broadcast
                                   else column_values[column])
            if op == "is null":
                return np.where(missing, TRUE, FALSE).astype(np.int8)
            return np.where(missing, FALSE, TRUE).astype(np.int8)
        if kind == "cmp":
            _, column, op, column_type, value_missing, value = node
            series = (broadcast[column] if column in broadcast
                      else column_values[column])
            missing = _series_isna(series)
            if value_missing:
                return np.full(nrows, UNKNOWN, dtype=np.int8)
            hits = _vector_compare(series, op, value)
            return _leaf_tri(missing, hits)
        # kind == "member"
        _, column, op, column_type, elements = node
        series = (broadcast[column] if column in broadcast
                  else column_values[column])
        missing = _series_isna(series)
        in_tri = _eval_in(series, missing, elements)
        return _tri_not(in_tri) if op == "not in" else in_tri

    # -- metadata-only pruning -------------------------------------------

    def can_row_group_be_true(self, stats_provider, partition_values):
        """Tri-state evaluation on row-group metadata.

        Returns the tri-state outcome of the whole predicate when only
        per-column min/max/null statistics and exact partition values are
        available. A row group is skipped only when this is :data:`FALSE`.
        """
        return self._eval_meta(self.tree, stats_provider, partition_values)

    def _eval_meta(self, node, stats_provider, partition_values):
        kind = node[0]
        if kind == "and":
            result = TRUE
            for child in node[1]:
                result = _scalar_tri_and(
                    result, self._eval_meta(child, stats_provider,
                                            partition_values))
            return result
        if kind == "or":
            result = FALSE
            for child in node[1]:
                result = _scalar_tri_or(
                    result, self._eval_meta(child, stats_provider,
                                            partition_values))
            return result
        if kind == "not":
            return _scalar_tri_not(self._eval_meta(
                node[1], stats_provider, partition_values))
        if kind == "null":
            _, column, op, column_type = node
            null_tri = stats_provider.null_tri(column, partition_values)
            if op == "is null":
                return null_tri
            return _scalar_tri_not(null_tri)
        if kind == "cmp":
            _, column, op, column_type, value_missing, value = node
            if column in partition_values:
                return _scalar_compare(partition_values[column], op, value,
                                       value_missing)
            if value_missing:
                return UNKNOWN
            return stats_provider.compare_tri(column, op, value)
        # kind == "member"
        _, column, op, column_type, elements = node
        if column in partition_values:
            tri = _scalar_in(partition_values[column], elements)
            return _scalar_tri_not(tri) if op == "not in" else tri
        tri = stats_provider.member_tri(column, elements)
        return _scalar_tri_not(tri) if op == "not in" else tri


# ---------------------------------------------------------------------------
# tri-state primitives
# ---------------------------------------------------------------------------

def _tri_not(a):
    out = a.copy()
    out[a == TRUE] = FALSE
    out[a == FALSE] = TRUE
    # UNKNOWN stays UNKNOWN
    return out


def _tri_and(a, b):
    # false AND x -> false; true AND x -> x; unknown AND unknown/true -> unknown
    out = np.where(a == FALSE, FALSE,
                   np.where(b == FALSE, FALSE,
                            np.where((a == TRUE) & (b == TRUE), TRUE,
                                     UNKNOWN))).astype(np.int8)
    return out


def _tri_or(a, b):
    out = np.where(a == TRUE, TRUE,
                   np.where(b == TRUE, TRUE,
                            np.where((a == FALSE) & (b == FALSE), FALSE,
                                     UNKNOWN))).astype(np.int8)
    return out


def _scalar_tri_not(a):
    if a == UNKNOWN:
        return UNKNOWN
    return not bool(a)


def _scalar_tri_and(a, b):
    if a == FALSE or b == FALSE:
        return FALSE
    if a == TRUE and b == TRUE:
        return TRUE
    return UNKNOWN


def _scalar_tri_or(a, b):
    if a == TRUE or b == TRUE:
        return TRUE
    if a == FALSE and b == FALSE:
        return FALSE
    return UNKNOWN


# ---------------------------------------------------------------------------
# exact (row-wise) leaf evaluation
# ---------------------------------------------------------------------------

def _series_isna(series):
    if isinstance(series, np.ndarray):
        if series.dtype.kind == "f":
            return np.isnan(series)
        if series.dtype.kind in ("i", "u", "b"):
            return np.zeros(len(series), dtype=bool)
        return pd.isna(np.asarray(series))
    return np.asarray(pd.isna(series))


def _vector_compare(series, op, value):
    """Boolean vector for a comparison; missing rows are handled separately."""
    if op == "==":
        raw = series == value
    elif op == "!=":
        raw = series != value
    elif op == "<":
        raw = series < value
    elif op == "<=":
        raw = series <= value
    elif op == ">":
        raw = series > value
    else:  # ">="
        raw = series >= value
    return _to_bool_array(raw)


def _to_bool_array(raw):
    """Convert a possibly nullable pandas boolean result to a numpy bool array.

    Missing positions (pandas.NA) become False here; the caller combines the
    result with an explicit missing mask so three-valued semantics survive.
    """
    if isinstance(raw, np.ndarray):
        if raw.dtype == bool:
            return raw
        return np.asarray(raw == True)  # noqa: E712
    if isinstance(raw, pd.Series):
        raw = raw.array
    # pandas extension arrays: fill NA with False without an ambiguous bool()
    na = pd.isna(raw)
    filled = raw.fillna(False) if hasattr(raw, "fillna") else raw
    out = np.asarray(filled, dtype=bool)
    out[np.asarray(na)] = False
    return out


def _leaf_tri(missing, hits):
    """Combine a missing-value mask with a non-missing comparison result."""
    result = np.full(len(missing), UNKNOWN, dtype=np.int8)
    result[~missing] = np.where(hits[~missing], TRUE, FALSE).astype(np.int8)
    return result


def _partition_series(value, nrows):
    """Broadcast one Hive partition scalar to a row-sized series."""
    if is_missing(value):
        return pd.Series([pd.NA] * nrows, dtype="object")
    if isinstance(value, (bool, np.bool_)):
        return pd.Series([bool(value)] * nrows, dtype="boolean")
    if isinstance(value, (int, np.integer)):
        return pd.Series([int(value)] * nrows, dtype="Int64")
    if isinstance(value, (float, np.floating)):
        return pd.Series([float(value)] * nrows, dtype="float64")
    return pd.Series([value] * nrows, dtype="object")


def _eval_in(series, missing, elements):
    """SQL ``x IN (elements)`` as a tri-state vector."""
    n = len(missing)
    if not elements:
        # an empty IN list is false on every row, even for a missing value
        return np.zeros(n, dtype=np.int8)
    has_missing_element = any(m for m, _ in elements)
    non_missing_values = [v for m, v in elements if not m]
    result = np.zeros(n, dtype=np.int8)  # FALSE until proven otherwise
    if non_missing_values:
        hits = _to_bool_array(series.isin(list(non_missing_values)))
        result[hits & ~missing] = TRUE
    if has_missing_element:
        # rows without an exact hit become unknown unless they are missing
        result[(result != TRUE) & ~missing] = UNKNOWN
        # a missing compared value with no hit is also unknown
        result[missing] = UNKNOWN
    else:
        result[missing] = UNKNOWN
    return result


def _scalar_compare(actual, op, value, value_missing):
    if is_missing(actual) or value_missing:
        return UNKNOWN
    if op == "==":
        return TRUE if actual == value else FALSE
    if op == "!=":
        # NaN on either side would already have been caught as missing
        return TRUE if actual != value else FALSE
    if op == "<":
        return TRUE if actual < value else FALSE
    if op == "<=":
        return TRUE if actual <= value else FALSE
    if op == ">":
        return TRUE if actual > value else FALSE
    return TRUE if actual >= value else FALSE


def _scalar_in(actual, elements):
    if is_missing(actual):
        # empty IN list is always false even for a missing compared value
        if not elements:
            return FALSE
        return UNKNOWN
    for element_missing, element in elements:
        if not element_missing and actual == element:
            return TRUE
    if any(m for m, _ in elements):
        return UNKNOWN
    return FALSE


# ---------------------------------------------------------------------------
# column typing
# ---------------------------------------------------------------------------

def file_column_type(schema_element):
    """Map a parquet schema element to a supported predicate column type.

    Raises :class:`PredicateError` for physical types not covered by the
    predicate feature.
    """
    se = schema_element
    physical = se.type
    if physical == parquet_thrift.Type.BOOLEAN:
        return T_BOOL
    if physical == parquet_thrift.Type.DOUBLE:
        return T_FLOAT64
    if physical in (parquet_thrift.Type.INT32, parquet_thrift.Type.INT64):
        ct = se.converted_type
        if ct is None:
            # plain INT32/INT64 parquet columns decode to signed int dtypes
            return T_INT64
        signed = {
            parquet_thrift.ConvertedType.INT_8,
            parquet_thrift.ConvertedType.INT_16,
            parquet_thrift.ConvertedType.INT_32,
            parquet_thrift.ConvertedType.INT_64,
        }
        if ct in signed:
            return T_INT64
        raise PredicateError(
            "predicates only support signed 64-bit integer columns; this "
            "column uses converted type %r"
            % (parquet_thrift.ConvertedType._VALUES_TO_NAMES.get(ct, ct),))
    if physical == parquet_thrift.Type.BYTE_ARRAY:
        if se.converted_type == parquet_thrift.ConvertedType.UTF8:
            return T_STRING
        raise PredicateError(
            "predicates only support UTF-8 string BYTE_ARRAY columns")
    raise PredicateError(
        "predicates do not support parquet physical type %r for this column"
        % (parquet_thrift.Type._VALUES_TO_NAMES.get(physical, physical),))


def partition_column_type(values):
    """Map the known Hive partition values to a supported column type."""
    non_missing = [v for v in values if not is_missing(v)]
    if not non_missing:
        raise PredicateError(
            "cannot determine a type for an entirely missing partition "
            "column used in a predicate")
    if all(isinstance(v, (bool, np.bool_)) for v in non_missing):
        return T_BOOL
    if all(isinstance(v, (int, np.integer)) and not isinstance(v, bool)
           for v in non_missing):
        return T_INT64
    if all(isinstance(v, (int, float, np.integer, np.floating))
           and not isinstance(v, bool) for v in non_missing):
        return T_FLOAT64
    if all(isinstance(v, str) for v in non_missing):
        return T_STRING
    raise PredicateError(
        "partition column values %r are not one of the supported predicate "
        "types (signed int64, float64, bool, UTF-8 string)" % (non_missing,))


# ---------------------------------------------------------------------------
# row-group metadata statistics
# ---------------------------------------------------------------------------

class RowGroupStats:
    """Lazily decoded, cached min/max/null statistics for one row group."""

    def __init__(self, pf, rg):
        self.pf = pf
        self.rg = rg
        self._chunks = None
        self._cache = {}

    def _chunk(self, column):
        if self._chunks is None:
            self._chunks = {
                ".".join(c.meta_data.path_in_schema): c
                for c in self.rg.columns
            }
        return self._chunks.get(column)

    def _decoded(self, column):
        if column in self._cache:
            return self._cache[column]
        chunk = self._chunk(column)
        info = None
        if chunk is not None:
            info = self._decode_chunk(column, chunk)
        self._cache[column] = info
        return info

    def _decode_chunk(self, column, chunk):
        md = chunk.meta_data
        stats = md.statistics
        if stats is None:
            return None
        try:
            se = self.pf.schema.schema_element(column)
        except KeyError:
            return None
        null_count = stats.null_count
        vmin = vmax = None
        raw_max = stats.max if stats.max is not None else stats.max_value
        raw_min = stats.min if stats.min is not None else stats.min_value
        try:
            if raw_max is not None:
                vmax = _decode_stat(raw_max, md, se)
            if raw_min is not None:
                vmin = _decode_stat(raw_min, md, se)
        except Exception:
            # un-decodable statistics must not cause missed rows
            return None
        if (vmax is not None and isinstance(vmax, float)
                and math.isnan(vmax)) or (
                vmin is not None and isinstance(vmin, float)
                and math.isnan(vmin)):
            # NaN in the range itself makes range reasoning unsound
            vmin = vmax = None
        return {"min": vmin, "max": vmax, "null_count": null_count,
                "num_values": md.num_values}

    def null_tri(self, column, partition_values):
        if column in partition_values:
            return TRUE if is_missing(partition_values[column]) else FALSE
        info = self._decoded(column)
        if info is None or info["null_count"] is None:
            return UNKNOWN
        if info["null_count"] == 0:
            return FALSE
        if info["null_count"] >= info["num_values"]:
            return TRUE
        return UNKNOWN

    def _range(self, column):
        info = self._decoded(column)
        if info is None or info["min"] is None or info["max"] is None:
            return None
        return info

    def compare_tri(self, column, op, value):
        info = self._range(column)
        if info is None:
            return UNKNOWN
        vmin, vmax = info["min"], info["max"]
        null_count = info["null_count"]
        all_null = (null_count is not None
                    and null_count >= info["num_values"])
        if all_null:
            return FALSE
        false_proven = _range_proves_false(op, value, vmin, vmax)
        if false_proven:
            # the non-null values cannot satisfy the comparison. Missing rows
            # evaluate to unknown, so the group may still match unless the
            # column is known to have no nulls.
            if null_count is None or null_count > 0:
                return UNKNOWN
            return FALSE
        if vmin == vmax and not _is_nan(vmin):
            # constant non-null value; missing rows are unknown,
            # so a definite FALSE needs a known zero null_count
            definite = _scalar_compare(vmin, op, value, False)
            if definite == FALSE and (null_count is None or null_count > 0):
                return UNKNOWN
            return definite
        return UNKNOWN

    def member_tri(self, column, elements):
        if not elements:
            # an empty IN list is false on every row, even for a missing
            # value, independently of any statistics
            return FALSE
        info = self._range(column)
        if info is None:
            return UNKNOWN
        vmin, vmax = info["min"], info["max"]
        null_count = info["null_count"]
        all_null = (null_count is not None
                    and null_count >= info["num_values"])
        non_missing = [v for m, v in elements if not m]
        has_missing_element = any(m for m, _ in elements)

        if all_null:
            # every compared value is missing, so the result is unknown unless
            # the list is empty (handled above) or a missing list element also
            # makes it unknown; in no case can it be true
            return UNKNOWN
        if vmin == vmax and not _is_nan(vmin):
            # constant non-null value; missing rows stay unknown, so a definite
            # FALSE can only be used when the column has a known zero null count
            tri = _scalar_in(vmin, elements)
            if tri == FALSE and (null_count is None or null_count > 0):
                return UNKNOWN
            return tri
        if non_missing and _range_disjoint_from_values(vmin, vmax,
                                                       non_missing):
            # no non-missing row can hit any non-missing list element; rows may
            # still be unknown when nulls or missing list elements exist
            if has_missing_element or null_count is None or null_count > 0:
                return UNKNOWN
            return FALSE
        return UNKNOWN


def _is_nan(value):
    return isinstance(value, float) and math.isnan(value)


def _decode_stat(raw, md, se):
    data = ensure_bytes(raw)
    out = encoding.read_plain(data, md.type, 1, stat=True)
    value = out[0] if hasattr(out, "__len__") else out
    if se.converted_type is not None or se.logicalType is not None:
        converted = converted_types.convert(out, se)
        value = converted[0] if hasattr(converted, "__len__") else converted
    if isinstance(value, np.generic):
        value = value.item()
    return value


def _range_proves_false(op, value, vmin, vmax):
    """True if no value within [vmin, vmax] can satisfy the comparison."""
    if op == "==":
        return value < vmin or value > vmax
    if op == "!=":
        # only provable on a constant column, handled by the caller
        return False
    if op == ">":
        return value >= vmax
    if op == ">=":
        return value > vmax
    if op == "<":
        return value <= vmin
    if op == "<=":
        return value < vmin
    return False


def _range_disjoint_from_values(vmin, vmax, values):
    """True if none of the (non-missing) list values lie in the range."""
    for value in values:
        if vmin <= value <= vmax:
            return False
    return True


# ---------------------------------------------------------------------------
# helpers shared with the ParquetFile methods
# ---------------------------------------------------------------------------

def partition_value_for_row_group(pf, rg, column):
    """Extract one Hive partition scalar from a row group's file path."""
    path = rg.columns[0].file_path
    if not path:
        return None
    for part in path.split("/"):
        if "=" not in part:
            continue
        key, raw = part.split("=", 1)
        if key == column:
            meta = pf.partition_meta.get(column)
            return val_to_num(raw, meta=meta)
    return None


def row_group_may_match(compiled, stats_provider, partition_values):
    """Return True unless metadata proves the predicate cannot be true.

    A ``False`` result means the whole row group can be skipped without
    reading any data page.
    """
    return compiled.can_row_group_be_true(stats_provider,
                                          partition_values) != FALSE
