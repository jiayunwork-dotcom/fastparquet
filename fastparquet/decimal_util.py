"""Exact decimal (Parquet DECIMAL) support.

This module is deliberately independent of the caller's ``decimal`` context:
every conversion between a Python :class:`decimal.Decimal` and the integer
"unscaled" representation used by Parquet is done with integer arithmetic and
:meth:`decimal.Decimal.as_tuple`.  No arithmetic ``Decimal`` operations are
performed, so precision, rounding mode and traps of the ambient context can
neither change the result nor be triggered.

Parquet stores DECIMAL values as a signed two's-complement integer in one of
four physical types:

======  =========================================
INT32   little-endian, 4 bytes
INT64   little-endian, 8 bytes
BYTE_ARRAY            big-endian, variable length
FIXED_LEN_BYTE_ARRAY  big-endian, ``type_length`` bytes
======  =========================================

The value represented is ``unscaled * 10 ** -scale``.
"""

import math
import numbers
from decimal import Decimal

import numpy as np

from fastparquet import parquet_thrift
from fastparquet.cencoding import ThriftObject
from fastparquet.speedups import pack_byte_array

MAX_PRECISION = 38

#: Marker stored on a *private copy* of a schema element while writing.  It is
#: never serialised (thrift only writes integer field ids 1..13).
EXACT_DECIMAL_KEY = "_fastparquet_exact_decimal"

DECIMAL_PHYSICAL_TYPES = (
    parquet_thrift.Type.INT32,
    parquet_thrift.Type.INT64,
    parquet_thrift.Type.BYTE_ARRAY,
    parquet_thrift.Type.FIXED_LEN_BYTE_ARRAY,
)


def _is_bool(value):
    return isinstance(value, (bool, np.bool_))


def is_missing(value):
    """Whether a user value means "no value" for an exact decimal column.

    Only ``None``, ``pd.NA`` and float NaN are missing.  ``Decimal('NaN')`` is
    *not* a missing value, it is an invalid input.
    """
    if value is None:
        return True
    # Avoid importing pandas here; pd.NA is a singleton with this repr/type
    # but an identity check against the pandas object is cheap and explicit.
    try:
        import pandas as pd
        if value is pd.NA:
            return True
    except ImportError:  # pragma: no cover
        pass
    if isinstance(value, (float, np.floating)):
        return bool(math.isnan(value))
    return False


def value_to_unscaled(value, precision, scale):
    """Convert one non-missing user value to its exact unscaled integer.

    Raises ``ValueError`` if the value is of an unsupported type, is not
    finite, requires rounding at ``scale``, or does not fit ``precision``.
    """
    if _is_bool(value):
        raise ValueError("bool values are not accepted")
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("non-finite Decimal (%s) is not accepted" % value)
        sign, digits, exponent = value.as_tuple()
        coeff = 0
        for digit in digits:
            coeff = coeff * 10 + digit
        shift = exponent + scale
        if shift >= 0:
            unscaled = coeff * 10 ** shift
        else:
            divisor = 10 ** (-shift)
            quotient, remainder = divmod(coeff, divisor)
            if remainder:
                raise ValueError(
                    "value %s cannot be represented exactly at scale %d"
                    % (value, scale))
            unscaled = quotient
        if sign:
            unscaled = -unscaled
    elif isinstance(value, numbers.Integral) and not _is_bool(value):
        unscaled = int(value) * 10 ** scale
    elif isinstance(value, numbers.Real) and not _is_bool(value):
        # excludes Decimal (handled above); any finite float is rejected
        raise ValueError(
            "float value %r is not accepted; pass Decimal or int" % (value,))
    else:
        raise ValueError(
            "unsupported value %r of type %s" % (value, type(value).__name__))

    if abs(unscaled) >= 10 ** precision:
        raise ValueError(
            "value %r does not fit precision %d (unscaled %d)"
            % (value, precision, unscaled))
    return unscaled


def unscaled_to_decimal(unscaled, scale):
    """Build a Decimal with exactly ``-scale`` as its exponent.

    Uses the (sign, digits, exponent) tuple constructor, which is unaffected
    by the current decimal context.
    """
    unscaled = int(unscaled)
    sign = 0 if unscaled >= 0 else 1
    digits = tuple(int(c) for c in str(abs(unscaled)))
    return Decimal((sign, digits, -scale))


def bytes_to_unscaled(raw):
    """Decode a Parquet DECIMAL byte string (big-endian two's complement)."""
    return int.from_bytes(bytes(raw), byteorder="big", signed=True)


def unscaled_to_bytes(unscaled, length):
    """Encode an unscaled integer as a fixed-size big-endian signed value."""
    unscaled = int(unscaled)
    try:
        return unscaled.to_bytes(length, byteorder="big", signed=True)
    except OverflowError:
        raise ValueError(
            "unscaled value %d does not fit in %d bytes" % (unscaled, length))


def unscaled_to_var_bytes(unscaled):
    """Minimal-length big-endian two's complement encoding (BYTE_ARRAY)."""
    unscaled = int(unscaled)
    length = 1
    while not (-(1 << (8 * length - 1)) <= unscaled < (1 << (8 * length - 1))):
        length += 1
    return unscaled.to_bytes(length, byteorder="big", signed=True)


def physical_for_precision(precision):
    """Choose a physical representation able to hold every value of precision.

    Returns ``(parquet_type, type_length_or_None)``.
    """
    if precision <= 9:
        return parquet_thrift.Type.INT32, None
    if precision <= 18:
        return parquet_thrift.Type.INT64, None
    # Smallest signed byte length able to hold 10**precision - 1.
    length = 1
    while 10 ** precision - 1 > (1 << (8 * length - 1)) - 1:
        length += 1
    return parquet_thrift.Type.FIXED_LEN_BYTE_ARRAY, length


def _max_precision_for_physical(physical, type_length):
    """Return the maximum precision the physical representation can hold."""
    if physical == parquet_thrift.Type.INT32:
        limit = (1 << 31) - 1
    elif physical == parquet_thrift.Type.INT64:
        limit = (1 << 63) - 1
    elif physical == parquet_thrift.Type.FIXED_LEN_BYTE_ARRAY:
        if not isinstance(type_length, numbers.Integral) or type_length < 1:
            return 0
        limit = (1 << (8 * int(type_length) - 1)) - 1
    else:
        # BYTE_ARRAY has no intrinsic limit; the format-wide cap is 38.
        return MAX_PRECISION
    # Largest p with 10**p - 1 <= limit.
    p = 0
    while p < MAX_PRECISION and 10 ** (p + 1) - 1 <= limit:
        p += 1
    return p


def _check_precision_scale(precision, scale, name):
    if _is_bool(precision) or not isinstance(precision, numbers.Integral):
        raise ValueError(
            "Column %r: decimal precision must be an integer, got %r"
            % (name, precision))
    if _is_bool(scale) or not isinstance(scale, numbers.Integral):
        raise ValueError(
            "Column %r: decimal scale must be an integer, got %r"
            % (name, scale))
    precision = int(precision)
    scale = int(scale)
    if not 1 <= precision <= MAX_PRECISION:
        raise ValueError(
            "Column %r: decimal precision must be between 1 and %d, got %d"
            % (name, MAX_PRECISION, precision))
    if not 0 <= scale <= precision:
        raise ValueError(
            "Column %r: decimal scale must be between 0 and precision (%d), "
            "got %d" % (name, precision, scale))
    return precision, scale


def validate_declaration(data, decimal_columns):
    """Validate and normalise the ``decimal_columns`` write argument.

    Returns an ordered dict ``{name: (precision, scale)}``.  Column names must
    occur exactly once in ``data``.
    """
    if decimal_columns is None:
        return {}
    if not isinstance(decimal_columns, dict):
        raise ValueError(
            "decimal_columns must be a dict of {column: (precision, scale)}")
    out = {}
    columns = list(data.columns)
    for name, spec in decimal_columns.items():
        occurrences = sum(1 for c in columns if c == name)
        if occurrences == 0:
            raise ValueError(
                "Column %r given in decimal_columns is not present in the "
                "DataFrame" % (name,))
        if occurrences > 1:
            raise ValueError(
                "Column %r given in decimal_columns appears more than once "
                "in the DataFrame" % (name,))
        if not isinstance(spec, (tuple, list)) or len(spec) != 2:
            raise ValueError(
                "Column %r: decimal declaration must be a (precision, scale) "
                "pair, got %r" % (name, spec))
        precision, scale = _check_precision_scale(spec[0], spec[1], name)
        out[name] = (precision, scale)
    return out


def _logical_decimal(logical):
    """Extract the DecimalType from a LogicalType (ThriftObject or dict)."""
    if logical is None:
        return None
    if isinstance(logical, dict):
        # LogicalType.DECIMAL is thrift field 5
        return logical.get(5)
    return getattr(logical, "DECIMAL", None)


def schema_decimal_info(se, name="?"):
    """Return ``(precision, scale)`` for a DECIMAL schema element.

    Returns ``None`` if the element is not a DECIMAL.  Both the deprecated
    ``converted_type``/``scale``/``precision`` fields and the modern
    ``LogicalType.DECIMAL`` annotation are understood.  If both are present
    they must agree.  Invalid schemas raise ``ValueError`` mentioning the
    column name.
    """
    old = None
    if se.converted_type == parquet_thrift.ConvertedType.DECIMAL:
        if se.scale is None or se.precision is None:
            raise ValueError(
                "Column %r: DECIMAL converted_type without scale/precision"
                % (name,))
        old = (se.precision, se.scale)

    new = None
    decimal_type = _logical_decimal(getattr(se, "logicalType", None))
    if decimal_type is not None:
        scale = (decimal_type.get(1) if isinstance(decimal_type, dict)
                 else decimal_type.scale)
        precision = (decimal_type.get(2) if isinstance(decimal_type, dict)
                     else decimal_type.precision)
        if scale is None or precision is None:
            raise ValueError(
                "Column %r: LogicalType.DECIMAL without scale/precision"
                % (name,))
        new = (precision, scale)

    if old is None and new is None:
        return None

    if old is not None and new is not None and tuple(old) != tuple(new):
        raise ValueError(
            "Column %r: conflicting DECIMAL annotations "
            "(converted_type precision/scale %s, LogicalType %s)"
            % (name, tuple(old), tuple(new)))

    precision, scale = old if old is not None else new
    precision, scale = _check_precision_scale(precision, scale, name)

    physical = se.type
    if physical not in DECIMAL_PHYSICAL_TYPES:
        raise ValueError(
            "Column %r: DECIMAL is not valid on physical type %r"
            % (name, parquet_thrift.Type._VALUES_TO_NAMES.get(physical, physical)))
    type_length = se.type_length
    max_precision = _max_precision_for_physical(physical, type_length)
    if precision > max_precision:
        raise ValueError(
            "Column %r: precision %d does not fit physical type %s%s"
            % (name, precision,
               parquet_thrift.Type._VALUES_TO_NAMES.get(physical, physical),
               "" if physical != parquet_thrift.Type.FIXED_LEN_BYTE_ARRAY
               else " of length %r" % (type_length,)))
    return precision, scale


def validate_column_values(series, name, precision, scale, allow_missing=True):
    """Validate every value of ``series`` (no conversion is retained).

    Raises ``ValueError`` naming the column and the zero-based physical row
    for the first bad value.
    """
    for row, value in enumerate(series):
        if is_missing(value):
            if not allow_missing:
                raise ValueError(
                    "Column %r: value at row %d is missing but the column is "
                    "not allowed to contain nulls" % (name, row))
            continue
        try:
            value_to_unscaled(value, precision, scale)
        except ValueError as exc:
            raise ValueError(
                "Column %r: value at row %d: %s" % (name, row, exc)) from exc


def stats_bytes(unscaled, se):
    """PLAIN encoding of a single statistic value (no length prefix)."""
    physical = se.type
    if physical == parquet_thrift.Type.INT32:
        return int(unscaled).to_bytes(4, byteorder="little", signed=True)
    if physical == parquet_thrift.Type.INT64:
        return int(unscaled).to_bytes(8, byteorder="little", signed=True)
    if physical == parquet_thrift.Type.FIXED_LEN_BYTE_ARRAY:
        return unscaled_to_bytes(unscaled, se.type_length)
    if physical == parquet_thrift.Type.BYTE_ARRAY:
        return unscaled_to_var_bytes(unscaled)
    raise ValueError("unsupported DECIMAL physical type %r" % (physical,))


def decode_values(values, se, scale):
    """Convert primitive read values (ints or byte strings) to Decimals."""
    physical = se.type
    n = len(values)
    out = np.empty(n, dtype=object)
    if physical in (parquet_thrift.Type.INT32, parquet_thrift.Type.INT64):
        out[:] = [unscaled_to_decimal(int(v), scale) for v in values]
    elif physical == parquet_thrift.Type.FIXED_LEN_BYTE_ARRAY:
        # Read the raw buffer: individual numpy ``S<n>`` scalars drop trailing
        # NUL bytes, corrupting negative values (and any value ending in 0x00).
        width = int(se.type_length)
        raw = np.asarray(values).tobytes()
        out[:] = [
            unscaled_to_decimal(
                bytes_to_unscaled(raw[i * width:(i + 1) * width]), scale)
            for i in range(n)
        ]
    elif physical == parquet_thrift.Type.BYTE_ARRAY:
        out[:] = [unscaled_to_decimal(bytes_to_unscaled(v), scale)
                  for v in values]
    else:  # pragma: no cover - guarded by schema validation
        raise ValueError("unsupported DECIMAL physical type %r" % (physical,))
    return out


def decode_stat_value(value, se, scale):
    """Convert one statistics value (as returned by ``api.statistics``)."""
    if value is None:
        return None
    physical = se.type
    if physical in (parquet_thrift.Type.INT32, parquet_thrift.Type.INT64):
        return unscaled_to_decimal(int(value), scale)
    raw = bytes(value)
    if physical == parquet_thrift.Type.FIXED_LEN_BYTE_ARRAY:
        # numpy fixed-width scalars drop trailing (low-order) NUL bytes; the
        # value is big-endian, so restore them at the end.
        width = int(se.type_length)
        raw = raw.ljust(width, b"\x00")
    return unscaled_to_decimal(bytes_to_unscaled(raw), scale)


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------

def make_schema_element(name, precision, scale, repetition_type):
    """Build a standard DECIMAL ``SchemaElement``.

    The physical type is the smallest of INT32/INT64/FIXED_LEN_BYTE_ARRAY that
    can hold the whole declared range.  Both the modern ``LogicalType`` and
    the deprecated ``converted_type`` annotation are written, with matching
    precision/scale.
    """
    physical, type_length = physical_for_precision(precision)
    logical = ThriftObject.from_fields(
        "LogicalType",
        DECIMAL=ThriftObject.from_fields(
            "DecimalType", precision=int(precision), scale=int(scale),
            i32=True),
    )
    se = ThriftObject.from_fields(
        "SchemaElement",
        name=name,
        type=physical,
        type_length=type_length,
        repetition_type=repetition_type,
        converted_type=parquet_thrift.ConvertedType.DECIMAL,
        scale=int(scale),
        precision=int(precision),
        logicalType=logical,
        i32=True,
    )
    se[EXACT_DECIMAL_KEY] = True
    return se


def prepare_series(series, precision, scale, name, physical=None,
                   type_length=None):
    """Validate and pre-encode one whole column.

    ``physical``/``type_length`` default to the representation chosen by
    :func:`physical_for_precision`; appended files pass their existing
    physical type so the encoding matches the file.

    Returns ``(encoded, unscaled, null_mask)`` where ``encoded`` is an object
    ndarray of PLAIN-ready physical values (``None`` for missing), ``unscaled``
    is an object ndarray of Python ints (``None`` for missing) and
    ``null_mask`` is a boolean ndarray.

    All validation happens here, *before* anything is written, so invalid data
    is always rejected prior to touching the target file.  The Series is never
    modified.
    """
    if _is_bool(precision) or _is_bool(scale):  # pragma: no cover - validated
        raise ValueError("bad precision/scale for column %r" % (name,))
    if physical is None:
        physical, type_length = physical_for_precision(precision)

    if hasattr(series, "to_numpy"):
        raw = series.to_numpy(dtype=object, copy=True)
    else:
        raw = np.asarray(list(series), dtype=object)
    encoded = np.empty(len(raw), dtype=object)
    unscaled_out = np.empty(len(raw), dtype=object)
    null_mask = np.zeros(len(raw), dtype=bool)
    for row, value in enumerate(raw):
        if is_missing(value):
            encoded[row] = None
            unscaled_out[row] = None
            null_mask[row] = True
            continue
        try:
            unscaled = value_to_unscaled(value, precision, scale)
        except ValueError as exc:
            raise ValueError(
                "Column %r: value at row %d: %s" % (name, row, exc)) from exc
        unscaled_out[row] = unscaled
        if physical == parquet_thrift.Type.INT32:
            if not (-(1 << 31) <= unscaled < (1 << 31)):
                raise ValueError(
                    "Column %r: value at row %d does not fit INT32"
                    % (name, row))
            encoded[row] = np.int32(unscaled)
        elif physical == parquet_thrift.Type.INT64:
            if not (-(1 << 63) <= unscaled < (1 << 63)):
                raise ValueError(
                    "Column %r: value at row %d does not fit INT64"
                    % (name, row))
            encoded[row] = np.int64(unscaled)
        elif physical == parquet_thrift.Type.FIXED_LEN_BYTE_ARRAY:
            try:
                encoded[row] = unscaled_to_bytes(unscaled, type_length)
            except ValueError as exc:
                raise ValueError(
                    "Column %r: value at row %d: %s"
                    % (name, row, exc)) from exc
        else:
            encoded[row] = unscaled_to_var_bytes(unscaled)
    return encoded, unscaled_out, null_mask


def encode_plain_exact(values, se, name):
    """PLAIN encoding for a null-free array of prepared values."""
    if len(values) == 0:
        return b""
    physical = se.type
    if physical == parquet_thrift.Type.BYTE_ARRAY:
        # prepared values are minimal big-endian byte strings
        return pack_byte_array(list(values))
    if physical == parquet_thrift.Type.FIXED_LEN_BYTE_ARRAY:
        # prepared values are already exactly type_length big-endian bytes
        return b"".join(bytes(v) for v in values)
    if physical == parquet_thrift.Type.INT32:
        return np.asarray(values, dtype=np.int32).tobytes()
    if physical == parquet_thrift.Type.INT64:
        return np.asarray(values, dtype=np.int64).tobytes()
    raise ValueError(
        "Column %r: unsupported DECIMAL physical type %r" % (name, physical))


def exact_stats_bytes(min_unscaled, max_unscaled, se):
    """Return ``(min_bytes, max_bytes)`` PLAIN encodings for statistics."""
    return (stats_bytes(min_unscaled, se), stats_bytes(max_unscaled, se))


def is_exact_explicit(se):
    """True when ``se`` was created by :func:`make_schema_element`."""
    try:
        return bool(se[EXACT_DECIMAL_KEY])
    except (TypeError, KeyError):
        return False


def find_schema_element(schema, name):
    """Find the schema element with exactly ``name`` in a schema list."""
    found = [se for se in schema if getattr(se, "name", None) == name]
    if len(found) != 1:
        return None
    return found[0]

