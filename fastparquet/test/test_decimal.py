"""Exact DECIMAL write/read support (opt-in).

Every file used here is constructed independently of the code under test: the
pyarrow vectors use pyarrow's writer, and the four physical representations
that pyarrow does not directly produce (notably BYTE_ARRAY) are written by a
small, self-contained builder in this module.
"""
import decimal
import io
import os
import struct
from decimal import Decimal, localcontext, Inexact, Rounded

import numpy as np
import pandas as pd
import pytest

import fastparquet
from fastparquet import ParquetFile, write, parquet_thrift as pt
from fastparquet.cencoding import ThriftObject, from_buffer
from fastparquet import writer
from fastparquet.test.util import tempdir  # noqa: F401

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

D = Decimal


# ---------------------------------------------------------------------------
# independent file builders
# ---------------------------------------------------------------------------

def _varint(n):
    out = bytearray()
    while True:
        b = n & 0x7F
        n >>= 7
        if n:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _rle_bitpacked_bits(bits):
    nbytes = (len(bits) + 7) // 8
    packed = bytearray(nbytes)
    for i, b in enumerate(bits):
        if b:
            packed[i // 8] |= 1 << (i % 8)
    return _varint((nbytes << 1) | 1) + bytes(packed)


def _unscaled(value, scale):
    sign, digits, exponent = value.as_tuple()
    coeff = 0
    for d in digits:
        coeff = coeff * 10 + d
    shift = exponent + scale
    if shift >= 0:
        out = coeff * 10 ** shift
    else:
        div = 10 ** (-shift)
        q, r = divmod(coeff, div)
        if r:
            raise ValueError("not exact")
        out = q
    return -out if sign else out


def _plain_bytes(u, physical, type_length):
    if physical == pt.Type.INT32:
        return struct.pack('<i', u)
    if physical == pt.Type.INT64:
        return struct.pack('<q', u)
    if physical == pt.Type.FIXED_LEN_BYTE_ARRAY:
        return u.to_bytes(type_length, 'big', signed=True)
    n = 1
    while not (-(1 << (8 * n - 1)) <= u < (1 << (8 * n - 1))):
        n += 1
    b = u.to_bytes(n, 'big', signed=True)
    return struct.pack('<I', len(b)) + b


def _stat_bytes(u, physical, type_length):
    if physical == pt.Type.INT32:
        return struct.pack('<i', u)
    if physical == pt.Type.INT64:
        return struct.pack('<q', u)
    if physical == pt.Type.FIXED_LEN_BYTE_ARRAY:
        return u.to_bytes(type_length, 'big', signed=True)
    n = 1
    while not (-(1 << (8 * n - 1)) <= u < (1 << (8 * n - 1))):
        n += 1
    return u.to_bytes(n, 'big', signed=True)


def build_parquet(path, values, precision, scale, physical, type_length=None,
                  annotation='both', page_version=1, name='amt'):
    """Write a minimal standard Parquet DECIMAL file, single PLAIN page."""
    n = len(values)
    bits = [1 if v is not None else 0 for v in values]
    defs = _rle_bitpacked_bits(bits)
    data = b''.join(_plain_bytes(_unscaled(v, scale), physical, type_length)
                    for v in values if v is not None)

    f = open(path, 'wb')
    f.write(b'PAR1')
    if page_version == 1:
        body = struct.pack('<I', len(defs)) + defs + data
        ph = pt.PageHeader(
            type=pt.PageType.DATA_PAGE,
            uncompressed_page_size=len(body), compressed_page_size=len(body),
            data_page_header=pt.DataPageHeader(
                num_values=n, encoding=pt.Encoding.PLAIN,
                definition_level_encoding=pt.Encoding.RLE,
                repetition_level_encoding=pt.Encoding.BIT_PACKED, i32=True),
            i32=True)
        offset = f.tell()
        writer.write_thrift(f, ph)
        f.write(body)
    else:
        nulls = n - sum(bits)
        body = defs + data
        ph = pt.PageHeader(
            type=pt.PageType.DATA_PAGE_V2,
            uncompressed_page_size=len(body), compressed_page_size=len(body),
            data_page_header_v2=pt.DataPageHeaderV2(
                num_values=n, num_nulls=nulls, num_rows=n,
                encoding=pt.Encoding.PLAIN,
                definition_levels_byte_length=len(defs),
                repetition_levels_byte_length=0, is_compressed=False,
                statistics=None, i32=True),
            i32=True)
        offset = f.tell()
        writer.write_thrift(f, ph)
        f.write(body)

    present = [v for v in values if v is not None]
    stats = dict(null_count=n - len(present))
    if present:
        us = [_unscaled(v, scale) for v in present]
        lo, hi = _stat_bytes(min(us), physical, type_length), \
            _stat_bytes(max(us), physical, type_length)
        stats.update(min=lo, max=hi, min_value=lo, max_value=hi)

    total = f.tell() - offset
    cmd = pt.ColumnMetaData(
        type=physical, encodings=[pt.Encoding.PLAIN], path_in_schema=[name],
        codec=pt.CompressionCodec.UNCOMPRESSED, num_values=n,
        total_uncompressed_size=total, total_compressed_size=total,
        key_value_metadata=[], data_page_offset=offset,
        statistics=pt.Statistics(**stats), i32list=[1, 4])
    chunk = pt.ColumnChunk(file_offset=offset, meta_data=cmd, file_path=None)
    rg = pt.RowGroup(num_rows=n, columns=[chunk], total_byte_size=total)

    se_kwargs = dict(name=name, type=physical, type_length=type_length,
                     repetition_type=pt.FieldRepetitionType.OPTIONAL, i32=True)
    if annotation in ('both', 'old'):
        se_kwargs.update(converted_type=pt.ConvertedType.DECIMAL,
                         scale=scale, precision=precision)
    if annotation in ('both', 'new'):
        se_kwargs['logicalType'] = ThriftObject.from_fields(
            'LogicalType',
            DECIMAL=ThriftObject.from_fields(
                'DecimalType', precision=precision, scale=scale, i32=True))
    se = pt.SchemaElement(**se_kwargs)
    root = pt.SchemaElement(name=b'schema', num_children=1, i32=True)
    fmd = pt.FileMetaData(num_rows=n, schema=[root, se], version=1,
                          created_by=b'independent-decimal-builder',
                          row_groups=[rg], key_value_metadata=[], i32list=[1])
    size = writer.write_thrift(f, fmd)
    f.write(struct.pack('<I', size))
    f.write(b'PAR1')
    f.close()


def rewrite_schema(path, mutate):
    """Rewrite a file's footer after applying ``mutate(fmd)``."""
    raw = open(path, 'rb').read()
    size = struct.unpack('<I', raw[-8:-4])[0]
    fmd = from_buffer(raw[-8 - size:-8], 'FileMetaData')
    mutate(fmd)
    f = open(path, 'wb')
    f.write(raw[:-8 - size])
    new_size = writer.write_thrift(f, fmd)
    f.write(struct.pack('<I', new_size))
    f.write(b'PAR1')
    f.close()


def file_bytes(path):
    with open(path, 'rb') as f:
        return f.read()


# ---------------------------------------------------------------------------
# writing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("precision,scale,values", [
    (9, 2, [D('1.23'), D('-4.50'), D('0'), None, D('9999999.99')]),
    (18, 4, [D('0.0001'), D('-12345678901234.5678'), None, D('0')]),
    (38, 10, [D('1234567890123456789012345678.0123456789'), None,
              D('-9999999999999999999999999999.9999999999')]),
    (5, 0, [D('0'), D('-1'), D('99999')]),
])
def test_roundtrip_exact_and_physical(tempdir, precision, scale, values):
    fn = os.path.join(tempdir, 'exact.parquet')
    df = pd.DataFrame({'amt': pd.Series(values, dtype=object),
                       'n': np.arange(len(values))})
    original = df.copy(deep=True)
    write(fn, df, decimal_columns={'amt': (precision, scale)})
    # input is untouched
    pd.testing.assert_frame_equal(df, original)

    pf = ParquetFile(fn, decimal_mode='decimal')
    se = pf.schema.root['children']['amt']
    # physical representation must cover the whole declared range
    expected_physical = (pt.Type.INT32 if precision <= 9
                         else pt.Type.INT64 if precision <= 18
                         else pt.Type.FIXED_LEN_BYTE_ARRAY)
    assert se.type == expected_physical
    assert se.converted_type == pt.ConvertedType.DECIMAL
    assert (se.precision, se.scale) == (precision, scale)
    assert se.logicalType.DECIMAL.precision == precision
    assert se.logicalType.DECIMAL.scale == scale

    out = pf.to_pandas()
    assert out['amt'].dtype == object
    got = out['amt'].tolist()
    assert [(v is None) for v in got] == [(v is None) for v in values]
    for actual, expected in zip(got, values):
        if expected is None:
            continue
        assert actual == expected
        assert actual.as_tuple().exponent == -scale
    assert out['n'].tolist() == list(range(len(values)))

    # default float mode is unchanged
    fl = ParquetFile(fn).to_pandas()
    assert fl['amt'].dtype == np.dtype('float64')

    # pyarrow reads the exact values and the schema
    table = pq.read_table(fn)
    assert table.schema.field('amt').type == pa.decimal128(precision, scale)
    assert table.column('amt').to_pylist() == out['amt'].tolist()


def test_zero_negative_and_trailing_zeros(tempdir):
    fn = os.path.join(tempdir, 'z.parquet')
    vals = [D('0.0000'), D('-0.0000'), D('1.2300'), D('-1.2300'), D('0')]
    write(fn, pd.DataFrame({'amt': vals}), decimal_columns={'amt': (10, 4)})
    out = ParquetFile(fn, decimal_mode='decimal').to_pandas()['amt'].tolist()
    assert out == [D('0.0000'), D('0.0000'), D('1.2300'), D('-1.2300'),
                   D('0.0000')]
    assert all(v.as_tuple().exponent == -4 for v in out)


def test_integer_dtypes_and_missing(tempdir):
    fn = os.path.join(tempdir, 'i.parquet')
    df = pd.DataFrame({
        'i64': np.array([1, 2, 3], dtype='int64'),
        'i32': np.array([-1, 0, 1], dtype='int32'),
        'u8': np.array([0, 1, 255], dtype='uint8'),
        'nullable': pd.array([1, None, 3], dtype='Int64'),
        'obj': pd.Series([1, None, 3], dtype=object),
    })
    write(fn, df, decimal_columns={c: (10, 0) for c in df.columns})
    out = ParquetFile(fn, decimal_mode='decimal').to_pandas()
    assert out['i64'].tolist() == [D(1), D(2), D(3)]
    assert out['i32'].tolist() == [D(-1), D(0), D(1)]
    assert out['u8'].tolist() == [D(0), D(1), D(255)]
    assert out['nullable'].tolist() == [D(1), None, D(3)]
    assert out['obj'].tolist() == [D(1), None, D(3)]
    assert all(out[c].dtype == object for c in out.columns)


def test_large_integers_not_via_float(tempdir):
    fn = os.path.join(tempdir, 'big.parquet')
    big = D(2 ** 53) + 1
    write(fn, pd.DataFrame({'amt': [big, -big]}),
          decimal_columns={'amt': (20, 0)})
    out = ParquetFile(fn, decimal_mode='decimal').to_pandas()['amt'].tolist()
    assert out == [big, -big]
    # the float path demonstrably loses the value; exact mode does not
    fl = ParquetFile(fn).to_pandas()['amt'].tolist()
    assert fl[0] == float(2 ** 53)


def test_empty_and_all_null_columns(tempdir):
    for label, values in [('empty', []), ('allnull', [None, None])]:
        for precision, scale in [(9, 2), (18, 4), (38, 10)]:
            fn = os.path.join(tempdir, f'{label}_{precision}.parquet')
            write(fn, pd.DataFrame({'amt': pd.Series(values, dtype=object)}),
                  decimal_columns={'amt': (precision, scale)},
                  row_group_offsets=1)
            pf = ParquetFile(fn, decimal_mode='decimal')
            out = pf.to_pandas()['amt'].tolist()
            assert out == values
            assert all(v is None for v in pf.statistics['min']['amt'])
            assert pq.read_table(fn).column('amt').to_pylist() == values


def test_decimal_columns_overrides_object_encoding(tempdir):
    fn = os.path.join(tempdir, 'prio.parquet')
    write(fn, pd.DataFrame({'amt': [D('1.23')]}),
          object_encoding='bytes', decimal_columns={'amt': (9, 2)})
    assert ParquetFile(fn, decimal_mode='decimal').to_pandas()['amt'].tolist() \
        == [D('1.23')]


@pytest.mark.parametrize("declaration", [
    {'amt': (0, 0)},
    {'amt': (39, 0)},
    {'amt': (5, 6)},
    {'amt': (-1, 0)},
    {'amt': (True, 0)},
    {'amt': (9, False)},
    {'amt': (9.0, 2)},
    {'amt': 9},
    {'missing': (9, 2)},
])
def test_invalid_declarations(tempdir, declaration):
    fn = os.path.join(tempdir, 'bad.parquet')
    with pytest.raises(ValueError):
        write(fn, pd.DataFrame({'amt': [D('1')]}),
              decimal_columns=declaration)
    assert not os.path.exists(fn)


def test_duplicate_column_rejected(tempdir):
    fn = os.path.join(tempdir, 'dup.parquet')
    df = pd.DataFrame([[D('1'), D('2')]], columns=['amt', 'amt'])
    with pytest.raises(ValueError, match='more than once'):
        write(fn, df, decimal_columns={'amt': (9, 2)})
    assert not os.path.exists(fn)


@pytest.mark.parametrize("value", [
    1.5, np.float64(2.5), 'x', b'x', True, np.bool_(False),
    D('NaN'), D('sNaN'), D('Infinity'), D('-Infinity'), D('1.234'),
    D('12345678901'), object(),
])
def test_invalid_values(tempdir, value):
    fn = os.path.join(tempdir, 'badval.parquet')
    with pytest.raises(ValueError) as exc:
        write(fn, pd.DataFrame({'amt': [D('1.00'), value]}),
              decimal_columns={'amt': (9, 2)})
    assert 'amt' in str(exc.value)
    assert 'row 1' in str(exc.value)
    assert not os.path.exists(fn)


def test_float_nan_is_missing_not_rejected(tempdir):
    fn = os.path.join(tempdir, 'nanmiss.parquet')
    write(fn, pd.DataFrame({'amt': [D('1.23'), float('nan')]}),
          decimal_columns={'amt': (9, 2)})
    assert ParquetFile(fn, decimal_mode='decimal').to_pandas()[
        'amt'].tolist() == [D('1.23'), None]


def test_rounding_rejected_not_truncated(tempdir):
    fn = os.path.join(tempdir, 'round.parquet')
    for v in [D('1.239'), D('1.235'), D('-1.239'), D('0.001')]:
        with pytest.raises(ValueError, match='exactly'):
            write(fn, pd.DataFrame({'amt': [v]}),
                  decimal_columns={'amt': (9, 2)})
        assert not os.path.exists(fn)


def test_precision_boundary(tempdir):
    fn = os.path.join(tempdir, 'prec.parquet')
    # 10**precision - 1 fits, 10**precision does not
    write(fn, pd.DataFrame({'amt': [D(10 ** 9 - 1), D(-(10 ** 9 - 1))]}),
          decimal_columns={'amt': (9, 0)})
    assert ParquetFile(fn, decimal_mode='decimal').to_pandas()['amt'].tolist() \
        == [D(10 ** 9 - 1), D(-(10 ** 9 - 1))]
    with pytest.raises(ValueError, match='precision'):
        write(fn, pd.DataFrame({'amt': [D(10 ** 9)]}),
              decimal_columns={'amt': (9, 0)})


def test_scale_equals_precision_boundary(tempdir):
    fn = os.path.join(tempdir, 's.parquet')
    write(fn, pd.DataFrame({'amt': [D('0.0001'), D('-0.0001')]}),
          decimal_columns={'amt': (4, 4)})
    out = ParquetFile(fn, decimal_mode='decimal').to_pandas()['amt'].tolist()
    assert out == [D('0.0001'), D('-0.0001')]


def test_has_nulls_false_rejected_up_front(tempdir):
    fn = os.path.join(tempdir, 'nn.parquet')
    with pytest.raises(ValueError, match='missing'):
        write(fn, pd.DataFrame({'amt': [D('1'), None]}), has_nulls=False,
              decimal_columns={'amt': (9, 2)})
    assert not os.path.exists(fn)
    # no nulls is fine
    write(fn, pd.DataFrame({'amt': [D('1'), D('2')]}), has_nulls=False,
          decimal_columns={'amt': (9, 2)})
    assert ParquetFile(fn, decimal_mode='decimal').to_pandas()['amt'].tolist() \
        == [D('1.00'), D('2.00')]


@pytest.mark.parametrize("scheme", ['hive', 'drill'])
def test_multi_file_schemes_rejected(tempdir, scheme):
    fn = os.path.join(tempdir, 'multi')
    with pytest.raises(ValueError):
        write(fn, pd.DataFrame({'amt': [D('1')], 'p': ['a']}),
              file_scheme=scheme, decimal_columns={'amt': (9, 2)})
    assert not os.path.exists(fn)


def test_partition_on_rejected(tempdir):
    fn = os.path.join(tempdir, 'part')
    with pytest.raises(ValueError):
        write(fn, pd.DataFrame({'amt': [D('1')], 'p': ['a']}),
              file_scheme='hive', partition_on=['p'],
              decimal_columns={'amt': (9, 2)})
    assert not os.path.exists(fn)


def test_existing_file_untouched_on_write_error(tempdir):
    fn = os.path.join(tempdir, 'existing.parquet')
    write(fn, pd.DataFrame({'amt': [D('1.23')]}),
          decimal_columns={'amt': (9, 2)})
    before = file_bytes(fn)
    with pytest.raises(ValueError):
        # overwrite=False means this would replace the file; the bad value
        # must be caught first
        write(fn, pd.DataFrame({'amt': [D('1.234')]}),
              decimal_columns={'amt': (9, 2)})
    assert file_bytes(fn) == before


def test_row_groups_compression_stats_options(tempdir):
    fn = os.path.join(tempdir, 'rg.parquet')
    values = [D('1.23'), None, D('-4.50'), D('0'), D('9.99')]
    write(fn, pd.DataFrame({'amt': values, 'n': range(5)}),
          decimal_columns={'amt': (9, 2)}, row_group_offsets=2,
          compression='GZIP')
    pf = ParquetFile(fn, decimal_mode='decimal')
    assert [rg.num_rows for rg in pf.row_groups] == [2, 2, 1]
    assert pf.to_pandas()['amt'].tolist() == [D('1.23'), None, D('-4.50'),
                                              D('0.00'), D('9.99')]
    assert pf.statistics['null_count']['amt'] == [1, 0, 0]
    assert pf.statistics['min']['amt'] == [D('1.23'), D('-4.50'), D('9.99')]
    assert pf.statistics['max']['amt'] == [D('1.23'), D('0.00'), D('9.99')]

    # stats=False -> no min/max, as before
    fn2 = os.path.join(tempdir, 'nostats.parquet')
    write(fn2, pd.DataFrame({'amt': values}), decimal_columns={'amt': (9, 2)},
          stats=False)
    st = ParquetFile(fn2, decimal_mode='decimal').statistics
    assert st['min']['amt'] == [None] and st['max']['amt'] == [None]
    assert st['null_count']['amt'] == [1]


def test_seekable_file_object(tempdir):
    payload = io.BytesIO()
    write(payload, pd.DataFrame({'amt': [D('1.23'), None]}),
          decimal_columns={'amt': (9, 2)})
    payload.seek(0)
    out = ParquetFile(payload, decimal_mode='decimal').to_pandas()
    assert out['amt'].tolist() == [D('1.23'), None]

    import tempfile as _tempfile
    with _tempfile.TemporaryFile() as f:
        write(f, pd.DataFrame({'amt': [D('-9.99')]}),
              decimal_columns={'amt': (9, 2)})
        f.seek(0)
        assert ParquetFile(f, decimal_mode='decimal').to_pandas()[
            'amt'].tolist() == [D('-9.99')]


# ---------------------------------------------------------------------------
# appending
# ---------------------------------------------------------------------------

def test_append_simple(tempdir):
    fn = os.path.join(tempdir, 'app.parquet')
    write(fn, pd.DataFrame({'id': [1, 2], 'amt': [D('1.23'), None]}),
          decimal_columns={'amt': (9, 2)})
    write(fn, pd.DataFrame({'id': [3, 4], 'amt': [D('-0.01'), D('5')]}),
          decimal_columns={'amt': (9, 2)}, append=True)
    pf = ParquetFile(fn, decimal_mode='decimal')
    out = pf.to_pandas()
    assert out['id'].tolist() == [1, 2, 3, 4]
    assert out['amt'].tolist() == [D('1.23'), None, D('-0.01'), D('5.00')]
    assert pq.read_table(fn).column('amt').to_pylist() == out['amt'].tolist()


def test_append_reuses_physical_and_rejects_mismatch(tempdir):
    fn = os.path.join(tempdir, 'appmix.parquet')
    # create an 18-digit column; append a declaration that would otherwise
    # use INT64 as well, but also one that forces a different physical type
    write(fn, pd.DataFrame({'amt': [D('1.23')]}),
          decimal_columns={'amt': (18, 2)})
    before = file_bytes(fn)
    for bad in [{'amt': (9, 2)}, {'amt': (18, 3)}, {'amt': (19, 2)}]:
        with pytest.raises(ValueError, match='amt'):
            write(fn, pd.DataFrame({'amt': [D('1.00')]}),
                  decimal_columns=bad, append=True)
        assert file_bytes(fn) == before
    write(fn, pd.DataFrame({'amt': [D('2.00')]}),
          decimal_columns={'amt': (18, 2)}, append=True)
    pf = ParquetFile(fn, decimal_mode='decimal')
    assert pf.schema.root['children']['amt'].type == pt.Type.INT64
    assert pf.to_pandas()['amt'].tolist() == [D('1.23'), D('2.00')]


def test_append_rejects_unrepresentable_without_touching_file(tempdir):
    fn = os.path.join(tempdir, 'appbad.parquet')
    write(fn, pd.DataFrame({'amt': [D('1.23')]}),
          decimal_columns={'amt': (9, 2)})
    before = file_bytes(fn)
    for bad_value in [D('1.234'), D('12345678.99')]:
        with pytest.raises(ValueError):
            write(fn, pd.DataFrame({'amt': [bad_value]}),
                  decimal_columns={'amt': (9, 2)}, append=True)
        assert file_bytes(fn) == before
    assert ParquetFile(fn, decimal_mode='decimal').to_pandas()[
        'amt'].tolist() == [D('1.23')]


def test_append_required_column_rejects_nulls(tempdir):
    fn = os.path.join(tempdir, 'appreq.parquet')
    write(fn, pd.DataFrame({'amt': [D('1.00')]}), has_nulls=False,
          decimal_columns={'amt': (9, 2)})
    before = file_bytes(fn)
    with pytest.raises(ValueError, match='missing'):
        write(fn, pd.DataFrame({'amt': [None]}), decimal_columns={'amt': (9, 2)},
              append=True)
    assert file_bytes(fn) == before


def test_append_non_decimal_existing_rejected(tempdir):
    fn = os.path.join(tempdir, 'appplain.parquet')
    write(fn, pd.DataFrame({'amt': [1.5]}))
    before = file_bytes(fn)
    with pytest.raises(ValueError, match='not DECIMAL'):
        write(fn, pd.DataFrame({'amt': [D('1')]}),
              decimal_columns={'amt': (9, 2)}, append=True)
    assert file_bytes(fn) == before


def test_non_decimal_columns_unchanged_when_appending(tempdir):
    fn = os.path.join(tempdir, 'appother.parquet')
    write(fn, pd.DataFrame({'amt': [D('1.23')], 's': ['a']}),
          decimal_columns={'amt': (9, 2)})
    write(fn, pd.DataFrame({'amt': [D('2.00')], 's': ['b']}),
          decimal_columns={'amt': (9, 2)}, append=True)
    out = ParquetFile(fn, decimal_mode='decimal').to_pandas()
    assert out['s'].tolist() == ['a', 'b']
    assert out['amt'].tolist() == [D('1.23'), D('2.00')]


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

def test_decimal_mode_validation(tempdir):
    fn = os.path.join(tempdir, 'm.parquet')
    write(fn, pd.DataFrame({'amt': [D('1')]}), decimal_columns={'amt': (9, 2)})
    with pytest.raises(ValueError):
        ParquetFile(fn, decimal_mode='something')


def test_pyarrow_vectors_dictionary_plain_v1_v2_compression(tempdir):
    values = [D('1.23'), None, D('-4.50'), D('0'), D('99999.99'), D('-0.01')]
    variants = {
        'plain': dict(use_dictionary=False),
        'dict': dict(use_dictionary=True),
        'v2': dict(data_page_version='2.0'),
        'snappy': dict(compression='snappy'),
        'gz_v2': dict(compression='gzip', data_page_version='2.0'),
    }
    expected = [D('1.23'), None, D('-4.50'), D('0.00'), D('99999.99'),
                D('-0.01')]
    for label, kwargs in variants.items():
        fn = os.path.join(tempdir, f'pa_{label}.parquet')
        table = pa.table({'amt': pa.array(values, type=pa.decimal128(9, 2)),
                          'n': list(range(len(values)))})
        pq.write_table(table, fn, **kwargs)
        pf = ParquetFile(fn, decimal_mode='decimal')
        out = pf.to_pandas()
        assert out['amt'].tolist() == expected, label
        assert out['amt'].dtype == object
        assert out['n'].dtype == np.dtype('int64')
        assert [v.as_tuple().exponent for v in out['amt'] if v is not None] \
            == [-2] * 5
        # float mode keeps old behaviour for the same file
        assert ParquetFile(fn).to_pandas()['amt'].dtype == np.dtype('float64')


@pytest.mark.parametrize("physical,type_length", [
    (pt.Type.INT32, None),
    (pt.Type.INT64, None),
    (pt.Type.BYTE_ARRAY, None),
    (pt.Type.FIXED_LEN_BYTE_ARRAY, 8),
])
@pytest.mark.parametrize("annotation", ['both', 'old', 'new'])
@pytest.mark.parametrize("page_version", [1, 2])
def test_all_physical_and_annotation_forms(tempdir, physical, type_length,
                                           annotation, page_version):
    values = [D('1.23'), None, D('-4.50'), D('0'), D('9999999.99')]
    fn = os.path.join(tempdir, 'ind.parquet')
    build_parquet(fn, values, 9, 2, physical, type_length,
                  annotation=annotation, page_version=page_version)
    out = ParquetFile(fn, decimal_mode='decimal').to_pandas()['amt'].tolist()
    expected = [D('1.23'), None, D('-4.50'), D('0.00'), D('9999999.99')]
    assert out == expected
    assert pq.read_table(fn).column('amt').to_pylist() == expected


def test_all_null_page_and_empty_file_independent(tempdir):
    fn = os.path.join(tempdir, 'nulls.parquet')
    build_parquet(fn, [None, None, None], 18, 4, pt.Type.FIXED_LEN_BYTE_ARRAY,
                  9)
    pf = ParquetFile(fn, decimal_mode='decimal')
    assert pf.to_pandas()['amt'].tolist() == [None, None, None]
    assert pf.statistics['null_count']['amt'] == [3]
    assert pf.statistics['min']['amt'] == [None]
    assert pq.read_table(fn).column('amt').to_pylist() == [None, None, None]


def test_conflicting_annotations_rejected(tempdir):
    fn = os.path.join(tempdir, 'conflict.parquet')
    build_parquet(fn, [D('1.23')], 9, 2, pt.Type.INT32)

    def mutate(fmd):
        fmd.schema[1][7] = 3   # old scale differs from LogicalType scale 2

    rewrite_schema(fn, mutate)
    with pytest.raises(ValueError, match='amt'):
        ParquetFile(fn, decimal_mode='decimal')
    # the float path is untouched for well-formed files; this one is rejected
    with pytest.raises(ValueError, match='amt'):
        ParquetFile(fn, decimal_mode='decimal').to_pandas()


@pytest.mark.parametrize("mutate,match", [
    (lambda fmd: fmd.schema[1].__setitem__(7, 0), 'precision'),
    (lambda fmd: fmd.schema[1].__setitem__(1, pt.Type.FLOAT), 'physical'),
])
def test_invalid_decimal_schema_rejected(tempdir, mutate, match):
    fn = os.path.join(tempdir, 'invalid.parquet')
    build_parquet(fn, [D('1.23')], 9, 2, pt.Type.INT32)

    def both(fmd):
        mutate(fmd)
        if match == 'precision':
            fmd.schema[1][8] = 0
            fmd.schema[1][10] = {5: {1: 0, 2: 0, 'i32': 1}}

    rewrite_schema(fn, both)
    with pytest.raises(ValueError, match=match):
        ParquetFile(fn, decimal_mode='decimal').to_pandas()


def test_explicit_dtypes_rejected_before_pages(tempdir):
    fn = os.path.join(tempdir, 'dt.parquet')
    write(fn, pd.DataFrame({'amt': [D('1.23')], 'n': [1]}),
          decimal_columns={'amt': (9, 2)})
    pf = ParquetFile(fn, decimal_mode='decimal')
    with pytest.raises(ValueError, match='amt'):
        pf.to_pandas(dtypes={'amt': 'float64'})
    with pytest.raises(ValueError, match='amt'):
        ParquetFile(fn, decimal_mode='decimal',
                    dtypes={'amt': np.dtype('float64')}).to_pandas()
    # object dtype is accepted
    out = pf.to_pandas(dtypes={'amt': object, 'n': np.dtype('int64')})
    assert out['amt'].tolist() == [D('1.23')]
    assert out['n'].tolist() == [1]


def test_nested_decimal_rejected(tempdir):
    fn = os.path.join(tempdir, 'nested.parquet')
    arr = pa.array([[D('1.23')], None, [D('-2.00'), D('3.00')]],
                   type=pa.list_(pa.decimal128(9, 2)))
    pq.write_table(pa.table({'amt': arr, 'x': [1, 2, 3]}), fn)
    pf = ParquetFile(fn, decimal_mode='decimal')
    with pytest.raises(ValueError, match='amt'):
        pf.to_pandas()
    assert pf.to_pandas(columns=['x'])['x'].tolist() == [1, 2, 3]
    # float-mode nested reads are unchanged
    assert ParquetFile(fn).to_pandas()['amt'].notnull().sum() == 2


def test_projection_slicing_iteration_multifile(tempdir):
    f1 = os.path.join(tempdir, 'f1.parquet')
    f2 = os.path.join(tempdir, 'f2.parquet')
    write(f1, pd.DataFrame({'amt': [D('1.23'), None], 'n': [1, 2]}),
          decimal_columns={'amt': (9, 2)})
    write(f2, pd.DataFrame({'amt': [D('-4.50'), D('0')], 'n': [3, 4]}),
          decimal_columns={'amt': (9, 2)})
    pf = ParquetFile([f1, f2], decimal_mode='decimal')
    out = pf.to_pandas()
    assert out['amt'].tolist() == [D('1.23'), None, D('-4.50'), D('0.00')]
    assert out['n'].tolist() == [1, 2, 3, 4]
    assert out['n'].dtype == np.dtype('int64')

    proj = pf.to_pandas(columns=['amt'])
    assert list(proj.columns) == ['amt']
    assert proj['amt'].tolist() == out['amt'].tolist()

    sliced = pf[0:1].to_pandas()
    assert sliced['amt'].tolist() == [D('1.23'), None]

    groups = [d['amt'].tolist() for d in pf.iter_row_groups()]
    assert groups == [[D('1.23'), None], [D('-4.50'), D('0.00')]]
    concatenated = pd.concat([d['amt'] for d in pf.iter_row_groups()],
                             ignore_index=True)
    assert concatenated.tolist() == out['amt'].tolist()


def test_statistics_exact_and_signs(tempdir):
    fn = os.path.join(tempdir, 'stats.parquet')
    write(fn, pd.DataFrame({'amt': [D('1.23'), D('-9.99'), None, D('0')]}),
          decimal_columns={'amt': (9, 2)}, row_group_offsets=2)
    pf = ParquetFile(fn, decimal_mode='decimal')
    st = pf.statistics
    assert st['min']['amt'] == [D('-9.99'), D('0.00')]
    assert st['max']['amt'] == [D('1.23'), D('0.00')]
    assert st['null_count']['amt'] == [0, 1]
    for value in st['min']['amt'] + st['max']['amt']:
        if value is not None:
            assert value.as_tuple().exponent == -2
    # float mode still returns floats
    fst = ParquetFile(fn).statistics
    assert fst['min']['amt'][0] == -9.99


@pytest.mark.parametrize("filters,expected", [
    ([[('n', '>=', 2)]], [2, 3]),
    ([('n', '>=', 2)], [2, 3]),
])
def test_legacy_filters_unaffected(tempdir, filters, expected):
    fn = os.path.join(tempdir, 'filt.parquet')
    write(fn, pd.DataFrame({'amt': [D('1.23'), D('2.00'), D('3.00')],
                            'n': [1, 2, 3]}),
          decimal_columns={'amt': (9, 2)}, row_group_offsets=1)
    pf = ParquetFile(fn, decimal_mode='decimal')
    out = pf.to_pandas(filters=filters, row_filter=True)
    assert out['n'].tolist() == expected
    assert out['amt'].tolist() == [D('2.00'), D('3.00')]


def test_predicate_on_non_decimal_unaffected(tempdir):
    fn = os.path.join(tempdir, 'pred.parquet')
    write(fn, pd.DataFrame({'amt': [D('1.23'), D('2.00')], 'n': [1, 5]}),
          decimal_columns={'amt': (9, 2)})
    pf = ParquetFile(fn, decimal_mode='decimal')
    out = pf.to_pandas(predicate={'column': 'n', 'op': '>', 'value': 2},
                       row_filter=True)
    assert out['amt'].tolist() == [D('2.00')]


# ---------------------------------------------------------------------------
# decimal context and state
# ---------------------------------------------------------------------------

def test_decimal_context_isolation(tempdir):
    vals = [D('123456789012345678901234567890.12345678'), D('-0.00000001'),
            D('0E-8'), D('-9999999999999999999999999999.99999999')]
    df = pd.DataFrame({'amt': vals})
    fn = os.path.join(tempdir, 'ctx.parquet')
    saved = decimal.getcontext().copy()
    try:
        with localcontext() as ctx:
            ctx.prec = 6
            ctx.rounding = decimal.ROUND_UP
            ctx.traps[Inexact] = True
            ctx.traps[Rounded] = True
            write(fn, df, decimal_columns={'amt': (38, 8)}, stats=True)
            pf = ParquetFile(fn, decimal_mode='decimal')
            out = pf.to_pandas()['amt'].tolist()
            st = pf.statistics
            # the context is unchanged
            assert ctx.prec == 6
            assert ctx.rounding == decimal.ROUND_UP
            assert ctx.traps[Inexact] and ctx.traps[Rounded]
    finally:
        decimal.setcontext(saved)
    assert out == vals
    assert [v.as_tuple().exponent for v in out] == [-8] * 4
    assert st['min']['amt'] == [D('-9999999999999999999999999999.99999999')]
    assert st['max']['amt'] == [D('123456789012345678901234567890.12345678')]
    # input DataFrame unchanged
    assert df['amt'].tolist() == vals


def test_mode_switching_and_failed_call_no_state_leak(tempdir):
    fn = os.path.join(tempdir, 'state.parquet')
    write(fn, pd.DataFrame({'amt': [D('1.23')]}),
          decimal_columns={'amt': (9, 2)})
    pf_f = ParquetFile(fn)
    pf_d = ParquetFile(fn, decimal_mode='decimal')
    assert pf_f.to_pandas()['amt'].tolist() == [1.23]
    with pytest.raises(ValueError):
        pf_d.to_pandas(dtypes={'amt': 'float64'})
    assert pf_d.to_pandas()['amt'].tolist() == [D('1.23')]
    assert pf_f.to_pandas()['amt'].tolist() == [1.23]
    assert pf_f.statistics['min']['amt'][0] == 1.23
    assert pf_d.statistics['min']['amt'][0] == D('1.23')


@pytest.mark.parametrize("page_version", [1, 2])
def test_row_group_boundaries_do_not_change_values(tempdir, page_version):
    values = ([D('1.23'), None, D('-4.50'), D('0')] * 5)
    results = []
    for offsets in [10, 3, [0, 4, 5, 12]]:
        fn = os.path.join(tempdir, f'rg_{offsets}.parquet')
        old = writer.DATAPAGE_VERSION
        writer.DATAPAGE_VERSION = page_version
        try:
            write(fn, pd.DataFrame({'amt': values, 'n': range(len(values))}),
                  decimal_columns={'amt': (9, 2)}, row_group_offsets=offsets,
                  compression='GZIP')
        finally:
            writer.DATAPAGE_VERSION = old
        out = ParquetFile(fn, decimal_mode='decimal').to_pandas()
        results.append((out['amt'].tolist(), out['n'].tolist()))
    for amt, n in results[1:]:
        assert amt == results[0][0]
        assert n == results[0][1]
    assert [v is None for v in results[0][0]] == [v is None for v in values]


# ---------------------------------------------------------------------------
# cross-engine exchange
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("precision,scale", [(9, 2), (18, 4), (38, 10)])
def test_pyarrow_reads_our_exact_files(tempdir, precision, scale):
    fn = os.path.join(tempdir, 'ours.parquet')
    values = [D('1.23'), D('-4.50'), None, D('0')]
    write(fn, pd.DataFrame({'amt': values}),
          decimal_columns={'amt': (precision, scale)})
    table = pq.read_table(fn)
    assert table.schema.field('amt').type == pa.decimal128(precision, scale)
    got = table.column('amt').to_pylist()
    assert [(v is None) for v in got] == [(v is None) for v in values]
    for actual, expected in zip(got, values):
        if expected is None:
            continue
        assert actual == expected
        assert actual.as_tuple().exponent == -scale
    # logical annotation also visible on the physical schema
    physical = pq.ParquetFile(fn).schema
    assert f'Decimal(precision={precision}, scale={scale})' in str(physical)


@pytest.mark.parametrize("precision,scale", [(9, 2), (18, 4)])
def test_we_read_pyarrow_integer_physical(tempdir, precision, scale):
    fn = os.path.join(tempdir, 'pa_int.parquet')
    values = [D('1.23'), None, D('-4.50'), D('0'), D('1234567.89')]
    table = pa.table({'amt': pa.array(values,
                                      type=pa.decimal128(precision, scale)),
                      'n': list(range(len(values)))})
    pq.write_table(table, fn, store_decimal_as_integer=True,
                   use_dictionary=False)
    physical_type = pq.ParquetFile(fn).schema.column(0).physical_type
    assert physical_type == ('INT32' if precision == 9 else 'INT64')
    out = ParquetFile(fn, decimal_mode='decimal').to_pandas()
    expected = [D('1.23'), None, D('-4.50'), D('0.00'), D('1234567.89')]
    assert out['amt'].tolist() == expected
    assert out['amt'].dtype == object


def test_we_read_pyarrow_fixed_len_multi_row_group(tempdir):
    fn = os.path.join(tempdir, 'pa_groups.parquet')
    values = [D('1.23'), None, D('-4.50'), D('0'), D('9.99')] * 4
    table = pa.table({'amt': pa.array(values, type=pa.decimal128(9, 2)),
                      'n': list(range(len(values)))})
    pq.write_table(table, fn, row_group_size=3, compression='snappy',
                   use_dictionary=True, write_statistics=True)
    pf = ParquetFile(fn, decimal_mode='decimal')
    out = pf.to_pandas()
    assert out['amt'].tolist() == [
        None if v is None else v for v in values]
    assert out['n'].tolist() == list(range(len(values)))
    # group-by-group consumption matches the concatenation
    pieces = [d['amt'] for d in pf.iter_row_groups()]
    assert pd.concat(pieces, ignore_index=True).tolist() == out['amt'].tolist()
    # independent statistics check: compare against recomputed min/max
    st = pf.statistics
    for idx in range(len(pf.row_groups)):
        group = pieces[idx].tolist()
        present = [v for v in group if v is not None]
        assert st['min']['amt'][idx] == (min(present) if present else None)
        assert st['max']['amt'][idx] == (max(present) if present else None)
        assert st['null_count']['amt'][idx] == sum(
            v is None for v in group)


def test_our_statistics_match_computed(tempdir):
    fn = os.path.join(tempdir, 'st.parquet')
    values = [D('1.23'), D('-9.99'), None, D('0'), D('1234567.89')]
    write(fn, pd.DataFrame({'amt': values}),
          decimal_columns={'amt': (9, 2)}, row_group_offsets=2)
    pf = ParquetFile(fn, decimal_mode='decimal')
    st = pf.statistics
    chunks = [d['amt'].tolist() for d in pf.iter_row_groups()]
    for i, group in enumerate(chunks):
        present = [v for v in group if v is not None]
        assert st['min']['amt'][i] == (min(present) if present else None)
        assert st['max']['amt'][i] == (max(present) if present else None)
        assert st['null_count']['amt'][i] == sum(v is None for v in group)


# ---------------------------------------------------------------------------
# resource behaviour: consume many row groups without accumulating state
# ---------------------------------------------------------------------------

def test_many_row_groups_streamed(tempdir, monkeypatch):
    import tracemalloc
    fn = os.path.join(tempdir, 'many.parquet')
    n_groups = 120
    rows = 5
    df = pd.DataFrame({
        'amt': [D(x) / 100 for x in range(n_groups * rows)],
        'n': range(n_groups * rows),
    })
    write(fn, df, decimal_columns={'amt': (9, 2)}, row_group_offsets=rows)
    pf = ParquetFile(fn, decimal_mode='decimal')
    assert len(pf.row_groups) == n_groups

    # Record how reading is performed: each group read uses its own row group.
    from fastparquet import core
    original = core.read_row_group
    seen = []

    def spy(file, rg, *args, **kwargs):
        seen.append((rg.num_rows, kwargs.get('decimal_mode')))
        return original(file, rg, *args, **kwargs)

    monkeypatch.setattr(core, 'read_row_group', spy)

    tracemalloc.start()
    baseline = tracemalloc.get_traced_memory()[0]
    peak = baseline
    count = 0
    expected = []
    for group in pf.iter_row_groups():
        expected.extend(group['amt'].tolist())
        count += len(group)
        current, this_peak = tracemalloc.get_traced_memory()
        peak = max(peak, this_peak)
        # The result is immediately discarded; no reference is kept.
        del group
    tracemalloc.stop()

    assert count == n_groups * rows
    assert len(seen) == n_groups
    assert all(mode == 'decimal' for _, mode in seen)
    # Memory must not grow with the number of consumed row groups.  A few
    # hundred KB of slack allows for caches and allocator noise.
    assert peak - baseline < 8 * 1024 * 1024, (peak - baseline)
    assert expected == df['amt'].tolist()


@pytest.mark.parametrize("page_version", [1, 2])
@pytest.mark.parametrize("compression", [None, 'GZIP', 'SNAPPY'])
def test_pyarrow_reads_v1_v2_compressed(tempdir, page_version, compression):
    fn = os.path.join(tempdir, 'vec.parquet')
    values = [D('1.23'), None, D('-4.50'), D('0'), D('9.99')]
    old = writer.DATAPAGE_VERSION
    writer.DATAPAGE_VERSION = page_version
    try:
        write(fn, pd.DataFrame({'amt': values}),
              decimal_columns={'amt': (9, 2)}, compression=compression)
    finally:
        writer.DATAPAGE_VERSION = old
    expected = [D('1.23'), None, D('-4.50'), D('0.00'), D('9.99')]
    assert pq.read_table(fn).column('amt').to_pylist() == expected
    assert ParquetFile(fn, decimal_mode='decimal').to_pandas()[
        'amt'].tolist() == expected


def test_fixed_len_trailing_zero_bytes(tempdir):
    """Negative / low-order-zero values must survive numpy S<n> truncation."""
    fn = os.path.join(tempdir, 'tz.parquet')
    # unscaled 256 ends in 0x00; negative fixed-length values end in 0x00 too
    values = [D('0.00000256'), D('-0.00000256'), D('0.00000100'),
              D('-0.00000100')]
    write(fn, pd.DataFrame({'amt': values}),
          decimal_columns={'amt': (38, 8)})
    pf = ParquetFile(fn, decimal_mode='decimal')
    out = pf.to_pandas()['amt'].tolist()
    assert out == values
    assert pq.read_table(fn).column('amt').to_pylist() == values
    assert pf.statistics['min']['amt'] == [D('-0.00000256')]
    assert pf.statistics['max']['amt'] == [D('0.00000256')]

    # same values supplied through pyarrow's fixed-length encoder
    fn2 = os.path.join(tempdir, 'tz_pa.parquet')
    pq.write_table(pa.table({'amt': pa.array(values,
                                             type=pa.decimal128(38, 8))}),
                   fn2, use_dictionary=False)
    assert ParquetFile(fn2, decimal_mode='decimal').to_pandas()[
        'amt'].tolist() == values


def test_float_mode_scales_integer_sourced_columns(tempdir):
    """Default ('float') mode must still apply the DECIMAL scale."""
    fn = os.path.join(tempdir, 'intsrc.parquet')
    write(fn, pd.DataFrame({'amt': pd.array([123, None, 350], dtype='Int64')}),
          decimal_columns={'amt': (9, 2)})
    fl = ParquetFile(fn).to_pandas()
    assert fl['amt'].dtype == np.dtype('float64')
    # integers denote whole units, so at scale 2 they are 123.00 and 350.00
    assert fl['amt'].tolist()[0] == 123.0
    assert np.isnan(fl['amt'].tolist()[1])
    assert fl['amt'].tolist()[2] == 350.0
    dec = ParquetFile(fn, decimal_mode='decimal').to_pandas()['amt'].tolist()
    assert dec == [D('123.00'), None, D('350.00')]


def test_append_preserves_existing_physical(tempdir):
    fn = os.path.join(tempdir, 'bytearr.parquet')
    build_parquet(fn, [D('1.23'), None], 9, 2, pt.Type.BYTE_ARRAY)
    write(fn, pd.DataFrame({'amt': [D('-4.50')]}),
          decimal_columns={'amt': (9, 2)}, append=True)
    pf = ParquetFile(fn, decimal_mode='decimal')
    assert pf.schema.root['children']['amt'].type == pt.Type.BYTE_ARRAY
    assert pf.to_pandas()['amt'].tolist() == [D('1.23'), None, D('-4.50')]
    assert pq.read_table(fn).column('amt').to_pylist() == [
        D('1.23'), None, D('-4.50')]
