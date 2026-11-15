API
===

.. currentmodule:: fastparquet

.. autosummary::
    ParquetFile
    ParquetFile.to_pandas
    ParquetFile.iter_row_groups
    ParquetFile.info
    ParquetFile.head
    ParquetFile.count
    ParquetFile.__getitem__
    write

.. autoclass:: ParquetFile
    :members:

.. automethod:: ParquetFile.to_pandas

.. automethod:: ParquetFile.iter_row_groups

.. autofunction:: write

.. autofunction:: update_file_custom_metadata

Exact DECIMAL values
--------------------

See :ref:`exact-decimals` in :doc:`details` for the opt-in exact decimal
write (``write(..., decimal_columns={col: (precision, scale)})``) and read
(``ParquetFile(..., decimal_mode='decimal')``) support.


.. raw:: html

    <script data-goatcounter="https://fastparquet.goatcounter.com/count"
        async src="//gc.zgo.at/count.js"></script>