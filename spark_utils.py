import numpy as np
import pandas as pd
from pyspark.sql import DataFrame
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import ArrayType, DoubleType
from typing import Iterator


def encode_dataframe(
    model,
    df: DataFrame,
    feature_col: str,
    output_col: str = "repr",
    encoding_window=None,
    causal: bool = False,
    sliding_length: int = None,
    sliding_padding: int = 0,
    batch_size: int = None,
) -> DataFrame:
    """Encode a PySpark DataFrame of time series with a TS2Vec model.

    Parameters
    ----------
    model : TS2Vec
        A trained TS2Vec model.
    df : DataFrame
        PySpark DataFrame containing a column of arrays representing
        individual time series.
    feature_col : str
        Name of the column containing the time series arrays.
    output_col : str, optional
        Name of the column to store the computed representations.
    encoding_window : optional
        Passed to :func:`TS2Vec.encode`.
    causal : bool, optional
        Passed to :func:`TS2Vec.encode`.
    sliding_length : int, optional
        Passed to :func:`TS2Vec.encode`.
    sliding_padding : int, optional
        Passed to :func:`TS2Vec.encode`.
    batch_size : int, optional
        Batch size used when calling ``TS2Vec.encode``.

    Returns
    -------
    DataFrame
        The input DataFrame with an additional column ``output_col`` containing
        the representations as arrays of ``double``.
    """

    schema = df.schema.add(output_col, ArrayType(DoubleType(), True))

    def encode_iter(iterator: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        for pdf in iterator:
            arr = np.stack(pdf[feature_col].apply(np.array).to_numpy())
            out = model.encode(
                arr,
                encoding_window=encoding_window,
                causal=causal,
                sliding_length=sliding_length,
                sliding_padding=sliding_padding,
                batch_size=batch_size,
            )
            pdf[output_col] = [o.tolist() for o in out]
            yield pdf

    return df.mapInPandas(encode_iter, schema)

def df_to_numpy(df: DataFrame, feature_col: str) -> np.ndarray:
    """Collect a PySpark DataFrame column of arrays to a numpy array."""
    pdf = df.select(feature_col).toPandas()
    return np.stack(pdf[feature_col].apply(np.array).to_numpy())
