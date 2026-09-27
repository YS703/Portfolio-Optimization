import numpy as np
import pandas as pd
import polars as pl

def get_correlation_matrix(data: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate the correlation matrix for the given DataFrame.

    Parameters:
        data (pl.DataFrame): A DataFrame where each column represents a different asset.
    Returns:
        pl.DataFrame: A DataFrame representing the correlation matrix of the input data.
    """
    ticker_names = [name for name in data.columns if name.lower() != "date"]

    price_arrays = {
        name: np.asarray(data[name].cast(pl.Float64, strict=True).to_list(), dtype=float)
        for name in ticker_names
    }
    matrix: dict[str, list[float] | list[str]] = {"ticker": ticker_names}

    for left_name in ticker_names:
        correlations = []
        left_values = price_arrays[left_name]
        for right_name in ticker_names:
            right_values = price_arrays[right_name]
            valid = np.isfinite(left_values) & np.isfinite(right_values)
            if valid.sum() < 2:
                correlations.append(float("nan"))
                continue

            left = left_values[valid]
            right = right_values[valid]
            if np.ptp(left) == 0 or np.ptp(right) == 0:
                correlations.append(float("nan"))
                continue

            correlations.append(float(np.corrcoef(left, right)[0, 1]))
        matrix[left_name] = correlations

    return pl.DataFrame(matrix)


def get_uncorrelated_assets(correlation_matrix: pl.DataFrame, threshold: float) -> list[str]:
     """
     Identify uncorrelated assets based on a correlation matrix and a specified threshold.

     Parameters:
         correlation_matrix (pl.DataFrame): A DataFrame representing the correlation matrix.
         threshold (float): The correlation threshold below which assets are considered uncorrelated.

     Returns:
         list[str]: A list of asset names that are uncorrelated with each other based on the given threshold.
     """
     uncorrelated_assets = []
     ticker_names = [name for name in correlation_matrix.columns if name != "ticker"]
     for asset in ticker_names:
         correlations = correlation_matrix.filter(pl.col("ticker") == asset).select(
             [name for name in ticker_names if name != asset]
         ).row(0)
         if all(np.isfinite(value) and abs(value) < threshold for value in correlations):
             uncorrelated_assets.append(asset)
     return uncorrelated_assets