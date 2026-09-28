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


def get_uncorrelated_assets(
    data: pl.DataFrame,
    correlation_matrix: pl.DataFrame,
    threshold: float,
) -> pl.DataFrame:
    """
    Return the date and price columns for a diversified subset of assets 
    by sequentially dropping highly correlated pairs.
    """
    ticker_names = [name for name in correlation_matrix.columns if name != "ticker"]
    
    corr_array = correlation_matrix.select(ticker_names).to_numpy()
    assets_to_drop = set()
    
    for i in range(len(ticker_names)):
        if ticker_names[i] in assets_to_drop:
            continue
            
        for j in range(i + 1, len(ticker_names)):
            if abs(corr_array[i, j]) >= threshold:
                assets_to_drop.add(ticker_names[j])

    uncorrelated_assets = [name for name in ticker_names if name not in assets_to_drop]

    selected_columns = [
        name
        for name in data.columns
        if name.lower() == "date" or name in uncorrelated_assets
    ]
    
    return data.select(selected_columns)