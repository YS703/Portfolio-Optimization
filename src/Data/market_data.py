import pandas as pd
import polars as pl
import yfinance as yf


def load_data(
    tickers_list: list[str],
    column: str,
    start_date: str,
    end_date: str,
    interval: str,
) -> pl.DataFrame:
    """
    Download the selected price field for multiple tickers into a wide DataFrame.
    """
    data = yf.download(
        tickers=tickers_list,
        start=start_date,
        end=end_date,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )

    if data.empty:
        raise RuntimeError(f"No data returned for tickers: {', '.join(tickers_list)}")

    if isinstance(data.columns, pd.MultiIndex):
        if column not in data.columns.levels[0]:
            raise ValueError(f"Column '{column}' is not available.")
        extracted = data[column].copy()
    else:
        if column not in data.columns:
            raise ValueError(f"Column '{column}' is not available for ticker '{tickers_list[0]}'.")
        extracted = data[[column]].copy()
        extracted.columns = [tickers_list[0]]

    extracted = extracted.reset_index()
    extracted.rename(columns={extracted.columns[0]: "date"}, inplace=True)

    return pl.from_pandas(extracted)
