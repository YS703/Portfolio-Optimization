# Portfolio Management

A Python portfolio-analysis project that downloads historical market prices, computes asset correlations, simulates monthly-contribution portfolios with random asset weights, and creates PDF reports for the simulation and selected portfolios.

## Current Workflow

Run `main.py` from the project root. The application:

1. Loads and validates settings from `config.yaml`.
2. Downloads the requested Yahoo Finance price field for all configured tickers.
3. Calculates a pairwise correlation matrix and identifies assets whose absolute correlations with all other assets are below the configured threshold.
4. Simulates candidate portfolios with random long-only weights and monthly contributions.
5. Selects the portfolios with the highest Sortino ratios.
6. Writes a risk-return PDF and a PDF of selected portfolio weights, including their mean allocation.

The correlation analysis currently reports the uncorrelated asset list but does not use that list to filter the tickers passed into the Monte Carlo simulation.

## Requirements and Setup

Python 3.10 or later is recommended. From the project root, create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the workflow with:

```powershell
python main.py
```

The market-data step requires an internet connection and a ticker supported by Yahoo Finance. Yahoo Finance treats the configured start date as inclusive and end date as exclusive. The configured interval must also be supported by Yahoo Finance.

## Configuration

All current settings are in `config.yaml`:

| Key | Purpose | Current example |
| --- | --- | --- |
| `tickers_list` | Yahoo Finance symbols used as candidate assets | `CW8.PA`, `SPY`, `AAPL`, `MSFT`, `AMZN`, `GOOGL` |
| `column` | Price field requested from Yahoo Finance | `Close` |
| `start_date` | Inclusive historical data start date | `2024-01-01` |
| `end_date` | Exclusive historical data end date | `2024-12-31` |
| `interval` | Yahoo Finance sampling interval | `1d` |
| `monthly_deposit` | Amount deposited and invested at each monthly contribution date | `200` |
| `correlation_threshold` | Absolute-correlation cutoff used by the correlation analysis | `0.95` |
| `n_simulations` | Number of random portfolios to simulate | `10000` |
| `target_return` | Annual minimum acceptable return used in downside and Sortino calculations | `0.05` |
| `n_best_portfolios` | Number of highest-Sortino portfolios selected for reporting | `5` |

The YAML loader validates that ticker names are a nonempty list, text settings are nonempty strings, the deposit is positive, the correlation threshold is between 0 and 1, simulation and selection counts are positive integers, and the target return is finite.

## Modules

- `main.py`: Coordinates configuration, data retrieval, correlation analysis, portfolio simulation, selection, and PDF report generation.
- `src/Data/load_config.py`: Reads and validates YAML configuration.
- `src/Data/market_data.py`: Downloads the selected price field for multiple tickers and returns a Polars DataFrame with a `date` column and one price column per ticker.
- `src/Optimization/correlation.py`: Calculates a labeled pairwise Pearson correlation matrix and identifies assets whose correlations with all other assets are below the threshold. The calculations currently use the downloaded price series.
- `src/Optimization/monte_carlo.py`: Samples long-only portfolio weights from a Dirichlet distribution and simulates monthly-contribution accounts using the `account` class.
- `src/Portfolio/account.py`: Tracks account cash, total deposited, and fractional asset holdings; supports deposits and purchases.
- `src/Analytics/reporting.py`: Creates the Markowitz-style risk-return PDF and the selected-portfolio weights PDF.
- `src/Backtest/engine.py`: Reserved module for the planned backtesting feature; it is currently empty.

## Portfolio Simulation Details

Each simulation starts with a zero-balance account. A contribution is made on the first available price observation in each calendar month and allocated across assets using that simulation's generated weights. Existing holdings are not rebalanced between contributions. Fractional shares are supported by the account model.

The simulation reports:

- `expected_return`: Geometrically annualized return based on compounded time-weighted portfolio performance.
- `downside_deviation`: Annualized root-mean-square downside of monthly portfolio returns relative to the monthly target return.
- `sortino_ratio`: `(expected_return - target_return) / downside_deviation`.
- `performance_pct`: Compounded time-weighted performance over the full data period, expressed as a percentage. Deposits themselves are not counted as gains.
- `ending_value`: Estimated final account value, including the market value of fractional holdings.
- One additional column per ticker: the candidate portfolio's allocation weights, which sum to 1.

The simulation currently uses 12 periods per year for the monthly return and downside calculations and 252 trading observations per year when annualizing the cumulative return. Historical length, asset universe, and market regime affect the reliability of the calculated metrics; a short history can make downside deviation and Sortino rankings especially sensitive.

## Generated Reports

Running `main.py` writes these files in the current working directory unless the reporting functions are called with custom paths:

- `markowitz_frontier.pdf`: Scatter plot of simulated portfolios, with annualized downside deviation on the x-axis, annualized expected return on the y-axis, and Sortino ratio represented by color. The selected top-Sortino portfolios and minimum-downside portfolio are highlighted.
- `best_portfolios_weights.pdf`: Grouped asset-weight chart for the selected top portfolios plus the per-asset mean weight across those portfolios. That mean allocation is the intended combined allocation to use.

## Planned Feature

A future feature will backtest selected portfolios in `src/Backtest/engine.py`. This is **not implemented yet**; the current workflow evaluates portfolios against historical data for selection but does not run a dedicated portfolio backtest with a separate engine.