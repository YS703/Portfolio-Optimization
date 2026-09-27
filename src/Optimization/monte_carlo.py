import numpy as np
import polars as pl

from src.Portfolio.account import account


def simulate_random_weights(
	data: pl.DataFrame,
	monthly_deposit: float,
	n_portfolios: int = 10_000,
	periods_per_year: int = 12,
	target_return: float = 0.0,
	random_seed: int | None = None,
) -> pl.DataFrame:
	"""Simulate monthly-contribution accounts and calculate Sortino ratios.

	Each portfolio starts with zero balance. On the first available price row of
	each calendar month, ``monthly_deposit`` is deposited and invested across
	assets using that portfolio's fixed target weights. Holdings are not
	rebalanced between deposits. ``data`` must contain a ``date`` column and one
	price column per asset. Metrics use monthly time-weighted account returns,
	so cash contributions are excluded from investment performance.
	"""
	if not isinstance(n_portfolios, int) or isinstance(n_portfolios, bool) or n_portfolios < 1:
		raise ValueError("n_portfolios must be a positive integer.")
	if not np.isfinite(monthly_deposit) or monthly_deposit <= 0:
		raise ValueError("monthly_deposit must be a positive finite amount.")
	if not isinstance(periods_per_year, int) or isinstance(periods_per_year, bool) or periods_per_year < 1:
		raise ValueError("periods_per_year must be a positive integer.")
	if not np.isfinite(target_return):
		raise ValueError("target_return must be finite.")

	date_column = next((name for name in data.columns if name.lower() == "date"), None)
	if date_column is None:
		raise ValueError("data must contain a date column for monthly deposits.")
	asset_names = [name for name in data.columns if name != date_column]
	if not asset_names:
		raise ValueError("data must contain at least one asset price column.")
	reserved_names = {
		"expected_return", "downside_deviation", "sortino_ratio",
		"performance_pct", "ending_value",
	}
	if reserved_names.intersection(asset_names):
		raise ValueError("Asset column names cannot conflict with output metric names.")

	ordered_data = data.sort(date_column) if date_column is not None else data
	prices = ordered_data.select(asset_names).cast(pl.Float64, strict=True).to_numpy()
	dates = ordered_data[date_column].to_list()
	usable_rows = np.isfinite(prices).all(axis=1) & (prices > 0).all(axis=1)
	prices = prices[usable_rows]
	dates = [date for date, usable in zip(dates, usable_rows, strict=True) if usable]
	if prices.shape[0] < 2:
		raise ValueError("At least two complete positive price observations are required.")

	months = [(date.year, date.month) for date in dates]
	month_start_indices = {
		index for index, month in enumerate(months)
		if index == 0 or month != months[index - 1]
	}

	rng = np.random.default_rng(random_seed)
	weights = rng.dirichlet(np.ones(len(asset_names)), size=n_portfolios)
	month_to_index = {month: index for index, month in enumerate(dict.fromkeys(months))}
	month_indices = np.asarray([month_to_index[month] for month in months])
	month_start_indices = set(month_start_indices)
	month_count = len(month_to_index)
	accounts = [account(initial_balance=0.0) for _ in range(n_portfolios)]
	holdings = np.zeros((n_portfolios, len(asset_names)), dtype=float)
	previous_equity = np.zeros(n_portfolios, dtype=float)
	monthly_growth = np.ones((month_count, n_portfolios), dtype=float)

	for row_index, asset_prices in enumerate(prices):
		portfolio_values = holdings @ asset_prices + np.fromiter(
			(investment_account.balance for investment_account in accounts),
			dtype=float,
			count=n_portfolios,
		)
		daily_returns = np.divide(
			portfolio_values,
			previous_equity,
			out=np.zeros(n_portfolios, dtype=float),
			where=previous_equity > 0,
		) - (previous_equity > 0)
		daily_returns[np.isclose(portfolio_values, previous_equity, rtol=1e-12, atol=1e-12)] = 0.0
		monthly_growth[month_indices[row_index]] *= 1.0 + daily_returns

		if row_index in month_start_indices:
			allocations = monthly_deposit * weights
			if len(asset_names) > 1:
				allocations[:, -1] = monthly_deposit - allocations[:, :-1].sum(axis=1)
			for portfolio_index, investment_account in enumerate(accounts):
				investment_account.deposit(monthly_deposit)
				for asset_index, asset in enumerate(asset_names):
					investment_account.invest(
						allocations[portfolio_index, asset_index],
						asset,
						asset_prices[asset_index],
					)
			holdings += allocations / asset_prices
			previous_equity = portfolio_values + monthly_deposit
		else:
			previous_equity = portfolio_values

	monthly_returns = monthly_growth - 1.0
	performance_percentages = (np.prod(monthly_growth, axis=0) - 1.0) * 100.0
	years = len(prices) / 252
	expected_returns = (1.0 + performance_percentages / 100.0) ** (1.0 / years) - 1.0
	target_per_period = target_return / periods_per_year
	downside_returns = np.minimum(monthly_returns - target_per_period, 0.0)
	downside_deviations = np.sqrt(np.mean(downside_returns**2, axis=0)) * np.sqrt(periods_per_year)
	ending_values = holdings @ prices[-1] + np.fromiter(
		(investment_account.balance for investment_account in accounts),
		dtype=float,
		count=n_portfolios,
	)

	sortino_ratios = np.divide(
		expected_returns - target_return,
		downside_deviations,
		out=np.full(n_portfolios, np.nan),
		where=downside_deviations > 0,
	)
	sortino_ratios[(downside_deviations == 0) & (expected_returns > target_return)] = np.inf

	result = pl.DataFrame(
		{
			"expected_return": expected_returns,
			"downside_deviation": downside_deviations,
			"sortino_ratio": sortino_ratios,
			"performance_pct": performance_percentages,
			"ending_value": ending_values,
		}
	)
	weights_frame = pl.DataFrame(
		{asset_name: weights[:, index] for index, asset_name in enumerate(asset_names)}
	)
	return result.hstack(weights_frame)
