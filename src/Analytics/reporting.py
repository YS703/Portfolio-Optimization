from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


def generate_correlation_pdf(
	correlation_matrix: pl.DataFrame,
	output_path: str | Path = "correlation_matrix.pdf",
) -> Path:
	"""Save a labeled heatmap of the asset correlation matrix as a PDF."""
	if "ticker" not in correlation_matrix.columns:
		raise ValueError("correlation_matrix must contain a 'ticker' label column.")
	asset_names = [name for name in correlation_matrix.columns if name != "ticker"]
	if not asset_names or correlation_matrix.is_empty():
		raise ValueError("correlation_matrix must contain at least one asset.")
	if correlation_matrix.height != len(asset_names):
		raise ValueError("correlation_matrix must have one row per asset.")
	if not set(correlation_matrix["ticker"].to_list()).issuperset(asset_names):
		raise ValueError("correlation_matrix ticker labels must match its asset columns.")

	values = correlation_matrix.select(asset_names).cast(pl.Float64, strict=True).to_numpy()
	finite_values = values[np.isfinite(values)]
	if np.isinf(values).any():
		raise ValueError("correlation_matrix values cannot be infinite.")
	if ((finite_values < -1.0) | (finite_values > 1.0)).any():
		raise ValueError("correlation_matrix values must be between -1 and 1.")

	row_order = {name: index for index, name in enumerate(correlation_matrix["ticker"].to_list())}
	values = values[[row_order[name] for name in asset_names], :]
	output_file = Path(output_path)
	output_file.parent.mkdir(parents=True, exist_ok=True)
	figure_size = max(6.0, len(asset_names) * 0.75)
	figure, axis = plt.subplots(figsize=(figure_size, figure_size), constrained_layout=True)
	color_map = plt.get_cmap("coolwarm").copy()
	color_map.set_bad("#d9d9d9")
	image = axis.imshow(np.ma.masked_invalid(values), cmap=color_map, vmin=-1.0, vmax=1.0)
	axis.set_xticks(np.arange(len(asset_names)), asset_names, rotation=45, ha="right")
	axis.set_yticks(np.arange(len(asset_names)), asset_names)
	axis.set_title("Asset Correlation Matrix")
	for row_index in range(len(asset_names)):
		for column_index in range(len(asset_names)):
			cell_value = values[row_index, column_index]
			axis.text(
				column_index,
				row_index,
				f"{cell_value:.2f}" if np.isfinite(cell_value) else "N/A",
				ha="center",
				va="center",
				color="black",
				fontsize=8,
			)
	figure.colorbar(image, ax=axis, label="Pearson correlation", shrink=0.8)
	figure.savefig(output_file, format="pdf", bbox_inches="tight")
	plt.close(figure)
	return output_file


def generate_markowitz_pdf(
	MC_weights: pl.DataFrame,
	best_portfolios: pl.DataFrame,
	output_path: str | Path = "markowitz_frontier.pdf",
) -> Path:
	"""Save a risk-return scatter plot of simulated portfolios as a PDF.

	Downside deviation is used as the risk axis, expected annual return as the
	return axis, and point color represents the Sortino ratio. The portfolios
	provided in ``best_portfolios`` are highlighted.
	"""
	required_columns = {"expected_return", "downside_deviation", "sortino_ratio"}
	missing_columns = required_columns.difference(MC_weights.columns)
	if missing_columns:
		raise ValueError(
			"MC_weights is missing required columns: "
			+ ", ".join(sorted(missing_columns))
		)
	if MC_weights.is_empty():
		raise ValueError("MC_weights must contain at least one portfolio.")
	missing_best_columns = required_columns.difference(best_portfolios.columns)
	if missing_best_columns:
		raise ValueError(
			"best_portfolios is missing required columns: "
			+ ", ".join(sorted(missing_best_columns))
		)

	expected_returns = MC_weights["expected_return"].to_numpy()
	downside_deviations = MC_weights["downside_deviation"].to_numpy()
	sortino_ratios = MC_weights["sortino_ratio"].to_numpy()
	valid_points = np.isfinite(expected_returns) & np.isfinite(downside_deviations)
	if not valid_points.any():
		raise ValueError("MC_weights contains no finite risk-return points to plot.")

	finite_sortino = sortino_ratios[np.isfinite(sortino_ratios)]
	if finite_sortino.size:
		color_max = max(float(np.max(finite_sortino)), 1.0)
		colors = np.nan_to_num(sortino_ratios, nan=0.0, posinf=color_max, neginf=0.0)
		colors = np.clip(colors, 0.0, color_max)
	else:
		color_max = 1.0
		colors = np.zeros_like(sortino_ratios, dtype=float)

	output_file = Path(output_path)
	output_file.parent.mkdir(parents=True, exist_ok=True)
	figure, axis = plt.subplots(figsize=(10, 6.5), constrained_layout=True)
	points = axis.scatter(
		downside_deviations[valid_points],
		expected_returns[valid_points],
		c=colors[valid_points],
		cmap="viridis",
		vmin=0.0,
		vmax=color_max,
		s=24,
		alpha=0.72,
		edgecolors="none",
	)
	colorbar = figure.colorbar(points, ax=axis)
	colorbar.set_label("Sortino ratio")

	best_returns = best_portfolios["expected_return"].to_numpy()
	best_downside = best_portfolios["downside_deviation"].to_numpy()
	best_valid = np.isfinite(best_returns) & np.isfinite(best_downside)
	if best_valid.any():
		axis.scatter(
			best_downside[best_valid],
			best_returns[best_valid],
			marker="*",
			s=160,
			color="#d1495b",
			edgecolors="white",
			linewidths=0.8,
			label="Top Sortino portfolios",
			zorder=3,
		)

	minimum_risk_index = np.flatnonzero(valid_points)[
		np.argmin(downside_deviations[valid_points])
	]
	axis.scatter(
		downside_deviations[minimum_risk_index],
		expected_returns[minimum_risk_index],
		marker="D",
		s=120,
		color="blue",
		edgecolors="white",
		linewidths=0.8,
		label="Minimum Downside Risk Portfolio",
		zorder=4,
	)
	axis.legend(frameon=False, loc="best")

	axis.set(
		title="Markowitz Portfolio Simulation",
		xlabel="Annualized downside deviation",
		ylabel="Annualized expected return",
	)
	axis.grid(True, alpha=0.22)
	figure.savefig(output_file, format="pdf", bbox_inches="tight")
	plt.close(figure)
	return output_file


def generate_best_portfolios_pdf(
	best_portfolios: pl.DataFrame,
	output_path: str | Path = "best_portfolios_weights.pdf",
) -> Path:
	"""Save a grouped weight-composition chart for the selected portfolios."""
	if best_portfolios.is_empty():
		raise ValueError("best_portfolios must contain at least one portfolio.")

	metric_columns = {
		"expected_return", "downside_deviation", "sortino_ratio",
		"performance_pct", "ending_value",
	}
	asset_names = [name for name in best_portfolios.columns if name not in metric_columns]
	if not asset_names:
		raise ValueError("best_portfolios must contain at least one asset-weight column.")

	weights = best_portfolios.select(asset_names).cast(pl.Float64, strict=True).to_numpy()
	if not np.isfinite(weights).all():
		raise ValueError("Portfolio weights must all be finite numbers.")
	mean_weights = weights.mean(axis=0)

	portfolio_labels = []
	if "sortino_ratio" in best_portfolios.columns:
		portfolio_labels = [
			f"Portfolio {index + 1} (Sortino {ratio:.2f})"
			for index, ratio in enumerate(best_portfolios["sortino_ratio"].to_list())
		]
	else:
		portfolio_labels = [f"Portfolio {index + 1}" for index in range(best_portfolios.height)]
	portfolio_labels.append("Mean of selected portfolios")
	weights = np.vstack((weights, mean_weights))

	output_file = Path(output_path)
	output_file.parent.mkdir(parents=True, exist_ok=True)
	figure_width = max(9.0, len(asset_names) * 1.5)
	figure, axis = plt.subplots(figsize=(figure_width, 6), constrained_layout=True)
	positions = np.arange(len(asset_names), dtype=float)
	portfolio_count = len(portfolio_labels)
	bar_width = 0.8 / portfolio_count
	colors = plt.get_cmap("tab10")

	for portfolio_index, (portfolio_weights, label) in enumerate(zip(weights, portfolio_labels, strict=True)):
		offset = (portfolio_index - (portfolio_count - 1) / 2) * bar_width
		bars = axis.bar(
			positions + offset,
			portfolio_weights * 100.0,
			bar_width,
			label=label,
			color="#d1495b" if portfolio_index == portfolio_count - 1 else colors(portfolio_index % 10),
			edgecolor="black" if portfolio_index == portfolio_count - 1 else "none",
			linewidth=0.7 if portfolio_index == portfolio_count - 1 else 0,
			zorder=3 if portfolio_index == portfolio_count - 1 else 2,
		)
		axis.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)

	axis.set_xticks(positions, asset_names)
	axis.set_ylim(0, max(100.0, float(np.max(weights) * 110.0)))
	axis.set(
		title="Selected Portfolios and Mean Allocation",
		xlabel="Asset",
		ylabel="Portfolio weight (%)",
	)
	axis.grid(axis="y", alpha=0.22)
	axis.set_axisbelow(True)
	axis.legend(frameon=False, loc="best")
	figure.savefig(output_file, format="pdf", bbox_inches="tight")
	plt.close(figure)
	return output_file
