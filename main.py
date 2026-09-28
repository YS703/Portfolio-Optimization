import logging

from src.Data.load_config import load_config
from src.Data.market_data import load_data
from src.Optimization.correlation import get_correlation_matrix, get_uncorrelated_assets  
from src.Optimization.monte_carlo import simulate_random_weights
from src.Analytics.reporting import (
    generate_best_portfolios_pdf,
    generate_correlation_pdf,
    generate_markowitz_pdf,
)

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    # 1. Loading configuration
    try:
        logger.info("1 - Loading configuration...")
        config = load_config("config.yaml")
    except Exception as e:
        logger.error(f"Fatal error during initialization : {e}")
        return

    # 2. Loading market data
    try:
        logger.info("2 - Loading market data...")
        data = load_data(
            tickers_list=config["tickers_list"],
            column=config["column"],
            start_date=config["start_date"],
            end_date=config["end_date"],
            interval=config["interval"],
        )
    except Exception as e:
        logger.error(f"Fatal error during market data loading : {e}")
        return

    # 3. Identifying uncorrelated assets
    try:
        logger.info("3 - Identifying uncorrelated assets...")
        threshold = config["correlation_threshold"]
        correlation_matrix = get_correlation_matrix(data)
        uncorrelated_assets_data = get_uncorrelated_assets(data, correlation_matrix, threshold)
    except Exception as e:
        logger.error(f"Fatal error during correlation analysis : {e}")
        return

    # 4. Simulating random portfolio weights
    try:
        logger.info("4 - Simulating random portfolio weights...")
        MC_weights = simulate_random_weights(
            uncorrelated_assets_data,
            monthly_deposit=config["monthly_deposit"],
            n_portfolios=config["n_simulations"],
            periods_per_year=12,
            target_return=config["target_return"],
            random_seed=42,
        )
        best_portfolios = MC_weights.sort("sortino_ratio", descending=True).head(
            config["n_best_portfolios"]
        )
    except Exception as e:
        logger.error(f"Fatal error during Monte Carlo simulation : {e}")
        return

    # 5. Displaying results
    try:
        logger.info("5 - Displaying results...")
        correlation_report_path = generate_correlation_pdf(correlation_matrix)
        report_path = generate_markowitz_pdf(MC_weights, best_portfolios)
        weights_report_path = generate_best_portfolios_pdf(best_portfolios)
    except Exception as e:
        logger.error(f"Fatal error during result display : {e}")
        return

    logger.info(
        f"Reports generated successfully: {correlation_report_path}, "
        f"{report_path}, {weights_report_path}"
    )

main()