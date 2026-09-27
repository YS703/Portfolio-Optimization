from pathlib import Path
from math import isfinite
from yaml import safe_load

REQUIRED_KEYS = (
    "tickers_list",
    "column",
    "start_date",
    "end_date",
    "interval",
    "monthly_deposit",
    "correlation_threshold",
    "n_simulations",
    "target_return",
    "n_best_portfolios",
)
SHARED_KEYS = ("column", "start_date", "end_date", "interval")

def validate_config(config: object) -> None:
    """
    Validate the shared market-data arguments and ticker list.
    """
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping")

    missing_keys = [key for key in REQUIRED_KEYS if key not in config]
    if missing_keys:
        raise ValueError(f"Missing required configuration keys: {', '.join(missing_keys)}")

    if not isinstance(config["tickers_list"], list) or not config["tickers_list"]:
        raise ValueError("Configuration key 'tickers_list' must be a non-empty list of ticker strings.")

    invalid_tickers = [
        ticker for ticker in config["tickers_list"]
        if not isinstance(ticker, str) or not ticker.strip()
    ]
    if invalid_tickers:
        raise ValueError("Every item in configuration key 'tickers_list' must be a non-empty string.")

    invalid_keys = [
        key for key in SHARED_KEYS
        if not isinstance(config[key], str) or not config[key].strip()
    ]
    if invalid_keys:
        raise ValueError(f"Configuration values must be non-empty strings: {', '.join(invalid_keys)}")

    monthly_deposit = config["monthly_deposit"]
    if (
        isinstance(monthly_deposit, bool)
        or not isinstance(monthly_deposit, (int, float))
        or not isfinite(monthly_deposit)
        or monthly_deposit <= 0
    ):
        raise ValueError("Configuration key 'monthly_deposit' must be a positive finite number.")

    correlation_threshold = config["correlation_threshold"]
    if (
        isinstance(correlation_threshold, bool)
        or not isinstance(correlation_threshold, (int, float))
        or not isfinite(correlation_threshold)
        or not 0 <= correlation_threshold <= 1
    ):
        raise ValueError("Configuration key 'correlation_threshold' must be between 0 and 1.")

    for key in ("n_simulations", "n_best_portfolios"):
        value = config[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"Configuration key '{key}' must be a positive integer.")

    target_return = config["target_return"]
    if (
        isinstance(target_return, bool)
        or not isinstance(target_return, (int, float))
        or not isfinite(target_return)
    ):
        raise ValueError("Configuration key 'target_return' must be a finite number.")


def load_config(config_file: str | Path) -> dict[str, str | list[str] | float | int]:
    """
    Load the ticker list and shared market-data arguments from YAML.
    """
    with Path(config_file).open(encoding="utf-8") as file:
        config = safe_load(file)

    validate_config(config)
    return {
        "tickers_list": config["tickers_list"],
        **{key: config[key] for key in SHARED_KEYS},
        "monthly_deposit": float(config["monthly_deposit"]),
        "correlation_threshold": float(config["correlation_threshold"]),
        "n_simulations": config["n_simulations"],
        "target_return": float(config["target_return"]),
        "n_best_portfolios": config["n_best_portfolios"],
    }
