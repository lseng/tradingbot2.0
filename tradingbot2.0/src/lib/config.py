"""
Unified configuration management.

This module provides a centralized way to load, validate, and access
configuration for the trading system. It supports:
- YAML file loading
- Environment variable overrides
- Type validation via dataclasses
- Default values from constants

Configuration Hierarchy (highest to lowest priority):
1. Environment variables (TRADINGBOT_*)
2. User-provided config file
3. Default values from constants.py

Example usage:
    # Load config with environment overrides
    config = load_config("config/trading.yaml")

    # Access typed config sections
    print(config.risk.max_daily_loss)
    print(config.model.path)
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from src.lib.constants import (
    # Risk defaults
    DEFAULT_STARTING_CAPITAL,
    DEFAULT_MAX_DAILY_LOSS,
    DEFAULT_MAX_DAILY_DRAWDOWN,
    DEFAULT_MAX_PER_TRADE_RISK,
    DEFAULT_MAX_CONSECUTIVE_LOSSES,
    DEFAULT_KILL_SWITCH_THRESHOLD,
    DEFAULT_MIN_ACCOUNT_BALANCE,
    DEFAULT_RISK_PER_TRADE_PCT,
    DEFAULT_MIN_CONFIDENCE,
    # Feature defaults
    SCALPING_RETURN_PERIODS,
    SCALPING_EMA_PERIODS,
    DEFAULT_RSI_PERIOD,
    DEFAULT_ATR_PERIOD,
    # Model defaults
    DEFAULT_LOOKAHEAD_SECONDS,
    DEFAULT_THRESHOLD_TICKS,
    DEFAULT_BATCH_SIZE,
    DEFAULT_LEARNING_RATE,
    DEFAULT_WEIGHT_DECAY,
    DEFAULT_EPOCHS,
    DEFAULT_EARLY_STOPPING_PATIENCE,
    DEFAULT_GRAD_CLIP,
    # Walk-forward defaults
    DEFAULT_TRAINING_MONTHS,
    DEFAULT_VALIDATION_MONTHS,
    DEFAULT_TEST_MONTHS,
    DEFAULT_STEP_MONTHS,
    DEFAULT_MIN_TRADES_PER_FOLD,
    # API defaults
    TOPSTEPX_BASE_URL,
    TOPSTEPX_WS_MARKET_URL,
    TOPSTEPX_WS_TRADE_URL,
    # Contract defaults
    MES_ROUND_TRIP_COST,
    DEFAULT_SLIPPAGE_TICKS_NORMAL,
)


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass
class DataConfig:
    """Configuration for data loading and processing."""
    # Parquet data path
    parquet_path: str = "data/historical/MES/MES_1s_2years.parquet"
    # Legacy TXT data path
    txt_path: str = "data/historical/MES/MES_full_1min_continuous_UNadjusted.txt"
    # Session filtering
    use_rth_only: bool = True
    # Train/test split
    train_ratio: float = 0.8
    # Date range (optional)
    start_date: Optional[str] = None
    end_date: Optional[str] = None


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    # Return lookback periods (seconds for scalping)
    return_periods: list[int] = field(
        default_factory=lambda: list(SCALPING_RETURN_PERIODS)
    )
    # EMA periods
    ema_periods: list[int] = field(
        default_factory=lambda: list(SCALPING_EMA_PERIODS)
    )
    # Technical indicators
    rsi_period: int = DEFAULT_RSI_PERIOD
    atr_period: int = DEFAULT_ATR_PERIOD
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bollinger_period: int = 20
    bollinger_std: float = 2.0
    # Include session-based VWAP
    include_vwap: bool = True
    # Include time features
    include_time_features: bool = True
    # Include multi-timeframe features
    include_mtf_features: bool = True
    # Multi-timeframe aggregation windows (seconds)
    mtf_windows: list[int] = field(default_factory=lambda: [60, 300])


@dataclass
class ModelConfig:
    """Configuration for ML model."""
    # Model type: 'feedforward', 'lstm', 'hybrid', 'transformer'
    model_type: str = "feedforward"
    # Path to saved model
    path: Optional[str] = None
    # Config path (JSON)
    config_path: Optional[str] = None
    # Feature scaler path
    scaler_path: Optional[str] = None
    # Number of classes (3 for UP/FLAT/DOWN)
    num_classes: int = 3
    # FeedForward settings
    hidden_dims: list[int] = field(default_factory=lambda: [256, 128, 64])
    dropout_rate: float = 0.3
    use_batch_norm: bool = True
    # LSTM settings
    lstm_hidden_dim: int = 128
    lstm_num_layers: int = 2
    lstm_dropout: float = 0.3
    lstm_bidirectional: bool = False
    seq_length: int = 60
    # Transformer settings
    transformer_d_model: int = 64
    transformer_nhead: int = 4
    transformer_num_layers: int = 2
    transformer_dim_feedforward: int = 256
    transformer_dropout: float = 0.1
    transformer_pooling: str = 'last'  # 'last', 'mean', or 'cls'
    # Target variable
    lookahead_seconds: int = DEFAULT_LOOKAHEAD_SECONDS
    threshold_ticks: float = DEFAULT_THRESHOLD_TICKS
    # Inference
    min_confidence: float = DEFAULT_MIN_CONFIDENCE


@dataclass
class TrainingConfig:
    """Configuration for model training."""
    # Training parameters
    epochs: int = DEFAULT_EPOCHS
    batch_size: int = DEFAULT_BATCH_SIZE
    learning_rate: float = DEFAULT_LEARNING_RATE
    weight_decay: float = DEFAULT_WEIGHT_DECAY
    # Early stopping
    early_stopping_patience: int = DEFAULT_EARLY_STOPPING_PATIENCE
    min_delta: float = 0.001
    # Learning rate scheduler
    lr_scheduler_patience: int = 5
    lr_scheduler_factor: float = 0.5
    # Gradient clipping
    max_grad_norm: float = DEFAULT_GRAD_CLIP
    # Device
    device: str = "auto"  # 'auto', 'cpu', 'cuda', 'mps'
    # Reproducibility
    seed: int = 42
    # Use class weights for imbalanced data
    use_class_weights: bool = True


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation."""
    training_months: int = DEFAULT_TRAINING_MONTHS
    validation_months: int = DEFAULT_VALIDATION_MONTHS
    test_months: int = DEFAULT_TEST_MONTHS
    step_months: int = DEFAULT_STEP_MONTHS
    min_trades_per_fold: int = DEFAULT_MIN_TRADES_PER_FOLD
    # Window type
    expanding: bool = False  # True = expanding, False = rolling


@dataclass
class RiskConfig:
    """Configuration for risk management."""
    # Account settings
    starting_capital: float = DEFAULT_STARTING_CAPITAL
    min_account_balance: float = DEFAULT_MIN_ACCOUNT_BALANCE
    # Daily limits
    max_daily_loss: float = DEFAULT_MAX_DAILY_LOSS
    max_daily_drawdown: float = DEFAULT_MAX_DAILY_DRAWDOWN
    # Per-trade limits
    max_per_trade_risk: float = DEFAULT_MAX_PER_TRADE_RISK
    risk_per_trade_pct: float = DEFAULT_RISK_PER_TRADE_PCT
    # Circuit breakers
    max_consecutive_losses: int = DEFAULT_MAX_CONSECUTIVE_LOSSES
    kill_switch_threshold: float = DEFAULT_KILL_SWITCH_THRESHOLD
    # Stop loss
    stop_loss_atr_multiplier: float = 1.5
    default_stop_ticks: int = 8
    min_stop_ticks: int = 4
    max_stop_ticks: int = 16
    # Take profit
    default_rr_ratio: float = 1.5  # Risk:Reward
    # EOD management
    flatten_at_eod: bool = True


@dataclass
class ExecutionConfig:
    """Configuration for order execution."""
    # Order type preferences
    entry_order_type: str = "market"  # 'market' or 'limit'
    exit_order_type: str = "market"
    # Slippage
    slippage_ticks: float = DEFAULT_SLIPPAGE_TICKS_NORMAL
    # Commission
    commission_round_trip: float = MES_ROUND_TRIP_COST
    # Timeouts
    order_timeout_seconds: float = 5.0
    fill_timeout_seconds: float = 10.0
    # Cooldowns
    entry_cooldown_seconds: float = 5.0


@dataclass
class APIConfig:
    """Configuration for API connections."""
    # TopstepX
    base_url: str = TOPSTEPX_BASE_URL
    ws_market_url: str = TOPSTEPX_WS_MARKET_URL
    ws_trade_url: str = TOPSTEPX_WS_TRADE_URL
    # Credentials (loaded from env)
    api_key: Optional[str] = None
    account_id: Optional[str] = None
    # Connection settings
    max_retries: int = 3
    retry_delay_seconds: float = 1.0
    connection_timeout_seconds: float = 30.0


@dataclass
class OutputConfig:
    """Configuration for output and logging."""
    # Output directory
    output_dir: str = "./results"
    logs_dir: str = "./logs"
    # Save options
    save_model: bool = True
    save_plots: bool = True
    save_trades: bool = True
    # Log level
    log_level: str = "INFO"
    # Verbose mode
    verbose: bool = False


@dataclass
class TradingConfig:
    """Main configuration container."""
    data: DataConfig = field(default_factory=DataConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    walk_forward: WalkForwardConfig = field(default_factory=WalkForwardConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    api: APIConfig = field(default_factory=APIConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    # Contract to trade
    contract: str = "MES"
    # Paper trading mode
    paper_trading: bool = True


# =============================================================================
# Configuration Loading
# =============================================================================

def load_config(
    config_path: Optional[str] = None,
    override_env: bool = True
) -> TradingConfig:
    """
    Load configuration from YAML file with optional environment overrides.

    Args:
        config_path: Path to YAML config file (optional)
        override_env: If True, apply environment variable overrides

    Returns:
        TradingConfig instance

    Example:
        config = load_config("config/trading.yaml")
        print(config.risk.max_daily_loss)  # $50.0
    """
    config = TradingConfig()

    # Load from YAML file if provided
    if config_path:
        config = _load_from_yaml(config_path, config)

    # Apply environment variable overrides
    if override_env:
        config = _apply_env_overrides(config)

    return config


def _load_from_yaml(config_path: str, base_config: TradingConfig) -> TradingConfig:
    """Load configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path) as f:
        yaml_data = yaml.safe_load(f)

    if yaml_data is None:
        return base_config

    # Update each section
    if "data" in yaml_data:
        base_config.data = _update_dataclass(base_config.data, yaml_data["data"])

    if "features" in yaml_data:
        base_config.features = _update_dataclass(base_config.features, yaml_data["features"])

    if "model" in yaml_data:
        base_config.model = _update_dataclass(base_config.model, yaml_data["model"])

    if "training" in yaml_data:
        base_config.training = _update_dataclass(base_config.training, yaml_data["training"])

    if "walk_forward" in yaml_data:
        base_config.walk_forward = _update_dataclass(base_config.walk_forward, yaml_data["walk_forward"])

    if "risk" in yaml_data:
        base_config.risk = _update_dataclass(base_config.risk, yaml_data["risk"])

    if "execution" in yaml_data or "strategy" in yaml_data:
        # Support both 'execution' and legacy 'strategy' key
        exec_data = yaml_data.get("execution", yaml_data.get("strategy", {}))
        base_config.execution = _update_dataclass(base_config.execution, exec_data)

    if "api" in yaml_data:
        base_config.api = _update_dataclass(base_config.api, yaml_data["api"])

    if "output" in yaml_data:
        base_config.output = _update_dataclass(base_config.output, yaml_data["output"])

    # Top-level fields
    if "contract" in yaml_data:
        base_config.contract = yaml_data["contract"]

    if "paper_trading" in yaml_data:
        base_config.paper_trading = yaml_data["paper_trading"]

    if "seed" in yaml_data:
        base_config.training.seed = yaml_data["seed"]

    return base_config


def _update_dataclass(instance: Any, data: dict) -> Any:
    """Update dataclass fields from dictionary."""
    if not data:
        return instance

    field_names = {f.name for f in instance.__dataclass_fields__.values()}

    for key, value in data.items():
        # Handle nested keys (e.g., 'lstm.hidden_dim' -> 'lstm_hidden_dim')
        normalized_key = key.replace(".", "_").replace("-", "_")

        if normalized_key in field_names:
            setattr(instance, normalized_key, value)
        elif key in field_names:
            setattr(instance, key, value)

    return instance


def _apply_env_overrides(config: TradingConfig) -> TradingConfig:
    """Apply environment variable overrides to config."""

    # API credentials (always from env for security)
    config.api.api_key = os.getenv("TOPSTEPX_API_KEY")
    config.api.account_id = os.getenv("TOPSTEPX_ACCOUNT_ID")

    # Allow override of key risk parameters
    if env_val := os.getenv("TRADINGBOT_MAX_DAILY_LOSS"):
        config.risk.max_daily_loss = float(env_val)

    if env_val := os.getenv("TRADINGBOT_STARTING_CAPITAL"):
        config.risk.starting_capital = float(env_val)

    if env_val := os.getenv("TRADINGBOT_MIN_CONFIDENCE"):
        config.model.min_confidence = float(env_val)

    if env_val := os.getenv("TRADINGBOT_PAPER_TRADING"):
        config.paper_trading = env_val.lower() in ("true", "1", "yes")

    # Data paths
    if env_val := os.getenv("TRADINGBOT_DATA_PATH"):
        config.data.parquet_path = env_val

    if env_val := os.getenv("TRADINGBOT_MODEL_PATH"):
        config.model.path = env_val

    # Output
    if env_val := os.getenv("TRADINGBOT_OUTPUT_DIR"):
        config.output.output_dir = env_val

    if env_val := os.getenv("TRADINGBOT_LOG_LEVEL"):
        config.output.log_level = env_val.upper()

    return config


def load_config_from_env() -> TradingConfig:
    """
    Load configuration purely from environment variables.

    Useful for containerized deployments where config files aren't available.

    Returns:
        TradingConfig instance
    """
    return load_config(config_path=None, override_env=True)


# =============================================================================
# Configuration Validation
# =============================================================================

class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


def validate_config(config: TradingConfig) -> list[str]:
    """
    Validate configuration values.

    Args:
        config: TradingConfig to validate

    Returns:
        List of validation warnings (empty if valid)

    Raises:
        ConfigValidationError: If critical validation fails
    """
    warnings = []
    errors = []

    # Risk validation
    if config.risk.starting_capital < config.risk.min_account_balance:
        errors.append(
            f"starting_capital ({config.risk.starting_capital}) must be >= "
            f"min_account_balance ({config.risk.min_account_balance})"
        )

    if config.risk.max_daily_loss > config.risk.starting_capital * 0.10:
        warnings.append(
            f"max_daily_loss ({config.risk.max_daily_loss}) is > 10% of capital - "
            f"consider reducing for capital preservation"
        )

    if config.risk.kill_switch_threshold <= config.risk.max_daily_loss:
        errors.append(
            f"kill_switch_threshold ({config.risk.kill_switch_threshold}) must be > "
            f"max_daily_loss ({config.risk.max_daily_loss})"
        )

    # Model validation
    if config.model.min_confidence < 0.5:
        warnings.append(
            f"min_confidence ({config.model.min_confidence}) below 0.5 may lead to "
            f"excessive trading on weak signals"
        )

    if config.model.min_confidence > 0.9:
        warnings.append(
            f"min_confidence ({config.model.min_confidence}) above 0.9 may result in "
            f"very few trades"
        )

    # Training validation
    if config.training.batch_size < 32:
        warnings.append(
            f"batch_size ({config.training.batch_size}) below 32 may cause unstable training"
        )

    if config.training.learning_rate > 0.01:
        warnings.append(
            f"learning_rate ({config.training.learning_rate}) above 0.01 may cause "
            f"training instability"
        )

    # Walk-forward validation
    if config.walk_forward.training_months < 3:
        warnings.append(
            f"training_months ({config.walk_forward.training_months}) below 3 may not "
            f"provide enough data for robust training"
        )

    if config.walk_forward.min_trades_per_fold < 50:
        warnings.append(
            f"min_trades_per_fold ({config.walk_forward.min_trades_per_fold}) below 50 "
            f"may not be statistically significant"
        )

    # Execution validation
    if config.execution.slippage_ticks < 0:
        errors.append("slippage_ticks cannot be negative")

    if config.execution.commission_round_trip < 0:
        errors.append("commission_round_trip cannot be negative")

    # API validation (only for live trading)
    if not config.paper_trading:
        if not config.api.api_key:
            errors.append("API key required for live trading - set TOPSTEPX_API_KEY")
        if not config.api.account_id:
            errors.append("Account ID required for live trading - set TOPSTEPX_ACCOUNT_ID")

    if errors:
        raise ConfigValidationError("Configuration validation failed:\n" +
                                   "\n".join(f"  - {e}" for e in errors))

    return warnings


# =============================================================================
# Configuration Export
# =============================================================================

def config_to_dict(config: TradingConfig) -> dict:
    """
    Convert TradingConfig to dictionary for serialization.

    Args:
        config: Configuration to convert

    Returns:
        Dictionary representation (YAML-safe)
    """
    from dataclasses import asdict
    result = asdict(config)

    # Remove sensitive data
    if "api" in result:
        result["api"]["api_key"] = "***" if result["api"].get("api_key") else None

    return result


def save_config(config: TradingConfig, path: str) -> None:
    """
    Save configuration to YAML file.

    Args:
        config: Configuration to save
        path: Output file path
    """
    data = config_to_dict(config)

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
