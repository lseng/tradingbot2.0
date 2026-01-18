"""
Tests for src/lib/ modules.

Tests cover:
- constants.py: Contract specifications, conversions
- time_utils.py: Timezone handling, session detection, EOD phases
- config.py: Configuration loading, validation
- logging_utils.py: Log formatting, TradingLogger

Run with: pytest tests/test_lib.py -v
"""

import logging
import os
import tempfile
from datetime import datetime, date, time, timedelta
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest
import yaml


# =============================================================================
# Constants Tests
# =============================================================================

class TestContractSpecs:
    """Tests for contract specifications."""

    def test_mes_tick_values(self):
        """Test MES tick size and value."""
        from src.lib.constants import MES_TICK_SIZE, MES_TICK_VALUE, MES_POINT_VALUE

        assert MES_TICK_SIZE == 0.25
        assert MES_TICK_VALUE == 1.25
        assert MES_POINT_VALUE == 5.00
        # 4 ticks = 1 point
        assert MES_POINT_VALUE == MES_TICK_VALUE * 4

    def test_mes_commission(self):
        """Test MES commission structure."""
        from src.lib.constants import (
            MES_COMMISSION_PER_SIDE,
            MES_EXCHANGE_FEE_PER_SIDE,
            MES_ROUND_TRIP_COST,
        )

        assert MES_COMMISSION_PER_SIDE == 0.20
        assert MES_EXCHANGE_FEE_PER_SIDE == 0.22
        # Total per side = 0.42 (use pytest.approx for float comparison)
        assert MES_COMMISSION_PER_SIDE + MES_EXCHANGE_FEE_PER_SIDE == pytest.approx(0.42)
        # Round trip = 0.84
        assert MES_ROUND_TRIP_COST == pytest.approx(0.84)

    def test_contract_spec_properties(self):
        """Test ContractSpec computed properties."""
        from src.lib.constants import MES_SPEC

        assert MES_SPEC.symbol == "MES"
        assert MES_SPEC.total_per_side == pytest.approx(0.42)
        assert MES_SPEC.round_trip_cost == pytest.approx(0.84)
        assert MES_SPEC.ticks_per_point == 4

    def test_contract_spec_conversions(self):
        """Test ContractSpec conversion methods."""
        from src.lib.constants import MES_SPEC

        # Price to ticks
        assert MES_SPEC.price_to_ticks(1.0) == 4.0  # 1 point = 4 ticks
        assert MES_SPEC.price_to_ticks(0.25) == 1.0  # 1 tick

        # Ticks to price
        assert MES_SPEC.ticks_to_price(4) == 1.0
        assert MES_SPEC.ticks_to_price(1) == 0.25

        # Ticks to dollars
        assert MES_SPEC.ticks_to_dollars(1) == 1.25
        assert MES_SPEC.ticks_to_dollars(4) == 5.00
        assert MES_SPEC.ticks_to_dollars(4, contracts=2) == 10.00

        # Dollars to ticks
        assert MES_SPEC.dollars_to_ticks(1.25) == 1.0
        assert MES_SPEC.dollars_to_ticks(5.00) == 4.0

    def test_breakeven_ticks(self):
        """Test breakeven tick calculation."""
        from src.lib.constants import MES_SPEC

        # Breakeven = round_trip_cost / tick_value
        # = 0.84 / 1.25 = 0.672 ticks
        breakeven = MES_SPEC.breakeven_ticks(contracts=1)
        assert 0.6 < breakeven < 0.7

    def test_get_contract_spec(self):
        """Test contract spec lookup."""
        from src.lib.constants import get_contract_spec

        mes = get_contract_spec("MES")
        assert mes.symbol == "MES"

        es = get_contract_spec("ES")
        assert es.symbol == "ES"

        # Case insensitive
        mes_lower = get_contract_spec("mes")
        assert mes_lower.symbol == "MES"

    def test_get_contract_spec_unknown(self):
        """Test unknown contract raises error."""
        from src.lib.constants import get_contract_spec

        with pytest.raises(ValueError, match="Unknown contract symbol"):
            get_contract_spec("UNKNOWN")

    def test_all_contract_specs(self):
        """Test all predefined contract specs exist."""
        from src.lib.constants import MES_SPEC, ES_SPEC, MNQ_SPEC, NQ_SPEC

        for spec in [MES_SPEC, ES_SPEC, MNQ_SPEC, NQ_SPEC]:
            assert spec.symbol is not None
            assert spec.tick_size > 0
            assert spec.tick_value > 0
            assert spec.point_value > 0

    def test_session_times(self):
        """Test session time constants."""
        from src.lib.constants import (
            RTH_START, RTH_END, ETH_START, ETH_END,
            EOD_FLATTEN_TIME,
        )

        assert RTH_START == time(9, 30)
        assert RTH_END == time(16, 0)
        assert ETH_START == time(18, 0)
        assert ETH_END == time(17, 0)
        assert EOD_FLATTEN_TIME == time(16, 30)


# =============================================================================
# Time Utils Tests
# =============================================================================

class TestTimezoneConversion:
    """Tests for timezone conversion functions."""

    def test_get_ny_now(self):
        """Test getting current NY time."""
        from src.lib.time_utils import get_ny_now
        from src.lib.constants import NY_TIMEZONE

        now = get_ny_now()
        assert now.tzinfo is not None
        assert now.tzinfo == NY_TIMEZONE

    def test_to_ny_time_utc(self):
        """Test converting UTC to NY."""
        from src.lib.time_utils import to_ny_time

        # UTC time
        utc_time = datetime(2025, 1, 15, 14, 0, 0, tzinfo=ZoneInfo("UTC"))

        # Convert to NY (should be 9:00 AM in winter)
        ny_time = to_ny_time(utc_time)
        assert ny_time.hour == 9
        assert ny_time.tzinfo == ZoneInfo("America/New_York")

    def test_to_ny_time_naive(self):
        """Test converting naive datetime (assumed UTC)."""
        from src.lib.time_utils import to_ny_time

        # Naive datetime (assumed UTC)
        naive_time = datetime(2025, 1, 15, 14, 0, 0)

        ny_time = to_ny_time(naive_time)
        assert ny_time.hour == 9
        assert ny_time.tzinfo is not None

    def test_to_utc(self):
        """Test converting to UTC."""
        from src.lib.time_utils import to_utc

        # NY time
        ny_time = datetime(2025, 1, 15, 9, 30, 0, tzinfo=ZoneInfo("America/New_York"))

        utc_time = to_utc(ny_time)
        assert utc_time.hour == 14
        assert utc_time.minute == 30


class TestSessionDetection:
    """Tests for session detection functions."""

    def test_is_rth_during_rth(self):
        """Test RTH detection during RTH."""
        from src.lib.time_utils import is_rth

        # 10:30 AM NY on a Wednesday
        rth_time = datetime(2025, 1, 15, 10, 30, tzinfo=ZoneInfo("America/New_York"))
        assert is_rth(rth_time) is True

    def test_is_rth_before_open(self):
        """Test RTH detection before market open."""
        from src.lib.time_utils import is_rth

        # 9:00 AM NY (before 9:30 open)
        before_open = datetime(2025, 1, 15, 9, 0, tzinfo=ZoneInfo("America/New_York"))
        assert is_rth(before_open) is False

    def test_is_rth_after_close(self):
        """Test RTH detection after market close."""
        from src.lib.time_utils import is_rth

        # 4:30 PM NY (after 4:00 close)
        after_close = datetime(2025, 1, 15, 16, 30, tzinfo=ZoneInfo("America/New_York"))
        assert is_rth(after_close) is False

    def test_is_rth_weekend(self):
        """Test RTH detection on weekend."""
        from src.lib.time_utils import is_rth

        # Saturday 10:30 AM
        saturday = datetime(2025, 1, 18, 10, 30, tzinfo=ZoneInfo("America/New_York"))
        assert is_rth(saturday) is False

    def test_is_eth_evening(self):
        """Test ETH detection in evening session."""
        from src.lib.time_utils import is_eth

        # 7:00 PM NY on a Monday (ETH)
        evening = datetime(2025, 1, 13, 19, 0, tzinfo=ZoneInfo("America/New_York"))
        assert is_eth(evening) is True

    def test_is_eth_during_rth(self):
        """Test ETH detection during RTH (should be False)."""
        from src.lib.time_utils import is_eth

        # 10:30 AM NY (RTH, not ETH)
        rth_time = datetime(2025, 1, 15, 10, 30, tzinfo=ZoneInfo("America/New_York"))
        assert is_eth(rth_time) is False

    def test_is_market_open(self):
        """Test market open detection."""
        from src.lib.time_utils import is_market_open

        # During RTH
        rth = datetime(2025, 1, 15, 10, 30, tzinfo=ZoneInfo("America/New_York"))
        assert is_market_open(rth) is True

        # During ETH
        eth = datetime(2025, 1, 13, 20, 0, tzinfo=ZoneInfo("America/New_York"))
        assert is_market_open(eth) is True

        # Saturday (closed)
        saturday = datetime(2025, 1, 18, 10, 0, tzinfo=ZoneInfo("America/New_York"))
        assert is_market_open(saturday) is False


class TestEODManagement:
    """Tests for EOD management functions."""

    def test_eod_phase_normal(self):
        """Test normal phase before 4 PM."""
        from src.lib.time_utils import get_eod_phase, EODPhase

        # 2:00 PM
        normal_time = datetime(2025, 1, 15, 14, 0, tzinfo=ZoneInfo("America/New_York"))
        assert get_eod_phase(normal_time) == EODPhase.NORMAL

    def test_eod_phase_reduced_size(self):
        """Test reduced size phase at 4 PM."""
        from src.lib.time_utils import get_eod_phase, EODPhase

        # 4:05 PM
        reduced_time = datetime(2025, 1, 15, 16, 5, tzinfo=ZoneInfo("America/New_York"))
        assert get_eod_phase(reduced_time) == EODPhase.REDUCED_SIZE

    def test_eod_phase_close_only(self):
        """Test close only phase at 4:15 PM."""
        from src.lib.time_utils import get_eod_phase, EODPhase

        # 4:20 PM
        close_only_time = datetime(2025, 1, 15, 16, 20, tzinfo=ZoneInfo("America/New_York"))
        assert get_eod_phase(close_only_time) == EODPhase.CLOSE_ONLY

    def test_eod_phase_aggressive_exit(self):
        """Test aggressive exit phase at 4:25 PM."""
        from src.lib.time_utils import get_eod_phase, EODPhase

        # 4:27 PM
        flatten_time = datetime(2025, 1, 15, 16, 27, tzinfo=ZoneInfo("America/New_York"))
        assert get_eod_phase(flatten_time) == EODPhase.AGGRESSIVE_EXIT

    def test_eod_phase_must_be_flat(self):
        """Test must be flat phase at 4:30 PM."""
        from src.lib.time_utils import get_eod_phase, EODPhase

        # 4:30 PM
        flat_time = datetime(2025, 1, 15, 16, 30, tzinfo=ZoneInfo("America/New_York"))
        assert get_eod_phase(flat_time) == EODPhase.MUST_BE_FLAT

    def test_eod_size_multiplier(self):
        """Test EOD position size multiplier."""
        from src.lib.time_utils import get_eod_size_multiplier

        # Normal = 1.0
        normal = datetime(2025, 1, 15, 14, 0, tzinfo=ZoneInfo("America/New_York"))
        assert get_eod_size_multiplier(normal) == 1.0

        # Reduced = 0.5
        reduced = datetime(2025, 1, 15, 16, 5, tzinfo=ZoneInfo("America/New_York"))
        assert get_eod_size_multiplier(reduced) == 0.5

        # Close only = 0.0
        close_only = datetime(2025, 1, 15, 16, 20, tzinfo=ZoneInfo("America/New_York"))
        assert get_eod_size_multiplier(close_only) == 0.0

    def test_can_open_new_position(self):
        """Test new position permission."""
        from src.lib.time_utils import can_open_new_position

        # Normal - can open
        normal = datetime(2025, 1, 15, 14, 0, tzinfo=ZoneInfo("America/New_York"))
        assert can_open_new_position(normal) is True

        # Close only - cannot open
        close_only = datetime(2025, 1, 15, 16, 20, tzinfo=ZoneInfo("America/New_York"))
        assert can_open_new_position(close_only) is False

    def test_should_flatten(self):
        """Test flatten signal."""
        from src.lib.time_utils import should_flatten

        # Before 4:25 - no flatten
        before = datetime(2025, 1, 15, 16, 20, tzinfo=ZoneInfo("America/New_York"))
        assert should_flatten(before) is False

        # After 4:25 - flatten
        after = datetime(2025, 1, 15, 16, 28, tzinfo=ZoneInfo("America/New_York"))
        assert should_flatten(after) is True

    def test_minutes_to_close(self):
        """Test minutes to close calculation."""
        from src.lib.time_utils import minutes_to_close

        # 3:30 PM - 30 minutes to 4 PM
        test_time = datetime(2025, 1, 15, 15, 30, tzinfo=ZoneInfo("America/New_York"))
        minutes = minutes_to_close(test_time)
        assert 29 < minutes < 31


class TestTradingCalendar:
    """Tests for trading calendar functions."""

    def test_is_trading_day_weekday(self):
        """Test trading day detection for weekday."""
        from src.lib.time_utils import is_trading_day

        # Wednesday January 15, 2025
        wednesday = date(2025, 1, 15)
        assert is_trading_day(wednesday) is True

    def test_is_trading_day_weekend(self):
        """Test trading day detection for weekend."""
        from src.lib.time_utils import is_trading_day

        # Saturday
        saturday = date(2025, 1, 18)
        assert is_trading_day(saturday) is False

        # Sunday
        sunday = date(2025, 1, 19)
        assert is_trading_day(sunday) is False

    def test_is_trading_day_holiday(self):
        """Test trading day detection for holiday."""
        from src.lib.time_utils import is_trading_day

        # New Year's Day 2025
        new_year = date(2025, 1, 1)
        assert is_trading_day(new_year) is False

    def test_get_next_trading_day(self):
        """Test next trading day calculation."""
        from src.lib.time_utils import get_next_trading_day

        # Friday -> Monday (but Jan 20, 2025 is MLK Day holiday)
        # So next trading day after Jan 17 is Jan 21 (Tuesday)
        friday = date(2025, 1, 17)
        next_day = get_next_trading_day(friday)
        assert next_day == date(2025, 1, 21)  # Tuesday (MLK Day on Monday)

    def test_get_previous_trading_day(self):
        """Test previous trading day calculation."""
        from src.lib.time_utils import get_previous_trading_day

        # Monday -> Friday
        monday = date(2025, 1, 20)
        prev_day = get_previous_trading_day(monday)
        assert prev_day.weekday() == 4  # Friday

    def test_count_trading_days(self):
        """Test trading days count."""
        from src.lib.time_utils import count_trading_days

        # One week with no holidays
        start = date(2025, 1, 13)  # Monday
        end = date(2025, 1, 17)    # Friday
        count = count_trading_days(start, end)
        assert count == 5  # Mon-Fri


# =============================================================================
# Config Tests
# =============================================================================

class TestConfigDefaults:
    """Tests for configuration defaults."""

    def test_default_trading_config(self):
        """Test default TradingConfig values."""
        from src.lib.config import TradingConfig

        config = TradingConfig()

        # Risk defaults
        assert config.risk.starting_capital == 1000.0
        assert config.risk.max_daily_loss == 50.0
        assert config.risk.min_account_balance == 700.0

        # Model defaults
        assert config.model.min_confidence == 0.60
        assert config.model.num_classes == 3

        # Paper trading default
        assert config.paper_trading is True

    def test_risk_config_defaults(self):
        """Test RiskConfig default values."""
        from src.lib.config import RiskConfig

        risk = RiskConfig()

        assert risk.max_daily_loss == 50.0
        assert risk.max_daily_drawdown == 75.0
        assert risk.max_per_trade_risk == 25.0
        assert risk.kill_switch_threshold == 300.0
        assert risk.max_consecutive_losses == 5

    def test_feature_config_defaults(self):
        """Test FeatureConfig default values."""
        from src.lib.config import FeatureConfig

        features = FeatureConfig()

        # Scalping periods (seconds)
        assert features.return_periods == [1, 5, 10, 30, 60]
        assert features.ema_periods == [9, 21, 50, 200]

        # Technical indicators
        assert features.rsi_period == 14
        assert features.atr_period == 14


class TestConfigLoading:
    """Tests for configuration loading."""

    def test_load_config_no_file(self):
        """Test loading config with no file (defaults only)."""
        from src.lib.config import load_config

        config = load_config()
        assert config.risk.starting_capital == 1000.0

    def test_load_config_from_yaml(self):
        """Test loading config from YAML file."""
        from src.lib.config import load_config

        # Create temp YAML file
        yaml_content = """
risk:
  starting_capital: 2000.0
  max_daily_loss: 100.0
model:
  min_confidence: 0.70
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            temp_path = f.name

        try:
            config = load_config(temp_path)

            assert config.risk.starting_capital == 2000.0
            assert config.risk.max_daily_loss == 100.0
            assert config.model.min_confidence == 0.70
        finally:
            os.unlink(temp_path)

    def test_load_config_file_not_found(self):
        """Test loading nonexistent config file."""
        from src.lib.config import load_config

        with pytest.raises(FileNotFoundError):
            load_config("/nonexistent/path.yaml")

    def test_load_config_env_override(self):
        """Test environment variable overrides."""
        from src.lib.config import load_config

        with patch.dict(os.environ, {
            "TRADINGBOT_MAX_DAILY_LOSS": "75.0",
            "TRADINGBOT_MIN_CONFIDENCE": "0.65"
        }):
            config = load_config()

            assert config.risk.max_daily_loss == 75.0
            assert config.model.min_confidence == 0.65

    def test_load_config_api_credentials(self):
        """Test API credentials from env."""
        from src.lib.config import load_config

        with patch.dict(os.environ, {
            "TOPSTEPX_API_KEY": "test-api-key",
            "TOPSTEPX_ACCOUNT_ID": "12345"
        }):
            config = load_config()

            assert config.api.api_key == "test-api-key"
            assert config.api.account_id == "12345"


class TestConfigValidation:
    """Tests for configuration validation."""

    def test_validate_valid_config(self):
        """Test validating a valid config."""
        from src.lib.config import TradingConfig, validate_config

        config = TradingConfig()
        warnings = validate_config(config)

        # Should pass without errors (may have warnings)
        assert isinstance(warnings, list)

    def test_validate_invalid_capital(self):
        """Test validation catches invalid capital settings."""
        from src.lib.config import TradingConfig, validate_config, ConfigValidationError

        config = TradingConfig()
        config.risk.starting_capital = 500.0  # Below min_account_balance

        with pytest.raises(ConfigValidationError, match="starting_capital"):
            validate_config(config)

    def test_validate_invalid_kill_switch(self):
        """Test validation catches invalid kill switch."""
        from src.lib.config import TradingConfig, validate_config, ConfigValidationError

        config = TradingConfig()
        config.risk.kill_switch_threshold = 40.0  # Below max_daily_loss

        with pytest.raises(ConfigValidationError, match="kill_switch_threshold"):
            validate_config(config)

    def test_validate_live_trading_no_api_key(self):
        """Test validation requires API key for live trading."""
        from src.lib.config import TradingConfig, validate_config, ConfigValidationError

        config = TradingConfig()
        config.paper_trading = False
        config.api.api_key = None

        with pytest.raises(ConfigValidationError, match="API key"):
            validate_config(config)

    def test_validate_warnings(self):
        """Test validation produces appropriate warnings."""
        from src.lib.config import TradingConfig, validate_config

        config = TradingConfig()
        config.model.min_confidence = 0.45  # Below 0.5

        warnings = validate_config(config)
        assert any("min_confidence" in w for w in warnings)


class TestConfigSerialization:
    """Tests for config serialization."""

    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        from src.lib.config import TradingConfig, config_to_dict

        config = TradingConfig()
        config.api.api_key = "secret-key"

        data = config_to_dict(config)

        # Check structure
        assert "risk" in data
        assert "model" in data
        assert "api" in data

        # Check sensitive data masked
        assert data["api"]["api_key"] == "***"

    def test_save_config(self):
        """Test saving config to YAML file."""
        from src.lib.config import TradingConfig, save_config, load_config

        config = TradingConfig()
        config.risk.starting_capital = 5000.0

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            temp_path = f.name

        try:
            save_config(config, temp_path)

            # Verify file exists and is valid YAML
            with open(temp_path) as f:
                data = yaml.safe_load(f)

            assert data["risk"]["starting_capital"] == 5000.0
        finally:
            os.unlink(temp_path)


# =============================================================================
# Logging Tests
# =============================================================================

class TestLoggingSetup:
    """Tests for logging setup."""

    def test_setup_logging_default(self):
        """Test default logging setup."""
        from src.lib.logging_utils import setup_logging

        logger = setup_logging()
        assert logger is not None
        assert logger.level == logging.INFO

    def test_setup_logging_debug(self):
        """Test debug level logging setup."""
        from src.lib.logging_utils import setup_logging

        logger = setup_logging(level="DEBUG")
        assert logger.level == logging.DEBUG

    def test_setup_logging_with_file(self):
        """Test logging setup with file output."""
        from src.lib.logging_utils import setup_logging

        with tempfile.TemporaryDirectory() as temp_dir:
            logger = setup_logging(level="INFO", log_dir=temp_dir)

            # Log a message
            logger.info("Test message")

            # Check log file created
            log_files = list(Path(temp_dir).glob("*.log"))
            assert len(log_files) == 1

    def test_get_logger(self):
        """Test getting named logger."""
        from src.lib.logging_utils import get_logger

        logger = get_logger("test.module")
        assert logger.name == "test.module"


class TestTradingFormatter:
    """Tests for TradingFormatter."""

    def test_formatter_basic(self):
        """Test basic message formatting."""
        from src.lib.logging_utils import TradingFormatter

        formatter = TradingFormatter(use_colors=False)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message",
            args=(),
            exc_info=None
        )

        formatted = formatter.format(record)
        assert "Test message" in formatted
        assert "[INFO" in formatted

    def test_formatter_with_extras(self):
        """Test formatting with extra fields."""
        from src.lib.logging_utils import TradingFormatter

        formatter = TradingFormatter(use_colors=False, include_extras=True)
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Trade executed",
            args=(),
            exc_info=None
        )
        record.order_id = "12345"
        record.price = 6050.25

        formatted = formatter.format(record)
        assert "order_id=12345" in formatted
        assert "price=6050.25" in formatted


class TestTradingLogger:
    """Tests for TradingLogger."""

    @pytest.fixture
    def trading_logger(self):
        """Create TradingLogger for testing."""
        from src.lib.logging_utils import TradingLogger, setup_logging
        setup_logging(level="DEBUG")
        return TradingLogger("test_trading")

    def test_signal_logging(self, trading_logger, caplog):
        """Test signal logging."""
        with caplog.at_level(logging.INFO):
            trading_logger.signal("LONG_ENTRY", 0.75, 6050.25)

        assert "SIGNAL" in caplog.text
        assert "LONG_ENTRY" in caplog.text
        assert "0.75" in caplog.text

    def test_order_logging(self, trading_logger, caplog):
        """Test order logging."""
        with caplog.at_level(logging.INFO):
            trading_logger.order("MARKET", "BUY", 1, order_id="123")

        assert "ORDER" in caplog.text
        assert "MARKET" in caplog.text
        assert "BUY" in caplog.text

    def test_fill_logging(self, trading_logger, caplog):
        """Test fill logging."""
        with caplog.at_level(logging.INFO):
            trading_logger.fill("BOUGHT", 1, 6050.50, order_id="123")

        assert "FILL" in caplog.text
        assert "BOUGHT" in caplog.text
        assert "6050.50" in caplog.text

    def test_trade_entry_logging(self, trading_logger, caplog):
        """Test trade entry logging."""
        with caplog.at_level(logging.INFO):
            trading_logger.trade_entry(
                direction="LONG",
                size=1,
                entry_price=6050.25,
                stop_price=6048.00,
                target_price=6054.00,
                confidence=0.78
            )

        assert "ENTRY" in caplog.text
        assert "LONG" in caplog.text
        assert "stop=" in caplog.text
        assert "target=" in caplog.text

    def test_trade_exit_logging(self, trading_logger, caplog):
        """Test trade exit logging."""
        with caplog.at_level(logging.INFO):
            trading_logger.trade_exit(
                direction="LONG",
                size=1,
                entry_price=6050.25,
                exit_price=6054.00,
                pnl=15.00,
                exit_reason="TARGET"
            )

        assert "EXIT" in caplog.text
        assert "+$15.00" in caplog.text
        assert "TARGET" in caplog.text

    def test_risk_event_logging(self, trading_logger, caplog):
        """Test risk event logging."""
        with caplog.at_level(logging.WARNING):
            trading_logger.risk_event("DAILY_LIMIT", "Daily loss limit reached: $50")

        assert "RISK" in caplog.text
        assert "DAILY_LIMIT" in caplog.text

    def test_session_start_logging(self, trading_logger, caplog):
        """Test session start logging."""
        with caplog.at_level(logging.INFO):
            trading_logger.session_start(
                capital=1000.0,
                contract="MES",
                paper_trading=True
            )

        assert "SESSION START" in caplog.text
        assert "PAPER" in caplog.text
        assert "$1000.00" in caplog.text

    def test_session_end_logging(self, trading_logger, caplog):
        """Test session end logging."""
        with caplog.at_level(logging.INFO):
            trading_logger.session_end(
                trades=12,
                net_pnl=45.00,
                win_rate=0.58
            )

        assert "SESSION END" in caplog.text
        assert "12 trades" in caplog.text
        assert "+$45.00" in caplog.text
        assert "58.0%" in caplog.text

    def test_performance_logging(self, trading_logger, caplog):
        """Test performance logging."""
        with caplog.at_level(logging.DEBUG):
            trading_logger.performance("inference", 5.2)

        assert "PERF" in caplog.text
        assert "inference" in caplog.text
        assert "5.20ms" in caplog.text


class TestLogHelpers:
    """Tests for logging helper functions."""

    def test_log_trade(self, capsys):
        """Test log_trade helper."""
        from src.lib.logging_utils import log_trade, setup_logging

        setup_logging(level="INFO")
        logger = logging.getLogger("test_log_trade")

        log_trade(logger, "ENTRY", "MES", 1, 6050.25)

        # Logs go to stderr
        captured = capsys.readouterr()
        assert "ENTRY" in captured.err
        assert "MES" in captured.err
        assert "6050.25" in captured.err

    def test_log_latency(self, capsys):
        """Test log_latency helper."""
        from src.lib.logging_utils import log_latency, setup_logging
        from src.lib.time_utils import get_ny_now

        setup_logging(level="DEBUG")
        logger = logging.getLogger("test_log_latency")

        start = get_ny_now()
        end = start + timedelta(milliseconds=5)

        latency = log_latency(logger, "test_operation", start, end)

        assert 4 < latency < 6
        # Logs go to stderr
        captured = capsys.readouterr()
        assert "test_operation" in captured.err


# =============================================================================
# Integration Tests
# =============================================================================

class TestLibIntegration:
    """Integration tests for src/lib/ modules."""

    def test_config_uses_constants(self):
        """Test that config defaults use constants."""
        from src.lib.config import TradingConfig
        from src.lib.constants import (
            DEFAULT_STARTING_CAPITAL,
            DEFAULT_MIN_CONFIDENCE,
            MES_ROUND_TRIP_COST,
        )

        config = TradingConfig()

        assert config.risk.starting_capital == DEFAULT_STARTING_CAPITAL
        assert config.model.min_confidence == DEFAULT_MIN_CONFIDENCE
        assert config.execution.commission_round_trip == MES_ROUND_TRIP_COST

    def test_time_utils_with_config(self):
        """Test time utils with config values."""
        from src.lib.time_utils import get_eod_phase, EODPhase
        from src.lib.config import TradingConfig

        config = TradingConfig()
        assert config.risk.flatten_at_eod is True

        # Verify EOD phases work correctly
        from datetime import datetime
        from zoneinfo import ZoneInfo

        flatten_time = datetime(2025, 1, 15, 16, 27, tzinfo=ZoneInfo("America/New_York"))
        assert get_eod_phase(flatten_time) == EODPhase.AGGRESSIVE_EXIT

    def test_full_import(self):
        """Test all imports from src.lib work."""
        from src.lib import (
            # Constants
            MES_TICK_SIZE,
            MES_SPEC,
            NY_TIMEZONE,
            # Time utils
            get_ny_now,
            is_rth,
            get_eod_phase,
            EODPhase,
            # Config
            TradingConfig,
            load_config,
            validate_config,
            # Logging
            setup_logging,
            get_logger,
            TradingLogger,
        )

        # Verify imports work
        assert MES_TICK_SIZE == 0.25
        assert callable(get_ny_now)
        assert callable(load_config)
        assert callable(setup_logging)
