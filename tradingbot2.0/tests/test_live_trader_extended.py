"""
Extended tests for LiveTrader module.

Tests cover the async methods and functionality:
- LiveTrader._startup sequence
- LiveTrader._trading_loop
- LiveTrader._process_bar
- LiveTrader._run_inference
- LiveTrader._execute_signal
- LiveTrader._handle_eod_flatten
- LiveTrader._sync_positions
- LiveTrader._load_model
- LiveTrader._shutdown sequence
- LiveTrader._save_session_state
- LiveTrader._generate_session_report
- run_live_trading function
"""

import pytest
import asyncio
import json
from datetime import datetime, time, date
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch, mock_open
import torch

from src.trading.live_trader import (
    TradingConfig,
    SessionMetrics,
    LiveTrader,
    run_live_trading,
    RTH_START,
    RTH_END,
    FLATTEN_START,
    FLATTEN_DEADLINE,
)
from src.trading.signal_generator import Signal, SignalType
from src.trading.rt_features import OHLCV
from src.risk import EODPhase


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def trading_config():
    """Create a test trading configuration."""
    return TradingConfig(
        api_base_url="https://test.api.com",
        ws_url="wss://test.rtc.com",
        contract_id="CON.F.US.MES.H26",
        model_path="test_model.pt",
        scaler_path="test_scaler.pkl",
        min_confidence=0.65,
        starting_capital=1000.0,
        max_daily_loss=50.0,
        max_per_trade_risk=25.0,
        paper_trading=True,
        log_dir="/tmp/test_logs",
    )


@pytest.fixture
def mock_api_client():
    """Create a mock API client."""
    client = AsyncMock()
    client.authenticate = AsyncMock()
    client.default_account_id = 12345
    return client


@pytest.fixture
def mock_rest_client():
    """Create a mock REST client."""
    rest = AsyncMock()
    rest.get_position = AsyncMock(return_value=None)
    return rest


@pytest.fixture
def mock_ws_client():
    """Create a mock WebSocket client."""
    ws = AsyncMock()
    ws.connect = AsyncMock()
    ws.disconnect = AsyncMock()
    ws.subscribe_quotes = AsyncMock()
    ws.on_quote = MagicMock()
    return ws


@pytest.fixture
def mock_position():
    """Create a mock position."""
    pos = MagicMock()
    pos.direction = 1
    pos.size = 1
    pos.entry_price = 5000.0
    return pos


# =============================================================================
# TradingConfig Tests
# =============================================================================

class TestTradingConfig:
    """Tests for TradingConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TradingConfig()

        assert config.api_base_url == "https://api.topstepx.com"
        assert config.ws_url == "wss://rtc.topstepx.com"
        assert config.min_confidence == 0.65
        assert config.starting_capital == 1000.0
        assert config.paper_trading is True

    def test_custom_values(self, trading_config):
        """Test custom configuration values."""
        assert trading_config.api_base_url == "https://test.api.com"
        assert trading_config.contract_id == "CON.F.US.MES.H26"
        assert trading_config.starting_capital == 1000.0

    def test_session_times(self):
        """Test session time constants."""
        config = TradingConfig()

        assert config.session_start == RTH_START
        assert config.session_end == RTH_END
        assert config.flatten_time == FLATTEN_START


# =============================================================================
# SessionMetrics Tests
# =============================================================================

class TestSessionMetrics:
    """Tests for SessionMetrics dataclass."""

    def test_default_values(self):
        """Test default session metrics values."""
        metrics = SessionMetrics()

        assert metrics.trades_executed == 0
        assert metrics.wins == 0
        assert metrics.losses == 0
        assert metrics.gross_pnl == 0.0
        assert metrics.net_pnl == 0.0

    def test_to_dict(self):
        """Test to_dict conversion."""
        metrics = SessionMetrics(
            trades_executed=10,
            wins=6,
            losses=4,
            gross_pnl=150.0,
            commissions=8.40,
            net_pnl=141.60,
            signals_generated=20,
            predictions_made=100,
            bars_processed=1000,
        )

        result = metrics.to_dict()

        assert result['trades_executed'] == 10
        assert result['wins'] == 6
        assert result['losses'] == 4
        assert result['win_rate'] == 60.0
        assert result['gross_pnl'] == 150.0
        assert result['net_pnl'] == 141.60

    def test_win_rate_zero_trades(self):
        """Test win rate with zero trades."""
        metrics = SessionMetrics(trades_executed=0)

        result = metrics.to_dict()

        assert result['win_rate'] == 0

    def test_duration_minutes(self):
        """Test duration calculation."""
        metrics = SessionMetrics()
        metrics.start_time = datetime(2026, 1, 16, 9, 30, 0)
        metrics.end_time = datetime(2026, 1, 16, 16, 0, 0)

        result = metrics.to_dict()

        # 6.5 hours = 390 minutes
        assert result['duration_minutes'] == 390.0

    def test_duration_no_times(self):
        """Test duration with no start/end times."""
        metrics = SessionMetrics()

        result = metrics.to_dict()

        assert result['duration_minutes'] == 0


# =============================================================================
# LiveTrader Initialization Tests
# =============================================================================

class TestLiveTraderInit:
    """Tests for LiveTrader initialization."""

    def test_init_default(self, trading_config):
        """Test basic initialization."""
        trader = LiveTrader(trading_config)

        assert trader.config == trading_config
        assert trader._api_key is None
        assert trader._account_id is None
        assert trader._running is False

    def test_init_with_credentials(self, trading_config):
        """Test initialization with credentials."""
        trader = LiveTrader(
            trading_config,
            api_key="test_key",
            account_id=12345,
        )

        assert trader._api_key == "test_key"
        assert trader._account_id == 12345

    def test_init_components_none(self, trading_config):
        """Test components are None before start."""
        trader = LiveTrader(trading_config)

        assert trader._client is None
        assert trader._rest is None
        assert trader._ws is None
        assert trader._risk_manager is None
        assert trader._position_manager is None


# =============================================================================
# LiveTrader._process_bar Tests
# =============================================================================

class TestLiveTraderProcessBar:
    """Tests for LiveTrader._process_bar method."""

    @pytest.mark.asyncio
    async def test_process_bar_no_features(self, trading_config):
        """Test _process_bar returns when no features available."""
        trader = LiveTrader(trading_config)

        # Mock feature engine returning None
        trader._feature_engine = MagicMock()
        trader._feature_engine.update = MagicMock(return_value=None)

        bar = OHLCV(
            timestamp=datetime.now(),
            open=5000.0,
            high=5001.0,
            low=4999.0,
            close=5000.5,
            volume=100,
        )

        # Should not raise
        await trader._process_bar(bar)

    @pytest.mark.asyncio
    async def test_process_bar_eod_no_new_positions(self, trading_config):
        """Test _process_bar during EOD no new positions phase."""
        trader = LiveTrader(trading_config)

        # Mock components
        trader._feature_engine = MagicMock()
        feature_vector = MagicMock()
        feature_vector.features = MagicMock()
        feature_vector.atr = 2.0
        trader._feature_engine.update = MagicMock(return_value=feature_vector)
        trader._feature_engine.get_atr = MagicMock(return_value=2.0)

        trader._eod_manager = MagicMock()
        trader._eod_manager.get_status.return_value = MagicMock(phase=EODPhase.CLOSE_ONLY)

        trader._position_manager = MagicMock()
        trader._position_manager.is_flat = MagicMock(return_value=True)

        trader._run_inference = AsyncMock(return_value=MagicMock(direction=0, confidence=0.5))
        trader._session_metrics = SessionMetrics()

        bar = OHLCV(
            timestamp=datetime.now(),
            open=5000.0,
            high=5001.0,
            low=4999.0,
            close=5000.5,
            volume=100,
        )

        await trader._process_bar(bar)

        # Should have called inference
        trader._run_inference.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_bar_generates_signal(self, trading_config):
        """Test _process_bar generates and executes signal."""
        trader = LiveTrader(trading_config)

        # Mock components
        trader._feature_engine = MagicMock()
        feature_vector = MagicMock()
        feature_vector.features = MagicMock()
        feature_vector.atr = 2.0
        trader._feature_engine.update = MagicMock(return_value=feature_vector)
        trader._feature_engine.get_atr = MagicMock(return_value=2.0)

        trader._eod_manager = MagicMock()
        trader._eod_manager.get_status.return_value = MagicMock(phase=EODPhase.NORMAL)

        trader._position_manager = MagicMock()
        trader._position_manager.is_flat = MagicMock(return_value=True)
        trader._position_manager.position = None
        trader._position_manager.update_pnl = MagicMock()
        trader._position_manager.get_unrealized_pnl = MagicMock(return_value=0.0)

        trader._risk_manager = MagicMock()
        trader._risk_manager.update_open_pnl = MagicMock()

        trader._signal_generator = MagicMock()
        signal = Signal(
            signal_type=SignalType.LONG_ENTRY,
            confidence=0.75,
            stop_ticks=8,
            target_ticks=16,
        )
        trader._signal_generator.generate = MagicMock(return_value=signal)

        prediction = MagicMock(direction=1, confidence=0.75)
        trader._run_inference = AsyncMock(return_value=prediction)
        trader._execute_signal = AsyncMock()
        trader._session_metrics = SessionMetrics()

        bar = OHLCV(
            timestamp=datetime.now(),
            open=5000.0,
            high=5001.0,
            low=4999.0,
            close=5000.5,
            volume=100,
        )

        await trader._process_bar(bar)

        trader._execute_signal.assert_called_once()
        assert trader._session_metrics.signals_generated == 1


# =============================================================================
# LiveTrader._run_inference Tests
# =============================================================================

class TestLiveTraderRunInference:
    """Tests for LiveTrader._run_inference method."""

    @pytest.mark.asyncio
    async def test_run_inference_no_model(self, trading_config):
        """Test _run_inference returns default when no model."""
        trader = LiveTrader(trading_config)
        trader._model = None

        feature_vector = MagicMock()

        result = await trader._run_inference(feature_vector)

        assert result.direction == 0
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_run_inference_with_model(self, trading_config):
        """Test _run_inference with model."""
        trader = LiveTrader(trading_config)

        # Create mock model
        mock_model = MagicMock()
        mock_model.return_value = torch.tensor([[0.1, 0.2, 0.7]])  # UP predicted
        trader._model = mock_model
        trader._scaler = None

        feature_vector = MagicMock()
        feature_vector.as_tensor = MagicMock(return_value=torch.tensor([[0.1, 0.2, 0.3]]))
        feature_vector.atr = 2.0

        result = await trader._run_inference(feature_vector)

        assert result.direction == 1  # UP
        assert result.confidence > 0

    @pytest.mark.asyncio
    async def test_run_inference_with_scaler(self, trading_config):
        """Test _run_inference with scaler."""
        import numpy as np

        trader = LiveTrader(trading_config)

        # Create mock model
        mock_model = MagicMock()
        mock_model.return_value = torch.tensor([[0.7, 0.2, 0.1]])  # DOWN predicted
        trader._model = mock_model

        # Create mock scaler with proper attributes (10.22 fix)
        mock_scaler = MagicMock()
        mock_scaler.transform = MagicMock(return_value=np.array([[0.1, 0.2, 0.3]]))
        mock_scaler.n_features_in_ = 3  # Match feature count for validation
        mock_scaler.mean_ = np.zeros(3)
        trader._scaler = mock_scaler

        # Feature vector with real numpy array for len() to work
        feature_vector = MagicMock()
        feature_vector.features = np.array([0.1, 0.2, 0.3])
        feature_vector.atr = 2.0

        result = await trader._run_inference(feature_vector)

        mock_scaler.transform.assert_called_once()
        assert result.direction == -1  # DOWN

    @pytest.mark.asyncio
    async def test_run_inference_error(self, trading_config):
        """Test _run_inference handles errors gracefully."""
        import numpy as np

        trader = LiveTrader(trading_config)

        mock_model = MagicMock(side_effect=RuntimeError("Model error"))
        trader._model = mock_model
        trader._scaler = None
        trader._scaler_validated = True  # Skip validation for error test

        feature_vector = MagicMock()
        feature_vector.features = np.array([0.1, 0.2, 0.3])  # 10.22: Add features
        feature_vector.as_tensor = MagicMock(return_value=torch.tensor([[0.1, 0.2, 0.3]]))

        result = await trader._run_inference(feature_vector)

        assert result.direction == 0
        assert result.confidence == 0.0


# =============================================================================
# LiveTrader._execute_signal Tests
# =============================================================================

class TestLiveTraderExecuteSignal:
    """Tests for LiveTrader._execute_signal method."""

    @pytest.mark.asyncio
    async def test_execute_signal_zero_size(self, trading_config):
        """Test _execute_signal skips when size is 0."""
        trader = LiveTrader(trading_config)
        trader._session_metrics = SessionMetrics()

        trader._position_sizer = MagicMock()
        trader._position_sizer.calculate = MagicMock(return_value=MagicMock(contracts=0))

        trader._risk_manager = MagicMock()
        trader._risk_manager.state = MagicMock(account_balance=1000.0)

        trader._order_executor = AsyncMock()

        signal = Signal(
            signal_type=SignalType.LONG_ENTRY,
            confidence=0.50,  # Low confidence
            stop_ticks=8,
            target_ticks=16,
        )

        await trader._execute_signal(signal, 5000.0)

        trader._order_executor.execute_signal.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_signal_success(self, trading_config):
        """Test successful signal execution."""
        trader = LiveTrader(trading_config)
        trader._session_metrics = SessionMetrics()

        trader._position_sizer = MagicMock()
        trader._position_sizer.calculate = MagicMock(return_value=MagicMock(contracts=1))

        trader._risk_manager = MagicMock()
        trader._risk_manager.state = MagicMock(account_balance=1000.0)

        trader._order_executor = AsyncMock()
        # 1.16 FIX: Mock success case must have requires_halt=False
        trader._order_executor.execute_signal = AsyncMock(
            return_value=MagicMock(success=True, entry_fill_price=5000.0, requires_halt=False)
        )

        signal = Signal(
            signal_type=SignalType.LONG_ENTRY,
            confidence=0.75,
            stop_ticks=8,
            target_ticks=16,
        )

        await trader._execute_signal(signal, 5000.0)

        trader._order_executor.execute_signal.assert_called_once()
        assert trader._session_metrics.trades_executed == 1


# =============================================================================
# LiveTrader._handle_eod_flatten Tests
# =============================================================================

class TestLiveTraderEODFlatten:
    """Tests for LiveTrader._handle_eod_flatten method."""

    @pytest.mark.asyncio
    async def test_eod_flatten_already_flat(self, trading_config):
        """Test EOD flatten when already flat."""
        trader = LiveTrader(trading_config)

        trader._position_manager = MagicMock()
        trader._position_manager.is_flat = MagicMock(return_value=True)

        trader._order_executor = AsyncMock()

        await trader._handle_eod_flatten()

        trader._order_executor.flatten_all.assert_not_called()

    @pytest.mark.asyncio
    async def test_eod_flatten_has_position(self, trading_config):
        """Test EOD flatten with open position - includes retry and verification (1.17 FIX)."""
        trader = LiveTrader(trading_config)

        # Mock position manager: not flat initially, then flat after flatten
        trader._position_manager = MagicMock()
        trader._position_manager.is_flat = MagicMock(side_effect=[False, True])

        # Mock REST client for position sync verification
        trader._rest = AsyncMock()
        trader._rest.get_position = AsyncMock(return_value=None)

        # Mock order executor
        trader._order_executor = AsyncMock()
        trader._order_executor.flatten_all = AsyncMock(return_value=True)
        trader._order_executor._cancel_all_orders = AsyncMock()

        await trader._handle_eod_flatten()

        # Should be called once if flatten succeeds and position is verified flat
        trader._order_executor.flatten_all.assert_called_once_with(trading_config.contract_id)
        # Should cancel orphan orders
        trader._order_executor._cancel_all_orders.assert_called_once()

    @pytest.mark.asyncio
    async def test_eod_flatten_error_fallback(self, trading_config):
        """Test EOD flatten falls back to local flatten on error."""
        trader = LiveTrader(trading_config)

        trader._position_manager = MagicMock()
        trader._position_manager.is_flat = MagicMock(return_value=False)
        trader._position_manager.flatten = MagicMock()

        trader._order_executor = AsyncMock()
        trader._order_executor.flatten_all = AsyncMock(side_effect=Exception("API error"))

        await trader._handle_eod_flatten()

        trader._position_manager.flatten.assert_called_once()


# =============================================================================
# LiveTrader._sync_positions Tests
# =============================================================================

class TestLiveTraderSyncPositions:
    """Tests for LiveTrader._sync_positions method."""

    @pytest.mark.asyncio
    async def test_sync_positions_no_position(self, trading_config):
        """Test sync positions when no open position."""
        trader = LiveTrader(trading_config)

        trader._rest = AsyncMock()
        trader._rest.get_position = AsyncMock(return_value=None)

        trader._position_manager = MagicMock()

        await trader._sync_positions()

        trader._position_manager.sync_from_api.assert_not_called()

    @pytest.mark.asyncio
    async def test_sync_positions_with_position(self, trading_config, mock_position):
        """Test sync positions with open position."""
        trader = LiveTrader(trading_config)

        trader._rest = AsyncMock()
        trader._rest.get_position = AsyncMock(return_value=mock_position)

        trader._position_manager = MagicMock()

        await trader._sync_positions()

        trader._position_manager.sync_from_api.assert_called_once_with(mock_position)

    @pytest.mark.asyncio
    async def test_sync_positions_error(self, trading_config):
        """Test sync positions handles errors."""
        trader = LiveTrader(trading_config)

        trader._rest = AsyncMock()
        trader._rest.get_position = AsyncMock(side_effect=Exception("API error"))

        # Should not raise
        await trader._sync_positions()


# =============================================================================
# LiveTrader._load_model Tests
# =============================================================================

class TestLiveTraderLoadModel:
    """Tests for LiveTrader._load_model method."""

    @pytest.mark.asyncio
    async def test_load_model_not_found(self, trading_config):
        """Test load model when file not found."""
        trading_config.model_path = "/nonexistent/model.pt"
        trader = LiveTrader(trading_config)

        await trader._load_model()

        assert trader._model is None

    @pytest.mark.asyncio
    async def test_load_model_success(self, trading_config, tmp_path):
        """Test successful model loading."""
        # Create a mock model file
        model_path = tmp_path / "test_model.pt"
        mock_model = MagicMock()

        trading_config.model_path = str(model_path)
        trading_config.scaler_path = str(tmp_path / "nonexistent_scaler.pkl")

        trader = LiveTrader(trading_config)

        with patch('torch.load', return_value=mock_model):
            with patch.object(Path, 'exists', return_value=True):
                await trader._load_model()

        # Model loaded but scaler not found
        assert trader._scaler is None


# =============================================================================
# LiveTrader._save_session_state Tests
# =============================================================================

class TestLiveTraderSaveSessionState:
    """Tests for LiveTrader._save_session_state method."""

    @pytest.mark.asyncio
    async def test_save_session_state(self, trading_config, tmp_path):
        """Test saving session state."""
        trading_config.log_dir = str(tmp_path)
        trader = LiveTrader(trading_config)
        trader._session_metrics = SessionMetrics(
            trades_executed=5,
            net_pnl=100.0,
        )

        await trader._save_session_state()

        # Check file was created
        metrics_file = tmp_path / f"metrics_{date.today().isoformat()}.json"
        assert metrics_file.exists()

        # Verify content
        with open(metrics_file) as f:
            data = json.load(f)
            assert data['trades_executed'] == 5
            assert data['net_pnl'] == 100.0


# =============================================================================
# LiveTrader._generate_session_report Tests
# =============================================================================

class TestLiveTraderGenerateReport:
    """Tests for LiveTrader._generate_session_report method."""

    def test_generate_session_report(self, trading_config, capsys):
        """Test session report generation."""
        trader = LiveTrader(trading_config)
        trader._session_metrics = SessionMetrics(
            trades_executed=10,
            wins=6,
            losses=4,
            gross_pnl=150.0,
            commissions=8.40,
            net_pnl=141.60,
            signals_generated=20,
            predictions_made=100,
            bars_processed=1000,
        )
        trader._session_metrics.start_time = datetime.now()
        trader._session_metrics.end_time = datetime.now()

        trader._generate_session_report()

        # Check logging output (would be in logs)
        # Just verify it doesn't raise


# =============================================================================
# LiveTrader._on_quote Tests
# =============================================================================

class TestLiveTraderOnQuote:
    """Tests for LiveTrader._on_quote callback."""

    def test_on_quote_aggregates_bar(self, trading_config):
        """Test _on_quote aggregates quotes into bars."""
        trader = LiveTrader(trading_config)
        trader._session_metrics = SessionMetrics()

        mock_aggregator = MagicMock()
        mock_aggregator.add_tick = MagicMock(return_value=None)
        trader._bar_aggregator = mock_aggregator

        quote = MagicMock()

        trader._on_quote(quote)

        mock_aggregator.add_tick.assert_called_once_with(quote)

    def test_on_quote_completed_bar(self, trading_config):
        """Test _on_quote adds completed bar to queue (10.5 backpressure)."""
        trader = LiveTrader(trading_config)
        trader._session_metrics = SessionMetrics()

        completed_bar = OHLCV(
            timestamp=datetime.now(),
            open=5000.0,
            high=5001.0,
            low=4999.0,
            close=5000.5,
            volume=100,
        )

        mock_aggregator = MagicMock()
        mock_aggregator.add_tick = MagicMock(return_value=completed_bar)
        trader._bar_aggregator = mock_aggregator

        quote = MagicMock()

        # 10.5: Now uses queue instead of create_task for backpressure
        trader._on_quote(quote)

        # Verify bar was added to queue
        assert trader._bar_queue.qsize() == 1
        queued_bar = trader._bar_queue.get_nowait()
        assert queued_bar == completed_bar


# =============================================================================
# LiveTrader._on_bar_complete Tests
# =============================================================================

class TestLiveTraderOnBarComplete:
    """Tests for LiveTrader._on_bar_complete callback."""

    def test_on_bar_complete_updates_metrics(self, trading_config):
        """Test _on_bar_complete updates metrics."""
        trader = LiveTrader(trading_config)
        trader._session_metrics = SessionMetrics()

        bar = OHLCV(
            timestamp=datetime.now(),
            open=5000.0,
            high=5001.0,
            low=4999.0,
            close=5000.5,
            volume=100,
        )

        trader._on_bar_complete(bar)

        assert trader._last_bar == bar
        assert trader._session_metrics.bars_processed == 1


# =============================================================================
# LiveTrader._on_alert Tests
# =============================================================================

class TestLiveTraderOnAlert:
    """Tests for LiveTrader._on_alert callback."""

    def test_on_alert_logs_warning(self, trading_config):
        """Test _on_alert logs warning."""
        trader = LiveTrader(trading_config)

        error = MagicMock()
        error.category = MagicMock(value="TEST")
        error.message = "Test error"

        # Should not raise
        trader._on_alert(error)


# =============================================================================
# LiveTrader._on_halt Tests
# =============================================================================

class TestLiveTraderOnHalt:
    """Tests for LiveTrader._on_halt callback."""

    @pytest.mark.asyncio
    async def test_on_halt_triggers_shutdown(self, trading_config):
        """Test _on_halt triggers shutdown."""
        trader = LiveTrader(trading_config)
        trader._position_manager = MagicMock()
        trader._position_manager.is_flat = MagicMock(return_value=True)

        await trader._on_halt("Test halt reason")

        assert trader._shutdown_event.is_set()


# =============================================================================
# LiveTrader.stop Tests
# =============================================================================

class TestLiveTraderStop:
    """Tests for LiveTrader.stop method."""

    @pytest.mark.asyncio
    async def test_stop_sets_shutdown_event(self, trading_config):
        """Test stop sets shutdown event."""
        trader = LiveTrader(trading_config)

        await trader.stop()

        assert trader._shutdown_event.is_set()


# =============================================================================
# run_live_trading Tests
# =============================================================================

class TestRunLiveTrading:
    """Tests for run_live_trading function."""

    @pytest.mark.asyncio
    async def test_run_live_trading_no_api_key(self):
        """Test run_live_trading raises without API key."""
        config = TradingConfig()

        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="TOPSTEPX_API_KEY"):
                await run_live_trading(config)

    @pytest.mark.asyncio
    async def test_run_live_trading_with_api_key(self):
        """Test run_live_trading with API key."""
        config = TradingConfig()

        with patch.dict('os.environ', {'TOPSTEPX_API_KEY': 'test_key'}):
            with patch.object(LiveTrader, 'start', new_callable=AsyncMock):
                # Mock asyncio.get_event_loop to avoid signal handler issues
                mock_loop = MagicMock()
                mock_loop.add_signal_handler = MagicMock()

                with patch('asyncio.get_event_loop', return_value=mock_loop):
                    await run_live_trading(config)
