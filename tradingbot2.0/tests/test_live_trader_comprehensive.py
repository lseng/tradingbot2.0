"""
Comprehensive Tests for LiveTrader.

Focuses on improving coverage for:
- Startup sequence (lines 218-298)
- Trading loop (lines 300-334)
- Quote handling (lines 336-347)
- Bar processing (lines 354-407)
- Signal execution (lines 458-493)
- EOD flatten (lines 495-508)
- Position sync (lines 510-523)
- Model loading (lines 525-549)
- Shutdown sequence (lines 551-586)
- Session state (lines 588-602)
- Alert handlers (lines 631-640)
- run_live_trading function (lines 649-695)
"""

import pytest
import asyncio
from datetime import datetime, date, time, timedelta
from unittest.mock import Mock, AsyncMock, MagicMock, patch
from dataclasses import asdict
import json

import torch

from src.trading.live_trader import (
    LiveTrader,
    TradingConfig,
    SessionMetrics,
    RTH_START,
    RTH_END,
    FLATTEN_START,
    FLATTEN_DEADLINE,
    run_live_trading,
)
from src.trading.signal_generator import Signal, SignalType, ModelPrediction
from src.trading.rt_features import OHLCV, FeatureVector
from src.trading.recovery import ErrorEvent, ErrorSeverity, ErrorCategory
from src.risk import EODPhase


class TestTradingConfig:
    """Tests for TradingConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TradingConfig()
        assert config.api_base_url == "https://api.topstepx.com"
        assert config.ws_url == "wss://rtc.topstepx.com"
        assert config.contract_id == "CON.F.US.MES.H26"
        assert config.min_confidence == 0.65
        assert config.starting_capital == 1000.0
        assert config.max_daily_loss == 50.0
        assert config.paper_trading is True

    def test_custom_values(self):
        """Test custom configuration."""
        config = TradingConfig(
            contract_id="CON.F.US.ES.H26",
            starting_capital=2000.0,
            paper_trading=False,
        )
        assert config.contract_id == "CON.F.US.ES.H26"
        assert config.starting_capital == 2000.0
        assert config.paper_trading is False

    def test_session_times(self):
        """Test session time defaults."""
        config = TradingConfig()
        assert config.session_start == RTH_START
        assert config.session_end == RTH_END
        assert config.flatten_time == FLATTEN_START


class TestSessionMetricsExtended:
    """Extended tests for SessionMetrics."""

    def test_to_dict_complete(self):
        """Test complete serialization."""
        metrics = SessionMetrics()
        metrics.session_date = date(2026, 1, 16)
        metrics.trades_executed = 15
        metrics.wins = 9
        metrics.losses = 6
        metrics.gross_pnl = 120.0
        metrics.commissions = 12.60
        metrics.net_pnl = 107.40
        metrics.max_drawdown = 35.0
        metrics.signals_generated = 25
        metrics.predictions_made = 1000
        metrics.bars_processed = 5000
        metrics.start_time = datetime(2026, 1, 16, 9, 30)
        metrics.end_time = datetime(2026, 1, 16, 16, 0)

        d = metrics.to_dict()

        assert d["session_date"] == "2026-01-16"
        assert d["trades_executed"] == 15
        assert d["wins"] == 9
        assert d["losses"] == 6
        assert d["win_rate"] == 60.0
        assert d["gross_pnl"] == 120.0
        assert d["commissions"] == 12.60
        assert d["net_pnl"] == 107.40
        assert d["max_drawdown"] == 35.0
        assert d["signals_generated"] == 25
        assert d["predictions_made"] == 1000
        assert d["bars_processed"] == 5000
        assert d["duration_minutes"] == 390.0  # 6.5 hours

    def test_to_dict_no_trades(self):
        """Test serialization with no trades."""
        metrics = SessionMetrics()
        metrics.trades_executed = 0

        d = metrics.to_dict()

        assert d["win_rate"] == 0

    def test_to_dict_no_times(self):
        """Test serialization without start/end times."""
        metrics = SessionMetrics()

        d = metrics.to_dict()

        assert d["start_time"] is None
        assert d["end_time"] is None
        assert d["duration_minutes"] == 0


class TestLiveTraderInit:
    """Tests for LiveTrader initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default config."""
        config = TradingConfig()
        trader = LiveTrader(config)

        assert trader.config == config
        assert trader._api_key is None
        assert trader._account_id is None
        assert trader._running is False
        assert trader._model is None

    def test_init_with_credentials(self):
        """Test initialization with API credentials."""
        config = TradingConfig()
        trader = LiveTrader(
            config,
            api_key="test-key",
            account_id=12345,
        )

        assert trader._api_key == "test-key"
        assert trader._account_id == 12345


class TestLiveTraderStartup:
    """Tests for LiveTrader startup sequence."""

    @pytest.fixture
    def mock_trader(self):
        """Create trader with mocked dependencies."""
        config = TradingConfig()
        trader = LiveTrader(config, api_key="test-key")

        # Mock all async components
        trader._client = AsyncMock()
        trader._rest = AsyncMock()
        trader._ws = AsyncMock()
        trader._risk_manager = Mock()
        trader._risk_manager.state = Mock(account_balance=1000.0)
        trader._eod_manager = Mock()
        trader._position_sizer = Mock()
        trader._position_manager = Mock()
        trader._signal_generator = Mock()
        trader._order_executor = AsyncMock()
        trader._feature_engine = Mock()
        trader._bar_aggregator = Mock()
        trader._recovery_handler = AsyncMock()

        return trader

    @pytest.mark.asyncio
    async def test_startup_authenticates(self):
        """Test that startup authenticates API client."""
        with patch('src.trading.live_trader.TopstepXClient') as MockClient:
            mock_client = AsyncMock()
            MockClient.return_value = mock_client

            with patch('src.trading.live_trader.TopstepXREST'):
                with patch('src.trading.live_trader.TopstepXWebSocket') as MockWS:
                    mock_ws = AsyncMock()
                    MockWS.return_value = mock_ws

                    with patch('src.trading.live_trader.RiskManager') as MockRM:
                        # Create a proper mock with account_balance
                        mock_rm = Mock()
                        mock_rm.state = Mock()
                        mock_rm.state.account_balance = 1000.0
                        MockRM.return_value = mock_rm

                        with patch('src.trading.live_trader.PositionManager'):
                            with patch('src.trading.live_trader.SignalGenerator'):
                                with patch('src.trading.live_trader.OrderExecutor'):
                                    with patch('src.trading.live_trader.RealTimeFeatureEngine'):
                                        with patch('src.trading.live_trader.BarAggregator'):
                                            with patch('src.trading.live_trader.RecoveryHandler'):
                                                config = TradingConfig()
                                                trader = LiveTrader(config, api_key="test")

                                                # Mock _load_model
                                                trader._load_model = AsyncMock()
                                                trader._sync_positions = AsyncMock()

                                                await trader._startup()

                                                mock_client.authenticate.assert_called_once()

    @pytest.mark.asyncio
    async def test_startup_sets_account_id(self):
        """Test that startup sets account ID when provided."""
        with patch('src.trading.live_trader.TopstepXClient') as MockClient:
            mock_client = AsyncMock()
            MockClient.return_value = mock_client

            with patch('src.trading.live_trader.TopstepXREST'):
                with patch('src.trading.live_trader.TopstepXWebSocket') as MockWS:
                    mock_ws = AsyncMock()
                    MockWS.return_value = mock_ws

                    with patch('src.trading.live_trader.RiskManager') as MockRM:
                        mock_rm = Mock()
                        mock_rm.state = Mock(account_balance=1000.0)
                        MockRM.return_value = mock_rm

                        with patch('src.trading.live_trader.PositionManager'):
                            with patch('src.trading.live_trader.SignalGenerator'):
                                with patch('src.trading.live_trader.OrderExecutor'):
                                    with patch('src.trading.live_trader.RealTimeFeatureEngine'):
                                        with patch('src.trading.live_trader.BarAggregator'):
                                            with patch('src.trading.live_trader.RecoveryHandler'):
                                                config = TradingConfig()
                                                trader = LiveTrader(
                                                    config,
                                                    api_key="test",
                                                    account_id=12345,
                                                )

                                                trader._load_model = AsyncMock()
                                                trader._sync_positions = AsyncMock()

                                                await trader._startup()

                                                assert mock_client.default_account_id == 12345


class TestLiveTraderTradingLoop:
    """Tests for trading loop."""

    @pytest.mark.asyncio
    async def test_trading_loop_premarket(self):
        """Test trading loop waits during pre-market."""
        config = TradingConfig(
            session_start=time(9, 30),
        )
        trader = LiveTrader(config, api_key="test")
        trader._running = True

        # Mock time to be before session start
        with patch('src.trading.live_trader.datetime') as mock_dt:
            mock_dt.now.return_value = Mock(
                time=Mock(return_value=time(9, 0))  # 9:00 AM
            )

            # Stop after first iteration
            async def stop_after_delay():
                await asyncio.sleep(0.05)
                trader._running = False

            asyncio.create_task(stop_after_delay())

            await trader._trading_loop()
            # Should have waited during pre-market

    @pytest.mark.asyncio
    async def test_trading_loop_eod_flatten(self):
        """Test trading loop triggers EOD flatten."""
        config = TradingConfig(
            flatten_time=time(16, 25),
        )
        trader = LiveTrader(config, api_key="test")
        trader._running = True
        trader._handle_eod_flatten = AsyncMock()

        # Mock risk manager for can_trade() check (10A.2)
        mock_risk_manager = Mock()
        mock_risk_manager.can_trade.return_value = True
        mock_risk_manager.state.status = Mock()  # Not MANUAL_REVIEW
        trader._risk_manager = mock_risk_manager

        # Mock time to be at flatten time
        with patch('src.trading.live_trader.datetime') as mock_dt:
            mock_dt.now.return_value = Mock(
                time=Mock(return_value=time(16, 26))  # Past flatten time
            )

            await trader._trading_loop()

            trader._handle_eod_flatten.assert_called_once()

    @pytest.mark.asyncio
    async def test_trading_loop_session_end(self):
        """Test trading loop ends at session end."""
        config = TradingConfig(
            session_start=time(9, 30),
            session_end=time(16, 0),
            flatten_time=time(16, 25),
        )
        trader = LiveTrader(config, api_key="test")
        trader._running = True

        # Mock time to be at session end
        with patch('src.trading.live_trader.datetime') as mock_dt:
            mock_dt.now.return_value = Mock(
                time=Mock(return_value=time(16, 5))  # Past session end
            )

            await trader._trading_loop()
            # Loop should end gracefully

    @pytest.mark.asyncio
    async def test_trading_loop_cancelled(self):
        """Test trading loop handles cancellation gracefully."""
        config = TradingConfig()
        trader = LiveTrader(config, api_key="test")
        trader._running = True

        # Mock time in session
        with patch('src.trading.live_trader.datetime') as mock_dt:
            mock_dt.now.return_value = Mock(
                time=Mock(return_value=time(10, 0))
            )

            # Create a task that will be cancelled
            task = asyncio.create_task(trader._trading_loop())

            # Cancel the task after a brief delay
            await asyncio.sleep(0.01)
            task.cancel()

            # The loop should handle cancellation gracefully
            try:
                await task
            except asyncio.CancelledError:
                pass  # This is expected
            # Either way, the test passes - loop handles cancellation


class TestLiveTraderQuoteHandling:
    """Tests for quote handling."""

    def test_on_quote_aggregates_bar(self):
        """Test quote is aggregated into bar."""
        config = TradingConfig()
        trader = LiveTrader(config, api_key="test")
        trader._bar_aggregator = Mock()
        trader._bar_aggregator.add_tick.return_value = None

        mock_quote = Mock()
        mock_quote.last_price = 6000.0

        trader._on_quote(mock_quote)

        trader._bar_aggregator.add_tick.assert_called_once_with(mock_quote)

    def test_on_quote_handles_completed_bar(self):
        """Test completed bar triggers async processing."""
        config = TradingConfig()
        trader = LiveTrader(config, api_key="test")

        completed_bar = OHLCV(
            timestamp=datetime.now(),
            open=6000.0,
            high=6001.0,
            low=5999.0,
            close=6000.5,
            volume=100,
        )

        trader._bar_aggregator = Mock()
        trader._bar_aggregator.add_tick.return_value = completed_bar

        mock_quote = Mock()

        with patch('asyncio.create_task') as mock_create_task:
            trader._on_quote(mock_quote)
            mock_create_task.assert_called_once()

    def test_on_quote_handles_error(self):
        """Test error handling in quote callback."""
        config = TradingConfig()
        trader = LiveTrader(config, api_key="test")
        trader._bar_aggregator = Mock()
        trader._bar_aggregator.add_tick.side_effect = ValueError("Test error")

        mock_quote = Mock()

        # Should not raise
        trader._on_quote(mock_quote)


class TestLiveTraderBarProcessing:
    """Tests for bar processing."""

    @pytest.mark.asyncio
    async def test_process_bar_no_features(self):
        """Test bar processing when not enough data."""
        config = TradingConfig()
        trader = LiveTrader(config, api_key="test")
        trader._feature_engine = Mock()
        trader._feature_engine.update.return_value = None

        bar = OHLCV(
            timestamp=datetime.now(),
            open=6000.0,
            high=6001.0,
            low=5999.0,
            close=6000.5,
        )

        await trader._process_bar(bar)

        # Should return early without generating signals

    @pytest.mark.asyncio
    async def test_process_bar_full_cycle(self):
        """Test full bar processing cycle."""
        config = TradingConfig()
        trader = LiveTrader(config, api_key="test")

        # Mock feature engine
        mock_features = Mock(spec=FeatureVector)
        mock_features.features = Mock()
        mock_features.atr = 2.0
        trader._feature_engine = Mock()
        trader._feature_engine.update.return_value = mock_features
        trader._feature_engine.get_atr.return_value = 2.0

        # Mock inference
        trader._run_inference = AsyncMock(return_value=ModelPrediction(
            direction=1,
            confidence=0.75,
        ))

        # Mock session metrics
        trader._session_metrics = SessionMetrics()

        # Mock EOD manager
        trader._eod_manager = Mock()
        trader._eod_manager.get_status.return_value = Mock(phase=EODPhase.NORMAL)

        # Mock signal generator
        mock_signal = Signal(signal_type=SignalType.LONG_ENTRY, confidence=0.75)
        trader._signal_generator = Mock()
        trader._signal_generator.generate.return_value = mock_signal

        # Mock position manager
        trader._position_manager = Mock()
        trader._position_manager.position = Mock()
        trader._position_manager.get_unrealized_pnl.return_value = 0.0
        trader._position_manager.update_pnl = Mock()

        # Mock risk manager
        trader._risk_manager = Mock()
        trader._risk_manager.update_open_pnl = Mock()

        # Mock execute signal
        trader._execute_signal = AsyncMock()

        bar = OHLCV(
            timestamp=datetime.now(),
            open=6000.0,
            high=6001.0,
            low=5999.0,
            close=6000.5,
        )

        await trader._process_bar(bar)

        assert trader._session_metrics.predictions_made == 1
        assert trader._session_metrics.signals_generated == 1
        trader._execute_signal.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_bar_eod_close_only(self):
        """Test bar processing during CLOSE_ONLY phase."""
        config = TradingConfig()
        trader = LiveTrader(config, api_key="test")

        # Setup mocks
        mock_features = Mock(spec=FeatureVector)
        trader._feature_engine = Mock()
        trader._feature_engine.update.return_value = mock_features

        trader._run_inference = AsyncMock(return_value=ModelPrediction(
            direction=1,
            confidence=0.8,
        ))

        trader._session_metrics = SessionMetrics()

        # EOD CLOSE_ONLY phase
        trader._eod_manager = Mock()
        trader._eod_manager.get_status.return_value = Mock(phase=EODPhase.CLOSE_ONLY)

        trader._position_manager = Mock()
        trader._position_manager.is_flat.return_value = False

        trader._handle_eod_flatten = AsyncMock()

        bar = OHLCV(
            timestamp=datetime.now(),
            open=6000.0,
            high=6001.0,
            low=5999.0,
            close=6000.5,
        )

        await trader._process_bar(bar)

        trader._handle_eod_flatten.assert_called_once()

    @pytest.mark.asyncio
    async def test_process_bar_eod_must_be_flat(self):
        """Test bar processing during MUST_BE_FLAT phase."""
        config = TradingConfig()
        trader = LiveTrader(config, api_key="test")

        mock_features = Mock(spec=FeatureVector)
        trader._feature_engine = Mock()
        trader._feature_engine.update.return_value = mock_features

        trader._run_inference = AsyncMock(return_value=ModelPrediction(
            direction=1,
            confidence=0.8,
        ))

        trader._session_metrics = SessionMetrics()

        trader._eod_manager = Mock()
        trader._eod_manager.get_status.return_value = Mock(phase=EODPhase.MUST_BE_FLAT)

        trader._handle_eod_flatten = AsyncMock()

        bar = OHLCV(
            timestamp=datetime.now(),
            open=6000.0,
            high=6001.0,
            low=5999.0,
            close=6000.5,
        )

        await trader._process_bar(bar)

        trader._handle_eod_flatten.assert_called_once()


class TestLiveTraderInference:
    """Tests for model inference."""

    @pytest.mark.asyncio
    async def test_run_inference_no_model(self):
        """Test inference when model not loaded."""
        config = TradingConfig()
        trader = LiveTrader(config, api_key="test")
        trader._model = None

        mock_features = Mock()

        result = await trader._run_inference(mock_features)

        assert result.direction == 0
        assert result.confidence == 0.0

    @pytest.mark.asyncio
    async def test_run_inference_with_model(self):
        """Test inference with loaded model."""
        config = TradingConfig()
        trader = LiveTrader(config, api_key="test")

        # Create a simple mock model
        mock_model = Mock()
        # Return logits for 3-class output
        mock_model.return_value = torch.tensor([[0.1, 0.2, 0.7]])
        trader._model = mock_model
        trader._scaler = None

        mock_features = Mock()
        mock_features.as_tensor.return_value = torch.tensor([[0.5] * 50])
        mock_features.atr = 1.5

        result = await trader._run_inference(mock_features)

        assert result.direction == 1  # UP (class 2)
        assert result.confidence > 0.0
        assert result.timestamp is not None

    @pytest.mark.asyncio
    async def test_run_inference_with_scaler(self):
        """Test inference with feature scaler."""
        import numpy as np

        config = TradingConfig()
        trader = LiveTrader(config, api_key="test")

        mock_model = Mock()
        mock_model.return_value = torch.tensor([[0.7, 0.2, 0.1]])
        trader._model = mock_model

        mock_scaler = Mock()
        mock_scaler.transform.return_value = np.array([[0.1] * 50])
        trader._scaler = mock_scaler

        mock_features = Mock()
        mock_features.features = np.array([0.5] * 50)
        mock_features.atr = 1.0

        result = await trader._run_inference(mock_features)

        mock_scaler.transform.assert_called_once()
        assert result.direction == -1  # DOWN (class 0)

    @pytest.mark.asyncio
    async def test_run_inference_error(self):
        """Test inference error handling."""
        config = TradingConfig()
        trader = LiveTrader(config, api_key="test")

        mock_model = Mock()
        mock_model.side_effect = RuntimeError("Model error")
        trader._model = mock_model

        mock_features = Mock()
        mock_features.as_tensor.return_value = torch.tensor([[0.5] * 50])

        result = await trader._run_inference(mock_features)

        assert result.direction == 0
        assert result.confidence == 0.0


class TestLiveTraderSignalExecution:
    """Tests for signal execution."""

    @pytest.mark.asyncio
    async def test_execute_signal_success(self):
        """Test successful signal execution."""
        config = TradingConfig()
        trader = LiveTrader(config, api_key="test")

        # Mock position sizer
        mock_size = Mock()
        mock_size.contracts = 1
        trader._position_sizer = Mock()
        trader._position_sizer.calculate.return_value = mock_size

        # Mock risk manager
        trader._risk_manager = Mock()
        trader._risk_manager.state = Mock(account_balance=1000.0)

        # Mock order executor
        mock_result = Mock()
        mock_result.success = True
        mock_result.entry_fill_price = 6000.0
        trader._order_executor = AsyncMock()
        trader._order_executor.execute_signal.return_value = mock_result

        trader._session_metrics = SessionMetrics()

        signal = Signal(signal_type=SignalType.LONG_ENTRY, confidence=0.75)

        await trader._execute_signal(signal, 6000.0)

        assert trader._session_metrics.trades_executed == 1

    @pytest.mark.asyncio
    async def test_execute_signal_zero_size(self):
        """Test signal execution with zero position size."""
        config = TradingConfig()
        trader = LiveTrader(config, api_key="test")

        mock_size = Mock()
        mock_size.contracts = 0
        trader._position_sizer = Mock()
        trader._position_sizer.calculate.return_value = mock_size

        trader._risk_manager = Mock()
        trader._risk_manager.state = Mock(account_balance=1000.0)

        trader._order_executor = AsyncMock()
        trader._session_metrics = SessionMetrics()

        signal = Signal(signal_type=SignalType.LONG_ENTRY, confidence=0.60)

        await trader._execute_signal(signal, 6000.0)

        trader._order_executor.execute_signal.assert_not_called()

    @pytest.mark.asyncio
    async def test_execute_signal_error(self):
        """Test signal execution error handling."""
        config = TradingConfig()
        trader = LiveTrader(config, api_key="test")

        mock_size = Mock()
        mock_size.contracts = 1
        trader._position_sizer = Mock()
        trader._position_sizer.calculate.return_value = mock_size

        trader._risk_manager = Mock()
        trader._risk_manager.state = Mock(account_balance=1000.0)

        trader._order_executor = AsyncMock()
        trader._order_executor.execute_signal.side_effect = RuntimeError("Order error")

        trader._session_metrics = SessionMetrics()

        signal = Signal(signal_type=SignalType.LONG_ENTRY, confidence=0.75)

        # Should not raise
        await trader._execute_signal(signal, 6000.0)


class TestLiveTraderEODFlatten:
    """Tests for EOD flatten."""

    @pytest.mark.asyncio
    async def test_eod_flatten_when_flat(self):
        """Test EOD flatten does nothing when already flat."""
        config = TradingConfig()
        trader = LiveTrader(config, api_key="test")

        trader._position_manager = Mock()
        trader._position_manager.is_flat.return_value = True

        trader._order_executor = AsyncMock()

        await trader._handle_eod_flatten()

        trader._order_executor.flatten_all.assert_not_called()

    @pytest.mark.asyncio
    async def test_eod_flatten_with_position(self):
        """Test EOD flatten closes position."""
        config = TradingConfig()
        trader = LiveTrader(config, api_key="test")

        trader._position_manager = Mock()
        trader._position_manager.is_flat.return_value = False

        trader._order_executor = AsyncMock()

        await trader._handle_eod_flatten()

        trader._order_executor.flatten_all.assert_called_once_with(config.contract_id)

    @pytest.mark.asyncio
    async def test_eod_flatten_error_recovery(self):
        """Test EOD flatten recovers from error."""
        config = TradingConfig()
        trader = LiveTrader(config, api_key="test")

        trader._position_manager = Mock()
        trader._position_manager.is_flat.return_value = False

        trader._order_executor = AsyncMock()
        trader._order_executor.flatten_all.side_effect = RuntimeError("Flatten error")

        await trader._handle_eod_flatten()

        # Should force local position update
        trader._position_manager.flatten.assert_called_once()


class TestLiveTraderPositionSync:
    """Tests for position synchronization."""

    @pytest.mark.asyncio
    async def test_sync_positions_with_position(self):
        """Test syncing when position exists."""
        config = TradingConfig()
        trader = LiveTrader(config, api_key="test")

        mock_position = Mock()
        mock_position.direction = 1
        mock_position.size = 2

        trader._rest = AsyncMock()
        trader._rest.get_position.return_value = mock_position

        trader._position_manager = Mock()

        await trader._sync_positions()

        trader._position_manager.sync_from_api.assert_called_once_with(mock_position)

    @pytest.mark.asyncio
    async def test_sync_positions_no_position(self):
        """Test syncing when no position exists."""
        config = TradingConfig()
        trader = LiveTrader(config, api_key="test")

        trader._rest = AsyncMock()
        trader._rest.get_position.return_value = None

        trader._position_manager = Mock()

        await trader._sync_positions()

        trader._position_manager.sync_from_api.assert_not_called()

    @pytest.mark.asyncio
    async def test_sync_positions_error(self):
        """Test syncing handles errors."""
        config = TradingConfig()
        trader = LiveTrader(config, api_key="test")

        trader._rest = AsyncMock()
        trader._rest.get_position.side_effect = RuntimeError("API error")

        trader._position_manager = Mock()

        # Should not raise
        await trader._sync_positions()


class TestLiveTraderModelLoading:
    """Tests for model loading."""

    @pytest.mark.asyncio
    async def test_load_model_not_found(self, tmp_path):
        """Test model loading when file doesn't exist."""
        config = TradingConfig(model_path=str(tmp_path / "nonexistent.pt"))
        trader = LiveTrader(config, api_key="test")

        await trader._load_model()

        assert trader._model is None

    @pytest.mark.asyncio
    async def test_load_model_success(self, tmp_path):
        """Test successful model loading."""
        # Create a simple model file with state dict only (weights_only=True compatible)
        model_path = tmp_path / "test_model.pt"
        model_path.touch()  # Create file

        # Mock torch.load to return a model
        with patch('torch.load') as mock_load:
            mock_model = Mock()
            mock_load.return_value = mock_model

            config = TradingConfig(model_path=str(model_path))
            trader = LiveTrader(config, api_key="test")

            await trader._load_model()

            assert trader._model is not None
            mock_model.eval.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_model_with_scaler(self, tmp_path):
        """Test model loading with scaler."""
        # Create model and scaler files
        model_path = tmp_path / "test_model.pt"
        model_path.touch()

        scaler_path = tmp_path / "test_scaler.pkl"
        scaler_path.touch()

        # Mock torch.load and joblib.load
        with patch('torch.load') as mock_torch_load:
            mock_model = Mock()
            mock_torch_load.return_value = mock_model

            with patch('joblib.load') as mock_joblib_load:
                mock_scaler = Mock()
                mock_joblib_load.return_value = mock_scaler

                config = TradingConfig(
                    model_path=str(model_path),
                    scaler_path=str(scaler_path),
                )
                trader = LiveTrader(config, api_key="test")

                await trader._load_model()

                assert trader._model is not None
                assert trader._scaler is not None


class TestLiveTraderShutdown:
    """Tests for shutdown sequence."""

    @pytest.mark.asyncio
    async def test_shutdown_sequence(self, tmp_path):
        """Test full shutdown sequence."""
        config = TradingConfig(log_dir=str(tmp_path))
        trader = LiveTrader(config, api_key="test")

        trader._running = True
        trader._session_metrics = SessionMetrics()
        trader._session_metrics.start_time = datetime.now()

        trader._order_executor = AsyncMock()
        trader._position_manager = Mock()
        trader._position_manager.is_flat.return_value = True

        trader._ws = AsyncMock()

        await trader._shutdown()

        assert trader._running is False
        trader._ws.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_flattens_positions(self, tmp_path):
        """Test shutdown flattens open positions."""
        config = TradingConfig(log_dir=str(tmp_path))
        trader = LiveTrader(config, api_key="test")

        trader._running = True
        trader._session_metrics = SessionMetrics()
        trader._session_metrics.start_time = datetime.now()

        trader._order_executor = AsyncMock()
        trader._position_manager = Mock()
        trader._position_manager.is_flat.return_value = False

        trader._ws = AsyncMock()
        trader._handle_eod_flatten = AsyncMock()

        await trader._shutdown()

        trader._handle_eod_flatten.assert_called_once()


class TestLiveTraderSessionState:
    """Tests for session state management."""

    @pytest.mark.asyncio
    async def test_save_session_state(self, tmp_path):
        """Test saving session state."""
        config = TradingConfig(log_dir=str(tmp_path))
        trader = LiveTrader(config, api_key="test")

        trader._session_metrics = SessionMetrics()
        trader._session_metrics.trades_executed = 5
        trader._session_metrics.net_pnl = 25.0

        await trader._save_session_state()

        # Check file was created
        metrics_file = tmp_path / f"metrics_{date.today().isoformat()}.json"
        assert metrics_file.exists()

        with open(metrics_file) as f:
            data = json.load(f)
            assert data["trades_executed"] == 5
            assert data["net_pnl"] == 25.0

    def test_generate_session_report(self, caplog):
        """Test session report generation."""
        import logging

        config = TradingConfig()
        trader = LiveTrader(config, api_key="test")

        trader._session_metrics = SessionMetrics()
        trader._session_metrics.session_date = date(2026, 1, 16)
        trader._session_metrics.trades_executed = 10
        trader._session_metrics.wins = 6
        trader._session_metrics.losses = 4
        trader._session_metrics.net_pnl = 50.0
        trader._session_metrics.bars_processed = 1000
        trader._session_metrics.start_time = datetime(2026, 1, 16, 9, 30)
        trader._session_metrics.end_time = datetime(2026, 1, 16, 16, 0)

        with caplog.at_level(logging.INFO):
            trader._generate_session_report()

        assert "Session Report" in caplog.text
        assert "Trades: 10" in caplog.text


class TestLiveTraderAlertHandlers:
    """Tests for alert handlers."""

    def test_on_alert(self, caplog):
        """Test alert handler."""
        import logging

        config = TradingConfig()
        trader = LiveTrader(config, api_key="test")

        error = ErrorEvent(
            timestamp=datetime.now(),
            category=ErrorCategory.CONNECTION,
            severity=ErrorSeverity.WARNING,
            message="Connection dropped",
        )

        with caplog.at_level(logging.WARNING):
            trader._on_alert(error)

        assert "ALERT" in caplog.text
        assert "Connection dropped" in caplog.text

    @pytest.mark.asyncio
    async def test_on_halt(self):
        """Test halt handler."""
        config = TradingConfig()
        trader = LiveTrader(config, api_key="test")
        trader._handle_eod_flatten = AsyncMock()

        await trader._on_halt("Critical error")

        trader._handle_eod_flatten.assert_called_once()
        assert trader._shutdown_event.is_set()


class TestLiveTraderStop:
    """Tests for stop functionality."""

    @pytest.mark.asyncio
    async def test_stop(self):
        """Test stop method."""
        config = TradingConfig()
        trader = LiveTrader(config, api_key="test")

        await trader.stop()

        assert trader._shutdown_event.is_set()


class TestRunLiveTradingFunction:
    """Tests for run_live_trading function."""

    @pytest.mark.asyncio
    async def test_run_without_api_key(self):
        """Test run_live_trading raises without API key."""
        with patch.dict('os.environ', {}, clear=True):
            with pytest.raises(ValueError, match="TOPSTEPX_API_KEY"):
                await run_live_trading()

    @pytest.mark.asyncio
    async def test_run_with_api_key(self):
        """Test run_live_trading with API key."""
        with patch.dict('os.environ', {'TOPSTEPX_API_KEY': 'test-key'}):
            with patch('src.trading.live_trader.LiveTrader') as MockTrader:
                mock_trader = AsyncMock()
                MockTrader.return_value = mock_trader

                with patch('asyncio.get_event_loop'):
                    # Start will be called
                    await run_live_trading()

                mock_trader.start.assert_called_once()


class TestOnBarComplete:
    """Tests for _on_bar_complete callback."""

    def test_on_bar_complete_updates_metrics(self):
        """Test bar complete callback updates metrics."""
        config = TradingConfig()
        trader = LiveTrader(config, api_key="test")
        trader._session_metrics = SessionMetrics()

        bar = OHLCV(
            timestamp=datetime.now(),
            open=6000.0,
            high=6001.0,
            low=5999.0,
            close=6000.5,
        )

        trader._on_bar_complete(bar)

        assert trader._last_bar == bar
        assert trader._session_metrics.bars_processed == 1


class TestLiveTraderStart:
    """Tests for start method."""

    @pytest.mark.asyncio
    async def test_start_calls_startup_and_loop(self):
        """Test start calls startup and trading loop."""
        config = TradingConfig()
        trader = LiveTrader(config, api_key="test")

        trader._startup = AsyncMock()
        trader._trading_loop = AsyncMock()
        trader._shutdown = AsyncMock()

        await trader.start()

        trader._startup.assert_called_once()
        trader._trading_loop.assert_called_once()
        trader._shutdown.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_handles_critical_error(self):
        """Test start handles critical errors."""
        config = TradingConfig()
        trader = LiveTrader(config, api_key="test")

        trader._startup = AsyncMock(side_effect=RuntimeError("Startup failed"))
        trader._shutdown = AsyncMock()
        trader._recovery_handler = AsyncMock()

        await trader.start()

        trader._recovery_handler.handle_critical_error.assert_called_once()
        trader._shutdown.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
