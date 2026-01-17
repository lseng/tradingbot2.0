"""
Live Trader - Main Trading Loop.

Orchestrates all components for live trading:
- WebSocket market data connection
- Real-time feature calculation
- ML model inference
- Signal generation
- Order execution
- Risk management
- EOD flatten

This is the main entry point for live trading.

Reference: specs/live-trading-execution.md
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, time, date
from pathlib import Path
from typing import Optional, Callable, Awaitable
import json
import signal

import torch

from src.api import (
    TopstepXClient,
    TopstepXREST,
    TopstepXWebSocket,
    Quote,
)
from src.risk import RiskManager, RiskLimits, EODManager, EODPhase, PositionSizer, TradingStatus, CircuitBreakers
from src.trading.position_manager import PositionManager, PositionChange
from src.trading.signal_generator import SignalGenerator, SignalConfig, Signal, SignalType, ModelPrediction
from src.trading.order_executor import OrderExecutor, ExecutorConfig
from src.trading.rt_features import RealTimeFeatureEngine, BarAggregator, RTFeaturesConfig, OHLCV
from src.trading.recovery import RecoveryHandler, RecoveryConfig, ErrorEvent, ErrorSeverity
from src.lib.constants import MES_TICK_VALUE

logger = logging.getLogger(__name__)

# Trading session constants
RTH_START = time(9, 30)
RTH_END = time(16, 0)
FLATTEN_START = time(16, 25)  # 4:25 PM - start flatten
FLATTEN_DEADLINE = time(16, 30)  # 4:30 PM - must be flat

# Queue constants for backpressure (10.5)
BAR_QUEUE_MAX_SIZE = 100  # Max bars in queue (~100 seconds buffer at 1s bars)
BAR_QUEUE_WARNING_THRESHOLD = 0.8  # Warn when queue is 80% full


@dataclass
class TradingConfig:
    """Configuration for live trading."""
    # API settings
    api_base_url: str = "https://api.topstepx.com"
    ws_url: str = "wss://rtc.topstepx.com"

    # Contract
    contract_id: str = "CON.F.US.MES.H26"  # Update for current front-month

    # Model settings
    model_path: str = "models/scalper_v1.pt"
    scaler_path: str = "models/feature_scaler.pkl"
    min_confidence: float = 0.65

    # Risk settings (from RiskLimits)
    starting_capital: float = 1000.0
    max_daily_loss: float = 50.0
    max_per_trade_risk: float = 25.0

    # Session times (NY timezone)
    session_start: time = RTH_START
    session_end: time = RTH_END
    flatten_time: time = FLATTEN_START

    # Logging
    log_dir: str = "logs"
    trade_log_file: str = "trades_{date}.csv"
    session_log_file: str = "trading_{date}.log"

    # Debug mode
    paper_trading: bool = True  # Set False for live trading


@dataclass
class SessionMetrics:
    """Metrics for the trading session."""
    session_date: date = field(default_factory=date.today)
    trades_executed: int = 0
    wins: int = 0
    losses: int = 0
    gross_pnl: float = 0.0
    commissions: float = 0.0
    net_pnl: float = 0.0
    max_drawdown: float = 0.0
    signals_generated: int = 0
    predictions_made: int = 0
    bars_processed: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "session_date": self.session_date.isoformat(),
            "trades_executed": self.trades_executed,
            "wins": self.wins,
            "losses": self.losses,
            "win_rate": self.wins / self.trades_executed * 100 if self.trades_executed > 0 else 0,
            "gross_pnl": self.gross_pnl,
            "commissions": self.commissions,
            "net_pnl": self.net_pnl,
            "max_drawdown": self.max_drawdown,
            "signals_generated": self.signals_generated,
            "predictions_made": self.predictions_made,
            "bars_processed": self.bars_processed,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "duration_minutes": (
                (self.end_time - self.start_time).total_seconds() / 60
                if self.start_time and self.end_time else 0
            ),
        }


class LiveTrader:
    """
    Main live trading orchestrator.

    Coordinates all trading components:
    - TopstepX API (REST + WebSocket)
    - Risk Manager
    - Position Manager
    - Signal Generator
    - Order Executor
    - Real-time Feature Engine
    - Recovery Handler

    Usage:
        trader = LiveTrader(config)
        await trader.start()  # Runs until session end
    """

    def __init__(
        self,
        config: TradingConfig,
        api_key: Optional[str] = None,
        account_id: Optional[int] = None,
    ):
        """
        Initialize live trader.

        Args:
            config: Trading configuration
            api_key: TopstepX API key (or set TOPSTEPX_API_KEY env var)
            account_id: TopstepX account ID
        """
        self.config = config
        self._api_key = api_key
        self._account_id = account_id

        # Core components (initialized in start())
        self._client: Optional[TopstepXClient] = None
        self._rest: Optional[TopstepXREST] = None
        self._ws: Optional[TopstepXWebSocket] = None

        # Trading components
        self._risk_manager: Optional[RiskManager] = None
        self._eod_manager: Optional[EODManager] = None
        self._position_sizer: Optional[PositionSizer] = None
        self._position_manager: Optional[PositionManager] = None
        self._signal_generator: Optional[SignalGenerator] = None
        self._order_executor: Optional[OrderExecutor] = None
        self._feature_engine: Optional[RealTimeFeatureEngine] = None
        self._bar_aggregator: Optional[BarAggregator] = None
        self._recovery_handler: Optional[RecoveryHandler] = None
        self._circuit_breaker: Optional[CircuitBreakers] = None

        # ML Model
        self._model = None
        self._scaler = None
        self._requires_scaler: bool = False  # 10.22: Track if model was trained with scaler
        self._scaler_validated: bool = False  # 10.22: Track if scaler validation done

        # Session state
        self._running: bool = False
        self._session_metrics = SessionMetrics()
        self._last_bar: Optional[OHLCV] = None
        self._last_prediction: Optional[ModelPrediction] = None
        self._last_realized_pnl: float = 0.0  # Track session realized P&L for circuit breakers

        # Bar processing queue with backpressure (10.5)
        self._bar_queue: asyncio.Queue[OHLCV] = asyncio.Queue(maxsize=BAR_QUEUE_MAX_SIZE)
        self._bar_processor_task: Optional[asyncio.Task] = None
        self._queue_warning_logged: bool = False  # Prevent log spam

        # 10C.5 FIX: Market conditions tracking for circuit breaker updates
        self._last_market_conditions_update: Optional[datetime] = None
        self._market_conditions_update_interval: int = 60  # Update every 60 seconds
        self._recent_atr_values: list[float] = []  # Rolling ATR for baseline calculation
        self._atr_lookback: int = 20  # Number of ATR values for baseline
        self._current_spread_ticks: float = 0.0  # Current spread in ticks
        self._recent_volumes: list[int] = []  # Rolling volume for average calculation
        self._volume_lookback: int = 20  # Number of volume values for average

        # Shutdown handling
        self._shutdown_event = asyncio.Event()

        logger.info(f"LiveTrader initialized for {config.contract_id}")

    async def start(self) -> None:
        """
        Start the trading session.

        Runs until session end or shutdown signal.
        """
        logger.info("=" * 60)
        logger.info("STARTING LIVE TRADING SESSION")
        logger.info("=" * 60)

        try:
            # Startup sequence
            await self._startup()

            # Main trading loop
            await self._trading_loop()

        except Exception as e:
            logger.critical(f"Critical error in trading session: {e}")
            if self._recovery_handler:
                await self._recovery_handler.handle_critical_error(e, "main_loop")
        finally:
            # Shutdown sequence
            await self._shutdown()

    async def _startup(self) -> None:
        """Execute startup sequence."""
        logger.info("Executing startup sequence...")
        self._session_metrics.start_time = datetime.now()

        # 1. Initialize API client
        logger.info("1. Initializing API client...")
        self._client = TopstepXClient(
            api_key=self._api_key,
            base_url=self.config.api_base_url,
        )
        await self._client.authenticate()
        logger.info("   API authenticated")

        # Set account ID
        if self._account_id:
            self._client.default_account_id = self._account_id

        self._rest = TopstepXREST(self._client)
        self._ws = TopstepXWebSocket(self._client)

        # 2. Initialize risk manager
        logger.info("2. Initializing risk manager...")
        risk_limits = RiskLimits(
            starting_capital=self.config.starting_capital,
            max_daily_loss=self.config.max_daily_loss,
            max_per_trade_risk=self.config.max_per_trade_risk,
            min_confidence=self.config.min_confidence,
        )
        state_file = Path(self.config.log_dir) / "risk_state.json"
        self._risk_manager = RiskManager(limits=risk_limits, state_file=state_file)
        self._eod_manager = EODManager()
        self._position_sizer = PositionSizer()
        self._circuit_breaker = CircuitBreakers()  # 10A.3: Circuit breaker integration

        # 10C.4 FIX: Link risk manager to circuit breakers (single source of truth for consecutive losses)
        self._risk_manager.set_circuit_breakers(self._circuit_breaker)

        # 10.19 FIX: Link risk manager to EOD manager (defense-in-depth EOD checking)
        self._risk_manager.set_eod_manager(self._eod_manager)

        logger.info(f"   Balance: ${self._risk_manager.state.account_balance:.2f}")

        # 3. Initialize position manager
        logger.info("3. Initializing position manager...")
        self._position_manager = PositionManager(self.config.contract_id)
        # Register callback to track trade results for circuit breakers
        self._position_manager.on_position_change(self._on_position_change)
        self._last_realized_pnl = 0.0  # Track session realized P&L for incremental calculation

        # 4. Sync existing positions
        logger.info("4. Syncing existing positions...")
        await self._sync_positions()

        # 5. Initialize signal generator
        logger.info("5. Initializing signal generator...")
        signal_config = SignalConfig(min_entry_confidence=self.config.min_confidence)
        self._signal_generator = SignalGenerator(signal_config)

        # 6. Initialize order executor
        logger.info("6. Initializing order executor...")
        self._order_executor = OrderExecutor(
            rest_client=self._rest,
            ws_client=self._ws,
            position_manager=self._position_manager,
        )

        # 7. Initialize feature engine
        logger.info("7. Initializing feature engine...")
        self._feature_engine = RealTimeFeatureEngine()
        self._bar_aggregator = BarAggregator(on_bar_complete=self._on_bar_complete)

        # 8. Load ML model
        logger.info("8. Loading ML model...")
        await self._load_model()

        # 9. Initialize recovery handler
        logger.info("9. Initializing recovery handler...")
        self._recovery_handler = RecoveryHandler(
            on_alert=self._on_alert,
            on_halt=self._on_halt,
        )

        # 10. Connect to WebSocket and subscribe
        logger.info("10. Connecting to WebSocket...")
        await self._ws.connect()
        self._ws.on_quote(self._on_quote)

        # 10C.2 FIX: Register position sync callback for reconnection events
        # After WebSocket reconnect, positions must be synced with API as source of truth
        self._ws.on_reconnect(self._sync_positions)
        logger.info("   Registered reconnection callback for position sync")

        await self._ws.subscribe_quotes([self.config.contract_id])
        logger.info(f"   Subscribed to {self.config.contract_id}")

        # 11. Start bar processor worker (10.5: backpressure handling)
        logger.info("11. Starting bar processor worker...")
        self._bar_processor_task = asyncio.create_task(self._bar_processor_worker())
        logger.info("   Bar processor worker started")

        self._running = True
        logger.info("Startup complete - trading session active")

    async def _trading_loop(self) -> None:
        """Main trading loop."""
        logger.info("Entering main trading loop...")

        while self._running and not self._shutdown_event.is_set():
            try:
                # Check session status
                current_time = datetime.now().time()

                # Pre-market
                if current_time < self.config.session_start:
                    await asyncio.sleep(1)
                    continue

                # Risk checks - daily limits and account drawdown (10A.2, 10A.4)
                if not self._risk_manager.can_trade():
                    reason = self._risk_manager.state.halt_reason or "Risk limits exceeded"
                    logger.critical(f"Trading halted by risk manager: {reason}")
                    # Flatten any open positions for safety
                    await self._handle_eod_flatten()
                    self._shutdown_event.set()
                    break

                # Check for manual review status (20% account drawdown)
                if self._risk_manager.state.status == TradingStatus.MANUAL_REVIEW:
                    logger.critical(
                        "Account drawdown exceeds 20% - MANUAL_REVIEW required. "
                        "Flattening positions and halting trading."
                    )
                    await self._handle_eod_flatten()
                    self._shutdown_event.set()
                    break

                # EOD flatten check
                if current_time >= self.config.flatten_time:
                    await self._handle_eod_flatten()
                    break

                # Session end
                if current_time >= self.config.session_end:
                    logger.info("Session end reached")
                    break

                # Process any pending tasks
                await asyncio.sleep(0.1)

            except asyncio.CancelledError:
                logger.info("Trading loop cancelled")
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                if self._recovery_handler:
                    await self._recovery_handler.handle_critical_error(e, "trading_loop")
                break

    def _on_quote(self, quote: Quote) -> None:
        """
        Handle incoming quote from WebSocket.

        Uses bounded queue with backpressure to prevent unbounded task accumulation
        during high-volume periods (10.5).
        """
        try:
            # 10C.5 FIX: Track spread for market conditions update
            # Use try/except to handle Mock objects in tests gracefully
            try:
                ask = float(quote.ask) if hasattr(quote, 'ask') else 0
                bid = float(quote.bid) if hasattr(quote, 'bid') else 0
                if ask > 0 and bid > 0:
                    spread_ticks = (ask - bid) / 0.25  # MES tick size
                    self._current_spread_ticks = spread_ticks
            except (TypeError, ValueError):
                # Ignore if quote attributes are not numeric (e.g., Mock objects)
                pass

            # Aggregate into 1-second bars
            completed_bar = self._bar_aggregator.add_tick(quote)

            if completed_bar:
                # 10.5: Add to bounded queue with backpressure
                try:
                    self._bar_queue.put_nowait(completed_bar)
                    self._queue_warning_logged = False  # Reset warning flag on success

                    # Monitor queue depth and warn when approaching capacity
                    queue_depth = self._bar_queue.qsize()
                    warning_threshold = int(BAR_QUEUE_MAX_SIZE * BAR_QUEUE_WARNING_THRESHOLD)
                    if queue_depth >= warning_threshold and not self._queue_warning_logged:
                        logger.warning(
                            f"Bar queue depth high: {queue_depth}/{BAR_QUEUE_MAX_SIZE} "
                            f"({queue_depth/BAR_QUEUE_MAX_SIZE*100:.0f}%) - "
                            "processing may be falling behind"
                        )
                        self._queue_warning_logged = True

                except asyncio.QueueFull:
                    # Backpressure: queue is full, drop this bar and log
                    logger.warning(
                        f"Bar queue full ({BAR_QUEUE_MAX_SIZE} bars) - dropping bar. "
                        "Processing is falling behind quotes."
                    )

        except Exception as e:
            logger.error(f"Error processing quote: {e}")

    async def _bar_processor_worker(self) -> None:
        """
        Worker coroutine that processes bars from the queue.

        Implements the consumer side of the backpressure mechanism (10.5).
        Runs until shutdown, processing bars as they arrive.
        """
        logger.info("Bar processor worker started")
        try:
            while self._running or not self._bar_queue.empty():
                try:
                    # Wait for next bar with timeout to check shutdown flag
                    bar = await asyncio.wait_for(
                        self._bar_queue.get(),
                        timeout=1.0
                    )
                    await self._process_bar(bar)
                    self._bar_queue.task_done()
                except asyncio.TimeoutError:
                    # No bar available, check if we should continue
                    continue
                except asyncio.CancelledError:
                    logger.info("Bar processor worker cancelled")
                    break
                except Exception as e:
                    logger.error(f"Error in bar processor worker: {e}")
                    # Continue processing despite errors

        except Exception as e:
            logger.error(f"Bar processor worker failed: {e}")
        finally:
            logger.info(
                f"Bar processor worker stopped. "
                f"Queue depth at shutdown: {self._bar_queue.qsize()}"
            )

    def _on_bar_complete(self, bar: OHLCV) -> None:
        """Callback when bar is complete."""
        self._last_bar = bar
        self._session_metrics.bars_processed += 1

        # 10A.7: Update bar range for reversal constraint tracking
        if self._signal_generator:
            self._signal_generator.update_bar_range(bar.high, bar.low, bar.close)

        # 10C.5 FIX: Track ATR and volume for market conditions
        if bar:
            # Calculate this bar's range (proxy for volatility/ATR)
            bar_range = bar.high - bar.low
            self._recent_atr_values.append(bar_range)
            if len(self._recent_atr_values) > self._atr_lookback:
                self._recent_atr_values.pop(0)

            # Track volume
            self._recent_volumes.append(bar.volume)
            if len(self._recent_volumes) > self._volume_lookback:
                self._recent_volumes.pop(0)

            # Periodically update market conditions
            self._update_circuit_breaker_market_conditions()

    def _update_circuit_breaker_market_conditions(self) -> None:
        """
        Update circuit breaker market conditions.

        10C.5 FIX: Periodically calls circuit_breaker.update_market_conditions()
        to enable volatility, spread, and volume circuit breakers.
        """
        if not self._circuit_breaker:
            return

        now = datetime.now()

        # Only update every _market_conditions_update_interval seconds
        if self._last_market_conditions_update:
            elapsed = (now - self._last_market_conditions_update).total_seconds()
            if elapsed < self._market_conditions_update_interval:
                return

        self._last_market_conditions_update = now

        # Calculate current and baseline ATR
        current_atr = None
        normal_atr = None
        if len(self._recent_atr_values) >= 5:
            # Current ATR: average of last 5 bars
            current_atr = sum(self._recent_atr_values[-5:]) / 5
            # Baseline ATR: average of all tracked bars
            normal_atr = sum(self._recent_atr_values) / len(self._recent_atr_values)

        # Calculate volume percentage of average
        volume_pct = None
        if len(self._recent_volumes) >= 5:
            avg_volume = sum(self._recent_volumes) / len(self._recent_volumes)
            if avg_volume > 0:
                current_volume = self._recent_volumes[-1] if self._recent_volumes else 0
                volume_pct = current_volume / avg_volume

        # Current spread (already tracked from quotes)
        spread_ticks = self._current_spread_ticks

        # Update circuit breaker
        self._circuit_breaker.update_market_conditions(
            current_atr=current_atr,
            normal_atr=normal_atr,
            spread_ticks=spread_ticks,
            volume_pct=volume_pct,
        )

        # Log market conditions periodically
        if current_atr and normal_atr and normal_atr > 0:
            vol_ratio = current_atr / normal_atr
            logger.debug(
                f"Market conditions updated: ATR ratio={vol_ratio:.2f}, "
                f"spread={spread_ticks:.1f} ticks, volume_pct={volume_pct:.1%}" if volume_pct else
                f"Market conditions updated: ATR ratio={vol_ratio:.2f}, "
                f"spread={spread_ticks:.1f} ticks"
            )

    def _on_position_change(self, change: PositionChange) -> None:
        """
        Callback when position changes.

        Reports trade results to RiskManager for circuit breaker tracking.
        This triggers consecutive loss/win tracking and pause logic.
        """
        try:
            # Only report completed trades (close, partial_close, flatten)
            if change.change_type not in ("close", "partial_close", "flatten", "reversal"):
                return

            # Calculate incremental P&L from this trade
            new_realized = change.new_position.realized_pnl
            trade_pnl = new_realized - self._last_realized_pnl
            self._last_realized_pnl = new_realized

            # Report to risk manager (this triggers circuit breaker checks)
            if self._risk_manager and trade_pnl != 0:
                self._risk_manager.record_trade_result(trade_pnl)
                logger.info(
                    f"Trade result reported: P&L=${trade_pnl:+.2f}, "
                    f"consecutive_losses={self._risk_manager.state.consecutive_losses}"
                )

            # 10A.3: Update circuit breaker with trade result
            if self._circuit_breaker:
                if trade_pnl > 0:
                    self._circuit_breaker.record_win()
                elif trade_pnl < 0:
                    self._circuit_breaker.record_loss()

        except Exception as e:
            logger.error(f"Error in position change callback: {e}")

    async def _process_bar(self, bar: OHLCV) -> None:
        """
        Process a completed 1-second bar.

        This is the core trading logic:
        1. Update features
        2. Run model inference
        3. Generate signal
        4. Execute if approved
        """
        try:
            # 1. Update features
            feature_vector = self._feature_engine.update(bar)

            if feature_vector is None:
                return  # Not enough data yet

            # 2. Run model inference
            prediction = await self._run_inference(feature_vector)
            self._last_prediction = prediction
            self._session_metrics.predictions_made += 1

            # 3. Check EOD phase
            eod_status = self._eod_manager.get_status()
            eod_phase = eod_status.phase
            if eod_phase == EODPhase.CLOSE_ONLY:
                # No new entries, only exits
                if not self._position_manager.is_flat():
                    await self._handle_eod_flatten()
                return
            elif eod_phase == EODPhase.MUST_BE_FLAT:
                await self._handle_eod_flatten()
                return

            # 4. Check circuit breaker before generating signals (10A.3)
            if self._circuit_breaker and not self._circuit_breaker.can_trade():
                pause_remaining = self._circuit_breaker.state.pause_until
                if pause_remaining:
                    logger.info(
                        f"Circuit breaker active - paused until {pause_remaining}. "
                        "Skipping signal generation."
                    )
                return

            # 5. Generate signal
            current_atr = self._feature_engine.get_atr()
            signal = self._signal_generator.generate(
                prediction=prediction,
                position=self._position_manager.position,
                risk_manager=self._risk_manager,
                current_atr=current_atr,
            )

            if signal and signal.signal_type != SignalType.HOLD:
                self._session_metrics.signals_generated += 1
                await self._execute_signal(signal, bar.close)

            # 5. Update P&L
            self._position_manager.update_pnl(bar.close)
            self._risk_manager.update_open_pnl(
                self._position_manager.get_unrealized_pnl()
            )

        except Exception as e:
            logger.error(f"Error processing bar: {e}")

    async def _run_inference(self, feature_vector) -> ModelPrediction:
        """Run ML model inference."""
        if self._model is None:
            # Return default prediction if no model
            return ModelPrediction(
                direction=0,
                confidence=0.0,
                timestamp=datetime.now(),
            )

        try:
            with torch.no_grad():
                # 10.22 fix: Validate scaler on first inference
                if not self._scaler_validated:
                    self._validate_scaler(feature_vector)
                    self._scaler_validated = True

                # Scale features if scaler available
                if self._scaler:
                    features = self._scaler.transform(
                        feature_vector.features.reshape(1, -1)
                    )
                    tensor = torch.tensor(features, dtype=torch.float32)
                else:
                    tensor = feature_vector.as_tensor()

                # Forward pass
                logits = self._model(tensor)

                # Get probabilities and prediction
                probs = torch.softmax(logits, dim=1)
                predicted_class = torch.argmax(probs, dim=1).item()
                confidence = probs[0, predicted_class].item()

                # Map class to direction: 0=DOWN, 1=FLAT, 2=UP
                direction_map = {0: -1, 1: 0, 2: 1}
                direction = direction_map.get(predicted_class, 0)

                return ModelPrediction(
                    direction=direction,
                    confidence=confidence,
                    predicted_move=feature_vector.atr * direction,
                    volatility=feature_vector.atr,
                    timestamp=datetime.now(),
                )

        except Exception as e:
            logger.error(f"Inference error: {e}")
            return ModelPrediction(
                direction=0,
                confidence=0.0,
                timestamp=datetime.now(),
            )

    def _validate_scaler(self, feature_vector) -> None:
        """
        Validate scaler compatibility with features (10.22 fix).

        This ensures:
        1. If model requires scaler, scaler is loaded
        2. Scaler's feature count matches feature vector dimensions

        Raises:
            RuntimeError: If scaler is missing but required
            RuntimeError: If scaler dimensions don't match features
        """
        num_features = len(feature_vector.features)

        # Check 1: Scaler required but missing
        if self._requires_scaler and self._scaler is None:
            raise RuntimeError(
                f"Model requires scaler but none is loaded! "
                f"Feature vector has {num_features} features but no scaler available. "
                f"Model predictions will be INVALID without proper scaling."
            )

        # Check 2: Scaler dimensions match
        if self._scaler is not None:
            scaler_features = getattr(self._scaler, 'n_features_in_', len(self._scaler.mean_))
            if scaler_features != num_features:
                raise RuntimeError(
                    f"Scaler feature count mismatch! "
                    f"Scaler expects {scaler_features} features but feature vector has {num_features}. "
                    f"Model cannot be used with these features."
                )
            logger.info(
                f"Scaler validation PASSED - {num_features} features match scaler dimensions"
            )

        # Warning: No scaler but features look like they need scaling
        if self._scaler is None:
            logger.warning(
                f"Running inference without scaler ({num_features} features). "
                f"If model was trained with scaled features, predictions may be invalid!"
            )

    async def _execute_signal(self, signal: Signal, current_price: float) -> None:
        """Execute a trading signal."""
        logger.info(
            f"Executing signal: {signal.signal_type.value}, "
            f"confidence={signal.confidence:.2%}"
        )

        try:
            # Get EOD position size multiplier (1.0 normal, 0.5 after 4PM, 0.0 no trading)
            eod_multiplier = 1.0
            if self._eod_manager:
                eod_multiplier = self._eod_manager.get_position_size_multiplier()

            # Calculate position size
            size = self._position_sizer.calculate(
                account_balance=self._risk_manager.state.account_balance,
                stop_ticks=signal.stop_ticks,
                confidence=signal.confidence,
                max_risk_override=self.config.max_per_trade_risk,
                eod_multiplier=eod_multiplier,
            )

            if size.contracts <= 0:
                logger.warning(f"Position size is 0, skipping trade: {size.reason}")
                return

            # Validate trade with risk manager (per-trade risk + confidence check)
            risk_amount = size.contracts * signal.stop_ticks * MES_TICK_VALUE
            if not self._risk_manager.approve_trade(risk_amount, signal.confidence):
                logger.warning(
                    f"Trade rejected by risk manager: risk=${risk_amount:.2f}, "
                    f"confidence={signal.confidence:.1%}"
                )
                return

            # Execute via order executor
            result = await self._order_executor.execute_signal(
                signal=signal,
                contract_id=self.config.contract_id,
                size=size.contracts,
                current_price=current_price,
            )

            if result and result.success:
                self._session_metrics.trades_executed += 1
                logger.info(
                    f"Trade executed: {signal.signal_type.value} @ {result.entry_fill_price}"
                )

        except Exception as e:
            logger.error(f"Signal execution error: {e}")

    async def _handle_eod_flatten(self) -> None:
        """Handle end-of-day flatten."""
        if self._position_manager.is_flat():
            return

        logger.warning("EOD FLATTEN - Closing all positions")

        try:
            await self._order_executor.flatten_all(self.config.contract_id)
            logger.info("EOD flatten complete")
        except Exception as e:
            logger.error(f"EOD flatten failed: {e}")
            # Force local position update
            self._position_manager.flatten()

    async def _sync_positions(self) -> None:
        """Sync positions from API."""
        try:
            api_position = await self._rest.get_position(self.config.contract_id)
            if api_position:
                self._position_manager.sync_from_api(api_position)
                logger.info(
                    f"Position synced: direction={api_position.direction}, "
                    f"size={api_position.size}"
                )
            else:
                logger.info("No open position")
        except Exception as e:
            logger.error(f"Position sync failed: {e}")

    async def _load_model(self) -> None:
        """Load ML model, scaler, and feature names from checkpoint."""
        model_path = Path(self.config.model_path)

        if not model_path.exists():
            logger.warning(f"Model file not found: {model_path}")
            logger.warning("Running in prediction-disabled mode")
            return

        try:
            # Load checkpoint (contains model_state_dict, feature_names, scaler info, etc.)
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

            # Handle both full checkpoints (dict) and bare model objects
            if isinstance(checkpoint, dict):
                # Full checkpoint from training
                from src.ml.models.neural_networks import FeedForwardNet, LSTMNet, HybridNet

                model_config = checkpoint.get('model_config', checkpoint.get('config', {}))
                model_type = model_config.get('type', model_config.get('model_type', 'feedforward'))
                input_dim = checkpoint.get('input_dim', model_config.get('input_size', 50))
                hidden_size = model_config.get('hidden_size', 128)
                num_layers = model_config.get('num_layers', 2)
                dropout = model_config.get('dropout', 0.2)
                num_classes = checkpoint.get('num_classes', 3)

                # Create model based on type
                if model_type.lower() in ['lstm', 'rnn']:
                    self._model = LSTMNet(
                        input_size=input_dim,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout,
                        num_classes=num_classes,
                    )
                elif model_type.lower() == 'hybrid':
                    self._model = HybridNet(
                        input_size=input_dim,
                        lstm_hidden=hidden_size // 2,
                        fc_hidden=hidden_size,
                        num_layers=num_layers,
                        dropout=dropout,
                        num_classes=num_classes,
                    )
                else:
                    self._model = FeedForwardNet(
                        input_size=input_dim,
                        hidden_sizes=[hidden_size, hidden_size // 2],
                        dropout=dropout,
                        num_classes=num_classes,
                    )

                self._model.load_state_dict(checkpoint['model_state_dict'])
                logger.info(f"Model loaded from checkpoint: type={model_type}, input_dim={input_dim}")

                # Extract and set feature names for validation (10.21 fix)
                feature_names = checkpoint.get('feature_names')
                if feature_names and self._feature_engine:
                    self._feature_engine.set_expected_feature_names(feature_names)
                    logger.info(f"Feature names loaded from checkpoint ({len(feature_names)} features)")
                elif not feature_names:
                    logger.warning(
                        "No feature_names in checkpoint - feature order validation disabled. "
                        "Model may receive features in wrong order!"
                    )

                # Load scaler from checkpoint if available (10.22 fix)
                if 'scaler_mean' in checkpoint and 'scaler_scale' in checkpoint:
                    from sklearn.preprocessing import StandardScaler
                    import numpy as np
                    self._scaler = StandardScaler()
                    self._scaler.mean_ = np.array(checkpoint['scaler_mean'])
                    self._scaler.scale_ = np.array(checkpoint['scaler_scale'])
                    self._scaler.var_ = self._scaler.scale_ ** 2
                    self._scaler.n_features_in_ = len(self._scaler.mean_)
                    self._requires_scaler = True  # 10.22: Model was trained with scaler
                    logger.info(f"Scaler loaded from checkpoint ({len(self._scaler.mean_)} features)")

            else:
                # Bare model object (legacy format)
                self._model = checkpoint
                logger.warning(
                    "Model loaded in legacy format (bare model object). "
                    "Feature order validation disabled - retrain with full checkpoint format."
                )

            self._model.eval()
            logger.info(f"Model loaded from {model_path}")

            # Load external scaler if checkpoint didn't have one
            if self._scaler is None:
                scaler_path = Path(self.config.scaler_path)
                if scaler_path.exists():
                    import joblib
                    self._scaler = joblib.load(scaler_path)
                    self._requires_scaler = True  # 10.22: External scaler implies model needs scaling
                    logger.info(f"Scaler loaded from {scaler_path}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            import traceback
            logger.error(traceback.format_exc())
            self._model = None

    async def _shutdown(self) -> None:
        """Execute shutdown sequence."""
        logger.info("Executing shutdown sequence...")
        self._running = False
        self._session_metrics.end_time = datetime.now()

        try:
            # 1. Cancel pending orders
            if self._order_executor:
                logger.info("1. Cancelling pending orders...")
                # Already handled by flatten

            # 2. Flatten positions
            if self._position_manager and not self._position_manager.is_flat():
                logger.info("2. Flattening positions...")
                await self._handle_eod_flatten()

            # 3. Disconnect WebSocket
            if self._ws:
                logger.info("3. Disconnecting WebSocket...")
                await self._ws.disconnect()

            # 3.5 Stop bar processor worker and drain queue (10.5)
            if self._bar_processor_task:
                logger.info("3.5. Stopping bar processor worker...")
                # Wait for queue to drain (up to 5 seconds)
                try:
                    await asyncio.wait_for(self._bar_queue.join(), timeout=5.0)
                    logger.info("   Bar queue drained successfully")
                except asyncio.TimeoutError:
                    logger.warning(
                        f"   Bar queue drain timeout, {self._bar_queue.qsize()} bars remaining"
                    )
                # Cancel the worker task
                self._bar_processor_task.cancel()
                try:
                    await self._bar_processor_task
                except asyncio.CancelledError:
                    pass
                logger.info("   Bar processor worker stopped")

            # 4. Save session state
            logger.info("4. Saving session state...")
            await self._save_session_state()

            # 5. Generate session report
            logger.info("5. Generating session report...")
            self._generate_session_report()

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

        logger.info("=" * 60)
        logger.info("TRADING SESSION ENDED")
        logger.info("=" * 60)

    async def _save_session_state(self) -> None:
        """Save session state to disk."""
        try:
            log_dir = Path(self.config.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

            # Save metrics
            metrics_file = log_dir / f"metrics_{date.today().isoformat()}.json"
            with open(metrics_file, 'w') as f:
                json.dump(self._session_metrics.to_dict(), f, indent=2)

            logger.info(f"Session metrics saved to {metrics_file}")

        except Exception as e:
            logger.error(f"Failed to save session state: {e}")

    def _generate_session_report(self) -> None:
        """Generate and log session report."""
        metrics = self._session_metrics.to_dict()

        report = f"""
Session Report - {metrics['session_date']}
{'=' * 50}
Duration: {metrics['duration_minutes']:.1f} minutes
Bars Processed: {metrics['bars_processed']}
Predictions Made: {metrics['predictions_made']}
Signals Generated: {metrics['signals_generated']}

Trading Results:
  Trades: {metrics['trades_executed']}
  Wins: {metrics['wins']} ({metrics['win_rate']:.1f}%)
  Losses: {metrics['losses']}

P&L:
  Gross: ${metrics['gross_pnl']:+.2f}
  Commissions: ${metrics['commissions']:.2f}
  Net: ${metrics['net_pnl']:+.2f}

Max Drawdown: ${metrics['max_drawdown']:.2f}
{'=' * 50}
"""
        logger.info(report)

    def _on_alert(self, error: ErrorEvent) -> None:
        """Handle alert from recovery handler."""
        logger.warning(f"ALERT: [{error.category.value}] {error.message}")
        # Could add email/SMS notification here

    async def _on_halt(self, reason: str) -> None:
        """Handle halt signal from recovery handler."""
        logger.critical(f"HALT TRIGGERED: {reason}")
        await self._handle_eod_flatten()
        self._shutdown_event.set()

    async def stop(self) -> None:
        """Stop the trading session."""
        logger.info("Stop requested")
        self._shutdown_event.set()


# Entry point for running live trading
async def run_live_trading(config: Optional[TradingConfig] = None) -> None:
    """
    Run live trading session.

    Args:
        config: Trading configuration
    """
    import os

    config = config or TradingConfig()

    # Get credentials from environment
    api_key = os.environ.get("TOPSTEPX_API_KEY")
    account_id = os.environ.get("TOPSTEPX_ACCOUNT_ID")

    if not api_key:
        raise ValueError("TOPSTEPX_API_KEY environment variable required")

    trader = LiveTrader(
        config=config,
        api_key=api_key,
        account_id=int(account_id) if account_id else None,
    )

    # Handle shutdown signals
    loop = asyncio.get_event_loop()

    def signal_handler():
        logger.info("Shutdown signal received")
        asyncio.create_task(trader.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, signal_handler)

    await trader.start()


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Run trading
    asyncio.run(run_live_trading())
