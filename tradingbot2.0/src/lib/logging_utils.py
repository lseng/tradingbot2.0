"""
Structured logging utilities for trading operations.

This module provides:
- Configured logging with rotation and formatting
- Trading-specific log formatters
- Trade logging with structured output
- Performance logging for latency tracking

Log Format:
    YYYY-MM-DD HH:MM:SS.mmm [LEVEL] module - message

Example usage:
    from src.lib.logging_utils import setup_logging, get_logger

    # Setup logging at application start
    setup_logging(level="INFO", log_dir="./logs")

    # Get logger in modules
    logger = get_logger(__name__)
    logger.info("Trade executed", extra={"order_id": "123"})
"""

import logging
import sys
from datetime import datetime
from enum import Enum
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Optional

from src.lib.time_utils import get_ny_now


class LogLevel(Enum):
    """Log level enumeration for type-safe level selection."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


# =============================================================================
# Log Formatting
# =============================================================================

class TradingFormatter(logging.Formatter):
    """
    Custom formatter for trading logs.

    Features:
    - Millisecond precision timestamps
    - NY timezone by default
    - Colored output for terminal (optional)
    - Compact format for log files
    """

    # ANSI color codes
    COLORS = {
        logging.DEBUG: "\033[36m",     # Cyan
        logging.INFO: "\033[32m",      # Green
        logging.WARNING: "\033[33m",   # Yellow
        logging.ERROR: "\033[31m",     # Red
        logging.CRITICAL: "\033[35m",  # Magenta
    }
    RESET = "\033[0m"

    def __init__(
        self,
        use_colors: bool = False,
        use_ny_time: bool = True,
        include_extras: bool = True
    ):
        """
        Initialize formatter.

        Args:
            use_colors: Enable ANSI colors for terminal output
            use_ny_time: Use NY timezone instead of local
            include_extras: Include extra fields in output
        """
        self.use_colors = use_colors
        self.use_ny_time = use_ny_time
        self.include_extras = include_extras

        # Base format without timestamp (we handle timestamp separately)
        fmt = "[%(levelname)-8s] %(name)s - %(message)s"
        super().__init__(fmt)

    def format(self, record: logging.LogRecord) -> str:
        """Format log record with timestamp and optional colors."""
        # Get timestamp
        if self.use_ny_time:
            timestamp = get_ny_now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

        # Format the message
        message = super().format(record)

        # Add extras if present and enabled
        if self.include_extras and hasattr(record, "__dict__"):
            extras = {
                k: v for k, v in record.__dict__.items()
                if k not in logging.LogRecord.__dict__
                and not k.startswith("_")
                and k not in ("message", "asctime", "args", "exc_info", "exc_text",
                             "stack_info", "lineno", "funcName", "created",
                             "msecs", "relativeCreated", "levelno", "levelname",
                             "pathname", "filename", "module", "name", "msg",
                             "processName", "process", "threadName", "thread",
                             "taskName")
            }
            if extras:
                extras_str = " ".join(f"{k}={v}" for k, v in extras.items())
                message = f"{message} [{extras_str}]"

        # Full message with timestamp
        full_message = f"{timestamp} {message}"

        # Apply colors for terminal
        if self.use_colors and sys.stderr.isatty():
            color = self.COLORS.get(record.levelno, "")
            return f"{color}{full_message}{self.RESET}"

        return full_message


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(
    level: str = "INFO",
    log_dir: Optional[str] = None,
    log_file: Optional[str] = None,
    use_colors: bool = True,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Setup application-wide logging configuration.

    Creates handlers for:
    - Console output (with colors if terminal)
    - File output with rotation (if log_dir provided)

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory for log files (optional)
        log_file: Specific log file name (default: trading_YYYY-MM-DD.log)
        use_colors: Enable colored console output
        max_bytes: Max size per log file before rotation
        backup_count: Number of backup files to keep

    Returns:
        Root logger

    Example:
        setup_logging(level="DEBUG", log_dir="./logs")
    """
    # Get root logger
    root_logger = logging.getLogger()

    # Parse level
    log_level = getattr(logging, level.upper(), logging.INFO)
    root_logger.setLevel(log_level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(log_level)
    console_formatter = TradingFormatter(use_colors=use_colors, include_extras=True)
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler (if log_dir provided)
    if log_dir:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)

        # Generate log filename if not provided
        if not log_file:
            today = get_ny_now().strftime("%Y-%m-%d")
            log_file = f"trading_{today}.log"

        full_path = log_path / log_file

        # Rotating file handler
        file_handler = RotatingFileHandler(
            full_path,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
        file_handler.setLevel(log_level)
        file_formatter = TradingFormatter(use_colors=False, include_extras=True)
        file_handler.setFormatter(file_formatter)
        root_logger.addHandler(file_handler)

    return root_logger


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for the given module name.

    Args:
        name: Logger name (usually __name__)

    Returns:
        Logger instance

    Example:
        logger = get_logger(__name__)
        logger.info("Message")
    """
    return logging.getLogger(name)


# =============================================================================
# Trading Logger
# =============================================================================

class TradingLogger:
    """
    Specialized logger for trading operations.

    Provides methods for logging:
    - Trade entries and exits
    - Signals and predictions
    - Risk events
    - Performance metrics
    - Errors and warnings

    All methods accept extra fields as kwargs for structured logging.
    """

    def __init__(self, name: str = "trading"):
        """
        Initialize trading logger.

        Args:
            name: Logger name
        """
        self._logger = logging.getLogger(name)

    def signal(
        self,
        signal_type: str,
        confidence: float,
        price: float,
        **kwargs: Any
    ) -> None:
        """
        Log a trading signal.

        Args:
            signal_type: Signal type (LONG_ENTRY, SHORT_ENTRY, EXIT, etc.)
            confidence: Model confidence (0-1)
            price: Current price
            **kwargs: Additional fields
        """
        self._logger.info(
            f"SIGNAL: {signal_type} conf={confidence:.2f} price={price:.2f}",
            extra={"signal_type": signal_type, "confidence": confidence, "price": price, **kwargs}
        )

    def order(
        self,
        order_type: str,
        side: str,
        size: int,
        price: Optional[float] = None,
        order_id: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """
        Log an order placement.

        Args:
            order_type: Order type (MARKET, LIMIT, STOP)
            side: Order side (BUY, SELL)
            size: Number of contracts
            price: Order price (None for market orders)
            order_id: Order ID if available
            **kwargs: Additional fields
        """
        price_str = f"@ {price:.2f}" if price else "@ market"
        self._logger.info(
            f"ORDER: {order_type} {side} {size} {price_str}",
            extra={"order_type": order_type, "side": side, "size": size,
                  "price": price, "order_id": order_id, **kwargs}
        )

    def fill(
        self,
        side: str,
        size: int,
        fill_price: float,
        order_id: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """
        Log an order fill.

        Args:
            side: Fill side (BOUGHT, SOLD)
            size: Filled quantity
            fill_price: Fill price
            order_id: Order ID
            **kwargs: Additional fields
        """
        self._logger.info(
            f"FILL: {side} {size} @ {fill_price:.2f}",
            extra={"side": side, "size": size, "fill_price": fill_price,
                  "order_id": order_id, **kwargs}
        )

    def trade_entry(
        self,
        direction: str,
        size: int,
        entry_price: float,
        stop_price: float,
        target_price: float,
        confidence: float,
        **kwargs: Any
    ) -> None:
        """
        Log a trade entry with full details.

        Args:
            direction: Trade direction (LONG, SHORT)
            size: Position size
            entry_price: Entry fill price
            stop_price: Stop loss price
            target_price: Take profit price
            confidence: Entry signal confidence
            **kwargs: Additional fields
        """
        self._logger.info(
            f"ENTRY: {direction} {size} @ {entry_price:.2f} "
            f"stop={stop_price:.2f} target={target_price:.2f} conf={confidence:.2f}",
            extra={"direction": direction, "size": size, "entry_price": entry_price,
                  "stop_price": stop_price, "target_price": target_price,
                  "confidence": confidence, **kwargs}
        )

    def trade_exit(
        self,
        direction: str,
        size: int,
        entry_price: float,
        exit_price: float,
        pnl: float,
        exit_reason: str,
        **kwargs: Any
    ) -> None:
        """
        Log a trade exit with P&L.

        Args:
            direction: Trade direction
            size: Position size
            entry_price: Entry price
            exit_price: Exit price
            pnl: Net P&L in dollars
            exit_reason: Reason for exit (STOP, TARGET, EOD, SIGNAL)
            **kwargs: Additional fields
        """
        pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
        self._logger.info(
            f"EXIT: {direction} {size} @ {exit_price:.2f} entry={entry_price:.2f} "
            f"P&L={pnl_str} reason={exit_reason}",
            extra={"direction": direction, "size": size, "entry_price": entry_price,
                  "exit_price": exit_price, "pnl": pnl, "exit_reason": exit_reason, **kwargs}
        )

    def risk_event(
        self,
        event_type: str,
        details: str,
        **kwargs: Any
    ) -> None:
        """
        Log a risk management event.

        Args:
            event_type: Event type (DAILY_LIMIT, CIRCUIT_BREAKER, EOD_FLATTEN, etc.)
            details: Event details
            **kwargs: Additional fields
        """
        self._logger.warning(
            f"RISK: {event_type} - {details}",
            extra={"event_type": event_type, "details": details, **kwargs}
        )

    def session_start(
        self,
        capital: float,
        contract: str,
        paper_trading: bool = False,
        **kwargs: Any
    ) -> None:
        """
        Log session start.

        Args:
            capital: Starting capital
            contract: Contract being traded
            paper_trading: Whether in paper trading mode
            **kwargs: Additional fields
        """
        mode = "PAPER" if paper_trading else "LIVE"
        self._logger.info(
            f"SESSION START: {mode} mode, capital=${capital:.2f}, contract={contract}",
            extra={"capital": capital, "contract": contract, "paper_trading": paper_trading, **kwargs}
        )

    def session_end(
        self,
        trades: int,
        net_pnl: float,
        win_rate: float,
        **kwargs: Any
    ) -> None:
        """
        Log session end with summary.

        Args:
            trades: Number of trades
            net_pnl: Net P&L for session
            win_rate: Win rate (0-1)
            **kwargs: Additional fields
        """
        pnl_str = f"+${net_pnl:.2f}" if net_pnl >= 0 else f"-${abs(net_pnl):.2f}"
        self._logger.info(
            f"SESSION END: {trades} trades, P&L={pnl_str}, win_rate={win_rate:.1%}",
            extra={"trades": trades, "net_pnl": net_pnl, "win_rate": win_rate, **kwargs}
        )

    def performance(
        self,
        operation: str,
        latency_ms: float,
        **kwargs: Any
    ) -> None:
        """
        Log performance metrics.

        Args:
            operation: Operation name (inference, features, order)
            latency_ms: Latency in milliseconds
            **kwargs: Additional fields
        """
        self._logger.debug(
            f"PERF: {operation} latency={latency_ms:.2f}ms",
            extra={"operation": operation, "latency_ms": latency_ms, **kwargs}
        )

    def error(self, message: str, exc_info: bool = False, **kwargs: Any) -> None:
        """
        Log an error.

        Args:
            message: Error message
            exc_info: Include exception info
            **kwargs: Additional fields
        """
        self._logger.error(message, exc_info=exc_info, extra=kwargs)

    def warning(self, message: str, **kwargs: Any) -> None:
        """
        Log a warning.

        Args:
            message: Warning message
            **kwargs: Additional fields
        """
        self._logger.warning(message, extra=kwargs)

    def info(self, message: str, **kwargs: Any) -> None:
        """
        Log an info message.

        Args:
            message: Info message
            **kwargs: Additional fields
        """
        self._logger.info(message, extra=kwargs)

    def debug(self, message: str, **kwargs: Any) -> None:
        """
        Log a debug message.

        Args:
            message: Debug message
            **kwargs: Additional fields
        """
        self._logger.debug(message, extra=kwargs)


# =============================================================================
# Convenience Functions
# =============================================================================

def log_trade(
    logger: logging.Logger,
    trade_type: str,
    symbol: str,
    size: int,
    price: float,
    pnl: Optional[float] = None,
    **kwargs: Any
) -> None:
    """
    Log a trade with consistent formatting.

    Args:
        logger: Logger instance
        trade_type: Type of trade (ENTRY, EXIT, STOP, TARGET)
        symbol: Contract symbol
        size: Position size
        price: Trade price
        pnl: P&L (for exits)
        **kwargs: Additional fields
    """
    if pnl is not None:
        pnl_str = f" P&L={'+'if pnl >= 0 else ''}{pnl:.2f}"
    else:
        pnl_str = ""

    logger.info(
        f"{trade_type}: {symbol} {size} @ {price:.2f}{pnl_str}",
        extra={"trade_type": trade_type, "symbol": symbol, "size": size,
              "price": price, "pnl": pnl, **kwargs}
    )


def log_latency(
    logger: logging.Logger,
    operation: str,
    start_time: datetime,
    end_time: Optional[datetime] = None
) -> float:
    """
    Log operation latency.

    Args:
        logger: Logger instance
        operation: Operation name
        start_time: Operation start time
        end_time: Operation end time (default: now)

    Returns:
        Latency in milliseconds
    """
    if end_time is None:
        end_time = datetime.now(start_time.tzinfo)

    latency_ms = (end_time - start_time).total_seconds() * 1000

    logger.debug(f"{operation}: {latency_ms:.2f}ms")

    return latency_ms
