"""
Trade and Equity Logging for Backtesting

This module provides comprehensive logging of trades and equity curves.
Detailed logs are essential for:
1. Debugging strategy logic
2. Analyzing trade patterns
3. Identifying edge decay over time
4. Validating backtest accuracy

Output Formats:
- Trade Log: CSV with all trade details
- Equity Curve: CSV with timestamp, equity, drawdown
- Summary Report: JSON with aggregated metrics
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Dict, Any
import csv
import json
from pathlib import Path


class ExitReason(Enum):
    """Reasons for exiting a trade."""
    TARGET = "target"  # Take profit hit
    STOP = "stop"  # Stop loss hit
    TRAILING_STOP = "trailing_stop"  # Trailing stop triggered
    EOD_FLATTEN = "eod_flatten"  # End of day flatten
    SIGNAL = "signal"  # Exit signal from model
    REVERSAL = "reversal"  # Reversing position
    RISK_LIMIT = "risk_limit"  # Risk manager stopped trade
    CIRCUIT_BREAKER = "circuit_breaker"  # Circuit breaker triggered
    MANUAL = "manual"  # Manual exit


@dataclass
class TradeRecord:
    """
    Complete record of a single trade.

    This captures all information needed to analyze trade performance
    and verify backtest accuracy.

    Attributes:
        trade_id: Unique identifier for the trade
        entry_time: Timestamp of entry
        exit_time: Timestamp of exit
        direction: 1 for long, -1 for short
        entry_price: Price at entry
        exit_price: Price at exit
        contracts: Number of contracts traded
        gross_pnl: P&L before costs
        commission: Commission paid
        slippage: Slippage cost
        net_pnl: P&L after all costs
        exit_reason: Why the trade was closed
        model_confidence: Model's confidence at entry
        predicted_class: Model's prediction (0=down, 1=flat, 2=up)
        stop_price: Stop loss price
        target_price: Take profit price
        bars_held: Number of bars position was held
        max_favorable_excursion: Best unrealized P&L during trade
        max_adverse_excursion: Worst unrealized P&L during trade
    """
    trade_id: int
    entry_time: datetime
    exit_time: datetime
    direction: int
    entry_price: float
    exit_price: float
    contracts: int
    gross_pnl: float
    commission: float
    slippage: float
    net_pnl: float
    exit_reason: ExitReason
    model_confidence: float = 0.0
    predicted_class: int = 1  # Default to FLAT
    stop_price: float = 0.0
    target_price: float = 0.0
    bars_held: int = 0
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0

    @property
    def is_winner(self) -> bool:
        """Was this a winning trade?"""
        return self.net_pnl > 0

    @property
    def return_pct(self) -> float:
        """Return as percentage of entry value."""
        entry_value = self.entry_price * self.contracts * 5.0  # MES point value
        if entry_value == 0:
            return 0.0
        return (self.net_pnl / entry_value) * 100

    @property
    def r_multiple(self) -> float:
        """Return as multiple of risk (R)."""
        risk = abs(self.entry_price - self.stop_price) * self.contracts * 5.0
        if risk == 0:
            return 0.0
        return self.net_pnl / risk

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "trade_id": self.trade_id,
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat(),
            "direction": "LONG" if self.direction == 1 else "SHORT",
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "contracts": self.contracts,
            "gross_pnl": round(self.gross_pnl, 2),
            "commission": round(self.commission, 2),
            "slippage": round(self.slippage, 2),
            "net_pnl": round(self.net_pnl, 2),
            "exit_reason": self.exit_reason.value,
            "model_confidence": round(self.model_confidence, 4),
            "predicted_class": self.predicted_class,
            "stop_price": self.stop_price,
            "target_price": self.target_price,
            "bars_held": self.bars_held,
            "mfe": round(self.max_favorable_excursion, 2),
            "mae": round(self.max_adverse_excursion, 2),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TradeRecord':
        """Create TradeRecord from dictionary."""
        return cls(
            trade_id=data["trade_id"],
            entry_time=datetime.fromisoformat(data["entry_time"]),
            exit_time=datetime.fromisoformat(data["exit_time"]),
            direction=1 if data["direction"] == "LONG" else -1,
            entry_price=data["entry_price"],
            exit_price=data["exit_price"],
            contracts=data["contracts"],
            gross_pnl=data["gross_pnl"],
            commission=data["commission"],
            slippage=data["slippage"],
            net_pnl=data["net_pnl"],
            exit_reason=ExitReason(data["exit_reason"]),
            model_confidence=data.get("model_confidence", 0.0),
            predicted_class=data.get("predicted_class", 1),
            stop_price=data.get("stop_price", 0.0),
            target_price=data.get("target_price", 0.0),
            bars_held=data.get("bars_held", 0),
            max_favorable_excursion=data.get("mfe", 0.0),
            max_adverse_excursion=data.get("mae", 0.0),
        )


@dataclass
class EquityPoint:
    """
    Single point on the equity curve.

    Attributes:
        timestamp: Time of this equity reading
        equity: Total account equity
        drawdown: Current drawdown in dollars
        drawdown_pct: Current drawdown as percentage
        position_size: Current position size (0 if flat)
        unrealized_pnl: Unrealized P&L of open position
    """
    timestamp: datetime
    equity: float
    drawdown: float = 0.0
    drawdown_pct: float = 0.0
    position_size: int = 0
    unrealized_pnl: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "equity": round(self.equity, 2),
            "drawdown": round(self.drawdown, 2),
            "drawdown_pct": round(self.drawdown_pct, 6),
            "position_size": self.position_size,
            "unrealized_pnl": round(self.unrealized_pnl, 2),
        }


class TradeLog:
    """
    Logger for recording and exporting trade records.

    This class maintains a log of all trades executed during a backtest
    and provides export capabilities for analysis.

    Usage:
        log = TradeLog()
        log.add_trade(trade_record)
        log.export_csv("trades.csv")
    """

    def __init__(self):
        self._trades: List[TradeRecord] = []
        self._next_id: int = 1

    def add_trade(
        self,
        entry_time: datetime,
        exit_time: datetime,
        direction: int,
        entry_price: float,
        exit_price: float,
        contracts: int,
        gross_pnl: float,
        commission: float,
        slippage: float,
        exit_reason: ExitReason,
        model_confidence: float = 0.0,
        predicted_class: int = 1,
        stop_price: float = 0.0,
        target_price: float = 0.0,
        bars_held: int = 0,
        max_favorable_excursion: float = 0.0,
        max_adverse_excursion: float = 0.0,
    ) -> TradeRecord:
        """
        Add a completed trade to the log.

        Args:
            entry_time: Timestamp of entry
            exit_time: Timestamp of exit
            direction: 1 for long, -1 for short
            entry_price: Price at entry
            exit_price: Price at exit
            contracts: Number of contracts
            gross_pnl: P&L before costs
            commission: Commission paid
            slippage: Slippage cost
            exit_reason: Why the trade was closed
            model_confidence: Model's confidence at entry
            predicted_class: Model's prediction
            stop_price: Stop loss price
            target_price: Take profit price
            bars_held: Number of bars held
            max_favorable_excursion: Best unrealized P&L
            max_adverse_excursion: Worst unrealized P&L

        Returns:
            The created TradeRecord
        """
        net_pnl = gross_pnl - commission - slippage

        trade = TradeRecord(
            trade_id=self._next_id,
            entry_time=entry_time,
            exit_time=exit_time,
            direction=direction,
            entry_price=entry_price,
            exit_price=exit_price,
            contracts=contracts,
            gross_pnl=gross_pnl,
            commission=commission,
            slippage=slippage,
            net_pnl=net_pnl,
            exit_reason=exit_reason,
            model_confidence=model_confidence,
            predicted_class=predicted_class,
            stop_price=stop_price,
            target_price=target_price,
            bars_held=bars_held,
            max_favorable_excursion=max_favorable_excursion,
            max_adverse_excursion=max_adverse_excursion,
        )

        self._trades.append(trade)
        self._next_id += 1
        return trade

    def get_trades(self) -> List[TradeRecord]:
        """Get all trade records."""
        return self._trades.copy()

    def get_trade_pnls(self) -> List[float]:
        """Get list of net P&Ls for all trades."""
        return [t.net_pnl for t in self._trades]

    def get_trade_count(self) -> int:
        """Get total number of trades."""
        return len(self._trades)

    def get_winning_trades(self) -> List[TradeRecord]:
        """Get all winning trades."""
        return [t for t in self._trades if t.is_winner]

    def get_losing_trades(self) -> List[TradeRecord]:
        """Get all losing trades."""
        return [t for t in self._trades if not t.is_winner]

    def get_trades_by_direction(self, direction: int) -> List[TradeRecord]:
        """Get trades filtered by direction (1=long, -1=short)."""
        return [t for t in self._trades if t.direction == direction]

    def get_trades_by_exit_reason(self, reason: ExitReason) -> List[TradeRecord]:
        """Get trades filtered by exit reason."""
        return [t for t in self._trades if t.exit_reason == reason]

    def export_csv(self, filepath: str) -> None:
        """
        Export trade log to CSV file.

        CSV columns:
        trade_id, entry_time, exit_time, direction, entry_price, exit_price,
        contracts, gross_pnl, commission, slippage, net_pnl, exit_reason,
        model_confidence, predicted_class, stop_price, target_price,
        bars_held, mfe, mae

        Args:
            filepath: Path to output CSV file
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                "trade_id", "entry_time", "exit_time", "direction",
                "entry_price", "exit_price", "contracts",
                "gross_pnl", "commission", "slippage", "net_pnl",
                "exit_reason", "model_confidence", "predicted_class",
                "stop_price", "target_price", "bars_held", "mfe", "mae"
            ])

            # Data rows
            for trade in self._trades:
                writer.writerow([
                    trade.trade_id,
                    trade.entry_time.strftime("%Y-%m-%d %H:%M:%S"),
                    trade.exit_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "LONG" if trade.direction == 1 else "SHORT",
                    f"{trade.entry_price:.2f}",
                    f"{trade.exit_price:.2f}",
                    trade.contracts,
                    f"{trade.gross_pnl:.2f}",
                    f"{trade.commission:.2f}",
                    f"{trade.slippage:.2f}",
                    f"{trade.net_pnl:.2f}",
                    trade.exit_reason.value,
                    f"{trade.model_confidence:.4f}",
                    trade.predicted_class,
                    f"{trade.stop_price:.2f}",
                    f"{trade.target_price:.2f}",
                    trade.bars_held,
                    f"{trade.max_favorable_excursion:.2f}",
                    f"{trade.max_adverse_excursion:.2f}",
                ])

    def export_json(self, filepath: str) -> None:
        """Export trade log to JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump([t.to_dict() for t in self._trades], f, indent=2)

    def clear(self) -> None:
        """Clear all trades and reset ID counter."""
        self._trades.clear()
        self._next_id = 1


class EquityCurve:
    """
    Logger for recording and exporting equity curve data.

    Tracks equity at regular intervals (typically per bar or per trade)
    for visualization and drawdown analysis.

    Usage:
        curve = EquityCurve(initial_equity=1000.0)
        curve.add_point(timestamp, equity=1050.0)
        curve.export_csv("equity.csv")
    """

    def __init__(self, initial_equity: float = 1000.0):
        self._points: List[EquityPoint] = []
        self._initial_equity = initial_equity
        self._peak_equity = initial_equity

    def add_point(
        self,
        timestamp: datetime,
        equity: float,
        position_size: int = 0,
        unrealized_pnl: float = 0.0,
    ) -> EquityPoint:
        """
        Add a point to the equity curve.

        Automatically calculates drawdown from peak.

        Args:
            timestamp: Time of this reading
            equity: Current total equity
            position_size: Current position size
            unrealized_pnl: Unrealized P&L

        Returns:
            The created EquityPoint
        """
        # Update peak
        if equity > self._peak_equity:
            self._peak_equity = equity

        # Calculate drawdown
        drawdown = self._peak_equity - equity
        drawdown_pct = drawdown / self._peak_equity if self._peak_equity > 0 else 0.0

        point = EquityPoint(
            timestamp=timestamp,
            equity=equity,
            drawdown=drawdown,
            drawdown_pct=drawdown_pct,
            position_size=position_size,
            unrealized_pnl=unrealized_pnl,
        )

        self._points.append(point)
        return point

    def get_points(self) -> List[EquityPoint]:
        """Get all equity points."""
        return self._points.copy()

    def get_equity_values(self) -> List[float]:
        """Get list of equity values only."""
        return [p.equity for p in self._points]

    def get_drawdown_values(self) -> List[float]:
        """Get list of drawdown percentages."""
        return [p.drawdown_pct for p in self._points]

    def get_final_equity(self) -> float:
        """Get final equity value."""
        if self._points:
            return self._points[-1].equity
        return self._initial_equity

    def get_peak_equity(self) -> float:
        """Get peak equity achieved."""
        return self._peak_equity

    def export_csv(self, filepath: str) -> None:
        """
        Export equity curve to CSV file.

        Args:
            filepath: Path to output CSV file
        """
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                "timestamp", "equity", "drawdown", "drawdown_pct",
                "position_size", "unrealized_pnl"
            ])

            # Data rows
            for point in self._points:
                writer.writerow([
                    point.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    f"{point.equity:.2f}",
                    f"{point.drawdown:.2f}",
                    f"{point.drawdown_pct:.6f}",
                    point.position_size,
                    f"{point.unrealized_pnl:.2f}",
                ])

    def export_json(self, filepath: str) -> None:
        """Export equity curve to JSON file."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, 'w') as f:
            json.dump([p.to_dict() for p in self._points], f, indent=2)

    def clear(self) -> None:
        """Clear all points and reset peak."""
        self._points.clear()
        self._peak_equity = self._initial_equity


@dataclass
class BacktestReport:
    """
    Complete backtest report with all outputs.

    This is the main output object from a backtest run.
    Contains trade log, equity curve, and summary metrics.
    """
    trade_log: TradeLog
    equity_curve: EquityCurve
    metrics: Optional[Any] = None  # PerformanceMetrics
    config: Optional[Dict[str, Any]] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    fold_id: Optional[int] = None  # For walk-forward

    def export_all(self, output_dir: str, prefix: str = "") -> Dict[str, str]:
        """
        Export all report components to files.

        Args:
            output_dir: Directory for output files
            prefix: Optional prefix for filenames

        Returns:
            Dictionary mapping output type to filepath
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        prefix_str = f"{prefix}_" if prefix else ""
        files = {}

        # Trade log
        trades_path = output_path / f"{prefix_str}trades.csv"
        self.trade_log.export_csv(str(trades_path))
        files["trades_csv"] = str(trades_path)

        # Equity curve
        equity_path = output_path / f"{prefix_str}equity.csv"
        self.equity_curve.export_csv(str(equity_path))
        files["equity_csv"] = str(equity_path)

        # Summary metrics
        if self.metrics is not None:
            summary_path = output_path / f"{prefix_str}summary.json"
            with open(summary_path, 'w') as f:
                # Handle potential pandas Timestamp objects
                start_str = None
                end_str = None
                if self.start_date is not None:
                    try:
                        start_str = self.start_date.isoformat()
                    except AttributeError:
                        start_str = str(self.start_date)
                if self.end_date is not None:
                    try:
                        end_str = self.end_date.isoformat()
                    except AttributeError:
                        end_str = str(self.end_date)

                summary = {
                    "metrics": self.metrics.to_dict() if hasattr(self.metrics, 'to_dict') else {},
                    "config": self.config or {},
                    "period": {
                        "start": start_str,
                        "end": end_str,
                    },
                    "fold_id": self.fold_id,
                }
                json.dump(summary, f, indent=2)
            files["summary_json"] = str(summary_path)

        return files

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary."""
        return {
            "trades": [t.to_dict() for t in self.trade_log.get_trades()],
            "equity": [p.to_dict() for p in self.equity_curve.get_points()],
            "metrics": self.metrics.to_dict() if self.metrics and hasattr(self.metrics, 'to_dict') else None,
            "config": self.config,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "fold_id": self.fold_id,
        }
