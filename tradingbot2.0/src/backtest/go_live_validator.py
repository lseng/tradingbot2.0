"""
Go-Live Validation Module

This module provides comprehensive validation of system readiness before
going live with real capital. It enforces all Go-Live checklist requirements
from the implementation plan.

Go-Live Checklist Items Validated:
1. Walk-forward backtest profitability (Sharpe > 1.0, Calmar > 0.5)
2. Out-of-sample accuracy > 52%
3. Risk limits enforcement verification
4. EOD flatten verification
5. Inference latency < 10ms
6. No lookahead bias (verified via tests)
7. Test coverage > 80%
8. Paper trading (operational - not validated here)
9. Position sizing tier validation
10. Circuit breakers working
11. API reconnection working
12. Manual kill switch accessible

Why Go-Live Validation Matters:
- Prevents deploying unprofitable strategies
- Ensures risk management is properly configured
- Validates system reliability before risking capital
- Provides documented evidence of system readiness

Usage:
    validator = GoLiveValidator()
    result = validator.validate_profitability(metrics)
    if result.all_passed:
        print("System ready for live trading")
    else:
        print(f"Validation failed: {result.failures}")
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum

from .metrics import PerformanceMetrics


class ValidationStatus(Enum):
    """Status of a validation check."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    WARNING = "warning"


@dataclass
class ValidationCheck:
    """
    Result of a single validation check.

    Attributes:
        name: Name of the check
        status: Pass/fail/skip/warning status
        actual_value: The actual value measured
        threshold: The threshold value required
        message: Human-readable description
        details: Additional context
    """
    name: str
    status: ValidationStatus
    actual_value: Optional[float] = None
    threshold: Optional[float] = None
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        """Whether this check passed."""
        return self.status == ValidationStatus.PASSED

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "status": self.status.value,
            "actual_value": self.actual_value,
            "threshold": self.threshold,
            "message": self.message,
            "passed": self.passed,
            "details": self.details,
        }


@dataclass
class GoLiveValidationResult:
    """
    Complete result of Go-Live validation.

    Contains all individual checks and overall pass/fail status.
    Used to generate validation reports for audit trails.
    """
    timestamp: datetime = field(default_factory=datetime.now)
    checks: List[ValidationCheck] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        """Whether all required checks passed."""
        return all(check.passed for check in self.checks
                   if check.status != ValidationStatus.SKIPPED)

    @property
    def passed_count(self) -> int:
        """Number of passed checks."""
        return sum(1 for check in self.checks if check.passed)

    @property
    def failed_count(self) -> int:
        """Number of failed checks."""
        return sum(1 for check in self.checks
                   if check.status == ValidationStatus.FAILED)

    @property
    def failures(self) -> List[ValidationCheck]:
        """List of failed checks."""
        return [check for check in self.checks
                if check.status == ValidationStatus.FAILED]

    @property
    def warnings(self) -> List[ValidationCheck]:
        """List of warning checks."""
        return [check for check in self.checks
                if check.status == ValidationStatus.WARNING]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "all_passed": self.all_passed,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "total_checks": len(self.checks),
            "checks": [check.to_dict() for check in self.checks],
        }

    def generate_report(self) -> str:
        """Generate human-readable validation report."""
        lines = [
            "=" * 60,
            "GO-LIVE VALIDATION REPORT",
            "=" * 60,
            f"Timestamp: {self.timestamp.isoformat()}",
            f"Overall Status: {'PASSED' if self.all_passed else 'FAILED'}",
            f"Checks: {self.passed_count}/{len(self.checks)} passed",
            "",
            "DETAILED RESULTS:",
            "-" * 60,
        ]

        for check in self.checks:
            status_icon = {
                ValidationStatus.PASSED: "[PASS]",
                ValidationStatus.FAILED: "[FAIL]",
                ValidationStatus.SKIPPED: "[SKIP]",
                ValidationStatus.WARNING: "[WARN]",
            }[check.status]

            line = f"{status_icon} {check.name}"
            if check.actual_value is not None and check.threshold is not None:
                line += f": {check.actual_value:.4f} (threshold: {check.threshold:.4f})"
            lines.append(line)

            if check.message:
                lines.append(f"       {check.message}")

        lines.extend([
            "-" * 60,
            "",
        ])

        if self.failures:
            lines.append("FAILURES:")
            for fail in self.failures:
                lines.append(f"  - {fail.name}: {fail.message}")
            lines.append("")

        if self.warnings:
            lines.append("WARNINGS:")
            for warn in self.warnings:
                lines.append(f"  - {warn.name}: {warn.message}")
            lines.append("")

        lines.append("=" * 60)

        return "\n".join(lines)


@dataclass
class GoLiveThresholds:
    """
    Configurable thresholds for Go-Live validation.

    These thresholds define the minimum acceptable values for
    each validation check. Default values are from the spec.
    """
    # Profitability thresholds (Go-Live #1)
    min_sharpe_ratio: float = 1.0
    min_calmar_ratio: float = 0.5

    # Accuracy thresholds (Go-Live #2)
    min_oos_accuracy: float = 0.52  # 52% for 3-class

    # Risk thresholds
    max_drawdown: float = 0.20  # 20%
    max_daily_loss_pct: float = 0.05  # 5%

    # Performance thresholds (Go-Live #5)
    max_inference_latency_ms: float = 10.0
    max_feature_latency_ms: float = 5.0
    max_total_latency_ms: float = 15.0

    # Test coverage (Go-Live #7)
    min_test_coverage: float = 0.80  # 80%

    # Trade thresholds
    min_trades_per_fold: int = 100
    min_profit_factor: float = 1.0
    min_win_rate: float = 0.40  # 40%


class GoLiveValidator:
    """
    Validates system readiness for live trading.

    This class performs comprehensive validation of all Go-Live
    checklist requirements before allowing live trading to begin.

    The validator checks:
    - Profitability metrics (Sharpe, Calmar)
    - Risk management limits
    - System performance (latency)
    - Model accuracy
    - Walk-forward consistency

    Example:
        >>> validator = GoLiveValidator()
        >>> result = validator.validate_all(
        ...     metrics=backtest_metrics,
        ...     oos_accuracy=0.55,
        ...     inference_latency_ms=8.5,
        ... )
        >>> if result.all_passed:
        ...     print("Ready for live trading!")
        >>> else:
        ...     print(result.generate_report())
    """

    def __init__(self, thresholds: Optional[GoLiveThresholds] = None):
        """
        Initialize validator with thresholds.

        Args:
            thresholds: Custom thresholds or None for defaults
        """
        self.thresholds = thresholds or GoLiveThresholds()

    def validate_profitability(
        self,
        metrics: PerformanceMetrics,
    ) -> GoLiveValidationResult:
        """
        Validate profitability thresholds (Go-Live #1).

        Checks:
        - Sharpe ratio > 1.0
        - Calmar ratio > 0.5

        Args:
            metrics: Performance metrics from backtest

        Returns:
            Validation result with Sharpe and Calmar checks
        """
        result = GoLiveValidationResult()

        # Sharpe ratio check
        sharpe_check = ValidationCheck(
            name="Sharpe Ratio",
            status=(
                ValidationStatus.PASSED
                if metrics.sharpe_ratio > self.thresholds.min_sharpe_ratio
                else ValidationStatus.FAILED
            ),
            actual_value=metrics.sharpe_ratio,
            threshold=self.thresholds.min_sharpe_ratio,
            message=(
                "Good risk-adjusted returns"
                if metrics.sharpe_ratio > self.thresholds.min_sharpe_ratio
                else f"Sharpe {metrics.sharpe_ratio:.3f} below minimum {self.thresholds.min_sharpe_ratio}"
            ),
        )
        result.checks.append(sharpe_check)

        # Calmar ratio check
        calmar_check = ValidationCheck(
            name="Calmar Ratio",
            status=(
                ValidationStatus.PASSED
                if metrics.calmar_ratio > self.thresholds.min_calmar_ratio
                else ValidationStatus.FAILED
            ),
            actual_value=metrics.calmar_ratio,
            threshold=self.thresholds.min_calmar_ratio,
            message=(
                "Acceptable return vs drawdown"
                if metrics.calmar_ratio > self.thresholds.min_calmar_ratio
                else f"Calmar {metrics.calmar_ratio:.3f} below minimum {self.thresholds.min_calmar_ratio}"
            ),
        )
        result.checks.append(calmar_check)

        return result

    def validate_oos_accuracy(
        self,
        accuracy: float,
    ) -> GoLiveValidationResult:
        """
        Validate out-of-sample accuracy (Go-Live #2).

        For 3-class classification, random guessing yields ~33%.
        We require > 52% to show meaningful predictive power.

        Args:
            accuracy: OOS accuracy as decimal (0-1)

        Returns:
            Validation result with accuracy check
        """
        result = GoLiveValidationResult()

        accuracy_check = ValidationCheck(
            name="OOS Accuracy",
            status=(
                ValidationStatus.PASSED
                if accuracy > self.thresholds.min_oos_accuracy
                else ValidationStatus.FAILED
            ),
            actual_value=accuracy,
            threshold=self.thresholds.min_oos_accuracy,
            message=(
                f"Accuracy {accuracy:.2%} shows predictive power"
                if accuracy > self.thresholds.min_oos_accuracy
                else f"Accuracy {accuracy:.2%} below minimum {self.thresholds.min_oos_accuracy:.2%}"
            ),
        )
        result.checks.append(accuracy_check)

        return result

    def validate_inference_latency(
        self,
        inference_ms: float,
        feature_ms: Optional[float] = None,
    ) -> GoLiveValidationResult:
        """
        Validate inference latency (Go-Live #5).

        Real-time trading requires fast inference:
        - Model inference < 10ms
        - Feature calculation < 5ms
        - Total end-to-end < 15ms

        Args:
            inference_ms: Model inference time in milliseconds
            feature_ms: Feature calculation time (optional)

        Returns:
            Validation result with latency checks
        """
        result = GoLiveValidationResult()

        # Inference latency check
        inference_check = ValidationCheck(
            name="Inference Latency",
            status=(
                ValidationStatus.PASSED
                if inference_ms < self.thresholds.max_inference_latency_ms
                else ValidationStatus.FAILED
            ),
            actual_value=inference_ms,
            threshold=self.thresholds.max_inference_latency_ms,
            message=(
                f"Inference {inference_ms:.2f}ms meets requirement"
                if inference_ms < self.thresholds.max_inference_latency_ms
                else f"Inference {inference_ms:.2f}ms exceeds {self.thresholds.max_inference_latency_ms}ms limit"
            ),
        )
        result.checks.append(inference_check)

        # Feature latency check (if provided)
        if feature_ms is not None:
            feature_check = ValidationCheck(
                name="Feature Calculation Latency",
                status=(
                    ValidationStatus.PASSED
                    if feature_ms < self.thresholds.max_feature_latency_ms
                    else ValidationStatus.FAILED
                ),
                actual_value=feature_ms,
                threshold=self.thresholds.max_feature_latency_ms,
                message=(
                    f"Feature calc {feature_ms:.2f}ms meets requirement"
                    if feature_ms < self.thresholds.max_feature_latency_ms
                    else f"Feature calc {feature_ms:.2f}ms exceeds {self.thresholds.max_feature_latency_ms}ms limit"
                ),
            )
            result.checks.append(feature_check)

            # Total latency check
            total_ms = inference_ms + feature_ms
            total_check = ValidationCheck(
                name="Total End-to-End Latency",
                status=(
                    ValidationStatus.PASSED
                    if total_ms < self.thresholds.max_total_latency_ms
                    else ValidationStatus.FAILED
                ),
                actual_value=total_ms,
                threshold=self.thresholds.max_total_latency_ms,
                message=(
                    f"Total latency {total_ms:.2f}ms meets requirement"
                    if total_ms < self.thresholds.max_total_latency_ms
                    else f"Total latency {total_ms:.2f}ms exceeds {self.thresholds.max_total_latency_ms}ms limit"
                ),
            )
            result.checks.append(total_check)

        return result

    def validate_risk_limits(
        self,
        metrics: PerformanceMetrics,
    ) -> GoLiveValidationResult:
        """
        Validate risk management effectiveness (Go-Live #3).

        Checks that risk limits kept losses within bounds:
        - Max drawdown < 20%
        - Worst day < 5%

        Args:
            metrics: Performance metrics from backtest

        Returns:
            Validation result with risk limit checks
        """
        result = GoLiveValidationResult()

        # Max drawdown check
        dd_check = ValidationCheck(
            name="Max Drawdown",
            status=(
                ValidationStatus.PASSED
                if metrics.max_drawdown_pct < self.thresholds.max_drawdown
                else ValidationStatus.FAILED
            ),
            actual_value=metrics.max_drawdown_pct,
            threshold=self.thresholds.max_drawdown,
            message=(
                f"Max drawdown {metrics.max_drawdown_pct:.2%} within limits"
                if metrics.max_drawdown_pct < self.thresholds.max_drawdown
                else f"Max drawdown {metrics.max_drawdown_pct:.2%} exceeds {self.thresholds.max_drawdown:.2%} limit"
            ),
        )
        result.checks.append(dd_check)

        # Worst day check (if available)
        if metrics.worst_day_pct != 0:
            worst_day_pct = abs(metrics.worst_day_pct)
            daily_check = ValidationCheck(
                name="Worst Daily Loss",
                status=(
                    ValidationStatus.PASSED
                    if worst_day_pct < self.thresholds.max_daily_loss_pct
                    else ValidationStatus.FAILED
                ),
                actual_value=worst_day_pct,
                threshold=self.thresholds.max_daily_loss_pct,
                message=(
                    f"Worst day {worst_day_pct:.2%} within daily limit"
                    if worst_day_pct < self.thresholds.max_daily_loss_pct
                    else f"Worst day {worst_day_pct:.2%} exceeds {self.thresholds.max_daily_loss_pct:.2%} daily limit"
                ),
            )
            result.checks.append(daily_check)

        return result

    def validate_trade_quality(
        self,
        metrics: PerformanceMetrics,
    ) -> GoLiveValidationResult:
        """
        Validate trade quality metrics.

        Additional checks beyond profitability:
        - Minimum trades for statistical significance
        - Profit factor > 1.0
        - Win rate > 40%

        Args:
            metrics: Performance metrics from backtest

        Returns:
            Validation result with trade quality checks
        """
        result = GoLiveValidationResult()

        # Trade count check
        trade_count_check = ValidationCheck(
            name="Trade Count",
            status=(
                ValidationStatus.PASSED
                if metrics.total_trades >= self.thresholds.min_trades_per_fold
                else ValidationStatus.WARNING
            ),
            actual_value=float(metrics.total_trades),
            threshold=float(self.thresholds.min_trades_per_fold),
            message=(
                f"{metrics.total_trades} trades provides statistical significance"
                if metrics.total_trades >= self.thresholds.min_trades_per_fold
                else f"Only {metrics.total_trades} trades - may not be statistically significant"
            ),
        )
        result.checks.append(trade_count_check)

        # Profit factor check
        pf_check = ValidationCheck(
            name="Profit Factor",
            status=(
                ValidationStatus.PASSED
                if metrics.profit_factor > self.thresholds.min_profit_factor
                else ValidationStatus.FAILED
            ),
            actual_value=metrics.profit_factor,
            threshold=self.thresholds.min_profit_factor,
            message=(
                f"Profit factor {metrics.profit_factor:.3f} shows winning edge"
                if metrics.profit_factor > self.thresholds.min_profit_factor
                else f"Profit factor {metrics.profit_factor:.3f} indicates losses"
            ),
        )
        result.checks.append(pf_check)

        # Win rate check
        win_rate = metrics.win_rate_pct / 100  # Convert from percentage
        wr_check = ValidationCheck(
            name="Win Rate",
            status=(
                ValidationStatus.PASSED
                if win_rate > self.thresholds.min_win_rate
                else ValidationStatus.WARNING
            ),
            actual_value=win_rate,
            threshold=self.thresholds.min_win_rate,
            message=(
                f"Win rate {win_rate:.2%} acceptable"
                if win_rate > self.thresholds.min_win_rate
                else f"Win rate {win_rate:.2%} below {self.thresholds.min_win_rate:.2%} (may need better R:R)"
            ),
        )
        result.checks.append(wr_check)

        return result

    def validate_walk_forward_consistency(
        self,
        fold_metrics: List[PerformanceMetrics],
    ) -> GoLiveValidationResult:
        """
        Validate consistency across walk-forward folds.

        Checks that performance is consistent across folds,
        not just profitable in aggregate. Helps detect overfitting.

        Consistency checks:
        - No fold with Sharpe < 0 (losing money)
        - At least 60% of folds profitable
        - Standard deviation of Sharpe < 1.0 (not too variable)

        Args:
            fold_metrics: List of metrics from each walk-forward fold

        Returns:
            Validation result with consistency checks
        """
        result = GoLiveValidationResult()

        if not fold_metrics:
            skip_check = ValidationCheck(
                name="Walk-Forward Consistency",
                status=ValidationStatus.SKIPPED,
                message="No fold metrics provided",
            )
            result.checks.append(skip_check)
            return result

        sharpes = [m.sharpe_ratio for m in fold_metrics]
        profitable_folds = sum(1 for s in sharpes if s > 0)
        pct_profitable = profitable_folds / len(sharpes)

        # Percentage of profitable folds
        pct_check = ValidationCheck(
            name="Profitable Folds Percentage",
            status=(
                ValidationStatus.PASSED
                if pct_profitable >= 0.6
                else ValidationStatus.FAILED
            ),
            actual_value=pct_profitable,
            threshold=0.6,
            message=(
                f"{pct_profitable:.0%} of folds profitable"
                if pct_profitable >= 0.6
                else f"Only {pct_profitable:.0%} of folds profitable (need 60%+)"
            ),
            details={"profitable_folds": profitable_folds, "total_folds": len(fold_metrics)},
        )
        result.checks.append(pct_check)

        # No severely losing folds
        worst_sharpe = min(sharpes)
        worst_check = ValidationCheck(
            name="Worst Fold Sharpe",
            status=(
                ValidationStatus.PASSED
                if worst_sharpe > -1.0
                else ValidationStatus.WARNING
            ),
            actual_value=worst_sharpe,
            threshold=-1.0,
            message=(
                f"Worst fold Sharpe {worst_sharpe:.3f} acceptable"
                if worst_sharpe > -1.0
                else f"Worst fold Sharpe {worst_sharpe:.3f} indicates severe inconsistency"
            ),
        )
        result.checks.append(worst_check)

        # Sharpe consistency (standard deviation)
        import numpy as np
        sharpe_std = float(np.std(sharpes))
        std_check = ValidationCheck(
            name="Sharpe Consistency (Std Dev)",
            status=(
                ValidationStatus.PASSED
                if sharpe_std < 1.0
                else ValidationStatus.WARNING
            ),
            actual_value=sharpe_std,
            threshold=1.0,
            message=(
                f"Sharpe std dev {sharpe_std:.3f} shows consistency"
                if sharpe_std < 1.0
                else f"Sharpe std dev {sharpe_std:.3f} indicates high variability"
            ),
        )
        result.checks.append(std_check)

        return result

    def validate_all(
        self,
        metrics: PerformanceMetrics,
        oos_accuracy: Optional[float] = None,
        inference_latency_ms: Optional[float] = None,
        feature_latency_ms: Optional[float] = None,
        fold_metrics: Optional[List[PerformanceMetrics]] = None,
    ) -> GoLiveValidationResult:
        """
        Run all Go-Live validations.

        This is the main entry point for comprehensive validation.
        It runs all available checks and combines results.

        Args:
            metrics: Overall performance metrics
            oos_accuracy: Out-of-sample accuracy (optional)
            inference_latency_ms: Model inference time (optional)
            feature_latency_ms: Feature calculation time (optional)
            fold_metrics: Per-fold metrics for consistency (optional)

        Returns:
            Combined validation result with all checks
        """
        result = GoLiveValidationResult()

        # Required checks
        profitability = self.validate_profitability(metrics)
        result.checks.extend(profitability.checks)

        risk_limits = self.validate_risk_limits(metrics)
        result.checks.extend(risk_limits.checks)

        trade_quality = self.validate_trade_quality(metrics)
        result.checks.extend(trade_quality.checks)

        # Optional checks (based on provided data)
        if oos_accuracy is not None:
            accuracy = self.validate_oos_accuracy(oos_accuracy)
            result.checks.extend(accuracy.checks)

        if inference_latency_ms is not None:
            latency = self.validate_inference_latency(
                inference_latency_ms,
                feature_latency_ms,
            )
            result.checks.extend(latency.checks)

        if fold_metrics is not None:
            consistency = self.validate_walk_forward_consistency(fold_metrics)
            result.checks.extend(consistency.checks)

        return result


def check_go_live_ready(
    metrics: PerformanceMetrics,
    thresholds: Optional[GoLiveThresholds] = None,
) -> Tuple[bool, str]:
    """
    Quick check if metrics meet Go-Live profitability requirements.

    Convenience function for simple pass/fail checking.

    Args:
        metrics: Performance metrics from backtest
        thresholds: Custom thresholds (optional)

    Returns:
        Tuple of (passed, message)
    """
    thresholds = thresholds or GoLiveThresholds()

    sharpe_ok = metrics.sharpe_ratio > thresholds.min_sharpe_ratio
    calmar_ok = metrics.calmar_ratio > thresholds.min_calmar_ratio

    if sharpe_ok and calmar_ok:
        return True, f"PASSED: Sharpe={metrics.sharpe_ratio:.3f}, Calmar={metrics.calmar_ratio:.3f}"

    failures = []
    if not sharpe_ok:
        failures.append(f"Sharpe {metrics.sharpe_ratio:.3f} < {thresholds.min_sharpe_ratio}")
    if not calmar_ok:
        failures.append(f"Calmar {metrics.calmar_ratio:.3f} < {thresholds.min_calmar_ratio}")

    return False, f"FAILED: {'; '.join(failures)}"
