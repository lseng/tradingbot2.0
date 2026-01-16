"""
Tests for Go-Live Validator Module

This test module validates the GoLiveValidator class which ensures
system readiness before live trading. These tests verify that:

1. Profitability thresholds are correctly enforced (Sharpe > 1.0, Calmar > 0.5)
2. Risk limit checks work properly
3. Walk-forward consistency is validated
4. All Go-Live checklist items can be validated

Why These Tests Matter:
- Prevents deploying unprofitable strategies
- Ensures validation logic catches edge cases
- Documents expected behavior for Go-Live requirements
- Provides regression testing as validation rules evolve

Test Categories:
- test_profitability_*: Sharpe and Calmar threshold tests
- test_risk_*: Risk limit validation tests
- test_latency_*: Inference latency validation tests
- test_walk_forward_*: Walk-forward consistency tests
- test_validation_result_*: Result formatting and reporting tests
"""

import pytest
from datetime import datetime
from typing import List

from src.backtest.go_live_validator import (
    GoLiveValidator,
    GoLiveValidationResult,
    GoLiveThresholds,
    ValidationCheck,
    ValidationStatus,
    check_go_live_ready,
)
from src.backtest.metrics import PerformanceMetrics


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def validator():
    """Create a GoLiveValidator with default thresholds."""
    return GoLiveValidator()


@pytest.fixture
def custom_thresholds():
    """Create custom thresholds for testing edge cases."""
    return GoLiveThresholds(
        min_sharpe_ratio=1.5,
        min_calmar_ratio=0.75,
        min_oos_accuracy=0.55,
        max_drawdown=0.15,
        max_daily_loss_pct=0.03,
        max_inference_latency_ms=5.0,
        max_feature_latency_ms=3.0,
        max_total_latency_ms=8.0,
        min_test_coverage=0.90,
        min_trades_per_fold=150,
        min_profit_factor=1.2,
        min_win_rate=0.45,
    )


@pytest.fixture
def profitable_metrics():
    """Create metrics that pass all profitability thresholds."""
    return PerformanceMetrics(
        sharpe_ratio=1.5,
        calmar_ratio=0.8,
        sortino_ratio=2.0,
        total_return_pct=0.25,
        max_drawdown_pct=0.10,
        max_drawdown_dollars=100.0,
        total_trades=150,
        winning_trades=90,
        losing_trades=60,
        win_rate_pct=60.0,
        profit_factor=1.8,
        expectancy=5.0,
        worst_day_pct=-0.02,
        initial_capital=1000.0,
        final_capital=1250.0,
    )


@pytest.fixture
def unprofitable_metrics():
    """Create metrics that fail profitability thresholds."""
    return PerformanceMetrics(
        sharpe_ratio=0.5,
        calmar_ratio=0.3,
        sortino_ratio=0.6,
        total_return_pct=-0.05,
        max_drawdown_pct=0.25,
        max_drawdown_dollars=250.0,
        total_trades=80,
        winning_trades=35,
        losing_trades=45,
        win_rate_pct=43.75,
        profit_factor=0.8,
        expectancy=-2.0,
        worst_day_pct=-0.08,
        initial_capital=1000.0,
        final_capital=950.0,
    )


@pytest.fixture
def borderline_metrics():
    """Create metrics at exactly the threshold boundaries."""
    return PerformanceMetrics(
        sharpe_ratio=1.0,  # Exactly at threshold
        calmar_ratio=0.5,  # Exactly at threshold
        sortino_ratio=1.5,
        total_return_pct=0.10,
        max_drawdown_pct=0.20,  # Exactly at threshold
        max_drawdown_dollars=200.0,
        total_trades=100,
        winning_trades=50,
        losing_trades=50,
        win_rate_pct=50.0,
        profit_factor=1.0,  # Exactly at threshold
        expectancy=0.0,
        worst_day_pct=-0.05,  # Exactly at threshold
        initial_capital=1000.0,
        final_capital=1100.0,
    )


def create_fold_metrics(sharpes: List[float]) -> List[PerformanceMetrics]:
    """Helper to create fold metrics with given Sharpe ratios."""
    return [
        PerformanceMetrics(
            sharpe_ratio=sharpe,
            calmar_ratio=max(0.1, sharpe * 0.5),
            total_trades=100,
        )
        for sharpe in sharpes
    ]


# =============================================================================
# Profitability Threshold Tests
# =============================================================================

class TestProfitabilityValidation:
    """Tests for Sharpe and Calmar ratio validation."""

    def test_profitable_strategy_passes(self, validator, profitable_metrics):
        """Verify that profitable strategy passes thresholds."""
        result = validator.validate_profitability(profitable_metrics)

        assert result.all_passed
        assert result.passed_count == 2
        assert result.failed_count == 0

    def test_unprofitable_strategy_fails(self, validator, unprofitable_metrics):
        """Verify that unprofitable strategy fails thresholds."""
        result = validator.validate_profitability(unprofitable_metrics)

        assert not result.all_passed
        assert result.failed_count == 2

        # Check specific failures
        sharpe_check = result.checks[0]
        assert sharpe_check.name == "Sharpe Ratio"
        assert not sharpe_check.passed
        assert sharpe_check.actual_value == 0.5

        calmar_check = result.checks[1]
        assert calmar_check.name == "Calmar Ratio"
        assert not calmar_check.passed
        assert calmar_check.actual_value == 0.3

    def test_sharpe_exactly_at_threshold_fails(self, validator, borderline_metrics):
        """Verify that Sharpe exactly at threshold FAILS (must be > not >=)."""
        result = validator.validate_profitability(borderline_metrics)

        sharpe_check = result.checks[0]
        assert not sharpe_check.passed
        assert sharpe_check.actual_value == 1.0
        assert sharpe_check.threshold == 1.0

    def test_calmar_exactly_at_threshold_fails(self, validator, borderline_metrics):
        """Verify that Calmar exactly at threshold FAILS (must be > not >=)."""
        result = validator.validate_profitability(borderline_metrics)

        calmar_check = result.checks[1]
        assert not calmar_check.passed
        assert calmar_check.actual_value == 0.5
        assert calmar_check.threshold == 0.5

    def test_sharpe_just_above_threshold_passes(self, validator):
        """Verify that Sharpe just above threshold passes."""
        metrics = PerformanceMetrics(sharpe_ratio=1.001, calmar_ratio=0.501)
        result = validator.validate_profitability(metrics)

        assert result.all_passed

    def test_negative_sharpe_fails(self, validator):
        """Verify that negative Sharpe ratio fails."""
        metrics = PerformanceMetrics(sharpe_ratio=-0.5, calmar_ratio=0.6)
        result = validator.validate_profitability(metrics)

        sharpe_check = result.checks[0]
        assert not sharpe_check.passed
        assert "below minimum" in sharpe_check.message

    def test_zero_calmar_fails(self, validator):
        """Verify that zero Calmar ratio fails."""
        metrics = PerformanceMetrics(sharpe_ratio=1.5, calmar_ratio=0.0)
        result = validator.validate_profitability(metrics)

        calmar_check = result.checks[1]
        assert not calmar_check.passed

    def test_custom_thresholds(self, custom_thresholds):
        """Verify that custom thresholds are enforced."""
        validator = GoLiveValidator(custom_thresholds)
        metrics = PerformanceMetrics(sharpe_ratio=1.3, calmar_ratio=0.7)

        result = validator.validate_profitability(metrics)

        # Should fail because custom thresholds are higher
        assert not result.all_passed

    def test_very_high_sharpe_passes(self, validator):
        """Verify that very high Sharpe ratio passes."""
        metrics = PerformanceMetrics(sharpe_ratio=3.5, calmar_ratio=2.0)
        result = validator.validate_profitability(metrics)

        assert result.all_passed
        sharpe_check = result.checks[0]
        assert "Good risk-adjusted returns" in sharpe_check.message


# =============================================================================
# OOS Accuracy Tests
# =============================================================================

class TestOOSAccuracyValidation:
    """Tests for out-of-sample accuracy validation."""

    def test_good_accuracy_passes(self, validator):
        """Verify that accuracy > 52% passes."""
        result = validator.validate_oos_accuracy(0.55)

        assert result.all_passed
        check = result.checks[0]
        assert check.name == "OOS Accuracy"
        assert check.passed
        assert "predictive power" in check.message

    def test_bad_accuracy_fails(self, validator):
        """Verify that accuracy < 52% fails."""
        result = validator.validate_oos_accuracy(0.45)

        assert not result.all_passed
        check = result.checks[0]
        assert not check.passed
        assert "below minimum" in check.message

    def test_accuracy_at_threshold_fails(self, validator):
        """Verify that accuracy exactly at 52% fails."""
        result = validator.validate_oos_accuracy(0.52)

        assert not result.all_passed

    def test_accuracy_just_above_threshold_passes(self, validator):
        """Verify that accuracy just above 52% passes."""
        result = validator.validate_oos_accuracy(0.521)

        assert result.all_passed

    def test_random_guess_accuracy_fails(self, validator):
        """Verify that random 3-class accuracy (~33%) fails."""
        result = validator.validate_oos_accuracy(0.33)

        assert not result.all_passed

    def test_perfect_accuracy_passes(self, validator):
        """Verify that perfect accuracy passes."""
        result = validator.validate_oos_accuracy(1.0)

        assert result.all_passed


# =============================================================================
# Inference Latency Tests
# =============================================================================

class TestInferenceLatencyValidation:
    """Tests for inference latency validation."""

    def test_fast_inference_passes(self, validator):
        """Verify that inference < 10ms passes."""
        result = validator.validate_inference_latency(8.0)

        assert result.all_passed
        check = result.checks[0]
        assert check.passed
        assert "meets requirement" in check.message

    def test_slow_inference_fails(self, validator):
        """Verify that inference >= 10ms fails."""
        result = validator.validate_inference_latency(12.0)

        assert not result.all_passed
        check = result.checks[0]
        assert not check.passed
        assert "exceeds" in check.message

    def test_inference_at_threshold_fails(self, validator):
        """Verify that inference exactly at 10ms fails."""
        result = validator.validate_inference_latency(10.0)

        # Threshold is < not <=
        assert not result.all_passed

    def test_feature_latency_checked_when_provided(self, validator):
        """Verify that feature latency is checked when provided."""
        result = validator.validate_inference_latency(
            inference_ms=5.0,
            feature_ms=3.0,
        )

        assert len(result.checks) == 3  # inference, feature, total
        assert result.all_passed

    def test_feature_latency_fails_when_too_slow(self, validator):
        """Verify that feature latency > 5ms fails."""
        result = validator.validate_inference_latency(
            inference_ms=5.0,
            feature_ms=7.0,
        )

        feature_check = next(c for c in result.checks if "Feature" in c.name)
        assert not feature_check.passed

    def test_total_latency_check(self, validator):
        """Verify that total latency is checked."""
        result = validator.validate_inference_latency(
            inference_ms=9.0,
            feature_ms=4.5,
        )

        total_check = next(c for c in result.checks if "Total" in c.name)
        # 9.0 + 4.5 = 13.5ms < 15ms threshold
        assert total_check.passed

    def test_total_latency_fails_when_too_slow(self, validator):
        """Verify that total latency > 15ms fails."""
        result = validator.validate_inference_latency(
            inference_ms=9.0,
            feature_ms=8.0,
        )

        total_check = next(c for c in result.checks if "Total" in c.name)
        # 9.0 + 8.0 = 17ms > 15ms threshold
        assert not total_check.passed


# =============================================================================
# Risk Limits Tests
# =============================================================================

class TestRiskLimitsValidation:
    """Tests for risk limit validation."""

    def test_good_risk_metrics_pass(self, validator, profitable_metrics):
        """Verify that metrics within risk limits pass."""
        result = validator.validate_risk_limits(profitable_metrics)

        assert result.all_passed
        dd_check = next(c for c in result.checks if "Drawdown" in c.name)
        assert dd_check.passed

    def test_excessive_drawdown_fails(self, validator, unprofitable_metrics):
        """Verify that drawdown > 20% fails."""
        result = validator.validate_risk_limits(unprofitable_metrics)

        dd_check = next(c for c in result.checks if "Max Drawdown" in c.name)
        assert not dd_check.passed
        assert unprofitable_metrics.max_drawdown_pct == 0.25

    def test_drawdown_at_threshold_fails(self, validator, borderline_metrics):
        """Verify that drawdown exactly at 20% fails."""
        result = validator.validate_risk_limits(borderline_metrics)

        dd_check = next(c for c in result.checks if "Max Drawdown" in c.name)
        assert not dd_check.passed

    def test_worst_day_checked_when_available(self, validator, unprofitable_metrics):
        """Verify that worst day check is performed when data available."""
        result = validator.validate_risk_limits(unprofitable_metrics)

        daily_check = next(
            (c for c in result.checks if "Daily" in c.name),
            None
        )
        assert daily_check is not None
        assert not daily_check.passed  # -8% exceeds 5% limit


# =============================================================================
# Trade Quality Tests
# =============================================================================

class TestTradeQualityValidation:
    """Tests for trade quality validation."""

    def test_good_trade_quality_passes(self, validator, profitable_metrics):
        """Verify that good trade quality passes."""
        result = validator.validate_trade_quality(profitable_metrics)

        # Should have trade count, profit factor, win rate checks
        assert len(result.checks) >= 3
        assert result.passed_count >= 2

    def test_insufficient_trades_warns(self, validator):
        """Verify that < 100 trades generates warning."""
        metrics = PerformanceMetrics(
            total_trades=50,
            profit_factor=1.5,
            win_rate_pct=55.0,
        )
        result = validator.validate_trade_quality(metrics)

        trade_check = next(c for c in result.checks if "Trade Count" in c.name)
        assert trade_check.status == ValidationStatus.WARNING

    def test_low_profit_factor_fails(self, validator):
        """Verify that profit factor < 1.0 fails."""
        metrics = PerformanceMetrics(
            total_trades=150,
            profit_factor=0.7,
            win_rate_pct=55.0,
        )
        result = validator.validate_trade_quality(metrics)

        pf_check = next(c for c in result.checks if "Profit Factor" in c.name)
        assert not pf_check.passed

    def test_low_win_rate_warns(self, validator):
        """Verify that win rate < 40% generates warning."""
        metrics = PerformanceMetrics(
            total_trades=150,
            profit_factor=1.5,
            win_rate_pct=30.0,  # 30% win rate
        )
        result = validator.validate_trade_quality(metrics)

        wr_check = next(c for c in result.checks if "Win Rate" in c.name)
        assert wr_check.status == ValidationStatus.WARNING


# =============================================================================
# Walk-Forward Consistency Tests
# =============================================================================

class TestWalkForwardConsistencyValidation:
    """Tests for walk-forward consistency validation."""

    def test_consistent_folds_pass(self, validator):
        """Verify that consistent profitable folds pass."""
        fold_metrics = create_fold_metrics([1.2, 1.5, 1.3, 1.4, 1.1])
        result = validator.validate_walk_forward_consistency(fold_metrics)

        assert result.all_passed

    def test_inconsistent_folds_fail(self, validator):
        """Verify that inconsistent folds fail."""
        # 40% profitable (2/5) - below 60% threshold
        fold_metrics = create_fold_metrics([1.2, -0.5, 0.8, -0.3, -0.2])
        result = validator.validate_walk_forward_consistency(fold_metrics)

        pct_check = next(c for c in result.checks if "Profitable Folds" in c.name)
        assert not pct_check.passed

    def test_severely_losing_fold_warns(self, validator):
        """Verify that very negative Sharpe fold generates warning."""
        fold_metrics = create_fold_metrics([1.5, 1.3, -1.5, 1.2, 1.4])
        result = validator.validate_walk_forward_consistency(fold_metrics)

        worst_check = next(c for c in result.checks if "Worst Fold" in c.name)
        assert worst_check.status == ValidationStatus.WARNING

    def test_high_variance_warns(self, validator):
        """Verify that high Sharpe variance generates warning."""
        # Use more extreme values to ensure std > 1.0
        fold_metrics = create_fold_metrics([0.2, 3.0, 0.3, 3.5, 0.1, 3.2])
        result = validator.validate_walk_forward_consistency(fold_metrics)

        std_check = next(c for c in result.checks if "Consistency" in c.name)
        # High variance between 0.1 and 3.5 - std should be > 1.0
        assert std_check.status == ValidationStatus.WARNING

    def test_empty_folds_skipped(self, validator):
        """Verify that empty fold list is handled."""
        result = validator.validate_walk_forward_consistency([])

        assert len(result.checks) == 1
        assert result.checks[0].status == ValidationStatus.SKIPPED

    def test_single_fold_handled(self, validator):
        """Verify that single fold is handled."""
        fold_metrics = create_fold_metrics([1.5])
        result = validator.validate_walk_forward_consistency(fold_metrics)

        # Should work but may warn about sample size
        assert len(result.checks) >= 1


# =============================================================================
# Comprehensive Validation Tests
# =============================================================================

class TestComprehensiveValidation:
    """Tests for validate_all() comprehensive validation."""

    def test_all_passing(self, validator, profitable_metrics):
        """Verify that fully compliant system passes all checks."""
        fold_metrics = create_fold_metrics([1.3, 1.5, 1.2, 1.4, 1.1])

        result = validator.validate_all(
            metrics=profitable_metrics,
            oos_accuracy=0.58,
            inference_latency_ms=7.0,
            feature_latency_ms=3.0,
            fold_metrics=fold_metrics,
        )

        assert result.all_passed
        assert result.failed_count == 0

    def test_failing_profitability(self, validator, unprofitable_metrics):
        """Verify that unprofitable system fails."""
        result = validator.validate_all(metrics=unprofitable_metrics)

        assert not result.all_passed
        assert len(result.failures) > 0

    def test_partial_data_handled(self, validator, profitable_metrics):
        """Verify that partial data is handled gracefully."""
        # Only provide metrics, no optional data
        result = validator.validate_all(metrics=profitable_metrics)

        # Should still validate profitability and risk
        assert len(result.checks) > 0

    def test_report_generation(self, validator, profitable_metrics):
        """Verify that report can be generated."""
        result = validator.validate_all(metrics=profitable_metrics)

        report = result.generate_report()

        assert "GO-LIVE VALIDATION REPORT" in report
        assert "Overall Status" in report
        assert "PASSED" in report or "FAILED" in report


# =============================================================================
# Validation Result Tests
# =============================================================================

class TestValidationResultFormatting:
    """Tests for ValidationResult formatting and serialization."""

    def test_to_dict(self, validator, profitable_metrics):
        """Verify that result can be serialized to dict."""
        result = validator.validate_all(metrics=profitable_metrics)

        result_dict = result.to_dict()

        assert "timestamp" in result_dict
        assert "all_passed" in result_dict
        assert "checks" in result_dict
        assert isinstance(result_dict["checks"], list)

    def test_check_to_dict(self):
        """Verify that individual check can be serialized."""
        check = ValidationCheck(
            name="Test Check",
            status=ValidationStatus.PASSED,
            actual_value=1.5,
            threshold=1.0,
            message="Test message",
        )

        check_dict = check.to_dict()

        assert check_dict["name"] == "Test Check"
        assert check_dict["status"] == "passed"
        assert check_dict["actual_value"] == 1.5
        assert check_dict["threshold"] == 1.0
        assert check_dict["passed"] is True

    def test_failures_property(self, validator, unprofitable_metrics):
        """Verify that failures property returns only failures."""
        result = validator.validate_all(metrics=unprofitable_metrics)

        failures = result.failures
        assert all(f.status == ValidationStatus.FAILED for f in failures)

    def test_warnings_property(self, validator):
        """Verify that warnings property returns only warnings."""
        metrics = PerformanceMetrics(
            sharpe_ratio=1.5,
            calmar_ratio=0.8,
            total_trades=50,  # Will warn
            profit_factor=1.5,
            win_rate_pct=35.0,  # Will warn
        )

        result = validator.validate_all(metrics=metrics)

        warnings = result.warnings
        assert all(w.status == ValidationStatus.WARNING for w in warnings)


# =============================================================================
# Quick Check Function Tests
# =============================================================================

class TestQuickCheckFunction:
    """Tests for check_go_live_ready() convenience function."""

    def test_passing_metrics(self, profitable_metrics):
        """Verify that passing metrics return True."""
        passed, message = check_go_live_ready(profitable_metrics)

        assert passed is True
        assert "PASSED" in message
        assert "Sharpe=" in message
        assert "Calmar=" in message

    def test_failing_metrics(self, unprofitable_metrics):
        """Verify that failing metrics return False."""
        passed, message = check_go_live_ready(unprofitable_metrics)

        assert passed is False
        assert "FAILED" in message

    def test_custom_thresholds(self, profitable_metrics, custom_thresholds):
        """Verify that custom thresholds work."""
        # profitable_metrics has Sharpe=1.5, which is >= custom threshold of 1.5
        # but we need > not >=
        passed, message = check_go_live_ready(profitable_metrics, custom_thresholds)

        # Sharpe 1.5 is not > 1.5, so should fail
        assert passed is False


# =============================================================================
# Edge Case Tests
# =============================================================================

class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_zero_metrics(self, validator):
        """Verify handling of zero-valued metrics."""
        metrics = PerformanceMetrics()  # All zeros
        result = validator.validate_profitability(metrics)

        assert not result.all_passed

    def test_negative_calmar(self, validator):
        """Verify handling of negative Calmar ratio."""
        metrics = PerformanceMetrics(sharpe_ratio=1.5, calmar_ratio=-0.3)
        result = validator.validate_profitability(metrics)

        calmar_check = result.checks[1]
        assert not calmar_check.passed

    def test_infinite_values_handled(self, validator):
        """Verify handling of infinite values."""
        metrics = PerformanceMetrics(
            sharpe_ratio=float('inf'),
            calmar_ratio=float('inf'),
        )
        result = validator.validate_profitability(metrics)

        # Infinite Sharpe and Calmar should pass (> threshold)
        assert result.all_passed

    def test_nan_values(self, validator):
        """Verify handling of NaN values."""
        import math
        metrics = PerformanceMetrics(
            sharpe_ratio=float('nan'),
            calmar_ratio=float('nan'),
        )
        result = validator.validate_profitability(metrics)

        # NaN comparisons should fail
        assert not result.all_passed

    def test_very_small_positive_sharpe(self, validator):
        """Verify handling of very small positive Sharpe."""
        metrics = PerformanceMetrics(
            sharpe_ratio=0.001,
            calmar_ratio=0.001,
        )
        result = validator.validate_profitability(metrics)

        assert not result.all_passed


# =============================================================================
# Integration Tests with Real-like Data
# =============================================================================

class TestIntegrationScenarios:
    """Integration tests with realistic scenarios."""

    def test_typical_scalping_strategy_success(self, validator):
        """Test typical successful scalping strategy profile."""
        metrics = PerformanceMetrics(
            sharpe_ratio=1.8,
            calmar_ratio=0.9,
            sortino_ratio=2.5,
            total_return_pct=0.35,
            max_drawdown_pct=0.12,
            max_drawdown_dollars=120.0,
            total_trades=500,
            winning_trades=275,
            losing_trades=225,
            win_rate_pct=55.0,
            profit_factor=1.6,
            expectancy=3.50,
            worst_day_pct=-0.025,
            initial_capital=1000.0,
            final_capital=1350.0,
        )

        fold_metrics = create_fold_metrics([1.5, 2.0, 1.8, 1.6, 2.2, 1.9])

        result = validator.validate_all(
            metrics=metrics,
            oos_accuracy=0.56,
            inference_latency_ms=6.5,
            feature_latency_ms=2.8,
            fold_metrics=fold_metrics,
        )

        assert result.all_passed
        assert result.failed_count == 0

    def test_typical_scalping_strategy_failure(self, validator):
        """Test typical failing scalping strategy profile."""
        metrics = PerformanceMetrics(
            sharpe_ratio=0.6,
            calmar_ratio=0.25,
            sortino_ratio=0.8,
            total_return_pct=-0.08,
            max_drawdown_pct=0.22,
            max_drawdown_dollars=220.0,
            total_trades=300,
            winning_trades=135,
            losing_trades=165,
            win_rate_pct=45.0,
            profit_factor=0.85,
            expectancy=-1.50,
            worst_day_pct=-0.06,
            initial_capital=1000.0,
            final_capital=920.0,
        )

        fold_metrics = create_fold_metrics([0.4, -0.3, 0.6, -0.1, 0.2, -0.5])

        result = validator.validate_all(
            metrics=metrics,
            oos_accuracy=0.48,
            inference_latency_ms=12.0,
            fold_metrics=fold_metrics,
        )

        assert not result.all_passed
        assert len(result.failures) >= 3  # Multiple failures expected

    def test_overfitted_strategy_detected(self, validator):
        """Test that overfitted strategy is detected via consistency."""
        # Good overall metrics but highly variable folds
        metrics = PerformanceMetrics(
            sharpe_ratio=1.5,
            calmar_ratio=0.7,
            total_trades=200,
            profit_factor=1.5,
            win_rate_pct=55.0,
            max_drawdown_pct=0.15,
        )

        # Highly variable fold performance suggests overfitting
        fold_metrics = create_fold_metrics([3.0, -1.5, 2.5, -0.8, 2.8, -1.2])

        result = validator.validate_all(
            metrics=metrics,
            fold_metrics=fold_metrics,
        )

        # Should have warnings about consistency
        warnings = result.warnings
        assert len(warnings) > 0
