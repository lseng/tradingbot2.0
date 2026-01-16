"""
Optimization Results Data Structures.

This module defines data structures for storing and analyzing optimization results:
- TrialResult: Results from a single parameter combination trial
- OptimizationResult: Complete results from an optimization run
- Overfitting analysis utilities

Key Concepts:
- In-sample (IS): Metrics from training/validation data
- Out-of-sample (OOS): Metrics from held-out test data
- Overfitting score: IS/OOS ratio (>1.0 indicates overfitting)

Usage:
    from src.optimization.results import OptimizationResult, TrialResult

    # Create trial result
    trial = TrialResult(
        trial_id=1,
        params={"stop_ticks": 8, "target_ticks": 16},
        metrics={"sharpe_ratio": 1.5, "profit_factor": 1.3},
    )

    # Check overfitting
    oos_sharpe = trial.get_overfitting_score("sharpe_ratio")
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
import json
import numpy as np


class OptimizationStatus(Enum):
    """Status of an optimization run."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PRUNED = "pruned"


@dataclass
class TrialResult:
    """
    Results from a single optimization trial.

    Attributes:
        trial_id: Unique identifier for the trial
        params: Parameter values used in this trial
        metrics: All computed metrics from backtest
        in_sample_metrics: Metrics from validation data (if available)
        out_of_sample_metrics: Metrics from holdout test data (if available)
        status: Trial completion status
        duration_seconds: Time taken for this trial
        error_message: Error message if trial failed
        metadata: Additional metadata (e.g., fold info, timestamps)
    """
    trial_id: int
    params: Dict[str, Any]
    metrics: Dict[str, float] = field(default_factory=dict)
    in_sample_metrics: Optional[Dict[str, float]] = None
    out_of_sample_metrics: Optional[Dict[str, float]] = None
    status: str = "completed"
    duration_seconds: float = 0.0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def get_metric(self, name: str, default: float = 0.0) -> float:
        """
        Get a metric value by name.

        Args:
            name: Metric name
            default: Default value if metric not found

        Returns:
            Metric value
        """
        return self.metrics.get(name, default)

    def get_overfitting_score(self, metric_name: str) -> Optional[float]:
        """
        Calculate overfitting score for a metric.

        Overfitting score = in_sample_metric / out_of_sample_metric
        - Score > 1.0 indicates overfitting (IS better than OOS)
        - Score ~= 1.0 indicates robust parameters
        - Score < 1.0 indicates parameters generalize well

        Args:
            metric_name: Name of the metric to analyze

        Returns:
            Overfitting score or None if OOS data unavailable
        """
        if self.in_sample_metrics is None or self.out_of_sample_metrics is None:
            return None

        is_value = self.in_sample_metrics.get(metric_name, 0.0)
        oos_value = self.out_of_sample_metrics.get(metric_name, 0.0)

        if oos_value == 0:
            return float('inf') if is_value > 0 else 1.0

        return is_value / oos_value

    def is_better_than(
        self,
        other: "TrialResult",
        metric: str,
        higher_is_better: bool = True
    ) -> bool:
        """
        Compare this trial to another based on a metric.

        Args:
            other: Other trial to compare
            metric: Metric name to compare
            higher_is_better: Whether higher values are better

        Returns:
            True if this trial is better
        """
        this_value = self.get_metric(metric)
        other_value = other.get_metric(metric)

        if higher_is_better:
            return this_value > other_value
        else:
            return this_value < other_value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "trial_id": self.trial_id,
            "params": self.params,
            "metrics": self.metrics,
            "in_sample_metrics": self.in_sample_metrics,
            "out_of_sample_metrics": self.out_of_sample_metrics,
            "status": self.status,
            "duration_seconds": self.duration_seconds,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrialResult":
        """Create from dictionary."""
        return cls(
            trial_id=data["trial_id"],
            params=data["params"],
            metrics=data.get("metrics", {}),
            in_sample_metrics=data.get("in_sample_metrics"),
            out_of_sample_metrics=data.get("out_of_sample_metrics"),
            status=data.get("status", "completed"),
            duration_seconds=data.get("duration_seconds", 0.0),
            error_message=data.get("error_message"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class OptimizationResult:
    """
    Complete results from an optimization run.

    Attributes:
        best_params: Best parameter values found
        best_metric: Best value of the target metric
        metric_name: Name of the target metric
        all_results: List of all trial results
        in_sample_metrics: Overall in-sample metrics (validation)
        out_of_sample_metrics: Overall out-of-sample metrics (test)
        overfitting_score: IS/OOS ratio for target metric
        parameter_space_name: Name of the parameter space used
        optimizer_type: Type of optimizer used
        start_time: When optimization started
        end_time: When optimization ended
        total_trials: Total number of trials run
        successful_trials: Number of successful trials
        config: Configuration used for optimization
    """
    best_params: Dict[str, Any]
    best_metric: float
    metric_name: str = "sharpe_ratio"
    all_results: List[TrialResult] = field(default_factory=list)
    in_sample_metrics: Optional[Dict[str, float]] = None
    out_of_sample_metrics: Optional[Dict[str, float]] = None
    overfitting_score: Optional[float] = None
    parameter_space_name: str = ""
    optimizer_type: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    total_trials: int = 0
    successful_trials: int = 0
    config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate derived fields after initialization."""
        if self.total_trials == 0 and self.all_results:
            self.total_trials = len(self.all_results)

        if self.successful_trials == 0 and self.all_results:
            self.successful_trials = sum(
                1 for r in self.all_results if r.status == "completed"
            )

    def get_best_trial(self) -> Optional[TrialResult]:
        """
        Get the best trial result.

        Returns:
            Best TrialResult or None if no results
        """
        if not self.all_results:
            return None

        completed = [r for r in self.all_results if r.status == "completed"]
        if not completed:
            return None

        return max(
            completed,
            key=lambda r: r.get_metric(self.metric_name, float('-inf'))
        )

    def get_top_n_trials(self, n: int = 10) -> List[TrialResult]:
        """
        Get the top N trials by target metric.

        Args:
            n: Number of trials to return

        Returns:
            List of best trials
        """
        completed = [r for r in self.all_results if r.status == "completed"]
        sorted_trials = sorted(
            completed,
            key=lambda r: r.get_metric(self.metric_name, float('-inf')),
            reverse=True
        )
        return sorted_trials[:n]

    def get_parameter_importance(self) -> Dict[str, float]:
        """
        Estimate parameter importance based on metric correlation.

        Returns:
            Dict of parameter name -> importance score (0-1)
        """
        if len(self.all_results) < 10:
            return {}

        completed = [r for r in self.all_results if r.status == "completed"]
        if not completed:
            return {}

        # Get all parameter names
        param_names = list(completed[0].params.keys())
        importance = {}

        target_values = [r.get_metric(self.metric_name) for r in completed]

        for param_name in param_names:
            # Get parameter values across trials
            param_values = []
            for trial in completed:
                val = trial.params.get(param_name)
                # Convert to numeric if possible
                if isinstance(val, (int, float)):
                    param_values.append(val)
                elif isinstance(val, str):
                    # For categorical, use hash
                    param_values.append(hash(val) % 1000)
                else:
                    param_values.append(0)

            # Calculate correlation
            if len(set(param_values)) > 1:  # Need variation
                corr = np.corrcoef(param_values, target_values)[0, 1]
                if np.isnan(corr):
                    corr = 0.0
                importance[param_name] = abs(corr)
            else:
                importance[param_name] = 0.0

        return importance

    def get_overfitting_analysis(
        self,
        metrics: List[str] = None
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze overfitting across multiple metrics.

        Args:
            metrics: List of metrics to analyze (default: common metrics)

        Returns:
            Dict of metric name -> {is_value, oos_value, ratio}
        """
        if self.in_sample_metrics is None or self.out_of_sample_metrics is None:
            return {}

        if metrics is None:
            metrics = ["sharpe_ratio", "profit_factor", "win_rate_pct", "calmar_ratio"]

        analysis = {}
        for metric in metrics:
            is_val = self.in_sample_metrics.get(metric)
            oos_val = self.out_of_sample_metrics.get(metric)

            if is_val is not None and oos_val is not None:
                ratio = is_val / oos_val if oos_val != 0 else float('inf')
                analysis[metric] = {
                    "in_sample": is_val,
                    "out_of_sample": oos_val,
                    "ratio": ratio,
                    "overfitting_detected": ratio > 1.5,  # >50% degradation
                }

        return analysis

    def get_parameter_stability(
        self,
        top_n: int = 10
    ) -> Dict[str, Tuple[float, float]]:
        """
        Analyze stability of parameters across top N trials.

        Args:
            top_n: Number of top trials to analyze

        Returns:
            Dict of param name -> (mean, std)
        """
        top_trials = self.get_top_n_trials(top_n)
        if len(top_trials) < 2:
            return {}

        param_names = list(top_trials[0].params.keys())
        stability = {}

        for param_name in param_names:
            values = []
            for trial in top_trials:
                val = trial.params.get(param_name)
                if isinstance(val, (int, float)):
                    values.append(val)

            if values:
                stability[param_name] = (
                    float(np.mean(values)),
                    float(np.std(values))
                )

        return stability

    def get_convergence_curve(self) -> List[float]:
        """
        Get the best metric value over trials (cumulative max).

        Returns:
            List of best values seen so far at each trial
        """
        if not self.all_results:
            return []

        curve = []
        best_so_far = float('-inf')

        for trial in sorted(self.all_results, key=lambda r: r.trial_id):
            if trial.status == "completed":
                val = trial.get_metric(self.metric_name)
                best_so_far = max(best_so_far, val)
            curve.append(best_so_far if best_so_far != float('-inf') else 0.0)

        return curve

    def duration_seconds(self) -> float:
        """Get total duration in seconds."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return sum(r.duration_seconds for r in self.all_results)

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"=== Optimization Results: {self.optimizer_type} ===",
            f"Parameter Space: {self.parameter_space_name}",
            f"Target Metric: {self.metric_name}",
            f"Trials: {self.successful_trials}/{self.total_trials} successful",
            f"Duration: {self.duration_seconds():.1f}s",
            "",
            "Best Parameters:",
        ]

        for name, value in self.best_params.items():
            if isinstance(value, float):
                lines.append(f"  {name}: {value:.4f}")
            else:
                lines.append(f"  {name}: {value}")

        lines.append(f"\nBest {self.metric_name}: {self.best_metric:.4f}")

        if self.overfitting_score is not None:
            status = "OK" if self.overfitting_score < 1.5 else "WARNING"
            lines.append(f"Overfitting Score: {self.overfitting_score:.2f} ({status})")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "best_params": self.best_params,
            "best_metric": self.best_metric,
            "metric_name": self.metric_name,
            "all_results": [r.to_dict() for r in self.all_results],
            "in_sample_metrics": self.in_sample_metrics,
            "out_of_sample_metrics": self.out_of_sample_metrics,
            "overfitting_score": self.overfitting_score,
            "parameter_space_name": self.parameter_space_name,
            "optimizer_type": self.optimizer_type,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_trials": self.total_trials,
            "successful_trials": self.successful_trials,
            "config": self.config,
        }

    def to_json(self, path: str) -> None:
        """
        Save results to JSON file.

        Args:
            path: Output file path
        """
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OptimizationResult":
        """Create from dictionary."""
        # Parse datetime
        start_time = None
        end_time = None
        if data.get("start_time"):
            start_time = datetime.fromisoformat(data["start_time"])
        if data.get("end_time"):
            end_time = datetime.fromisoformat(data["end_time"])

        # Parse trial results
        all_results = [
            TrialResult.from_dict(r) for r in data.get("all_results", [])
        ]

        return cls(
            best_params=data["best_params"],
            best_metric=data["best_metric"],
            metric_name=data.get("metric_name", "sharpe_ratio"),
            all_results=all_results,
            in_sample_metrics=data.get("in_sample_metrics"),
            out_of_sample_metrics=data.get("out_of_sample_metrics"),
            overfitting_score=data.get("overfitting_score"),
            parameter_space_name=data.get("parameter_space_name", ""),
            optimizer_type=data.get("optimizer_type", ""),
            start_time=start_time,
            end_time=end_time,
            total_trials=data.get("total_trials", 0),
            successful_trials=data.get("successful_trials", 0),
            config=data.get("config", {}),
        )

    @classmethod
    def from_json(cls, path: str) -> "OptimizationResult":
        """
        Load results from JSON file.

        Args:
            path: Input file path

        Returns:
            OptimizationResult instance
        """
        with open(path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


def calculate_overfitting_score(
    in_sample_metric: float,
    out_of_sample_metric: float
) -> float:
    """
    Calculate overfitting score from IS and OOS metrics.

    Args:
        in_sample_metric: Metric value from training/validation
        out_of_sample_metric: Metric value from test set

    Returns:
        Overfitting score (IS/OOS ratio)
    """
    if out_of_sample_metric == 0:
        return float('inf') if in_sample_metric > 0 else 1.0
    return in_sample_metric / out_of_sample_metric


def is_overfitting(
    overfitting_score: float,
    threshold: float = 1.5
) -> bool:
    """
    Check if overfitting score indicates overfitting.

    Args:
        overfitting_score: IS/OOS ratio
        threshold: Score above which overfitting is detected

    Returns:
        True if overfitting detected
    """
    return overfitting_score > threshold


def merge_results(
    results: List[OptimizationResult],
    metric_name: str = None
) -> OptimizationResult:
    """
    Merge multiple optimization results into one.

    Useful for combining results from parallel optimization runs.

    Args:
        results: List of OptimizationResult to merge
        metric_name: Target metric (uses first result's if not specified)

    Returns:
        Merged OptimizationResult
    """
    if not results:
        raise ValueError("Cannot merge empty results list")

    metric_name = metric_name or results[0].metric_name

    # Combine all trials
    all_trials = []
    for i, result in enumerate(results):
        for trial in result.all_results:
            # Renumber trial IDs to be unique
            new_trial = TrialResult(
                trial_id=len(all_trials),
                params=trial.params,
                metrics=trial.metrics,
                in_sample_metrics=trial.in_sample_metrics,
                out_of_sample_metrics=trial.out_of_sample_metrics,
                status=trial.status,
                duration_seconds=trial.duration_seconds,
                error_message=trial.error_message,
                metadata={**trial.metadata, "source_result": i},
            )
            all_trials.append(new_trial)

    # Find best trial
    completed = [t for t in all_trials if t.status == "completed"]
    if completed:
        best_trial = max(
            completed,
            key=lambda t: t.get_metric(metric_name, float('-inf'))
        )
        best_params = best_trial.params
        best_metric = best_trial.get_metric(metric_name)
    else:
        best_params = {}
        best_metric = 0.0

    # Determine time range
    start_times = [r.start_time for r in results if r.start_time]
    end_times = [r.end_time for r in results if r.end_time]

    return OptimizationResult(
        best_params=best_params,
        best_metric=best_metric,
        metric_name=metric_name,
        all_results=all_trials,
        optimizer_type="merged",
        start_time=min(start_times) if start_times else None,
        end_time=max(end_times) if end_times else None,
        total_trials=len(all_trials),
        successful_trials=len(completed),
    )
