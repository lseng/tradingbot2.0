"""
Parameter Space Definitions for Strategy Optimization.

This module defines the parameter spaces for optimization, including:
- ParameterConfig: Configuration for individual parameters
- DefaultParameterSpaces: Predefined ranges for MES scalping strategy

Parameter Types:
- Float: Continuous values (e.g., confidence threshold)
- Integer: Discrete values (e.g., stop ticks)
- Categorical: Discrete choices (e.g., order type)

Usage:
    from src.optimization.parameter_space import (
        ParameterConfig,
        DefaultParameterSpaces,
        ParameterSpace,
    )

    # Use default MES scalping parameters
    space = DefaultParameterSpaces.mes_scalping()

    # Custom parameter
    custom_param = ParameterConfig(
        name="my_param",
        min_value=0.0,
        max_value=1.0,
        step=0.1,
        param_type="float"
    )
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import itertools
import numpy as np


class ParameterType(Enum):
    """Types of parameters for optimization."""
    FLOAT = "float"
    INT = "int"
    CATEGORICAL = "categorical"


@dataclass
class ParameterConfig:
    """
    Configuration for a single parameter in optimization.

    Attributes:
        name: Parameter name (used as key in results)
        min_value: Minimum value (for float/int types)
        max_value: Maximum value (for float/int types)
        step: Step size for grid search (for float/int types)
        param_type: Type of parameter (float, int, categorical)
        choices: List of choices (for categorical type)
        default: Default value
        description: Human-readable description
        log_scale: Whether to use log scale for sampling
    """
    name: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    param_type: str = "float"
    choices: Optional[List[Any]] = None
    default: Optional[Any] = None
    description: str = ""
    log_scale: bool = False

    def __post_init__(self):
        """Validate parameter configuration."""
        ptype = ParameterType(self.param_type)

        if ptype in (ParameterType.FLOAT, ParameterType.INT):
            if self.min_value is None or self.max_value is None:
                raise ValueError(
                    f"Parameter '{self.name}': min_value and max_value required "
                    f"for {self.param_type} type"
                )
            if self.min_value > self.max_value:
                raise ValueError(
                    f"Parameter '{self.name}': min_value ({self.min_value}) must be "
                    f"<= max_value ({self.max_value})"
                )

        elif ptype == ParameterType.CATEGORICAL:
            if not self.choices:
                raise ValueError(
                    f"Parameter '{self.name}': choices required for categorical type"
                )

    def get_grid_values(self) -> List[Any]:
        """
        Get all values for grid search.

        Returns:
            List of values to try in grid search
        """
        ptype = ParameterType(self.param_type)

        if ptype == ParameterType.CATEGORICAL:
            return list(self.choices)

        if self.step is None:
            # Default: 10 steps between min and max
            n_steps = 10
            if ptype == ParameterType.INT:
                values = np.linspace(self.min_value, self.max_value, n_steps)
                return sorted(set(int(round(v)) for v in values))
            else:
                return list(np.linspace(self.min_value, self.max_value, n_steps))

        # Generate values with step
        if ptype == ParameterType.INT:
            step = max(1, int(self.step))
            values = list(range(int(self.min_value), int(self.max_value) + 1, step))
        else:
            # Float with step
            n_steps = int((self.max_value - self.min_value) / self.step) + 1
            values = [
                round(self.min_value + i * self.step, 10)
                for i in range(n_steps)
            ]
            # Ensure max_value is included
            if values[-1] < self.max_value:
                values.append(self.max_value)

        return values

    def sample_random(self, rng: Optional[np.random.Generator] = None) -> Any:
        """
        Sample a random value from this parameter's range.

        Args:
            rng: Random number generator (uses default if None)

        Returns:
            Sampled value
        """
        if rng is None:
            rng = np.random.default_rng()

        ptype = ParameterType(self.param_type)

        if ptype == ParameterType.CATEGORICAL:
            return rng.choice(self.choices)

        if self.log_scale:
            # Log-uniform sampling
            log_min = np.log(self.min_value) if self.min_value > 0 else -10
            log_max = np.log(self.max_value)
            value = np.exp(rng.uniform(log_min, log_max))
        else:
            value = rng.uniform(self.min_value, self.max_value)

        if ptype == ParameterType.INT:
            return int(round(value))

        # Round to step if specified
        if self.step is not None:
            value = round(value / self.step) * self.step
            value = round(value, 10)  # Avoid floating point issues

        return value

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "min_value": self.min_value,
            "max_value": self.max_value,
            "step": self.step,
            "param_type": self.param_type,
            "choices": self.choices,
            "default": self.default,
            "description": self.description,
            "log_scale": self.log_scale,
        }


@dataclass
class ParameterSpace:
    """
    Collection of parameters defining the optimization space.

    Attributes:
        parameters: List of parameter configurations
        name: Name of this parameter space
        description: Description of the space
    """
    parameters: List[ParameterConfig] = field(default_factory=list)
    name: str = "unnamed"
    description: str = ""

    def __post_init__(self):
        """Validate parameter names are unique."""
        names = [p.name for p in self.parameters]
        if len(names) != len(set(names)):
            duplicates = [n for n in names if names.count(n) > 1]
            raise ValueError(f"Duplicate parameter names: {set(duplicates)}")

    def add_parameter(self, param: ParameterConfig) -> "ParameterSpace":
        """
        Add a parameter to the space.

        Args:
            param: Parameter configuration to add

        Returns:
            Self for chaining
        """
        if any(p.name == param.name for p in self.parameters):
            raise ValueError(f"Parameter '{param.name}' already exists in space")
        self.parameters.append(param)
        return self

    def remove_parameter(self, name: str) -> "ParameterSpace":
        """
        Remove a parameter from the space.

        Args:
            name: Name of parameter to remove

        Returns:
            Self for chaining
        """
        self.parameters = [p for p in self.parameters if p.name != name]
        return self

    def get_parameter(self, name: str) -> Optional[ParameterConfig]:
        """
        Get a parameter by name.

        Args:
            name: Parameter name

        Returns:
            Parameter configuration or None if not found
        """
        for param in self.parameters:
            if param.name == name:
                return param
        return None

    def get_grid_combinations(self) -> Iterator[Dict[str, Any]]:
        """
        Generate all parameter combinations for grid search.

        Yields:
            Dict of parameter name -> value for each combination
        """
        if not self.parameters:
            yield {}
            return

        # Get all values for each parameter
        param_values = {}
        for param in self.parameters:
            param_values[param.name] = param.get_grid_values()

        # Generate cartesian product
        names = list(param_values.keys())
        value_lists = [param_values[n] for n in names]

        for values in itertools.product(*value_lists):
            yield dict(zip(names, values))

    def count_grid_combinations(self) -> int:
        """
        Count total number of grid search combinations.

        Returns:
            Number of combinations
        """
        if not self.parameters:
            return 0

        count = 1
        for param in self.parameters:
            count *= len(param.get_grid_values())
        return count

    def sample_random(
        self,
        n: int = 1,
        seed: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Sample random parameter combinations.

        Args:
            n: Number of combinations to sample
            seed: Random seed for reproducibility

        Returns:
            List of parameter dictionaries
        """
        rng = np.random.default_rng(seed)
        samples = []

        for _ in range(n):
            sample = {}
            for param in self.parameters:
                sample[param.name] = param.sample_random(rng)
            samples.append(sample)

        return samples

    def get_defaults(self) -> Dict[str, Any]:
        """
        Get default values for all parameters.

        Returns:
            Dict of parameter name -> default value
        """
        defaults = {}
        for param in self.parameters:
            if param.default is not None:
                defaults[param.name] = param.default
            else:
                # Use midpoint as default
                ptype = ParameterType(param.param_type)
                if ptype == ParameterType.CATEGORICAL:
                    defaults[param.name] = param.choices[0]
                elif ptype == ParameterType.INT:
                    defaults[param.name] = int(
                        (param.min_value + param.max_value) / 2
                    )
                else:
                    defaults[param.name] = (param.min_value + param.max_value) / 2
        return defaults

    def validate_params(self, params: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a parameter dictionary against this space.

        Args:
            params: Parameters to validate

        Returns:
            Tuple of (is_valid, list of error messages)
        """
        errors = []

        for param in self.parameters:
            if param.name not in params:
                errors.append(f"Missing parameter: {param.name}")
                continue

            value = params[param.name]
            ptype = ParameterType(param.param_type)

            if ptype == ParameterType.CATEGORICAL:
                if value not in param.choices:
                    errors.append(
                        f"{param.name}: {value} not in choices {param.choices}"
                    )
            else:
                if value < param.min_value or value > param.max_value:
                    errors.append(
                        f"{param.name}: {value} outside range "
                        f"[{param.min_value}, {param.max_value}]"
                    )

        return len(errors) == 0, errors

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": [p.to_dict() for p in self.parameters],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParameterSpace":
        """Create from dictionary."""
        params = [ParameterConfig(**p) for p in data.get("parameters", [])]
        return cls(
            parameters=params,
            name=data.get("name", "unnamed"),
            description=data.get("description", ""),
        )


class DefaultParameterSpaces:
    """
    Factory for predefined parameter spaces.

    Provides optimized parameter ranges for different strategies.
    """

    @staticmethod
    def mes_scalping() -> ParameterSpace:
        """
        Default parameter space for MES scalping strategy.

        Based on specifications:
        - stop_ticks: 4-16 (step 2)
        - target_ticks: 4-24 (step 2)
        - confidence_threshold: 0.55-0.85 (step 0.05)
        - risk_pct: 0.01-0.03 (step 0.005)
        - atr_multiplier: 1.0-3.0 (step 0.25)

        Returns:
            ParameterSpace with MES scalping parameters
        """
        return ParameterSpace(
            name="mes_scalping",
            description="MES Micro E-mini S&P 500 scalping strategy parameters",
            parameters=[
                ParameterConfig(
                    name="stop_ticks",
                    min_value=4,
                    max_value=16,
                    step=2,
                    param_type="int",
                    default=8,
                    description="Stop loss distance in ticks (1 tick = $1.25 for MES)"
                ),
                ParameterConfig(
                    name="target_ticks",
                    min_value=4,
                    max_value=24,
                    step=2,
                    param_type="int",
                    default=16,
                    description="Take profit distance in ticks"
                ),
                ParameterConfig(
                    name="confidence_threshold",
                    min_value=0.55,
                    max_value=0.85,
                    step=0.05,
                    param_type="float",
                    default=0.60,
                    description="Minimum model confidence to enter trade"
                ),
                ParameterConfig(
                    name="risk_pct",
                    min_value=0.01,
                    max_value=0.03,
                    step=0.005,
                    param_type="float",
                    default=0.02,
                    description="Risk per trade as percentage of capital"
                ),
                ParameterConfig(
                    name="atr_multiplier",
                    min_value=1.0,
                    max_value=3.0,
                    step=0.25,
                    param_type="float",
                    default=1.5,
                    description="ATR multiplier for dynamic stop loss"
                ),
            ]
        )

    @staticmethod
    def quick_search() -> ParameterSpace:
        """
        Reduced parameter space for quick optimization runs.

        Uses fewer values per parameter for faster iteration.

        Returns:
            ParameterSpace with reduced search space
        """
        return ParameterSpace(
            name="quick_search",
            description="Reduced parameter space for quick optimization",
            parameters=[
                ParameterConfig(
                    name="stop_ticks",
                    min_value=6,
                    max_value=12,
                    step=3,
                    param_type="int",
                    default=8,
                    description="Stop loss distance in ticks"
                ),
                ParameterConfig(
                    name="target_ticks",
                    min_value=8,
                    max_value=20,
                    step=4,
                    param_type="int",
                    default=16,
                    description="Take profit distance in ticks"
                ),
                ParameterConfig(
                    name="confidence_threshold",
                    min_value=0.55,
                    max_value=0.75,
                    step=0.10,
                    param_type="float",
                    default=0.60,
                    description="Minimum model confidence"
                ),
            ]
        )

    @staticmethod
    def risk_only() -> ParameterSpace:
        """
        Parameter space focusing on risk parameters only.

        Useful when stop/target are already optimized.

        Returns:
            ParameterSpace with risk parameters
        """
        return ParameterSpace(
            name="risk_only",
            description="Risk management parameters only",
            parameters=[
                ParameterConfig(
                    name="risk_pct",
                    min_value=0.005,
                    max_value=0.05,
                    step=0.005,
                    param_type="float",
                    default=0.02,
                    description="Risk per trade as percentage"
                ),
                ParameterConfig(
                    name="max_daily_loss",
                    min_value=25.0,
                    max_value=100.0,
                    step=25.0,
                    param_type="float",
                    default=50.0,
                    description="Maximum daily loss limit"
                ),
                ParameterConfig(
                    name="max_consecutive_losses",
                    min_value=2,
                    max_value=5,
                    step=1,
                    param_type="int",
                    default=3,
                    description="Max consecutive losses before cooldown"
                ),
            ]
        )

    @staticmethod
    def entry_filters() -> ParameterSpace:
        """
        Parameter space for entry signal filters.

        Optimizes when to take trades based on market conditions.

        Returns:
            ParameterSpace with entry filter parameters
        """
        return ParameterSpace(
            name="entry_filters",
            description="Entry signal filter parameters",
            parameters=[
                ParameterConfig(
                    name="confidence_threshold",
                    min_value=0.50,
                    max_value=0.90,
                    step=0.05,
                    param_type="float",
                    default=0.60,
                    description="Minimum model confidence"
                ),
                ParameterConfig(
                    name="min_atr",
                    min_value=0.5,
                    max_value=3.0,
                    step=0.5,
                    param_type="float",
                    default=1.0,
                    description="Minimum ATR for entry (points)"
                ),
                ParameterConfig(
                    name="max_atr",
                    min_value=3.0,
                    max_value=10.0,
                    step=1.0,
                    param_type="float",
                    default=5.0,
                    description="Maximum ATR for entry (avoid high volatility)"
                ),
                ParameterConfig(
                    name="min_volume_ratio",
                    min_value=0.5,
                    max_value=2.0,
                    step=0.25,
                    param_type="float",
                    default=0.8,
                    description="Minimum volume vs average ratio"
                ),
            ]
        )

    @staticmethod
    def custom(
        params: List[Tuple[str, float, float, float, str]],
        name: str = "custom"
    ) -> ParameterSpace:
        """
        Create a custom parameter space from tuples.

        Args:
            params: List of (name, min, max, step, type) tuples
            name: Name for the space

        Returns:
            ParameterSpace with custom parameters

        Example:
            space = DefaultParameterSpaces.custom([
                ("stop_ticks", 4, 12, 2, "int"),
                ("confidence", 0.5, 0.8, 0.1, "float"),
            ])
        """
        param_configs = []
        for p in params:
            param_configs.append(ParameterConfig(
                name=p[0],
                min_value=p[1],
                max_value=p[2],
                step=p[3],
                param_type=p[4],
            ))

        return ParameterSpace(
            name=name,
            parameters=param_configs,
        )


def create_parameter_space_from_config(
    config: Dict[str, Dict[str, Any]]
) -> ParameterSpace:
    """
    Create a parameter space from a configuration dictionary.

    Args:
        config: Dictionary with parameter configurations

    Returns:
        ParameterSpace instance

    Example:
        config = {
            "stop_ticks": {"min": 4, "max": 16, "step": 2, "type": "int"},
            "confidence": {"min": 0.5, "max": 0.9, "step": 0.1, "type": "float"},
        }
        space = create_parameter_space_from_config(config)
    """
    params = []

    for name, cfg in config.items():
        param = ParameterConfig(
            name=name,
            min_value=cfg.get("min"),
            max_value=cfg.get("max"),
            step=cfg.get("step"),
            param_type=cfg.get("type", "float"),
            choices=cfg.get("choices"),
            default=cfg.get("default"),
            description=cfg.get("description", ""),
            log_scale=cfg.get("log_scale", False),
        )
        params.append(param)

    return ParameterSpace(parameters=params)
