"""
Synthetic Theft Injection Framework
Programmatically inject theft patterns for stress testing model recall

Author: Capstone Team Einstein
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class TheftPattern(Enum):
    """Types of theft patterns to inject."""
    SUDDEN_DROP = "sudden_drop"
    GRADUAL_DECLINE = "gradual_decline"
    PERIODIC_ZEROS = "periodic_zeros"
    CONSTANT_LOW = "constant_low"
    STEP_CHANGE = "step_change"
    WEEKEND_THEFT = "weekend_theft"


@dataclass
class TheftInjectionConfig:
    """Configuration for theft pattern injection."""
    pattern: TheftPattern
    start_day: int
    duration_days: int
    magnitude: float  # 0.0 to 1.0 (fraction reduction)
    noise_factor: float = 0.1


def inject_sudden_drop(
    consumption: np.ndarray,
    start_day: int,
    duration: int,
    magnitude: float,
    noise: float = 0.1
) -> np.ndarray:
    """
    Inject sudden consumption drop pattern.
    
    Simulates meter bypass or tampering that causes immediate reduction.
    """
    modified = consumption.copy()
    baseline = np.mean(consumption[:start_day]) if start_day > 0 else np.mean(consumption)
    
    for i in range(start_day, min(start_day + duration, len(modified))):
        reduction = baseline * magnitude
        noise_val = np.random.normal(0, noise * baseline)
        modified[i] = max(0, modified[i] - reduction + noise_val)
    
    return modified


def inject_gradual_decline(
    consumption: np.ndarray,
    start_day: int,
    duration: int,
    magnitude: float,
    noise: float = 0.1
) -> np.ndarray:
    """
    Inject gradual consumption decline pattern.
    
    Simulates progressive meter tampering to avoid detection.
    """
    modified = consumption.copy()
    baseline = np.mean(consumption[:start_day]) if start_day > 0 else np.mean(consumption)
    
    for i in range(start_day, min(start_day + duration, len(modified))):
        days_in = i - start_day
        reduction_factor = (days_in / duration) * magnitude
        reduction = baseline * reduction_factor
        noise_val = np.random.normal(0, noise * baseline)
        modified[i] = max(0, modified[i] - reduction + noise_val)
    
    return modified


def inject_periodic_zeros(
    consumption: np.ndarray,
    start_day: int,
    duration: int,
    magnitude: float,  # Used as probability of zero
    noise: float = 0.1
) -> np.ndarray:
    """
    Inject periodic zero consumption pattern.
    
    Simulates intermittent meter bypass.
    """
    modified = consumption.copy()
    
    for i in range(start_day, min(start_day + duration, len(modified))):
        if np.random.random() < magnitude:
            modified[i] = np.random.uniform(0, noise * 10)
    
    return modified


def inject_constant_low(
    consumption: np.ndarray,
    start_day: int,
    duration: int,
    magnitude: float,
    noise: float = 0.1
) -> np.ndarray:
    """
    Inject constant low consumption pattern.
    
    Simulates fixed meter manipulation.
    """
    modified = consumption.copy()
    baseline = np.mean(consumption[:start_day]) if start_day > 0 else np.mean(consumption)
    
    target_value = baseline * (1 - magnitude)
    
    for i in range(start_day, min(start_day + duration, len(modified))):
        noise_val = np.random.normal(0, noise * baseline)
        modified[i] = max(0, target_value + noise_val)
    
    return modified


def inject_step_change(
    consumption: np.ndarray,
    start_day: int,
    duration: int,
    magnitude: float,
    noise: float = 0.1
) -> np.ndarray:
    """
    Inject step change pattern.
    
    Simulates meter replacement with manipulated calibration.
    """
    modified = consumption.copy()
    baseline = np.mean(consumption[:start_day]) if start_day > 0 else np.mean(consumption)
    
    for i in range(start_day, len(modified)):
        reduction = baseline * magnitude
        noise_val = np.random.normal(0, noise * baseline)
        modified[i] = max(0, modified[i] - reduction + noise_val)
    
    return modified


def inject_weekend_theft(
    consumption: np.ndarray,
    start_day: int,
    duration: int,
    magnitude: float,
    noise: float = 0.1
) -> np.ndarray:
    """
    Inject weekend-only theft pattern.
    
    Simulates theft only on weekends to avoid detection.
    """
    modified = consumption.copy()
    baseline = np.mean(consumption)
    
    for i in range(start_day, min(start_day + duration, len(modified))):
        # Assume day 0 is Monday, so days 5 and 6 are weekend
        if (i % 7) in [5, 6]:
            reduction = baseline * magnitude
            noise_val = np.random.normal(0, noise * baseline)
            modified[i] = max(0, modified[i] - reduction + noise_val)
    
    return modified


INJECTION_FUNCTIONS = {
    TheftPattern.SUDDEN_DROP: inject_sudden_drop,
    TheftPattern.GRADUAL_DECLINE: inject_gradual_decline,
    TheftPattern.PERIODIC_ZEROS: inject_periodic_zeros,
    TheftPattern.CONSTANT_LOW: inject_constant_low,
    TheftPattern.STEP_CHANGE: inject_step_change,
    TheftPattern.WEEKEND_THEFT: inject_weekend_theft,
}


def inject_theft_pattern(
    consumption: np.ndarray,
    config: TheftInjectionConfig
) -> np.ndarray:
    """
    Inject theft pattern according to configuration.
    
    Args:
        consumption: Original consumption array
        config: Injection configuration
        
    Returns:
        Modified consumption array with injected pattern
    """
    inject_func = INJECTION_FUNCTIONS[config.pattern]
    return inject_func(
        consumption,
        config.start_day,
        config.duration_days,
        config.magnitude,
        config.noise_factor
    )


def create_theft_test_dataset(
    normal_data: pd.DataFrame,
    n_theft_samples: int,
    patterns: Optional[List[TheftPattern]] = None,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create test dataset with injected theft patterns.
    
    Args:
        normal_data: DataFrame with normal consumption data
        n_theft_samples: Number of theft samples to create
        patterns: List of patterns to use (default: all)
        random_state: Random seed
        
    Returns:
        Tuple of (modified_data, injection_log)
    """
    np.random.seed(random_state)
    
    if patterns is None:
        patterns = list(TheftPattern)
    
    # Identify consumption columns
    exclude_cols = ['CONS_NO', 'FLAG']
    consumption_cols = [c for c in normal_data.columns if c not in exclude_cols]
    
    # Sample normal customers to convert to theft
    sample_indices = np.random.choice(
        normal_data[normal_data['FLAG'] == 0].index,
        size=min(n_theft_samples, len(normal_data[normal_data['FLAG'] == 0])),
        replace=False
    )
    
    modified_data = normal_data.copy()
    injection_log = []
    
    for idx in sample_indices:
        # Select random pattern
        pattern = np.random.choice(patterns)
        
        # Random configuration
        n_days = len(consumption_cols)
        start_day = np.random.randint(0, max(1, n_days // 2))
        duration = np.random.randint(5, n_days - start_day)
        magnitude = np.random.uniform(0.4, 0.9)
        
        config = TheftInjectionConfig(
            pattern=pattern,
            start_day=start_day,
            duration_days=duration,
            magnitude=magnitude
        )
        
        # Get original consumption
        original = modified_data.loc[idx, consumption_cols].values.astype(float)
        
        # Inject pattern
        modified = inject_theft_pattern(original, config)
        
        # Update data
        modified_data.loc[idx, consumption_cols] = modified
        modified_data.loc[idx, 'FLAG'] = 1
        
        # Log injection
        injection_log.append({
            'index': idx,
            'customer_id': modified_data.loc[idx, 'CONS_NO'],
            'pattern': pattern.value,
            'start_day': start_day,
            'duration': duration,
            'magnitude': magnitude
        })
    
    return modified_data, pd.DataFrame(injection_log)


def calculate_detection_rate(
    model_predictions: np.ndarray,
    injection_log: pd.DataFrame,
    threshold: float = 0.5
) -> dict:
    """
    Calculate detection rate for injected theft patterns.
    
    Args:
        model_predictions: Model probability predictions
        injection_log: Log of injected patterns
        threshold: Classification threshold
        
    Returns:
        Detection statistics by pattern type
    """
    results = {}
    
    for pattern in TheftPattern:
        pattern_indices = injection_log[
            injection_log['pattern'] == pattern.value
        ]['index'].values
        
        if len(pattern_indices) == 0:
            continue
        
        pattern_preds = model_predictions[pattern_indices]
        detected = np.sum(pattern_preds >= threshold)
        
        results[pattern.value] = {
            'total': len(pattern_indices),
            'detected': int(detected),
            'detection_rate': float(detected / len(pattern_indices)),
            'avg_probability': float(np.mean(pattern_preds))
        }
    
    return results


if __name__ == "__main__":
    # Example usage
    print("Synthetic Theft Injection Framework")
    print("=" * 40)
    
    # Create sample data
    n_customers = 100
    n_days = 26
    
    consumption_data = np.random.normal(100, 20, (n_customers, n_days))
    consumption_data = np.maximum(consumption_data, 0)
    
    cols = [f'day_{i+1}' for i in range(n_days)]
    df = pd.DataFrame(consumption_data, columns=cols)
    df['CONS_NO'] = [f'CUST_{i:04d}' for i in range(n_customers)]
    df['FLAG'] = 0
    
    print(f"Original data: {len(df)} customers, {n_days} days")
    print(f"Original theft rate: {df['FLAG'].mean():.2%}")
    
    # Inject theft patterns
    modified_df, log = create_theft_test_dataset(
        df, 
        n_theft_samples=20,
        random_state=42
    )
    
    print(f"\nAfter injection:")
    print(f"Modified theft rate: {modified_df['FLAG'].mean():.2%}")
    print(f"\nInjection summary:")
    print(log['pattern'].value_counts())
