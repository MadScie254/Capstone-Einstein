"""
Einstein - Electricity Theft Detection System
Preprocessing Pipeline

This module contains the deterministic feature engineering pipeline
that transforms raw consumption data into model-ready features.

Author: Capstone Team Einstein
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
import joblib
from scipy import stats
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
import warnings

warnings.filterwarnings('ignore')

# Pipeline version for reproducibility tracking
PIPELINE_VERSION = "1.0.0"


@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    # Rolling window sizes (in days)
    rolling_windows: List[int] = None
    # Threshold for zero/missing detection
    zero_threshold: float = 0.001
    # Minimum valid days required
    min_valid_days: int = 10
    
    def __post_init__(self):
        if self.rolling_windows is None:
            self.rolling_windows = [3, 7, 14]


class FeaturePipeline:
    """
    Production-ready feature engineering pipeline for electricity theft detection.
    
    This is the sneaky bit where we catch the meter trying to lie. 🕵️
    All transformations are deterministic and versioned for reproducibility.
    """
    
    def __init__(self, config: Optional[FeatureConfig] = None):
        """Initialize the feature pipeline with configuration."""
        self.config = config or FeatureConfig()
        self.version = PIPELINE_VERSION
        self.fitted = False
        self.feature_names: List[str] = []
        
        # Statistics from training data (for normalization)
        self.train_stats: Dict[str, float] = {}
        
        logger.info(f"FeaturePipeline initialized (version {self.version})")
    
    def fit(self, df: pd.DataFrame) -> 'FeaturePipeline':
        """
        Fit the pipeline on training data to compute normalization statistics.
        
        Args:
            df: Training DataFrame with consumption columns
            
        Returns:
            Self for method chaining
        """
        logger.info(f"Fitting pipeline on {len(df)} samples")
        
        # Compute training statistics for normalization
        consumption_cols = self._get_consumption_columns(df)
        all_values = df[consumption_cols].values.flatten()
        valid_values = all_values[~np.isnan(all_values) & (all_values > self.config.zero_threshold)]
        
        self.train_stats = {
            'mean': float(np.mean(valid_values)) if len(valid_values) > 0 else 1.0,
            'std': float(np.std(valid_values)) if len(valid_values) > 0 else 1.0,
            'median': float(np.median(valid_values)) if len(valid_values) > 0 else 1.0,
            'p25': float(np.percentile(valid_values, 25)) if len(valid_values) > 0 else 0.5,
            'p75': float(np.percentile(valid_values, 75)) if len(valid_values) > 0 else 1.5,
        }
        
        self.fitted = True
        logger.info(f"Pipeline fitted. Training stats: mean={self.train_stats['mean']:.2f}")
        
        return self
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform raw consumption data into engineered features.
        
        Args:
            df: DataFrame with consumption columns (dates as columns)
            
        Returns:
            DataFrame with engineered features
        """
        if not self.fitted:
            logger.warning("Pipeline not fitted, using default statistics")
            self.train_stats = {'mean': 100.0, 'std': 50.0, 'median': 80.0, 'p25': 50.0, 'p75': 120.0}
        
        logger.info(f"Transforming {len(df)} samples")
        
        # Get consumption columns (excluding CONS_NO and FLAG)
        consumption_cols = self._get_consumption_columns(df)
        
        # Extract consumption matrix
        consumption_matrix = df[consumption_cols].values.astype(float)
        
        # Handle missing/invalid values
        consumption_matrix = np.where(
            (consumption_matrix == '') | pd.isna(consumption_matrix),
            np.nan,
            consumption_matrix
        )
        
        # Generate all features
        features = {}
        
        # Basic statistics
        features.update(self._compute_basic_stats(consumption_matrix))
        
        # Pattern features
        features.update(self._compute_pattern_features(consumption_matrix))
        
        # Anomaly indicators
        features.update(self._compute_anomaly_features(consumption_matrix))
        
        # Rolling features
        features.update(self._compute_rolling_features(consumption_matrix))
        
        # Temporal features
        features.update(self._compute_temporal_features(consumption_matrix))
        
        # Spectral features (New: Surgical Accuracy)
        features.update(self._compute_spectral_features(consumption_matrix))
        
        # Create feature DataFrame
        feature_df = pd.DataFrame(features)
        
        # Store feature names
        self.feature_names = list(feature_df.columns)
        
        # Add customer ID if present
        if 'CONS_NO' in df.columns:
            feature_df['CONS_NO'] = df['CONS_NO'].values
        
        # Add label if present (for training)
        if 'FLAG' in df.columns:
            feature_df['FLAG'] = df['FLAG'].values
        
        logger.info(f"Generated {len(self.feature_names)} features")
        
        return feature_df
    
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step."""
        return self.fit(df).transform(df)
    
    def _get_consumption_columns(self, df: pd.DataFrame) -> List[str]:
        """Get columns that contain consumption data (exclude ID and label)."""
        exclude_cols = {'CONS_NO', 'FLAG', 'customer_id', 'meter_id', 'label'}
        return [col for col in df.columns if col not in exclude_cols]
    
    def _compute_basic_stats(self, matrix: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute basic statistical features."""
        features = {}
        
        # Handle NaN values
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Mean consumption
            features['consumption_mean'] = np.nanmean(matrix, axis=1)
            
            # Standard deviation
            features['consumption_std'] = np.nanstd(matrix, axis=1)
            
            # Median
            features['consumption_median'] = np.nanmedian(matrix, axis=1)
            
            # Min and Max
            features['consumption_min'] = np.nanmin(matrix, axis=1)
            features['consumption_max'] = np.nanmax(matrix, axis=1)
            
            # Range
            features['consumption_range'] = features['consumption_max'] - features['consumption_min']
            
            # Coefficient of variation
            features['consumption_cv'] = np.where(
                features['consumption_mean'] > 0,
                features['consumption_std'] / features['consumption_mean'],
                0
            )
            
            # Percentiles
            features['consumption_p25'] = np.nanpercentile(matrix, 25, axis=1)
            features['consumption_p75'] = np.nanpercentile(matrix, 75, axis=1)
            
            # IQR
            features['consumption_iqr'] = features['consumption_p75'] - features['consumption_p25']
            
            # Skewness (simplified)
            mean_centered = matrix - features['consumption_mean'].reshape(-1, 1)
            features['consumption_skew'] = np.nanmean(mean_centered ** 3, axis=1) / (features['consumption_std'] ** 3 + 1e-10)
            
            # Kurtosis (simplified)
            features['consumption_kurtosis'] = np.nanmean(mean_centered ** 4, axis=1) / (features['consumption_std'] ** 4 + 1e-10) - 3
        
        # Replace inf/nan with 0
        for key in features:
            features[key] = np.nan_to_num(features[key], nan=0.0, posinf=0.0, neginf=0.0)
        
        return features
    
    def _compute_pattern_features(self, matrix: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute consumption pattern features."""
        features = {}
        n_samples, n_days = matrix.shape
        
        # Trend (linear regression slope) - Vectorized
        # Slope = (N*Σxy - ΣxΣy) / (N*Σx² - (Σx)²)
        x = np.arange(n_days)
        # Handle missings by filling with row mean for trend calc (approx)
        row_means = np.nanmean(matrix, axis=1).reshape(-1, 1)
        matrix_filled = np.where(np.isnan(matrix), row_means, matrix)
        
        N = n_days
        sum_x = np.sum(x)
        sum_x_sq = np.sum(x**2)
        sum_y = np.sum(matrix_filled, axis=1)
        sum_xy = np.sum(matrix_filled * x, axis=1)
        
        numerator = N * sum_xy - sum_x * sum_y
        denominator = N * sum_x_sq - sum_x**2
        
        # Avoid division by zero
        slopes = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
        features['consumption_trend'] = slopes
        
        # First half vs second half ratio
        half = n_days // 2
        first_half_mean = np.nanmean(matrix[:, :half], axis=1)
        second_half_mean = np.nanmean(matrix[:, half:], axis=1)
        features['first_second_ratio'] = np.where(
            second_half_mean > 0,
            first_half_mean / second_half_mean,
            1.0
        )
        
        # Autocorrelation (lag 1) - Vectorized
        # Corr(X_t, X_{t+1})
        # Use filled matrix from above to handle NaNs
        matrix_trunc = matrix_filled[:, :-1]
        matrix_shifted = matrix_filled[:, 1:]
        
        # Center the data
        mean_trunc = np.mean(matrix_trunc, axis=1, keepdims=True)
        mean_shifted = np.mean(matrix_shifted, axis=1, keepdims=True)
        
        trunc_centered = matrix_trunc - mean_trunc
        shifted_centered = matrix_shifted - mean_shifted
        
        numerator = np.sum(trunc_centered * shifted_centered, axis=1)
        denominator = np.sqrt(np.sum(trunc_centered**2, axis=1) * np.sum(shifted_centered**2, axis=1))
        
        # Handle division by zero
        autocorr = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator!=0)
        features['autocorrelation'] = autocorr
        
        # Day-over-day changes
        daily_changes = np.diff(matrix, axis=1)
        features['avg_daily_change'] = np.nanmean(np.abs(daily_changes), axis=1)
        features['max_daily_change'] = np.nanmax(np.abs(daily_changes), axis=1)
        
        # Normalize by mean
        mean_consumption = np.nanmean(matrix, axis=1).reshape(-1, 1)
        normalized_matrix = matrix / (mean_consumption + 1e-10)
        features['normalized_std'] = np.nanstd(normalized_matrix, axis=1)
        
        # Replace inf/nan
        for key in features:
            features[key] = np.nan_to_num(features[key], nan=0.0, posinf=0.0, neginf=0.0)
        
        return features
    
    def _compute_anomaly_features(self, matrix: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute features that indicate potential meter tampering or theft.
        This is where the magic happens - catching those sneaky patterns! 🎯
        """
        features = {}
        n_samples, n_days = matrix.shape
        
        # Zero/near-zero consumption ratio
        zero_mask = (matrix < self.config.zero_threshold) | np.isnan(matrix)
        features['zero_ratio'] = np.sum(zero_mask, axis=1) / n_days
        
        # Sudden drop detection (>50% drop from previous day)
        daily_changes = np.diff(matrix, axis=1)
        prev_values = matrix[:, :-1]
        drop_threshold = -0.5  # 50% drop
        relative_changes = daily_changes / (prev_values + 1e-10)
        sudden_drops = relative_changes < drop_threshold
        features['sudden_drop_count'] = np.nansum(sudden_drops, axis=1)
        features['sudden_drop_pct'] = features['sudden_drop_count'] / (n_days - 1)
        
        # Sudden spike detection (>100% increase)
        spike_threshold = 1.0  # 100% increase
        sudden_spikes = relative_changes > spike_threshold
        features['sudden_spike_count'] = np.nansum(sudden_spikes, axis=1)
        
        # Consecutive zeros
        max_consecutive_zeros = []
        for i in range(n_samples):
            row = zero_mask[i]
            max_streak = 0
            current_streak = 0
            for val in row:
                if val:
                    current_streak += 1
                    max_streak = max(max_streak, current_streak)
                else:
                    current_streak = 0
            max_consecutive_zeros.append(max_streak)
        features['max_consecutive_zeros'] = np.array(max_consecutive_zeros)
        
        # Constant consumption detection (suspiciously stable)
        features['is_constant'] = (features['consumption_cv'] if 'consumption_cv' in features 
                                   else np.nanstd(matrix, axis=1) / (np.nanmean(matrix, axis=1) + 1e-10)) < 0.01
        features['is_constant'] = features['is_constant'].astype(float)
        
        # Outlier ratio (values outside 2 std from mean)
        row_means = np.nanmean(matrix, axis=1).reshape(-1, 1)
        row_stds = np.nanstd(matrix, axis=1).reshape(-1, 1)
        z_scores = np.abs((matrix - row_means) / (row_stds + 1e-10))
        features['outlier_ratio'] = np.nansum(z_scores > 2, axis=1) / n_days
        
        # Missing data ratio
        features['missing_ratio'] = np.sum(np.isnan(matrix), axis=1) / n_days
        
        # Low consumption periods (below 25th percentile of training data)
        low_threshold = self.train_stats.get('p25', 50.0)
        features['low_consumption_ratio'] = np.sum(matrix < low_threshold, axis=1) / n_days
        
        # Weekend vs weekday pattern (assuming first day is Wednesday based on 01/01/2014)
        # Days 3,4 are Sat/Sun, 10,11 are Sat/Sun, etc.
        weekend_indices = []
        for d in range(n_days):
            # 01/01/2014 is Wednesday, so Saturday is day 3, Sunday is day 4
            day_of_week = (d + 2) % 7  # 0=Mon, 5=Sat, 6=Sun
            if day_of_week >= 5:
                weekend_indices.append(d)
        
        if weekend_indices:
            weekday_indices = [i for i in range(n_days) if i not in weekend_indices]
            weekend_mean = np.nanmean(matrix[:, weekend_indices], axis=1) if weekend_indices else np.zeros(n_samples)
            weekday_mean = np.nanmean(matrix[:, weekday_indices], axis=1) if weekday_indices else np.ones(n_samples)
            features['weekend_weekday_ratio'] = weekend_mean / (weekday_mean + 1e-10)
        else:
            features['weekend_weekday_ratio'] = np.ones(n_samples)
        
        # Replace inf/nan
        for key in features:
            features[key] = np.nan_to_num(features[key], nan=0.0, posinf=0.0, neginf=0.0)
        
        return features
    
    def _compute_rolling_features(self, matrix: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute rolling window features."""
        features = {}
        n_samples, n_days = matrix.shape
        
        for window in self.config.rolling_windows:
            if window >= n_days:
                continue
            
            # Rolling mean
            rolling_means = []
            rolling_stds = []
            for i in range(n_samples):
                row = matrix[i]
                means = []
                stds = []
                for j in range(n_days - window + 1):
                    window_data = row[j:j+window]
                    means.append(np.nanmean(window_data))
                    stds.append(np.nanstd(window_data))
                rolling_means.append(means)
                rolling_stds.append(stds)
            
            rolling_means = np.array(rolling_means)
            rolling_stds = np.array(rolling_stds)
            
            # Stats on rolling windows
            features[f'rolling_{window}d_mean_avg'] = np.nanmean(rolling_means, axis=1)
            features[f'rolling_{window}d_mean_std'] = np.nanstd(rolling_means, axis=1)
            features[f'rolling_{window}d_std_avg'] = np.nanmean(rolling_stds, axis=1)
            
            # Max deviation from rolling mean
            features[f'rolling_{window}d_max_dev'] = np.nanmax(
                np.abs(rolling_means - features[f'rolling_{window}d_mean_avg'].reshape(-1, 1)),
                axis=1
            )
        
        # Replace inf/nan
        for key in features:
            features[key] = np.nan_to_num(features[key], nan=0.0, posinf=0.0, neginf=0.0)
        
        return features
    
    def _compute_temporal_features(self, matrix: np.ndarray) -> Dict[str, np.ndarray]:
        """Compute temporal pattern features."""
        features = {}
        n_samples, n_days = matrix.shape
        
        # Weekly pattern strength (if enough data)
        if n_days >= 14:
            # Compare day 0 with day 7, day 1 with day 8, etc.
            week1 = matrix[:, :7]
            week2 = matrix[:, 7:14]
            
            weekly_corr = []
            for i in range(n_samples):
                w1 = week1[i]
                w2 = week2[i]
                valid = ~(np.isnan(w1) | np.isnan(w2))
                if np.sum(valid) >= 3:
                    corr = np.corrcoef(w1[valid], w2[valid])[0, 1]
                    weekly_corr.append(corr if not np.isnan(corr) else 0.0)
                else:
                    weekly_corr.append(0.0)
            features['weekly_pattern_strength'] = np.array(weekly_corr)
        else:
            features['weekly_pattern_strength'] = np.zeros(n_samples)
        
        # Entropy of consumption (measure of unpredictability)
        entropy = []
        for i in range(n_samples):
            row = matrix[i]
            valid = row[~np.isnan(row)]
            if len(valid) > 0 and np.sum(valid) > 0:
                # Normalize to probability distribution
                probs = valid / np.sum(valid)
                probs = probs[probs > 0]  # Remove zeros for log
                ent = -np.sum(probs * np.log2(probs + 1e-10))
                entropy.append(ent)
            else:
                entropy.append(0.0)
        features['consumption_entropy'] = np.array(entropy)
        
        # Peak to average ratio
        features['peak_to_avg_ratio'] = np.nanmax(matrix, axis=1) / (np.nanmean(matrix, axis=1) + 1e-10)
        
        # Trough to average ratio
        features['trough_to_avg_ratio'] = np.nanmin(matrix, axis=1) / (np.nanmean(matrix, axis=1) + 1e-10)
        
        # Replace inf/nan
        for key in features:
            features[key] = np.nan_to_num(features[key], nan=0.0, posinf=0.0, neginf=0.0)
        
        return features

    def _compute_spectral_features(self, matrix: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute frequency domain features using FFT.
        This detects hidden periodicities or lack thereof (random noise).
        """
        features = {}
        n_samples, n_days = matrix.shape
        
        # FFT requires no NaNs. We interpolate or fill with mean.
        # Simple forward fill + backward fill + fill mean
        matrix_filled = matrix.copy()
        
        # Row-wise fill
        for i in range(n_samples):
            row = matrix_filled[i]
            if np.isnan(row).any():
                # Simple linear interpolation
                valid = ~np.isnan(row)
                if valid.sum() > 1:
                    x = np.arange(n_days)
                    row[~valid] = np.interp(x[~valid], x[valid], row[valid])
                else:
                    row[:] = 0
            matrix_filled[i] = row
            
        # Compute FFT
        # rfft returns the positive frequencies
        fft_vals = np.fft.rfft(matrix_filled, axis=1)
        power_spectrum = np.abs(fft_vals)**2
        
        # Spectral Entropy
        # Normalize power spectrum to be a probability distribution
        ps_sum = np.sum(power_spectrum, axis=1, keepdims=True) + 1e-10
        ps_norm = power_spectrum / ps_sum
        features['spectral_entropy'] = -np.sum(ps_norm * np.log2(ps_norm + 1e-10), axis=1)
        
        # Dominant Frequency (excluding DC component at index 0)
        # We only care about the index of the max power
        if power_spectrum.shape[1] > 1:
            features['dominant_freq_idx'] = np.argmax(power_spectrum[:, 1:], axis=1) + 1
        else:
            features['dominant_freq_idx'] = np.zeros(n_samples)
            
        # Energy concentration (ratio of top 3 frequencies energy to total)
        if power_spectrum.shape[1] > 3:
            sorted_ps = np.sort(power_spectrum[:, 1:], axis=1)[:, ::-1] # Sort descending
            top3_energy = np.sum(sorted_ps[:, :3], axis=1)
            total_energy = np.sum(power_spectrum[:, 1:], axis=1) + 1e-10
            features['spectral_energy_concentration'] = top3_energy / total_energy
        else:
            features['spectral_energy_concentration'] = np.ones(n_samples)

        # Replace inf/nan
        for key in features:
            features[key] = np.nan_to_num(features[key], nan=0.0, posinf=0.0, neginf=0.0)
            
        return features
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return self.feature_names.copy()
    
    def save(self, path: str) -> None:
        """Save the fitted pipeline to disk."""
        joblib.dump({
            'version': self.version,
            'config': self.config,
            'train_stats': self.train_stats,
            'feature_names': self.feature_names,
            'fitted': self.fitted
        }, path)
        logger.info(f"Pipeline saved to {path}")
    
    @classmethod
    def load(cls, path: str) -> 'FeaturePipeline':
        """Load a fitted pipeline from disk."""
        data = joblib.load(path)
        
        pipeline = cls(config=data['config'])
        pipeline.version = data['version']
        pipeline.train_stats = data['train_stats']
        pipeline.feature_names = data['feature_names']
        pipeline.fitted = data['fitted']
        
        logger.info(f"Pipeline loaded from {path} (version {pipeline.version})")
        return pipeline


def create_synthetic_data(n_samples: int = 1000, n_days: int = 26, 
                          theft_ratio: float = 0.1, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic smart meter data for testing.
    
    This creates realistic consumption patterns with injected theft behaviors:
    - Normal customers: smooth consumption with daily/weekly patterns
    - Theft customers: sudden drops, constant readings, or missing data
    
    Args:
        n_samples: Number of customers
        n_days: Number of days of consumption
        theft_ratio: Proportion of customers with theft
        seed: Random seed for reproducibility
        
    Returns:
        DataFrame with synthetic consumption data
    """
    np.random.seed(seed)
    
    # Generate date columns
    dates = pd.date_range('2014-01-01', periods=n_days, freq='D')
    date_cols = [d.strftime('%m/%d/%Y') for d in dates]
    
    data = []
    
    n_theft = int(n_samples * theft_ratio)
    n_normal = n_samples - n_theft
    
    # Generate normal customers
    for i in range(n_normal):
        # Base consumption (50-200 kWh/day)
        base = np.random.uniform(50, 200)
        
        # Daily variation (±20%)
        daily_var = np.random.normal(0, 0.2, n_days)
        
        # Weekly pattern (weekends slightly lower)
        weekly_pattern = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.85, 0.85] * 4)[:n_days]
        
        consumption = base * (1 + daily_var) * weekly_pattern
        consumption = np.maximum(consumption, 5)  # Minimum consumption
        
        # Occasional missing values (1% chance)
        missing_mask = np.random.random(n_days) < 0.01
        consumption = np.where(missing_mask, np.nan, consumption)
        
        row = {col: consumption[j] for j, col in enumerate(date_cols)}
        row['CONS_NO'] = f'NORMAL_{i:06d}'
        row['FLAG'] = 0
        data.append(row)
    
    # Generate theft customers with various patterns
    theft_types = ['sudden_drop', 'constant', 'zeros', 'missing', 'gradual_decline']
    
    for i in range(n_theft):
        theft_type = np.random.choice(theft_types)
        base = np.random.uniform(80, 250)
        
        if theft_type == 'sudden_drop':
            # Normal at first, then sudden 50-90% drop
            drop_day = np.random.randint(5, n_days-5)
            drop_factor = np.random.uniform(0.1, 0.5)
            consumption = np.ones(n_days) * base * (1 + np.random.normal(0, 0.1, n_days))
            consumption[drop_day:] *= drop_factor
            
        elif theft_type == 'constant':
            # Suspiciously constant readings
            consumption = np.ones(n_days) * base * np.random.uniform(0.3, 0.7)
            consumption += np.random.normal(0, 0.5, n_days)  # Very small variation
            
        elif theft_type == 'zeros':
            # Many zero readings
            consumption = np.ones(n_days) * base * (1 + np.random.normal(0, 0.15, n_days))
            zero_days = np.random.choice(n_days, size=np.random.randint(5, 15), replace=False)
            consumption[zero_days] = 0
            
        elif theft_type == 'missing':
            # Many missing readings
            consumption = np.ones(n_days) * base * (1 + np.random.normal(0, 0.15, n_days))
            missing_days = np.random.choice(n_days, size=np.random.randint(8, 18), replace=False)
            consumption = consumption.astype(float)
            consumption[missing_days] = np.nan
            
        else:  # gradual_decline
            # Gradual reduction over time
            decline_rate = np.linspace(1.0, np.random.uniform(0.2, 0.5), n_days)
            consumption = base * decline_rate * (1 + np.random.normal(0, 0.1, n_days))
        
        consumption = np.maximum(consumption, 0)
        
        row = {col: consumption[j] for j, col in enumerate(date_cols)}
        row['CONS_NO'] = f'THEFT_{i:06d}'
        row['FLAG'] = 1
        data.append(row)
    
    # Shuffle the data
    df = pd.DataFrame(data)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    logger.info(f"Generated synthetic data: {n_normal} normal, {n_theft} theft customers")
    
    return df


if __name__ == "__main__":
    # Quick test
    print("Generating synthetic data...")
    df = create_synthetic_data(n_samples=100, theft_ratio=0.2)
    print(f"Data shape: {df.shape}")
    print(f"Theft ratio: {df['FLAG'].mean():.2%}")
    
    print("\nFitting and transforming...")
    pipeline = FeaturePipeline()
    features = pipeline.fit_transform(df)
    print(f"Features shape: {features.shape}")
    print(f"Feature names: {pipeline.get_feature_names()[:10]}...")
    
    print("\n✅ Preprocessing pipeline test passed!")
