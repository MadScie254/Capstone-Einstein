"""
Einstein - Electricity Theft Detection System
Test Suite - Preprocessing Tests

Author: Capstone Team Einstein
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src' / 'backend'))

from app.preprocess import FeaturePipeline, FeatureConfig, create_synthetic_data


class TestSyntheticDataGeneration:
    """Test synthetic data generation."""
    
    def test_create_synthetic_data_shape(self):
        """Test that synthetic data has correct shape."""
        df = create_synthetic_data(n_samples=100, theft_ratio=0.2)
        
        assert len(df) == 100
        assert 'CONS_NO' in df.columns
        assert 'FLAG' in df.columns
    
    def test_create_synthetic_data_theft_ratio(self):
        """Test that theft ratio is approximately correct."""
        df = create_synthetic_data(n_samples=1000, theft_ratio=0.15)
        
        actual_ratio = df['FLAG'].mean()
        assert 0.10 <= actual_ratio <= 0.20  # Allow some variance
    
    def test_create_synthetic_data_values(self):
        """Test that consumption values are reasonable."""
        df = create_synthetic_data(n_samples=100)
        
        consumption_cols = [c for c in df.columns if c not in ['CONS_NO', 'FLAG']]
        consumption_values = df[consumption_cols].values.flatten()
        
        # Remove NaN values
        valid_values = consumption_values[~np.isnan(consumption_values)]
        
        # Most values should be positive
        assert (valid_values >= 0).mean() > 0.95


class TestFeaturePipeline:
    """Test feature engineering pipeline."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        return create_synthetic_data(n_samples=50, theft_ratio=0.2)
    
    @pytest.fixture
    def pipeline(self):
        """Create pipeline instance."""
        return FeaturePipeline()
    
    def test_pipeline_init(self, pipeline):
        """Test pipeline initialization."""
        assert pipeline is not None
        assert pipeline.version == "1.0.0"
        assert pipeline.fitted == False
    
    def test_pipeline_fit(self, pipeline, sample_data):
        """Test pipeline fitting."""
        pipeline.fit(sample_data)
        
        assert pipeline.fitted == True
        assert 'mean' in pipeline.train_stats
        assert 'std' in pipeline.train_stats
    
    def test_pipeline_transform(self, pipeline, sample_data):
        """Test pipeline transformation."""
        features = pipeline.fit_transform(sample_data)
        
        assert len(features) == len(sample_data)
        assert len(pipeline.feature_names) > 0
    
    def test_pipeline_feature_count(self, pipeline, sample_data):
        """Test that we generate expected number of features."""
        features = pipeline.fit_transform(sample_data)
        
        # Exclude ID and label columns
        feature_cols = [c for c in features.columns if c not in ['CONS_NO', 'FLAG']]
        
        # Should have at least 20 features
        assert len(feature_cols) >= 20
    
    def test_pipeline_no_nan_output(self, pipeline, sample_data):
        """Test that output has no NaN values in feature columns."""
        features = pipeline.fit_transform(sample_data)
        
        feature_cols = [c for c in features.columns if c not in ['CONS_NO', 'FLAG']]
        
        has_nan = features[feature_cols].isna().any().any()
        assert not has_nan, "Features should not contain NaN values"
    
    def test_pipeline_save_load(self, pipeline, sample_data, tmp_path):
        """Test pipeline serialization."""
        # Fit and save
        pipeline.fit_transform(sample_data)
        save_path = tmp_path / "pipeline.joblib"
        pipeline.save(str(save_path))
        
        # Load and verify
        loaded = FeaturePipeline.load(str(save_path))
        
        assert loaded.version == pipeline.version
        assert loaded.fitted == True
        assert loaded.train_stats == pipeline.train_stats


class TestFeatureValues:
    """Test specific feature computations."""
    
    @pytest.fixture
    def known_data(self):
        """Create data with known patterns."""
        # Create simple test data
        data = {
            'day1': [100.0, 0.0, 100.0],
            'day2': [100.0, 0.0, 50.0],
            'day3': [100.0, 0.0, 25.0],
            'day4': [100.0, 0.0, 100.0],
            'day5': [100.0, 0.0, 100.0],
            'CONS_NO': ['A', 'B', 'C'],
            'FLAG': [0, 1, 1]
        }
        return pd.DataFrame(data)
    
    def test_zero_ratio_detection(self, known_data):
        """Test that zero consumption is detected."""
        pipeline = FeaturePipeline()
        features = pipeline.fit_transform(known_data)
        
        # Customer B (index 1) has all zeros
        assert features.loc[1, 'zero_ratio'] > 0.9
        
        # Customer A (index 0) has no zeros
        assert features.loc[0, 'zero_ratio'] == 0.0
    
    def test_consumption_mean(self, known_data):
        """Test mean consumption calculation."""
        pipeline = FeaturePipeline()
        features = pipeline.fit_transform(known_data)
        
        # Customer A should have mean ~100
        assert 95 <= features.loc[0, 'consumption_mean'] <= 105


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        pipeline = FeaturePipeline()
        
        # Create empty DataFrame with correct structure
        empty_df = pd.DataFrame(columns=['day1', 'day2', 'CONS_NO', 'FLAG'])
        
        features = pipeline.transform(empty_df)
        assert len(features) == 0
    
    def test_single_row(self):
        """Test handling of single row."""
        pipeline = FeaturePipeline()
        
        data = pd.DataFrame({
            'day1': [100.0],
            'day2': [95.0],
            'day3': [105.0],
            'CONS_NO': ['A'],
            'FLAG': [0]
        })
        
        features = pipeline.fit_transform(data)
        assert len(features) == 1
    
    def test_all_nan_row(self):
        """Test handling of row with all NaN values."""
        pipeline = FeaturePipeline()
        
        data = pd.DataFrame({
            'day1': [np.nan],
            'day2': [np.nan],
            'day3': [np.nan],
            'CONS_NO': ['A'],
            'FLAG': [0]
        })
        
        features = pipeline.fit_transform(data)
        
        # Should not crash, values should be 0 or similar
        feature_cols = [c for c in features.columns if c not in ['CONS_NO', 'FLAG']]
        assert not features[feature_cols].isna().any().any()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
