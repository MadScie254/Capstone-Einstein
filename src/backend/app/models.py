"""
Einstein - Electricity Theft Detection System
Model Loading and Scoring

This module handles loading trained models and performing inference.
Supports ensemble scoring with XGBoost, IsolationForest, and LSTM Autoencoder.

Author: Capstone Team Einstein
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import joblib
import json
from pathlib import Path
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    # Create placeholder for nn.Module when PyTorch is not available
    class _PlaceholderModule:
        pass
    class nn:
        Module = _PlaceholderModule
    logger.warning("PyTorch not available, LSTM autoencoder will be disabled")

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available, explainability will use fallback")

warnings.filterwarnings('ignore')


@dataclass
class ModelConfig:
    """Configuration for model ensemble."""
    xgb_weight: float = 0.5
    isolation_weight: float = 0.3
    autoencoder_weight: float = 0.2
    theft_threshold: float = 0.5
    high_risk_threshold: float = 0.8


class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder for sequence anomaly detection.
    
    Reconstruction error signals potential theft - meters that don't follow
    normal patterns will have higher reconstruction errors.
    """
    
    def __init__(self, input_size: int = 26, hidden_size: int = 32, 
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Encoder
        self.encoder = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Decoder
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Output layer
        self.output = nn.Linear(hidden_size, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder-decoder."""
        # x shape: (batch, seq_len, 1)
        batch_size, seq_len, _ = x.shape
        
        # Encode
        _, (hidden, cell) = self.encoder(x)
        
        # Repeat hidden state for decoder input
        decoder_input = hidden[-1].unsqueeze(1).repeat(1, seq_len, 1)
        
        # Decode
        decoder_output, _ = self.decoder(decoder_input)
        
        # Output
        reconstruction = self.output(decoder_output)
        
        return reconstruction
    
    def get_reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Compute reconstruction error for anomaly scoring."""
        with torch.no_grad():
            reconstruction = self.forward(x)
            error = torch.mean((x - reconstruction) ** 2, dim=(1, 2))
        return error


class TheftDetectionModels:
    """
    Ensemble model manager for electricity theft detection.
    
    Combines predictions from:
    1. XGBoost - Supervised classification
    2. Isolation Forest - Unsupervised anomaly detection (per cluster)
    3. LSTM Autoencoder - Sequence-based anomaly detection
    """
    
    def __init__(self, artifacts_dir: str = "artifacts", config: Optional[ModelConfig] = None):
        """
        Initialize the model manager.
        
        Args:
            artifacts_dir: Directory containing model artifacts
            config: Model configuration
        """
        self.artifacts_dir = Path(artifacts_dir)
        self.config = config or ModelConfig()
        
        # Models
        self.xgb_model = None
        self.isolation_models: Dict[int, Any] = {}  # cluster_id -> model
        self.autoencoder = None
        self.cluster_model = None
        
        # SHAP explainer
        self.shap_explainer = None
        
        # Metadata
        self.metadata: Dict[str, Any] = {}
        self.feature_names: List[str] = []
        
        # Thresholds (calibrated from training)
        self.thresholds: Dict[str, float] = {}
        
        self.loaded = False
        
        logger.info(f"TheftDetectionModels initialized (artifacts_dir: {artifacts_dir})")
    
    def load_models(self) -> bool:
        """
        Load all model artifacts from disk.
        
        Returns:
            True if at least one model was loaded successfully
        """
        success = False
        
        # Load metadata
        metadata_path = self.artifacts_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                self.metadata = json.load(f)
            logger.info(f"Loaded metadata: version {self.metadata.get('version', 'unknown')}")
        
        # Load XGBoost model
        xgb_path = self.artifacts_dir / "model_xgb.joblib"
        if xgb_path.exists():
            try:
                self.xgb_model = joblib.load(xgb_path)
                logger.info("Loaded XGBoost model")
                success = True
                
                # Initialize SHAP explainer
                if SHAP_AVAILABLE:
                    try:
                        self.shap_explainer = shap.TreeExplainer(self.xgb_model)
                        logger.info("SHAP explainer initialized")
                    except Exception as e:
                        logger.warning(f"Could not initialize SHAP explainer: {e}")
            except Exception as e:
                logger.error(f"Failed to load XGBoost model: {e}")
        
        # Load Isolation Forest models (per cluster)
        for path in self.artifacts_dir.glob("model_isolation_cluster_*.joblib"):
            try:
                cluster_id = int(path.stem.split('_')[-1])
                self.isolation_models[cluster_id] = joblib.load(path)
                logger.info(f"Loaded Isolation Forest for cluster {cluster_id}")
                success = True
            except Exception as e:
                logger.error(f"Failed to load Isolation Forest from {path}: {e}")
        
        # Load single Isolation Forest if no cluster models
        if not self.isolation_models:
            isolation_path = self.artifacts_dir / "model_isolation.joblib"
            if isolation_path.exists():
                try:
                    self.isolation_models[0] = joblib.load(isolation_path)
                    logger.info("Loaded single Isolation Forest model")
                    success = True
                except Exception as e:
                    logger.error(f"Failed to load Isolation Forest: {e}")
        
        # Load cluster model
        cluster_path = self.artifacts_dir / "cluster_model.joblib"
        if cluster_path.exists():
            try:
                self.cluster_model = joblib.load(cluster_path)
                logger.info("Loaded cluster model")
            except Exception as e:
                logger.warning(f"Could not load cluster model: {e}")
        
        # Load LSTM Autoencoder
        if TORCH_AVAILABLE:
            ae_path = self.artifacts_dir / "model_autoencoder.pt"
            if ae_path.exists():
                try:
                    state_dict = torch.load(ae_path, map_location='cpu')
                    self.autoencoder = LSTMAutoencoder()
                    self.autoencoder.load_state_dict(state_dict)
                    self.autoencoder.eval()
                    logger.info("Loaded LSTM Autoencoder")
                    success = True
                except Exception as e:
                    logger.error(f"Failed to load autoencoder: {e}")
        
        # Load thresholds
        thresholds_path = self.artifacts_dir / "thresholds.json"
        if thresholds_path.exists():
            with open(thresholds_path, 'r') as f:
                self.thresholds = json.load(f)
            logger.info("Loaded calibrated thresholds")
        else:
            # Default thresholds
            self.thresholds = {
                'xgb': 0.5,
                'isolation': -0.1,
                'autoencoder': 0.1
            }
        
        # Load feature names
        feature_names_path = self.artifacts_dir / "feature_names.json"
        if feature_names_path.exists():
            with open(feature_names_path, 'r') as f:
                self.feature_names = json.load(f)
        
        self.loaded = success
        logger.info(f"Model loading complete. Status: {'SUCCESS' if success else 'PARTIAL/FAILED'}")
        
        return success
    
    def predict_proba(self, features: pd.DataFrame, 
                      consumption_matrix: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Generate theft probability predictions using ensemble.
        
        Args:
            features: Engineered features from preprocessing pipeline
            consumption_matrix: Raw consumption data for autoencoder (optional)
            
        Returns:
            Dictionary with probabilities and model scores
        """
        if not self.loaded and not self._try_load():
            raise RuntimeError("Models not loaded. Call load_models() first.")
        
        n_samples = len(features)
        
        # Exclude non-feature columns
        feature_cols = [col for col in features.columns 
                       if col not in ['CONS_NO', 'FLAG', 'customer_id', 'cluster']]
        X = features[feature_cols].values.astype(np.float32)
        
        # Handle NaN values
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        results = {
            'probability': np.zeros(n_samples),
            'xgb_score': None,
            'isolation_score': None,
            'autoencoder_score': None,
            'risk_level': []
        }
        
        weights_used = []
        
        # XGBoost prediction
        if self.xgb_model is not None:
            try:
                xgb_proba = self.xgb_model.predict_proba(X)[:, 1]
                results['xgb_score'] = xgb_proba.tolist()
                results['probability'] += self.config.xgb_weight * xgb_proba
                weights_used.append(self.config.xgb_weight)
            except Exception as e:
                logger.error(f"XGBoost prediction failed: {e}")
        
        # Isolation Forest prediction
        if self.isolation_models:
            try:
                # Determine cluster for each sample
                if self.cluster_model is not None and 'cluster' in features.columns:
                    clusters = features['cluster'].values
                elif self.cluster_model is not None:
                    clusters = self.cluster_model.predict(X)
                else:
                    clusters = np.zeros(n_samples, dtype=int)
                
                iso_scores = np.zeros(n_samples)
                for i, cluster_id in enumerate(clusters):
                    if int(cluster_id) in self.isolation_models:
                        model = self.isolation_models[int(cluster_id)]
                    else:
                        # Use first available model as fallback
                        model = list(self.isolation_models.values())[0]
                    
                    # Isolation Forest returns -1 for outliers, 1 for inliers
                    # Convert to probability-like score
                    raw_score = model.decision_function(X[i:i+1])[0]
                    # Normalize: lower scores = more anomalous
                    iso_scores[i] = 1 / (1 + np.exp(raw_score))  # Sigmoid transform
                
                results['isolation_score'] = iso_scores.tolist()
                results['probability'] += self.config.isolation_weight * iso_scores
                weights_used.append(self.config.isolation_weight)
            except Exception as e:
                logger.error(f"Isolation Forest prediction failed: {e}")
        
        # Autoencoder prediction
        if self.autoencoder is not None and consumption_matrix is not None:
            try:
                # Prepare input
                x_tensor = torch.FloatTensor(consumption_matrix).unsqueeze(-1)
                
                # Handle NaN values
                x_tensor = torch.nan_to_num(x_tensor, nan=0.0)
                
                # Normalize
                x_mean = x_tensor.mean(dim=1, keepdim=True)
                x_std = x_tensor.std(dim=1, keepdim=True) + 1e-10
                x_normalized = (x_tensor - x_mean) / x_std
                
                # Get reconstruction error
                recon_error = self.autoencoder.get_reconstruction_error(x_normalized).numpy()
                
                # Normalize reconstruction error to [0, 1]
                ae_threshold = self.thresholds.get('autoencoder', np.median(recon_error))
                ae_scores = 1 / (1 + np.exp(-(recon_error - ae_threshold) / 0.1))
                
                results['autoencoder_score'] = ae_scores.tolist()
                results['probability'] += self.config.autoencoder_weight * ae_scores
                weights_used.append(self.config.autoencoder_weight)
            except Exception as e:
                logger.error(f"Autoencoder prediction failed: {e}")
        
        # Normalize by total weight
        if weights_used:
            total_weight = sum(weights_used)
            results['probability'] /= total_weight
        
        # Assign risk levels
        for prob in results['probability']:
            if prob >= self.config.high_risk_threshold:
                results['risk_level'].append('HIGH')
            elif prob >= self.config.theft_threshold:
                results['risk_level'].append('MEDIUM')
            else:
                results['risk_level'].append('LOW')
        
        results['probability'] = results['probability'].tolist()
        
        return results
    
    def explain(self, features: pd.DataFrame, 
                sample_idx: int = 0) -> Dict[str, Any]:
        """
        Generate SHAP-based explanations for predictions.
        
        Args:
            features: Engineered features
            sample_idx: Index of sample to explain
            
        Returns:
            Dictionary with feature importances and explanations
        """
        feature_cols = [col for col in features.columns 
                       if col not in ['CONS_NO', 'FLAG', 'customer_id', 'cluster']]
        X = features[feature_cols].values.astype(np.float32)
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        
        result = {
            'feature_importance': {},
            'top_features': [],
            'explanation_text': ""
        }
        
        # SHAP explanation
        if self.shap_explainer is not None and SHAP_AVAILABLE:
            try:
                shap_values = self.shap_explainer.shap_values(X[sample_idx:sample_idx+1])
                
                # For binary classification, shap_values might be a list
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Positive class
                
                # Create feature importance dict
                for i, col in enumerate(feature_cols):
                    result['feature_importance'][col] = float(shap_values[0][i])
                
                # Sort by absolute importance
                sorted_features = sorted(
                    result['feature_importance'].items(),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )
                result['top_features'] = sorted_features[:10]
                
            except Exception as e:
                logger.warning(f"SHAP explanation failed: {e}")
        
        # Fallback to model feature importance
        if not result['top_features'] and self.xgb_model is not None:
            try:
                importances = self.xgb_model.feature_importances_
                for i, col in enumerate(feature_cols):
                    result['feature_importance'][col] = float(importances[i])
                
                sorted_features = sorted(
                    result['feature_importance'].items(),
                    key=lambda x: x[1],
                    reverse=True
                )
                result['top_features'] = sorted_features[:10]
            except Exception as e:
                logger.warning(f"Feature importance extraction failed: {e}")
        
        # Generate human-readable explanation
        if result['top_features']:
            explanation_parts = []
            for feat, importance in result['top_features'][:5]:
                direction = "increases" if importance > 0 else "decreases"
                explanation_parts.append(
                    f"'{feat}' {direction} theft likelihood"
                )
            result['explanation_text'] = "; ".join(explanation_parts)
        
        return result
    
    def get_cluster_profiles(self) -> Dict[str, Any]:
        """Get cluster profile information."""
        profiles_path = self.artifacts_dir / "cluster_profiles.json"
        if profiles_path.exists():
            with open(profiles_path, 'r') as f:
                return json.load(f)
        
        return {
            'n_clusters': len(self.isolation_models),
            'profiles': {}
        }
    
    def _try_load(self) -> bool:
        """Attempt to load models if not already loaded."""
        if not self.loaded:
            return self.load_models()
        return True


def create_dummy_models(artifacts_dir: str = "artifacts") -> None:
    """
    Create dummy model artifacts for testing when real models aren't available.
    """
    from sklearn.ensemble import IsolationForest, RandomForestClassifier
    
    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)
    
    logger.info("Creating dummy models for testing...")
    
    # Create dummy XGBoost-like model (using RandomForest)
    dummy_clf = RandomForestClassifier(n_estimators=10, random_state=42)
    # Fit on dummy data
    X_dummy = np.random.randn(100, 35)
    y_dummy = np.random.randint(0, 2, 100)
    dummy_clf.fit(X_dummy, y_dummy)
    joblib.dump(dummy_clf, artifacts_path / "model_xgb.joblib")
    
    # Create dummy Isolation Forest
    dummy_iso = IsolationForest(n_estimators=50, random_state=42)
    dummy_iso.fit(X_dummy)
    joblib.dump(dummy_iso, artifacts_path / "model_isolation_cluster_0.joblib")
    
    # Create metadata
    metadata = {
        'version': '1.0.0',
        'created': '2024-01-01',
        'models': ['xgb', 'isolation'],
        'pipeline_version': '1.0.0'
    }
    with open(artifacts_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Create thresholds
    thresholds = {
        'xgb': 0.5,
        'isolation': -0.1,
        'autoencoder': 0.1
    }
    with open(artifacts_path / "thresholds.json", 'w') as f:
        json.dump(thresholds, f, indent=2)
    
    logger.info(f"Dummy models created in {artifacts_dir}")


if __name__ == "__main__":
    # Quick test
    print("Creating dummy models...")
    create_dummy_models()
    
    print("\nLoading models...")
    models = TheftDetectionModels()
    models.load_models()
    
    print("\nGenerating test predictions...")
    # Create dummy features
    test_features = pd.DataFrame({
        'consumption_mean': [100.0, 50.0],
        'consumption_std': [20.0, 5.0],
        'zero_ratio': [0.0, 0.3],
        'sudden_drop_count': [0, 3]
    })
    
    # Pad with zeros for remaining features
    for i in range(31):
        test_features[f'feature_{i}'] = np.random.randn(2)
    
    results = models.predict_proba(test_features)
    print(f"Probabilities: {results['probability']}")
    print(f"Risk levels: {results['risk_level']}")
    
    print("\n✅ Models test passed!")
