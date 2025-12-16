"""
Pipeline Execution Script
Generates data, runs preprocessing, and trains models end-to-end

Author: Capstone Team Einstein
"""

import sys
from pathlib import Path
import json
from datetime import datetime
import argparse

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src' / 'backend'))

import numpy as np
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, precision_recall_curve, auc


def run_pipeline(
    data_path: str = None,
    artifacts_dir: str = None,
    generate_synthetic: bool = True,
    n_synthetic_samples: int = 1000,
    verbose: bool = True
):
    """
    Run end-to-end pipeline.
    
    Args:
        data_path: Path to input CSV (optional)
        artifacts_dir: Output directory for artifacts
        generate_synthetic: Whether to generate synthetic data if no input
        n_synthetic_samples: Number of synthetic samples
        verbose: Print progress messages
    """
    # Setup paths
    if artifacts_dir is None:
        artifacts_dir = PROJECT_ROOT / 'artifacts'
    else:
        artifacts_dir = Path(artifacts_dir)
    
    artifacts_dir.mkdir(exist_ok=True)
    
    data_dir = PROJECT_ROOT / 'data'
    
    def log(msg):
        if verbose:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")
    
    log("="*60)
    log("NTL Detection Pipeline - Execution Started")
    log("="*60)
    
    # Step 1: Load or generate data
    log("\nStep 1: Data Preparation")
    
    if data_path and Path(data_path).exists():
        log(f"  Loading data from {data_path}")
        df = pd.read_csv(data_path)
    elif (data_dir / 'dataset.csv').exists():
        log(f"  Loading data from dataset.csv (Big Data Mode)")
        df = pd.read_csv(data_dir / 'dataset.csv')
    elif (data_dir / 'datasetsmall.csv').exists():
        log(f"  Loading data from datasetsmall.csv")
        df = pd.read_csv(data_dir / 'datasetsmall.csv')
    elif generate_synthetic:
        log(f"  Generating {n_synthetic_samples} synthetic samples")
        from app.preprocess import create_synthetic_data
        df = create_synthetic_data(n_samples=n_synthetic_samples, theft_ratio=0.15, seed=42)
        
        # Save generated data
        output_path = data_dir / 'synthetic_data.csv'
        df.to_csv(output_path, index=False)
        log(f"  Saved synthetic data to {output_path}")
    else:
        raise FileNotFoundError("No data source available")
    
    log(f"  Data shape: {df.shape}")
    log(f"  Theft ratio: {df['FLAG'].mean():.2%}")
    
    # Step 2: Feature Engineering
    log("\nStep 2: Feature Engineering")
    
    from app.preprocess import FeaturePipeline, FeatureConfig
    
    config = FeatureConfig(
        rolling_windows=[3, 7, 14],
        zero_threshold=0.01,
        min_valid_days=10
    )
    
    pipeline = FeaturePipeline(config=config)
    features = pipeline.fit_transform(df)
    
    log(f"  Features generated: {len(pipeline.feature_names)}")
    log(f"  Output shape: {features.shape}")
    
    # Save pipeline
    pipeline.save(str(artifacts_dir / 'pipeline_v1.joblib'))
    log(f"  Saved pipeline to pipeline_v1.joblib")
    
    # Save features (using CSV for compatibility)
    features.to_csv(artifacts_dir / 'preprocessed.csv', index=False)
    log(f"  Saved features to preprocessed.csv")
    
    # Save feature schema
    feature_schema = {
        'version': pipeline.version,
        'created_at': datetime.now().isoformat(),
        'n_features': len(pipeline.feature_names),
        'feature_names': pipeline.feature_names
    }
    
    with open(artifacts_dir / 'feature_schema.json', 'w') as f:
        json.dump(feature_schema, f, indent=2)
    
    # Step 3: Train Models
    log("\nStep 3: Model Training")
    
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import IsolationForest
    from sklearn.metrics import precision_recall_curve, auc
    from xgboost import XGBClassifier
    
    # Prepare data
    feature_cols = [c for c in features.columns if c not in ['CONS_NO', 'FLAG', 'quality_flag', 'cluster']]
    X = features[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0)
    y = features['FLAG'].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    log(f"  Training samples: {len(X_train)}")
    log(f"  Test samples: {len(X_test)}")
    
    # --- Champion-Challenger Model Training ---
    log("  Training Champion (XGBoost) and Challenger (Random Forest)...")
    
    # Split data (using last 20% as validation - temporal split)
    train_size = int(len(features) * 0.8)
    X_train = features.drop(columns=['CONS_NO', 'FLAG']).iloc[:train_size]
    y_train = features['FLAG'].iloc[:train_size]
    X_val = features.drop(columns=['CONS_NO', 'FLAG']).iloc[train_size:]
    y_val = features['FLAG'].iloc[train_size:]
    
    # 1. Train Champion: XGBoost
    xgb_model = XGBClassifier(
        n_estimators=100, 
        max_depth=6, 
        learning_rate=0.1, 
        eval_metric='logloss',
        random_state=42
    )
    xgb_model.fit(X_train, y_train)
    
    # Evaluate Champion
    y_pred_xgb = xgb_model.predict_proba(X_val)[:, 1]
    auprc_xgb = average_precision_score(y_val, y_pred_xgb)
    log(f"  XGBoost (Champion) AUPRC: {auprc_xgb:.4f}")
    
    # 2. Train Challenger: Random Forest (Surgical Ensemble)
    from sklearn.ensemble import RandomForestClassifier
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    # Evaluate Challenger
    y_pred_rf = rf_model.predict_proba(X_val)[:, 1]
    auprc_rf = average_precision_score(y_val, y_pred_rf)
    log(f"  Random Forest (Challenger) AUPRC: {auprc_rf:.4f}")
    
    # Select Winner
    if auprc_rf > auprc_xgb:
        winner_name = "RandomForest"
        winner_model = rf_model
        winner_score = auprc_rf
        log(f"  🏆 NEW CHAMPION: Random Forest outperforms XGBoost by {auprc_rf - auprc_xgb:.4f}")
    else:
        winner_name = "XGBoost"
        winner_model = xgb_model
        winner_score = auprc_xgb
        log(f"  🏆 CHAMPION RETAINED: XGBoost wins")
    
    # Save the winner
    model_path = artifacts_dir / 'model_xgb.joblib' # Keep filename for consistency
    joblib.dump(winner_model, model_path)
    log(f"  Saved best model ({winner_name}) to artifacts/model_xgb.joblib")
    
    # Save Challenger too for comparison view
    rf_path = artifacts_dir / 'model_rf.joblib'
    joblib.dump(rf_model, rf_path)

    # --- Unsupervised Model ---
    log("  Training Unsupervised Model (Isolation Forest)...")
    
    # Train Isolation Forest
    log("  Training Isolation Forest...")
    iso_model = IsolationForest(
        n_estimators=100,
        contamination=0.1,
        random_state=42,
        n_jobs=-1
    )
    iso_model.fit(X_train)
    
    joblib.dump(iso_model, artifacts_dir / 'model_isolation_cluster_0.joblib')
    log(f"  Saved Isolation Forest model")
    
    # Step 4: Save Metadata
    log("\nStep 4: Saving Metadata")
    
    metadata = {
        'version': '1.0.0',
        'created_at': datetime.now().isoformat(),
        'pipeline_version': pipeline.version,
        'models': {
            'xgb': {
                'type': 'XGBClassifier',
                'n_estimators': 100,
                'auprc': float(auprc)
            },
            'isolation_forest': {
                'type': 'IsolationForest',
                'contamination': 0.1
            }
        },
        'training': {
            'n_samples': len(X_train),
            'n_features': len(feature_cols),
            'theft_rate': float(y_train.mean())
        },
        'feature_names': feature_cols
    }
    
    with open(artifacts_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save thresholds
    thresholds = {
        'xgb_default': 0.5,
        'xgb_operational': float(np.percentile(y_proba, 90)),
        'isolation_percentile': 90,
        'high_risk': 0.8,
        'medium_risk': 0.5
    }
    
    with open(artifacts_dir / 'thresholds.json', 'w') as f:
        json.dump(thresholds, f, indent=2)
    
    log("="*60)
    log("Pipeline Execution Complete")
    log("="*60)
    log(f"\nArtifacts saved to: {artifacts_dir}")
    log("\nGenerated files:")
    for f in artifacts_dir.glob('*'):
        log(f"  - {f.name}")
    
    return {
        'auprc': auprc,
        'n_samples': len(df),
        'n_features': len(feature_cols),
        'artifacts_dir': str(artifacts_dir)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run NTL detection pipeline')
    parser.add_argument('--data', type=str, help='Path to input CSV')
    parser.add_argument('--artifacts', type=str, help='Output directory')
    parser.add_argument('--samples', type=int, default=1000, help='Synthetic sample count')
    parser.add_argument('--quiet', action='store_true', help='Suppress output')
    
    args = parser.parse_args()
    
    results = run_pipeline(
        data_path=args.data,
        artifacts_dir=args.artifacts,
        n_synthetic_samples=args.samples,
        verbose=not args.quiet
    )
    
    print(f"\nPipeline completed successfully. AUPRC: {results['auprc']:.4f}")
