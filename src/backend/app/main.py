"""
Einstein - Electricity Theft Detection System
FastAPI Backend

Production-ready API for electricity theft detection scoring and explainability.

Author: Capstone Team Einstein
Version: 1.0.0

HOW TO RUN:
    cd src/backend
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

API Documentation available at:
    http://localhost:8000/docs (Swagger UI)
    http://localhost:8000/redoc (ReDoc)
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, UploadFile, File, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import io
from loguru import logger

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from preprocess import FeaturePipeline, create_synthetic_data, PIPELINE_VERSION
from models import TheftDetectionModels, create_dummy_models
from utils import (
    generate_pdf_report, 
    format_consumption_for_chart,
    precision_at_k,
    calculate_business_metrics,
    hash_customer_id
)

# =============================================================================
# Configuration
# =============================================================================

# Get artifacts directory
ARTIFACTS_DIR = os.environ.get(
    "ARTIFACTS_DIR", 
    str(Path(__file__).parent.parent.parent.parent / "artifacts")
)

# API configuration
API_VERSION = "1.0.0"
API_TITLE = "Einstein - Electricity Theft Detection API"
API_DESCRIPTION = """
## ⚡ Electricity Theft Detection System

This API provides endpoints for:
- **Scoring**: Predict theft probability for customer consumption data
- **Explainability**: Get SHAP-based feature explanations
- **Cluster Analysis**: Explore customer consumption profiles
- **Reporting**: Generate PDF reports

### Quick Start

1. Use `/score` to submit customer consumption data
2. Get theft probability and risk level
3. Use `/explain` for detailed feature contributions
4. Download PDF report via `/report`
"""

# =============================================================================
# Pydantic Models
# =============================================================================

class ConsumptionData(BaseModel):
    """Input model for scoring a single customer."""
    consumption: List[float] = Field(
        ..., 
        description="Daily consumption values (kWh)",
        min_length=1,
        max_length=365
    )
    customer_id: Optional[str] = Field(
        default=None,
        description="Customer identifier (will be hashed for privacy)"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "consumption": [100.5, 98.2, 105.3, 0.0, 95.7, 92.1, 88.5, 
                              110.2, 107.8, 45.2, 48.3, 95.0, 97.8, 102.1,
                              99.5, 101.2, 103.8, 98.7, 94.2, 96.5, 100.1,
                              105.5, 99.8, 97.2, 101.1, 98.9],
                "customer_id": "CUST_001"
            }
        }


class BatchConsumptionData(BaseModel):
    """Input model for scoring multiple customers."""
    customers: List[ConsumptionData] = Field(
        ...,
        description="List of customer consumption data"
    )


class ScoreResponse(BaseModel):
    """Response model for scoring endpoint."""
    customer_id: str
    probability: float = Field(..., ge=0.0, le=1.0)
    risk_level: str = Field(..., pattern="^(LOW|MEDIUM|HIGH)$")
    xgb_score: Optional[float] = None
    isolation_score: Optional[float] = None
    autoencoder_score: Optional[float] = None
    confidence: str
    timestamp: str


class BatchScoreResponse(BaseModel):
    """Response model for batch scoring."""
    results: List[ScoreResponse]
    total_processed: int
    high_risk_count: int
    processing_time_ms: float


class ExplanationResponse(BaseModel):
    """Response model for explanation endpoint."""
    customer_id: str
    probability: float
    top_features: List[Dict[str, Any]]
    explanation_text: str
    consumption_stats: Dict[str, float]


class ClusterProfile(BaseModel):
    """Model for cluster profile data."""
    cluster_id: int
    n_customers: int
    avg_consumption: float
    theft_rate: float
    description: str


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    version: str
    pipeline_version: str
    models_loaded: bool
    timestamp: str


# =============================================================================
# Application Lifespan
# =============================================================================

# Global instances
pipeline: Optional[FeaturePipeline] = None
models: Optional[TheftDetectionModels] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup and shutdown."""
    global pipeline, models
    
    logger.info("🚀 Starting Einstein Theft Detection API...")
    
    # Initialize feature pipeline
    pipeline_path = Path(ARTIFACTS_DIR) / "pipeline_v1.joblib"
    if pipeline_path.exists():
        try:
            pipeline = FeaturePipeline.load(str(pipeline_path))
            logger.info("✅ Feature pipeline loaded from artifacts")
        except Exception as e:
            logger.warning(f"Could not load pipeline: {e}")
            pipeline = FeaturePipeline()
            logger.info("📝 Using fresh feature pipeline")
    else:
        pipeline = FeaturePipeline()
        logger.info("📝 No saved pipeline found, using fresh instance")
    
    # Initialize models
    models = TheftDetectionModels(artifacts_dir=ARTIFACTS_DIR)
    
    # Try to load models
    if Path(ARTIFACTS_DIR).exists():
        try:
            models.load_models()
            logger.info("✅ Models loaded successfully")
        except Exception as e:
            logger.warning(f"Could not load models: {e}")
    
    # Create dummy models if none loaded (for demo/testing)
    if not models.loaded:
        logger.warning("⚠️ No trained models found, creating dummy models for demo")
        try:
            create_dummy_models(ARTIFACTS_DIR)
            models.load_models()
            logger.info("✅ Dummy models created and loaded")
        except Exception as e:
            logger.error(f"Could not create dummy models: {e}")
    
    logger.info(f"📁 Artifacts directory: {ARTIFACTS_DIR}")
    logger.info("🎯 API ready to serve requests!")
    
    yield
    
    # Cleanup
    logger.info("👋 Shutting down Einstein API...")


# =============================================================================
# FastAPI Application
# =============================================================================

app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Endpoints
# =============================================================================

@app.get("/", tags=["General"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["General"])
async def health_check():
    """
    Health check endpoint.
    
    Returns API status, version info, and model loading status.
    """
    return HealthResponse(
        status="healthy",
        version=API_VERSION,
        pipeline_version=PIPELINE_VERSION,
        models_loaded=models.loaded if models else False,
        timestamp=datetime.now().isoformat()
    )


@app.post("/score", response_model=ScoreResponse, tags=["Scoring"])
async def score_customer(data: ConsumptionData):
    """
    Score a single customer's consumption data for theft probability.
    
    This endpoint accepts daily consumption values and returns:
    - Theft probability (0-1)
    - Risk level (LOW/MEDIUM/HIGH)
    - Individual model scores
    - Confidence indicator
    """
    global pipeline, models
    
    if pipeline is None or models is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Prepare data as DataFrame
        consumption = np.array(data.consumption)
        n_days = len(consumption)
        
        # Create date columns (matching expected format)
        dates = pd.date_range('2014-01-01', periods=n_days, freq='D')
        date_cols = [d.strftime('%m/%d/%Y') for d in dates]
        
        row_data = {col: consumption[i] for i, col in enumerate(date_cols)}
        row_data['CONS_NO'] = data.customer_id or f"API_{datetime.now().timestamp()}"
        
        df = pd.DataFrame([row_data])
        
        # Extract features
        features = pipeline.transform(df)
        
        # Get predictions
        results = models.predict_proba(
            features, 
            consumption_matrix=consumption.reshape(1, -1)
        )
        
        # Determine confidence based on data quality
        valid_days = np.sum(~np.isnan(consumption) & (consumption > 0))
        if valid_days < n_days * 0.5:
            confidence = "low"
        elif valid_days < n_days * 0.8:
            confidence = "medium"
        else:
            confidence = "high"
        
        # Hash customer ID for privacy
        hashed_id = hash_customer_id(row_data['CONS_NO'])
        
        return ScoreResponse(
            customer_id=hashed_id,
            probability=results['probability'][0],
            risk_level=results['risk_level'][0],
            xgb_score=results['xgb_score'][0] if results['xgb_score'] else None,
            isolation_score=results['isolation_score'][0] if results['isolation_score'] else None,
            autoencoder_score=results['autoencoder_score'][0] if results['autoencoder_score'] else None,
            confidence=confidence,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Scoring error: {e}")
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")


@app.post("/score/batch", response_model=BatchScoreResponse, tags=["Scoring"])
async def score_customers_batch(data: BatchConsumptionData):
    """
    Score multiple customers in a single request.
    
    More efficient than multiple single requests for bulk processing.
    """
    import time
    start_time = time.time()
    
    results = []
    high_risk_count = 0
    
    for customer_data in data.customers:
        try:
            result = await score_customer(customer_data)
            results.append(result)
            if result.risk_level == "HIGH":
                high_risk_count += 1
        except HTTPException as e:
            # Include error in results but continue processing
            results.append(ScoreResponse(
                customer_id=customer_data.customer_id or "unknown",
                probability=0.0,
                risk_level="LOW",
                confidence="error",
                timestamp=datetime.now().isoformat()
            ))
    
    processing_time = (time.time() - start_time) * 1000
    
    return BatchScoreResponse(
        results=results,
        total_processed=len(results),
        high_risk_count=high_risk_count,
        processing_time_ms=processing_time
    )


@app.post("/score/csv", tags=["Scoring"])
async def score_csv_upload(file: UploadFile = File(...)):
    """
    Score customers from an uploaded CSV file.
    
    CSV should have columns: dates as headers (MM/DD/YYYY format), 
    CONS_NO for customer ID, and optionally FLAG for labels.
    """
    global pipeline, models
    
    if pipeline is None or models is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Read CSV
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        logger.info(f"Processing CSV with {len(df)} rows")
        
        # Transform features
        features = pipeline.transform(df)
        
        # Get consumption matrix
        consumption_cols = [col for col in df.columns if col not in ['CONS_NO', 'FLAG']]
        consumption_matrix = df[consumption_cols].values.astype(float)
        
        # Get predictions
        results = models.predict_proba(features, consumption_matrix=consumption_matrix)
        
        # Prepare response
        response_data = []
        for i in range(len(df)):
            response_data.append({
                'customer_id': hash_customer_id(df['CONS_NO'].iloc[i]) if 'CONS_NO' in df.columns else f"row_{i}",
                'probability': results['probability'][i],
                'risk_level': results['risk_level'][i]
            })
        
        # Summary statistics
        high_risk = sum(1 for r in response_data if r['risk_level'] == 'HIGH')
        medium_risk = sum(1 for r in response_data if r['risk_level'] == 'MEDIUM')
        
        return {
            'total_customers': len(response_data),
            'high_risk_count': high_risk,
            'medium_risk_count': medium_risk,
            'low_risk_count': len(response_data) - high_risk - medium_risk,
            'results': response_data
        }
        
    except Exception as e:
        logger.error(f"CSV processing error: {e}")
        raise HTTPException(status_code=500, detail=f"CSV processing failed: {str(e)}")


@app.post("/explain", response_model=ExplanationResponse, tags=["Explainability"])
async def explain_prediction(data: ConsumptionData):
    """
    Get detailed explanation for a theft prediction.
    
    Returns:
    - Top contributing features with SHAP values
    - Human-readable explanation text
    - Consumption statistics
    """
    global pipeline, models
    
    if pipeline is None or models is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Prepare data
        consumption = np.array(data.consumption)
        n_days = len(consumption)
        
        dates = pd.date_range('2014-01-01', periods=n_days, freq='D')
        date_cols = [d.strftime('%m/%d/%Y') for d in dates]
        
        row_data = {col: consumption[i] for i, col in enumerate(date_cols)}
        row_data['CONS_NO'] = data.customer_id or "EXPLAIN_REQUEST"
        
        df = pd.DataFrame([row_data])
        features = pipeline.transform(df)
        
        # Get prediction
        results = models.predict_proba(features, consumption_matrix=consumption.reshape(1, -1))
        
        # Get explanation
        explanation = models.explain(features, sample_idx=0)
        
        # Calculate consumption stats
        valid_consumption = consumption[~np.isnan(consumption)]
        stats = {
            'mean': float(np.mean(valid_consumption)) if len(valid_consumption) > 0 else 0,
            'std': float(np.std(valid_consumption)) if len(valid_consumption) > 0 else 0,
            'min': float(np.min(valid_consumption)) if len(valid_consumption) > 0 else 0,
            'max': float(np.max(valid_consumption)) if len(valid_consumption) > 0 else 0,
            'zero_days': int(np.sum(consumption < 0.01)),
            'missing_days': int(np.sum(np.isnan(consumption)))
        }
        
        # Format top features
        top_features = [
            {'feature': feat, 'importance': imp, 'direction': 'increases' if imp > 0 else 'decreases'}
            for feat, imp in explanation.get('top_features', [])
        ]
        
        return ExplanationResponse(
            customer_id=hash_customer_id(row_data['CONS_NO']),
            probability=results['probability'][0],
            top_features=top_features,
            explanation_text=explanation.get('explanation_text', 'No explanation available.'),
            consumption_stats=stats
        )
        
    except Exception as e:
        logger.error(f"Explanation error: {e}")
        raise HTTPException(status_code=500, detail=f"Explanation failed: {str(e)}")


@app.get("/clusters", tags=["Analysis"])
async def get_cluster_profiles():
    """
    Get customer cluster profiles.
    
    Returns consumption patterns and theft rates for each customer segment.
    """
    global models
    
    if models is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    profiles = models.get_cluster_profiles()
    
    # Add default profiles if none exist
    if not profiles.get('profiles'):
        profiles = {
            'n_clusters': 3,
            'profiles': {
                '0': {
                    'name': 'Low Consumers',
                    'avg_consumption': 75.0,
                    'n_customers': 0,
                    'description': 'Customers with below-average consumption'
                },
                '1': {
                    'name': 'Standard Consumers',
                    'avg_consumption': 150.0,
                    'n_customers': 0,
                    'description': 'Customers with typical consumption patterns'
                },
                '2': {
                    'name': 'High Consumers',
                    'avg_consumption': 300.0,
                    'n_customers': 0,
                    'description': 'Commercial or high-usage residential customers'
                }
            }
        }
    
    return profiles


@app.post("/report", tags=["Reporting"])
async def generate_report(data: ConsumptionData):
    """
    Generate a PDF report for a customer assessment.
    
    Returns a downloadable PDF with:
    - Risk assessment summary
    - Top contributing features
    - Consumption statistics
    - Recommended actions
    """
    global pipeline, models
    
    if pipeline is None or models is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    try:
        # Get score and explanation
        consumption = np.array(data.consumption)
        n_days = len(consumption)
        
        dates = pd.date_range('2014-01-01', periods=n_days, freq='D')
        date_cols = [d.strftime('%m/%d/%Y') for d in dates]
        
        row_data = {col: consumption[i] for i, col in enumerate(date_cols)}
        row_data['CONS_NO'] = data.customer_id or "REPORT_REQUEST"
        
        df = pd.DataFrame([row_data])
        features = pipeline.transform(df)
        
        results = models.predict_proba(features, consumption_matrix=consumption.reshape(1, -1))
        explanation = models.explain(features, sample_idx=0)
        
        # Generate PDF
        pdf_bytes = generate_pdf_report(
            customer_id=hash_customer_id(row_data['CONS_NO']),
            probability=results['probability'][0],
            risk_level=results['risk_level'][0],
            top_features=explanation.get('top_features', []),
            consumption_data=data.consumption,
            explanation=explanation.get('explanation_text', '')
        )
        
        if pdf_bytes is None:
            raise HTTPException(status_code=500, detail="PDF generation not available")
        
        # Return PDF as download
        return StreamingResponse(
            io.BytesIO(pdf_bytes),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename=theft_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            }
        )
        
    except Exception as e:
        logger.error(f"Report generation error: {e}")
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")


@app.get("/metrics", tags=["Analysis"])
async def get_model_metrics():
    """
    Get model performance metrics.
    
    Returns precision, recall, and other metrics from model evaluation.
    """
    # Load cached metrics if available
    metrics_path = Path(ARTIFACTS_DIR) / "evaluation_metrics.json"
    
    if metrics_path.exists():
        with open(metrics_path, 'r') as f:
            return json.load(f)
    
    # Return placeholder metrics
    return {
        'note': 'Run evaluation notebook to generate actual metrics',
        'placeholder_metrics': {
            'xgb': {
                'precision_at_1pct': 0.75,
                'precision_at_5pct': 0.60,
                'auprc': 0.82
            },
            'isolation_forest': {
                'precision_at_1pct': 0.55,
                'precision_at_5pct': 0.45,
                'auprc': 0.68
            },
            'ensemble': {
                'precision_at_1pct': 0.78,
                'precision_at_5pct': 0.65,
                'auprc': 0.85
            }
        }
    }


@app.get("/chart-data/{customer_id}", tags=["Visualization"])
async def get_chart_data(customer_id: str, consumption: str = Query(...)):
    """
    Get formatted data for frontend charts.
    
    Accepts consumption as comma-separated values in query string.
    """
    try:
        consumption_list = [float(x.strip()) for x in consumption.split(',')]
        return format_consumption_for_chart(consumption_list)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid consumption data: {str(e)}")


@app.post("/synthetic", tags=["Testing"])
async def generate_synthetic_data(
    n_samples: int = Query(100, ge=10, le=10000),
    theft_ratio: float = Query(0.1, ge=0.0, le=0.5)
):
    """
    Generate synthetic test data.
    
    Useful for testing and demonstrations.
    """
    df = create_synthetic_data(n_samples=n_samples, theft_ratio=theft_ratio)
    
    # Convert to JSON-serializable format
    df_dict = df.to_dict(orient='records')
    
    # Replace NaN with None
    for record in df_dict:
        for key, value in record.items():
            if isinstance(value, float) and np.isnan(value):
                record[key] = None
    
    return {
        'total_samples': len(df_dict),
        'theft_count': int(df['FLAG'].sum()),
        'normal_count': int(len(df) - df['FLAG'].sum()),
        'samples': df_dict[:10],  # Return first 10 as preview
        'message': f'Generated {n_samples} synthetic samples with {theft_ratio:.0%} theft rate'
    }


# =============================================================================
# Error Handlers
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler for unexpected errors."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if os.environ.get("DEBUG") else "An unexpected error occurred"
        }
    )


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    
    # Get configuration from environment
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))
    debug = os.environ.get("DEBUG", "false").lower() == "true"
    
    logger.info(f"Starting server on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=debug,
        workers=1
    )
