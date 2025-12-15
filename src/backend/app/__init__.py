"""
Einstein - Electricity Theft Detection System
Backend Package

This package provides the FastAPI backend for electricity theft detection.
"""

from .preprocess import FeaturePipeline, create_synthetic_data, PIPELINE_VERSION
from .models import TheftDetectionModels, LSTMAutoencoder
from .utils import (
    generate_pdf_report,
    precision_at_k,
    calculate_business_metrics,
    hash_customer_id,
    format_consumption_for_chart
)

__version__ = "1.0.0"
__all__ = [
    "FeaturePipeline",
    "create_synthetic_data",
    "PIPELINE_VERSION",
    "TheftDetectionModels",
    "LSTMAutoencoder",
    "generate_pdf_report",
    "precision_at_k",
    "calculate_business_metrics",
    "hash_customer_id",
    "format_consumption_for_chart"
]
