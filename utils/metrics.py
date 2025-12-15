"""
Einstein - Electricity Theft Detection System
Utility Functions - Metrics

Author: Capstone Team Einstein
"""

import numpy as np
from typing import Dict, Any


def precision_at_k(y_true: np.ndarray, y_scores: np.ndarray, k: float) -> float:
    """
    Calculate precision at top k% of predictions.
    
    Args:
        y_true: True labels (0/1)
        y_scores: Prediction scores
        k: Top k percentage (0.0-1.0)
        
    Returns:
        Precision at k
    """
    n = len(y_true)
    top_k = max(1, int(n * k))
    
    top_indices = np.argsort(y_scores)[-top_k:]
    precision = np.mean(y_true[top_indices])
    
    return float(precision)


def time_to_detection(
    y_true: np.ndarray, 
    y_pred: np.ndarray
) -> Dict[str, Any]:
    """
    Calculate detection metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with detection metrics
    """
    true_positives = (y_true == 1) & (y_pred == 1)
    false_negatives = (y_true == 1) & (y_pred == 0)
    
    return {
        'total_theft_cases': int(np.sum(y_true)),
        'detected_cases': int(np.sum(true_positives)),
        'missed_cases': int(np.sum(false_negatives)),
        'detection_rate': float(np.sum(true_positives) / (np.sum(y_true) + 1e-10))
    }


def calculate_business_metrics(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    theft_threshold: float = 0.5,
    avg_theft_value: float = 1000.0
) -> Dict[str, float]:
    """
    Calculate business impact metrics.
    
    Args:
        y_true: True labels
        y_scores: Prediction probabilities
        theft_threshold: Threshold for positive prediction
        avg_theft_value: Average value of theft case
        
    Returns:
        Dictionary with business metrics
    """
    y_pred = (y_scores >= theft_threshold).astype(int)
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    
    revenue_recovered = tp * avg_theft_value
    investigation_cost = (tp + fp) * (avg_theft_value * 0.1)
    missed_revenue = fn * avg_theft_value
    
    return {
        'true_positives': int(tp),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_negatives': int(tn),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(2 * precision * recall / (precision + recall + 1e-10)),
        'revenue_recovered': float(revenue_recovered),
        'net_benefit': float(revenue_recovered - investigation_cost - missed_revenue)
    }
