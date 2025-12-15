"""
Einstein - Electricity Theft Detection System
Utility Functions

This module contains shared utilities for reporting, metrics, and data handling.

Author: Capstone Team Einstein
Version: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime
from pathlib import Path
import hashlib
import io
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    logger.warning("ReportLab not available, PDF generation will be disabled")


def hash_customer_id(customer_id: str, salt: str = "einstein_2024") -> str:
    """
    Hash customer ID for privacy.
    
    Args:
        customer_id: Original customer ID
        salt: Salt for hashing
        
    Returns:
        Hashed customer ID
    """
    combined = f"{salt}:{customer_id}"
    return hashlib.sha256(combined.encode()).hexdigest()[:32].upper()


def anonymize_dataframe(df: pd.DataFrame, id_column: str = "CONS_NO") -> pd.DataFrame:
    """
    Anonymize DataFrame by hashing customer IDs.
    
    Args:
        df: DataFrame with customer data
        id_column: Name of ID column
        
    Returns:
        DataFrame with hashed IDs
    """
    df = df.copy()
    if id_column in df.columns:
        df[id_column] = df[id_column].apply(hash_customer_id)
    return df


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
    
    # Get indices of top k predictions
    top_indices = np.argsort(y_scores)[-top_k:]
    
    # Calculate precision
    precision = np.mean(y_true[top_indices])
    
    return float(precision)


def time_to_detection(y_true: np.ndarray, y_pred: np.ndarray, 
                      timestamps: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Calculate time-to-detection metrics.
    
    For labeled theft cases, measures how quickly the model detects theft
    from the start of the anomalous period.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        timestamps: Optional timestamps
        
    Returns:
        Dictionary with detection metrics
    """
    # Find true positives
    true_positives = (y_true == 1) & (y_pred == 1)
    false_negatives = (y_true == 1) & (y_pred == 0)
    
    results = {
        'total_theft_cases': int(np.sum(y_true)),
        'detected_cases': int(np.sum(true_positives)),
        'missed_cases': int(np.sum(false_negatives)),
        'detection_rate': float(np.sum(true_positives) / (np.sum(y_true) + 1e-10))
    }
    
    # If timestamps provided, calculate time metrics
    if timestamps is not None and len(timestamps) > 0:
        # Placeholder for sequential detection analysis
        results['avg_detection_delay'] = 0.0
    
    return results


def calculate_business_metrics(y_true: np.ndarray, y_scores: np.ndarray,
                               theft_threshold: float = 0.5,
                               avg_theft_value: float = 1000.0) -> Dict[str, float]:
    """
    Calculate business impact metrics.
    
    Args:
        y_true: True labels
        y_scores: Prediction probabilities
        theft_threshold: Threshold for positive prediction
        avg_theft_value: Average value of theft case in currency
        
    Returns:
        Dictionary with business metrics
    """
    y_pred = (y_scores >= theft_threshold).astype(int)
    
    # Confusion matrix elements
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    
    # Calculate metrics
    precision = tp / (tp + fp + 1e-10)
    recall = tp / (tp + fn + 1e-10)
    
    # Business impact
    revenue_recovered = tp * avg_theft_value
    investigation_cost = (tp + fp) * (avg_theft_value * 0.1)  # 10% of theft value
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
        'investigation_cost': float(investigation_cost),
        'missed_revenue': float(missed_revenue),
        'net_benefit': float(revenue_recovered - investigation_cost - missed_revenue)
    }


def format_consumption_for_chart(consumption: List[float], 
                                  dates: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Format consumption data for frontend charts.
    
    Args:
        consumption: List of consumption values
        dates: Optional list of date strings
        
    Returns:
        Dictionary formatted for Recharts
    """
    n = len(consumption)
    
    if dates is None:
        # Generate default dates
        dates = [f"Day {i+1}" for i in range(n)]
    
    chart_data = []
    for i, (date, value) in enumerate(zip(dates, consumption)):
        chart_data.append({
            'date': date,
            'consumption': float(value) if not np.isnan(value) else None,
            'index': i
        })
    
    return {
        'data': chart_data,
        'min': float(np.nanmin(consumption)) if len(consumption) > 0 else 0,
        'max': float(np.nanmax(consumption)) if len(consumption) > 0 else 100,
        'mean': float(np.nanmean(consumption)) if len(consumption) > 0 else 50
    }


class PDFReportGenerator:
    """
    Generate PDF reports for theft detection results.
    """
    
    def __init__(self):
        if not REPORTLAB_AVAILABLE:
            raise ImportError("ReportLab is required for PDF generation")
        
        self.styles = getSampleStyleSheet()
        self._add_custom_styles()
    
    def _add_custom_styles(self):
        """Add custom paragraph styles."""
        self.styles.add(ParagraphStyle(
            name='Title',
            parent=self.styles['Heading1'],
            fontSize=24,
            spaceAfter=30,
            textColor=colors.HexColor('#1a1a2e')
        ))
        
        self.styles.add(ParagraphStyle(
            name='SectionHeader',
            parent=self.styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            textColor=colors.HexColor('#16213e')
        ))
        
        self.styles.add(ParagraphStyle(
            name='BodyText',
            parent=self.styles['Normal'],
            fontSize=10,
            spaceAfter=8
        ))
    
    def generate_report(self, 
                        customer_id: str,
                        probability: float,
                        risk_level: str,
                        top_features: List[Tuple[str, float]],
                        consumption_data: List[float],
                        explanation: str = "") -> bytes:
        """
        Generate a PDF report for a customer assessment.
        
        Args:
            customer_id: Customer ID
            probability: Theft probability
            risk_level: Risk level (LOW/MEDIUM/HIGH)
            top_features: Top contributing features
            consumption_data: Consumption values
            explanation: Text explanation
            
        Returns:
            PDF as bytes
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4,
                               rightMargin=72, leftMargin=72,
                               topMargin=72, bottomMargin=72)
        
        story = []
        
        # Title
        story.append(Paragraph("⚡ Electricity Theft Detection Report", self.styles['Title']))
        story.append(Spacer(1, 12))
        
        # Report metadata
        report_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        story.append(Paragraph(f"<b>Generated:</b> {report_date}", self.styles['BodyText']))
        story.append(Paragraph(f"<b>Customer ID:</b> {customer_id}", self.styles['BodyText']))
        story.append(Spacer(1, 20))
        
        # Risk Assessment
        story.append(Paragraph("Risk Assessment", self.styles['SectionHeader']))
        
        # Risk level with color
        risk_colors = {
            'LOW': colors.green,
            'MEDIUM': colors.orange,
            'HIGH': colors.red
        }
        risk_color = risk_colors.get(risk_level, colors.gray)
        
        risk_table_data = [
            ['Theft Probability', f'{probability:.1%}'],
            ['Risk Level', risk_level],
        ]
        
        risk_table = Table(risk_table_data, colWidths=[2.5*inch, 2*inch])
        risk_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.white),
        ]))
        story.append(risk_table)
        story.append(Spacer(1, 20))
        
        # Top Contributing Features
        story.append(Paragraph("Top Contributing Features", self.styles['SectionHeader']))
        
        if top_features:
            feature_data = [['Feature', 'Contribution']]
            for feat, importance in top_features[:5]:
                direction = "↑" if importance > 0 else "↓"
                feature_data.append([feat, f"{direction} {abs(importance):.4f}"])
            
            feature_table = Table(feature_data, colWidths=[3*inch, 1.5*inch])
            feature_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a1a2e')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#dddddd')),
            ]))
            story.append(feature_table)
        else:
            story.append(Paragraph("No feature data available.", self.styles['BodyText']))
        
        story.append(Spacer(1, 20))
        
        # Explanation
        if explanation:
            story.append(Paragraph("Analysis Summary", self.styles['SectionHeader']))
            story.append(Paragraph(explanation, self.styles['BodyText']))
            story.append(Spacer(1, 20))
        
        # Consumption Statistics
        story.append(Paragraph("Consumption Statistics", self.styles['SectionHeader']))
        
        valid_consumption = [c for c in consumption_data if not np.isnan(c)]
        if valid_consumption:
            stats_data = [
                ['Mean Consumption', f'{np.mean(valid_consumption):.2f} kWh'],
                ['Std Deviation', f'{np.std(valid_consumption):.2f} kWh'],
                ['Min', f'{np.min(valid_consumption):.2f} kWh'],
                ['Max', f'{np.max(valid_consumption):.2f} kWh'],
                ['Zero/Missing Days', f'{len(consumption_data) - len(valid_consumption)}'],
            ]
            
            stats_table = Table(stats_data, colWidths=[2.5*inch, 2*inch])
            stats_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f0f0f0')),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.white),
            ]))
            story.append(stats_table)
        
        story.append(Spacer(1, 30))
        
        # Recommendation
        story.append(Paragraph("Recommended Action", self.styles['SectionHeader']))
        
        if risk_level == 'HIGH':
            recommendation = ("⚠️ <b>Immediate investigation recommended.</b> "
                            "Schedule field inspection within 48 hours. "
                            "Review meter installation and tampering indicators.")
        elif risk_level == 'MEDIUM':
            recommendation = ("📋 <b>Further review recommended.</b> "
                            "Add to monitoring queue. Request recent consumption photographs. "
                            "Compare with neighboring meters.")
        else:
            recommendation = ("✅ <b>No immediate action required.</b> "
                            "Continue standard monitoring. Customer appears normal.")
        
        story.append(Paragraph(recommendation, self.styles['BodyText']))
        
        # Footer
        story.append(Spacer(1, 40))
        story.append(Paragraph(
            "<i>This report was generated by Einstein Theft Detection System. "
            "All predictions are probabilistic and should be verified by field investigation.</i>",
            self.styles['BodyText']
        ))
        
        # Build PDF
        doc.build(story)
        
        return buffer.getvalue()


def generate_pdf_report(customer_id: str,
                        probability: float,
                        risk_level: str,
                        top_features: List[Tuple[str, float]],
                        consumption_data: List[float],
                        explanation: str = "") -> Optional[bytes]:
    """
    Convenience function to generate PDF report.
    
    Returns None if ReportLab is not available.
    """
    if not REPORTLAB_AVAILABLE:
        logger.warning("PDF generation not available (ReportLab not installed)")
        return None
    
    try:
        generator = PDFReportGenerator()
        return generator.generate_report(
            customer_id=customer_id,
            probability=probability,
            risk_level=risk_level,
            top_features=top_features,
            consumption_data=consumption_data,
            explanation=explanation
        )
    except Exception as e:
        logger.error(f"PDF generation failed: {e}")
        return None


if __name__ == "__main__":
    # Quick test
    print("Testing utility functions...")
    
    # Test hashing
    hashed = hash_customer_id("CUSTOMER_001")
    print(f"Hashed ID: {hashed}")
    
    # Test precision@k
    y_true = np.array([0, 0, 1, 0, 1, 1, 0, 1, 0, 0])
    y_scores = np.array([0.1, 0.2, 0.9, 0.3, 0.8, 0.7, 0.2, 0.85, 0.1, 0.3])
    
    p_at_10 = precision_at_k(y_true, y_scores, 0.1)
    p_at_50 = precision_at_k(y_true, y_scores, 0.5)
    print(f"Precision@10%: {p_at_10:.2f}")
    print(f"Precision@50%: {p_at_50:.2f}")
    
    # Test business metrics
    metrics = calculate_business_metrics(y_true, y_scores)
    print(f"Business metrics: {metrics}")
    
    # Test chart formatting
    consumption = [100.0, 95.0, np.nan, 105.0, 98.0]
    chart_data = format_consumption_for_chart(consumption)
    print(f"Chart data points: {len(chart_data['data'])}")
    
    # Test PDF generation
    if REPORTLAB_AVAILABLE:
        pdf = generate_pdf_report(
            customer_id="TEST_001",
            probability=0.75,
            risk_level="MEDIUM",
            top_features=[('sudden_drop_count', 0.35), ('zero_ratio', 0.28)],
            consumption_data=[100, 95, 0, 0, 85, 90],
            explanation="High number of sudden consumption drops detected."
        )
        print(f"PDF generated: {len(pdf)} bytes")
    
    print("\n✅ Utilities test passed!")
