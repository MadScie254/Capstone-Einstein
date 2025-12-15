"""
Einstein - Electricity Theft Detection System
Utility Functions - Reporting

Author: Capstone Team Einstein
"""

from typing import List, Tuple, Optional, Any
import numpy as np
from datetime import datetime

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
    from reportlab.lib.styles import getSampleStyleSheet
    import io
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False


def generate_theft_report(
    customer_id: str,
    probability: float,
    risk_level: str,
    top_features: List[Tuple[str, float]],
    consumption_data: List[float],
    explanation: str = ""
) -> Optional[bytes]:
    """
    Generate a PDF report for theft detection results.
    
    Args:
        customer_id: Customer identifier
        probability: Theft probability (0-1)
        risk_level: LOW/MEDIUM/HIGH
        top_features: List of (feature_name, importance) tuples
        consumption_data: Daily consumption values
        explanation: Human-readable explanation
        
    Returns:
        PDF bytes or None if reportlab not available
    """
    if not REPORTLAB_AVAILABLE:
        return None
    
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    story.append(Paragraph("⚡ Electricity Theft Detection Report", styles['Title']))
    story.append(Spacer(1, 20))
    
    # Metadata
    story.append(Paragraph(f"<b>Date:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Paragraph(f"<b>Customer ID:</b> {customer_id}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Risk Assessment
    story.append(Paragraph("Risk Assessment", styles['Heading2']))
    story.append(Paragraph(f"<b>Theft Probability:</b> {probability:.1%}", styles['Normal']))
    story.append(Paragraph(f"<b>Risk Level:</b> {risk_level}", styles['Normal']))
    story.append(Spacer(1, 20))
    
    # Feature Importance
    if top_features:
        story.append(Paragraph("Top Contributing Factors", styles['Heading2']))
        table_data = [['Feature', 'Importance']]
        for feat, imp in top_features[:5]:
            table_data.append([feat, f'{imp:.4f}'])
        
        table = Table(table_data, colWidths=[250, 100])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ]))
        story.append(table)
    
    story.append(Spacer(1, 20))
    
    # Explanation
    if explanation:
        story.append(Paragraph("Analysis Summary", styles['Heading2']))
        story.append(Paragraph(explanation, styles['Normal']))
    
    doc.build(story)
    return buffer.getvalue()
