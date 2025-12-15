"""
Einstein NTL Detection System - Streamlit Demo
Production-grade demonstration interface for stakeholder presentations

Author: Capstone Team Einstein
Version: 1.0.0
"""

import sys
from pathlib import Path
import json
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Configuration
st.set_page_config(
    page_title="NTL Detection System",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional styling - minimal, clean, serious
st.markdown("""
<style>
    /* Professional dark theme */
    .main {
        background-color: #0e1117;
    }
    
    /* Clean header styling */
    .stTitle {
        font-weight: 600;
        color: #ffffff;
    }
    
    /* Metric cards */
    div[data-testid="metric-container"] {
        background-color: #1c1f26;
        border: 1px solid #2d3139;
        border-radius: 8px;
        padding: 16px;
    }
    
    /* Remove excessive styling */
    .stAlert {
        border-radius: 4px;
    }
    
    /* Professional button styling */
    .stButton > button {
        background-color: #1a56db;
        border: none;
        border-radius: 4px;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        background-color: #1e429f;
    }
    
    /* Disclaimer styling */
    .disclaimer {
        background-color: #1c1f26;
        border-left: 3px solid #fbbf24;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 16px 0;
    }
    
    /* Risk level indicators */
    .risk-high { color: #ef4444; font-weight: 600; }
    .risk-medium { color: #f59e0b; font-weight: 600; }
    .risk-low { color: #10b981; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / 'artifacts'
sys.path.insert(0, str(PROJECT_ROOT / 'src' / 'backend'))

# =============================================================================
# Helper Functions
# =============================================================================

def load_models():
    """Load model artifacts."""
    try:
        from app.preprocess import FeaturePipeline
        from app.models import TheftDetectionModels
        
        pipeline = None
        if (ARTIFACTS_DIR / 'pipeline_v1.joblib').exists():
            pipeline = FeaturePipeline.load(str(ARTIFACTS_DIR / 'pipeline_v1.joblib'))
        
        models = TheftDetectionModels(artifacts_dir=str(ARTIFACTS_DIR))
        models.load_models()
        
        return pipeline, models, True
    except Exception as e:
        st.warning(f"Models not loaded: {str(e)[:100]}")
        return None, None, False


def create_consumption_chart(consumption_data, anomaly_threshold=None):
    """Create professional consumption time series chart."""
    days = list(range(1, len(consumption_data) + 1))
    
    fig = go.Figure()
    
    # Main consumption line
    fig.add_trace(go.Scatter(
        x=days,
        y=consumption_data,
        mode='lines+markers',
        name='Daily Consumption',
        line=dict(color='#3b82f6', width=2),
        marker=dict(size=6)
    ))
    
    # Anomaly threshold line
    if anomaly_threshold:
        fig.add_hline(
            y=anomaly_threshold,
            line_dash="dash",
            line_color="#ef4444",
            annotation_text="Anomaly Threshold"
        )
    
    # Mean reference line
    mean_val = np.nanmean(consumption_data)
    fig.add_hline(
        y=mean_val,
        line_dash="dot",
        line_color="#6b7280",
        annotation_text=f"Mean: {mean_val:.1f}"
    )
    
    fig.update_layout(
        title="Consumption Pattern Analysis",
        xaxis_title="Day",
        yaxis_title="Consumption (kWh)",
        template="plotly_dark",
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    
    return fig


def create_feature_importance_chart(features):
    """Create horizontal bar chart for feature importance."""
    if not features:
        return None
    
    df = pd.DataFrame(features)
    df = df.sort_values('importance', ascending=True).tail(10)
    
    colors = ['#ef4444' if x > 0 else '#10b981' for x in df['importance']]
    
    fig = go.Figure(go.Bar(
        x=df['importance'],
        y=df['feature'],
        orientation='h',
        marker_color=colors
    ))
    
    fig.update_layout(
        title="Top Contributing Factors",
        xaxis_title="Importance",
        template="plotly_dark",
        height=300,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig


def get_risk_level(probability):
    """Determine risk level from probability."""
    if probability >= 0.8:
        return "HIGH", "#ef4444"
    elif probability >= 0.5:
        return "MEDIUM", "#f59e0b"
    else:
        return "LOW", "#10b981"


# =============================================================================
# Main Application
# =============================================================================

def main():
    # Header
    st.title("Non-Technical Loss Detection System")
    st.markdown("*Decision support tool for electricity theft investigation prioritization*")
    
    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        <strong>Notice:</strong> This system provides probabilistic assessments for investigation prioritization. 
        All flagged cases require human review and field verification before any action is taken.
    </div>
    """, unsafe_allow_html=True)
    
    # Load models
    pipeline, models, models_loaded = load_models()
    
    # Sidebar - System Status
    with st.sidebar:
        st.header("System Status")
        
        status_color = "🟢" if models_loaded else "🟡"
        st.markdown(f"{status_color} **Model Status:** {'Operational' if models_loaded else 'Demo Mode'}")
        
        if (ARTIFACTS_DIR / 'metadata.json').exists():
            with open(ARTIFACTS_DIR / 'metadata.json') as f:
                metadata = json.load(f)
            st.markdown(f"**Version:** {metadata.get('version', 'N/A')}")
        
        st.divider()
        st.header("Input Options")
        input_mode = st.radio(
            "Select input method:",
            ["Upload CSV", "Manual Entry", "Demo Data"],
            label_visibility="collapsed"
        )
    
    # Main content area
    col_input, col_results = st.columns([1, 1])
    
    with col_input:
        st.subheader("Consumption Data Input")
        
        consumption_data = None
        customer_id = st.text_input("Customer ID (optional)", placeholder="Enter customer identifier")
        
        if input_mode == "Upload CSV":
            uploaded_file = st.file_uploader(
                "Upload consumption data (CSV)",
                type=['csv'],
                help="CSV file with daily consumption values"
            )
            
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    if len(df.columns) > 2:
                        # Wide format - take first row
                        numeric_cols = df.select_dtypes(include=[np.number]).columns
                        consumption_data = df[numeric_cols].iloc[0].values.astype(float)
                    else:
                        # Long format
                        consumption_data = df.iloc[:, -1].values.astype(float)
                    st.success(f"Loaded {len(consumption_data)} days of consumption data")
                except Exception as e:
                    st.error(f"Error reading file: {str(e)}")
        
        elif input_mode == "Manual Entry":
            manual_input = st.text_area(
                "Enter daily consumption values (comma-separated)",
                placeholder="100.5, 98.2, 105.3, 95.7, 92.1, ...",
                height=100
            )
            
            if manual_input:
                try:
                    consumption_data = np.array([float(x.strip()) for x in manual_input.split(',')])
                except ValueError:
                    st.error("Invalid format. Use comma-separated numbers.")
        
        else:  # Demo Data
            st.info("Using demonstration data with suspicious patterns")
            # Demo data with theft indicators
            consumption_data = np.array([
                120.5, 118.2, 125.3, 115.7, 122.1, 118.5, 130.2,
                127.8, 45.2, 38.3, 15.0, 12.8, 8.1, 5.5,
                6.2, 15.0, 88.5, 95.2, 98.7, 94.2, 96.5, 100.1,
                105.5, 99.8, 97.2, 101.1
            ])
        
        # Analysis button
        analyze_clicked = st.button(
            "Analyze Consumption Pattern",
            type="primary",
            disabled=consumption_data is None,
            use_container_width=True
        )
    
    with col_results:
        st.subheader("Analysis Results")
        
        if analyze_clicked and consumption_data is not None:
            with st.spinner("Analyzing consumption pattern..."):
                # Perform analysis
                if pipeline and models:
                    try:
                        # Create dataframe for pipeline
                        cols = [f'day_{i+1}' for i in range(len(consumption_data))]
                        df_input = pd.DataFrame([consumption_data], columns=cols)
                        df_input['CONS_NO'] = customer_id or 'DEMO'
                        df_input['FLAG'] = 0
                        
                        features = pipeline.transform(df_input)
                        results = models.predict_proba(features, consumption_data.reshape(1, -1))
                        
                        probability = results['probability'][0]
                        risk_level, risk_color = get_risk_level(probability)
                        
                    except Exception as e:
                        st.error(f"Analysis error: {str(e)}")
                        probability = 0.65  # Fallback demo
                        risk_level, risk_color = get_risk_level(probability)
                else:
                    # Demo mode
                    probability = 0.78
                    risk_level, risk_color = get_risk_level(probability)
                
                # Display results
                col_prob, col_risk = st.columns(2)
                
                with col_prob:
                    st.metric(
                        label="Theft Probability",
                        value=f"{probability:.1%}"
                    )
                
                with col_risk:
                    st.markdown(f"""
                    <div style="background-color: #1c1f26; padding: 16px; border-radius: 8px; text-align: center;">
                        <p style="color: #9ca3af; margin: 0; font-size: 14px;">Risk Level</p>
                        <p style="color: {risk_color}; font-size: 24px; font-weight: 600; margin: 0;">{risk_level}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Consumption chart
                mean_consumption = np.nanmean(consumption_data)
                anomaly_threshold = mean_consumption * 0.4
                fig = create_consumption_chart(consumption_data, anomaly_threshold)
                st.plotly_chart(fig, use_container_width=True)
                
                # Key observations
                st.subheader("Key Observations")
                
                zero_days = np.sum(consumption_data < 1)
                sudden_drops = np.sum(np.diff(consumption_data) < -mean_consumption * 0.5)
                
                observations = []
                if zero_days > 0:
                    observations.append(f"- {zero_days} days with near-zero consumption detected")
                if sudden_drops > 0:
                    observations.append(f"- {sudden_drops} sudden consumption drops (>50%) observed")
                if np.std(consumption_data) / mean_consumption > 0.5:
                    observations.append("- High consumption variability compared to baseline")
                
                if observations:
                    for obs in observations:
                        st.markdown(obs)
                else:
                    st.markdown("- No significant anomalies detected")
                
                # Recommendations
                st.subheader("Recommended Actions")
                
                if risk_level == "HIGH":
                    st.warning("""
                    **Priority Investigation Recommended**
                    - Schedule field inspection within 7 days
                    - Review meter installation and seals
                    - Check for physical bypass indicators
                    """)
                elif risk_level == "MEDIUM":
                    st.info("""
                    **Standard Review Recommended**
                    - Add to investigation queue
                    - Monitor consumption for next 30 days
                    - Review historical billing patterns
                    """)
                else:
                    st.success("""
                    **No Immediate Action Required**
                    - Continue standard monitoring
                    - No evidence of non-technical loss
                    """)
        
        else:
            st.info("Select an input method and provide consumption data to begin analysis.")
    
    # Footer
    st.divider()
    col_f1, col_f2, col_f3 = st.columns(3)
    
    with col_f1:
        st.caption("NTL Detection System v1.0.0")
    with col_f2:
        st.caption(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d')}")
    with col_f3:
        st.caption("For authorized utility personnel only")


if __name__ == "__main__":
    main()
