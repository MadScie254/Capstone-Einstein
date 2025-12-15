#!/bin/bash
# =============================================================================
# Einstein - Electricity Theft Detection System
# Quick Run Script - End-to-end demo on synthetic data
# =============================================================================

set -e

echo "⚡ Einstein Theft Detection System - Quick Run Script"
echo "====================================================="

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Project root
PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${GREEN}📁 Project root: $PROJECT_ROOT${NC}"

# Step 1: Check Python environment
echo -e "\n${YELLOW}Step 1: Checking Python environment...${NC}"
python --version
pip --version

# Step 2: Install dependencies
echo -e "\n${YELLOW}Step 2: Installing dependencies...${NC}"
pip install -r requirements.txt --quiet

# Step 3: Create artifacts directory
echo -e "\n${YELLOW}Step 3: Creating artifacts directory...${NC}"
mkdir -p artifacts

# Step 4: Generate synthetic data
echo -e "\n${YELLOW}Step 4: Generating synthetic data...${NC}"
python -c "
import sys
sys.path.insert(0, 'src/backend')
from app.preprocess import create_synthetic_data

# Generate synthetic data
df = create_synthetic_data(n_samples=1000, theft_ratio=0.15, seed=42)
df.to_csv('data/sample_synth.csv', index=False)
print(f'✅ Generated {len(df)} synthetic samples')
print(f'   - Normal: {len(df[df[\"FLAG\"] == 0])}')
print(f'   - Theft: {len(df[df[\"FLAG\"] == 1])}')
"

# Step 5: Run preprocessing
echo -e "\n${YELLOW}Step 5: Running preprocessing pipeline...${NC}"
python -c "
import sys
sys.path.insert(0, 'src/backend')
import pandas as pd
from app.preprocess import FeaturePipeline

# Load data
df = pd.read_csv('data/sample_synth.csv')

# Run preprocessing  
pipeline = FeaturePipeline()
features = pipeline.fit_transform(df)

# Save artifacts
pipeline.save('artifacts/pipeline_v1.joblib')
features.to_parquet('artifacts/preprocessed.parquet')

print(f'✅ Preprocessing complete')
print(f'   - Features shape: {features.shape}')
print(f'   - Pipeline saved to artifacts/pipeline_v1.joblib')
"

# Step 6: Create dummy models (for demo)
echo -e "\n${YELLOW}Step 6: Creating dummy models...${NC}"
python -c "
import sys
sys.path.insert(0, 'src/backend')
from app.models import create_dummy_models

create_dummy_models('artifacts')
print('✅ Dummy models created in artifacts/')
"

# Step 7: Test scoring
echo -e "\n${YELLOW}Step 7: Testing scoring pipeline...${NC}"
python -c "
import sys
sys.path.insert(0, 'src/backend')
import pandas as pd
import numpy as np
from app.preprocess import FeaturePipeline
from app.models import TheftDetectionModels

# Load pipeline and models
pipeline = FeaturePipeline.load('artifacts/pipeline_v1.joblib')
models = TheftDetectionModels(artifacts_dir='artifacts')
models.load_models()

# Test on sample data
df = pd.read_csv('data/sample_synth.csv')
sample = df.head(5)

# Transform and predict
features = pipeline.transform(sample)
consumption_cols = [c for c in sample.columns if c not in ['CONS_NO', 'FLAG']]
consumption = sample[consumption_cols].values.astype(float)

results = models.predict_proba(features, consumption)

print('✅ Scoring test results:')
for i in range(5):
    actual = sample['FLAG'].iloc[i]
    pred = results['probability'][i]
    risk = results['risk_level'][i]
    print(f'   Sample {i+1}: Actual={actual}, Prob={pred:.2%}, Risk={risk}')
"

# Step 8: Start backend (optional)
echo -e "\n${YELLOW}Step 8: Starting backend server...${NC}"
echo -e "Run the following command to start the API:"
echo -e "${GREEN}cd src/backend && uvicorn app.main:app --reload --host 0.0.0.0 --port 8000${NC}"

# Step 9: Start Streamlit (optional)
echo -e "\n${YELLOW}Step 9: Start Streamlit demo...${NC}"
echo -e "Run the following command in another terminal:"
echo -e "${GREEN}cd streamlit_demo && streamlit run app.py${NC}"

echo -e "\n${GREEN}✅ Quick run complete!${NC}"
echo "====================================================="
echo "📊 API Docs: http://localhost:8000/docs"
echo "🎨 Dashboard: http://localhost:3000"  
echo "📱 Streamlit: http://localhost:8501"
