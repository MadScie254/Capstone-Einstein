# Non-Technical Loss Detection System

## System Overview

This repository contains a production-grade machine learning system for detecting non-technical losses (NTL) in electricity distribution networks. The system is designed for deployment at utility companies and regulatory agencies.

### Capabilities

- Customer consumption pattern analysis
- Theft probability scoring with explainability
- Cluster-based behavioral segmentation
- Investigation prioritization and workflow support
- Audit logging and model governance

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Data Ingestion Layer                         │
│   Smart Meter Data → Validation → Temporal Alignment            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                 Feature Engineering Pipeline                     │
│   35+ Features: Rolling Stats, Anomaly Indicators, Patterns     │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Detection Model Ensemble                       │
│   XGBoost (supervised) + Isolation Forest (unsupervised)        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Operational Interface                         │
│   REST API │ Operations Dashboard │ Investigation Workflow       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- Conda or virtual environment manager

### Installation

```bash
# Clone repository
git clone <repository-url>
cd Capstone-Einstein

# Create environment
conda create -n ntl_detection python=3.10
conda activate ntl_detection

# Install dependencies
pip install -r requirements.txt
```

### Generate Artifacts

```bash
# Run preprocessing and model training
python scripts/run_pipeline.py
```

### Start Services

```bash
# Backend API
cd src/backend
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Streamlit Demo (separate terminal)
cd streamlit_demo
streamlit run app.py
```

### Access Points

| Service | URL | Description |
|---------|-----|-------------|
| API Documentation | http://localhost:8000/docs | OpenAPI Swagger UI |
| Streamlit Demo | http://localhost:8501 | Stakeholder demonstration |
| Health Check | http://localhost:8000/health | System status |

---

## Project Structure

```
Capstone-Einstein/
├── config/
│   └── config.yaml              # System configuration
├── data/
│   └── datasetsmall.csv         # Input dataset
├── docs/
│   ├── model_card.md            # Model governance
│   └── datasheet.md             # Dataset documentation
├── notebooks/
│   ├── 00_Project_Context.ipynb # Executive overview
│   ├── 01_EDA_Preprocessing.ipynb
│   ├── 02_Clustering.ipynb
│   ├── 03_Model_Training.ipynb
│   └── 04_Evaluation.ipynb
├── src/
│   ├── backend/                 # FastAPI application
│   │   └── app/
│   │       ├── main.py
│   │       ├── models.py
│   │       ├── preprocess.py
│   │       └── utils.py
│   └── frontend/                # React dashboard
├── streamlit_demo/              # Demonstration interface
├── artifacts/                   # Generated models (gitignored)
├── tests/                       # Test suite
└── requirements.txt
```

---

## API Reference

### Score Endpoint

```bash
POST /score
Content-Type: application/json

{
  "consumption": [100.5, 98.2, 105.3, ...],
  "customer_id": "CUST_001"
}
```

Response:

```json
{
  "probability": 0.73,
  "risk_level": "HIGH",
  "customer_id": "CUST_001",
  "confidence": "MEDIUM",
  "model_version": "1.0.0"
}
```

### Batch Scoring

```bash
POST /score/batch
```

### Explanation

```bash
POST /explain
```

---

## Configuration

System behavior is controlled via `config/config.yaml`:

```yaml
thresholds:
  high_risk: 0.80
  medium_risk: 0.50
  operational: 0.65

monitoring:
  psi_critical: 0.25
```

---

## Deployment

### Docker

```bash
# Build images
docker-compose -f infra/docker-compose.yml build

# Start services
docker-compose -f infra/docker-compose.yml up -d
```

### Cloud Platforms

| Platform | Backend | Frontend |
|----------|---------|----------|
| Railway | Supported | - |
| Render | Supported | - |
| Vercel | - | Supported |

---

## Governance

### Model Documentation

- **Model Card:** `docs/model_card.md` - Purpose, limitations, ethical considerations
- **Datasheet:** `docs/datasheet.md` - Dataset provenance and appropriate use

### Audit Requirements

All predictions are logged with:
- Model version
- Threshold applied
- Timestamp
- Customer identifier (hashed)

### Human-in-the-Loop

This system produces decision support only. All flagged cases require:
1. Supervisor review
2. Field investigation
3. Evidence collection
4. Human authorization before action

---

## Testing

```bash
# Run test suite
pytest tests/ -v

# Coverage report
pytest tests/ --cov=src/backend --cov-report=html
```

---

## Monitoring

### Drift Detection

- Population Stability Index (PSI) calculated weekly
- Alert threshold: PSI > 0.25

### Retraining Triggers

- PSI exceeds critical threshold
- Precision@5% drops below 40%
- Quarterly scheduled review

---

## Support

For technical questions, contact the Data Science team.

---

## License

For authorized utility and research use only. Commercial deployment requires explicit authorization.

---

**Version:** 1.0.0  
**Last Updated:** December 2024
