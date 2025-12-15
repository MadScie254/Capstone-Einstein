# Model Card: Non-Technical Loss Detection System

## Model Details

| Field | Value |
|-------|-------|
| Model Name | Einstein NTL Detection Ensemble |
| Version | 1.0.0 |
| Date | December 2024 |
| Type | Binary Classification (Theft/Non-Theft) |
| Framework | XGBoost + Isolation Forest Ensemble |
| Owner | Capstone Team Einstein |

## Intended Use

### Primary Use Case

Detection of non-technical losses (NTL) in electricity distribution networks, including:
- Meter tampering and bypass
- Consumption underreporting
- Billing manipulation
- Unauthorized connections

### Intended Users

- Utility revenue assurance teams
- Field inspection coordinators
- Regulatory compliance officers

### Out-of-Scope Uses

This model should NOT be used for:
- Automatic customer disconnection
- Penalty billing without investigation
- Credit scoring or lending decisions
- Any purpose outside utility NTL detection

## Training Data

### Dataset Description

- **Source:** Smart meter consumption data
- **Period:** 26 days of daily consumption readings
- **Format:** Wide format (customers as rows, days as columns)
- **Labels:** Binary (0 = normal, 1 = confirmed theft)

### Data Preprocessing

- Missing values: Seasonal median imputation
- Zero consumption: Flagged as feature, not removed
- Outliers: Winsorized at 99th percentile

### Class Distribution

| Class | Count | Percentage |
|-------|-------|------------|
| Normal (0) | ~85% | Majority |
| Theft (1) | ~15% | Minority |

## Model Architecture

### Ensemble Components

1. **XGBoost Classifier**
   - 200 estimators, max_depth=6
   - Handles class imbalance via scale_pos_weight
   - Early stopping on validation AUPRC

2. **Isolation Forest (per cluster)**
   - 100 estimators per cluster
   - Contamination set to cluster theft rate
   - Cluster-specific anomaly detection

### Feature Engineering

35+ engineered features including:
- Consumption statistics (mean, std, CV)
- Anomaly indicators (zero_ratio, sudden_drop_count)
- Rolling statistics (7-day, 14-day windows)
- Temporal patterns (autocorrelation, trend)

## Evaluation Results

### Performance Metrics

| Metric | Value |
|--------|-------|
| AUPRC | See artifacts/evaluation_metrics.json |
| Precision@1% | Target: >70% |
| Precision@5% | Target: >50% |

### Fairness Considerations

Performance should be monitored across:
- Geographic regions
- Customer segments (residential, commercial)
- Consumption levels

## Limitations

### Known Limitations

1. **Cold Start:** Customers with <30 days of data cannot be reliably scored
2. **Subtle Theft:** Gradual, low-magnitude theft may evade detection
3. **Data Quality:** Performance degrades with poor meter data quality
4. **Seasonal Patterns:** Limited by training data seasonality

### Failure Modes

| Scenario | Expected Behavior |
|----------|-------------------|
| New meter installation | May flag as anomaly; exclude from scoring for 30 days |
| Seasonal vacation | May trigger false positive; incorporate calendar features |
| Legitimate high variance | Higher false positive risk; use cluster-specific thresholds |

## Ethical Considerations

### Privacy

- Customer IDs are hashed before model training
- Raw consumption data is not stored in logs
- Aggregated metrics only in dashboards

### Bias Mitigation

- Cluster-specific thresholds normalize across customer types
- Regular audits of flag distribution by segment
- Human review required before enforcement action

### Human-in-the-Loop Requirement

This model produces decision support, not decisions. All flagged cases require:
1. Supervisor review
2. Field investigation
3. Evidence collection
4. Human authorization before action

## Monitoring and Maintenance

### Drift Detection

- Feature distribution drift (PSI) monitored monthly
- Score distribution drift tracked weekly
- Threshold recalibration quarterly

### Retraining Criteria

Retrain model when:
- PSI > 0.25 on any critical feature
- Precision@5% drops below 40%
- Significant external shock (tariff change, pandemic)

## Contact

For questions about this model, contact the Data Science team.

---

*This model card follows the format proposed by Mitchell et al. (2019) "Model Cards for Model Reporting"*
