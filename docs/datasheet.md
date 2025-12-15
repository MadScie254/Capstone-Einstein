# Datasheet: Smart Meter Consumption Dataset

## Motivation

### Purpose

This dataset was created to develop and evaluate machine learning models for detecting non-technical losses (electricity theft) in utility distribution networks.

### Creators

Capstone Team Einstein for academic research and utility analytics demonstration.

### Funding

Academic research project; no external funding or commercial interests.

## Composition

### Instance Description

Each instance represents one electricity customer (meter) with:
- Daily consumption readings over 26 consecutive days
- Customer identifier (hashed for privacy)
- Binary label indicating confirmed theft status

### Instance Count

Approximately 1,000+ customer records in the small dataset.

### Data Format

| Column Type | Description | Example |
|-------------|-------------|---------|
| Date columns | Daily consumption in kWh | 142.5 |
| CONS_NO | Hashed customer identifier | abc123def456 |
| FLAG | Theft label (0=normal, 1=theft) | 0 or 1 |

### Missing Data

- Approximately 5-15% of readings may be missing
- Missing data due to meter communication failures
- Represented as empty cells or NaN values

### Confidentiality

- Customer identifiers are cryptographically hashed
- No personally identifiable information (PII) included
- Geographic identifiers removed

## Collection Process

### Collection Mechanism

Data collected from Advanced Metering Infrastructure (AMI) systems via:
- Automated meter reading (AMR)
- Power line communication (PLC)
- RF mesh networks

### Timeframe

Approximately one month of consecutive daily readings.

### Labeling

Labels derived from:
- Field investigation outcomes
- Meter tampering evidence
- Billing audit confirmations

### Labeling Limitations

- Selection bias: Only inspected customers are labeled
- Temporal mismatch: Theft may occur before detection
- Confirmation bias: Inspectors may find expected results

## Preprocessing

### Applied Preprocessing

During data collection:
- Consumption values rounded to one decimal place
- Extreme outliers retained for analysis
- Time zone standardized

### Not Yet Applied

The following should be applied during model training:
- Missing value imputation
- Feature normalization
- Outlier treatment

## Uses

### Intended Uses

- Academic research on NTL detection
- Utility analytics system development
- Machine learning model training and evaluation

### Inappropriate Uses

- Direct customer billing without verification
- Credit scoring or financial decisions
- Any surveillance purposes

## Distribution

### Access

Dataset distributed through academic channels with appropriate data use agreements.

### License

For research and educational purposes only. Commercial use requires explicit authorization.

## Maintenance

### Updates

Dataset is static snapshot; no regular updates planned.

### Point of Contact

Capstone Team Einstein - academic project maintainers.

---

*This datasheet follows the format proposed by Gebru et al. (2018) "Datasheets for Datasets"*
