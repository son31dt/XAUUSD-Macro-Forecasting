# XAU/USD Intraday Forecasting: Macroeconomic Features vs Foundation Models

## Overview
This repository contains a quantitative data science research project comparing the performance of **XGBoost** (with and without macroeconomic feature injection) against **TimesFM** (Google's Zero-Shot Foundation Model) for intraday gold price prediction (XAU/USD, 15-minute intervals).

## Key Findings
- **Macroeconomic Alpha**: Injecting macroeconomic data (e.g., NFP, CPI, Fed Funds Rate) significantly improves model performance during high-volatility news releases.
- **Event-Driven Surge**: During news release hours, the Mean Directional Accuracy (MDA) of the XGBoost Macro model (Scenario A2) surged by **+8.82%** compared to the technical-only baseline.
- **TimesFM Limitations**: While TimesFM zero-shot inference is competitive in predicting the price direction, it struggles to adapt to absolute price shifts dynamically without fine-tuning, resulting in higher Mean Absolute Error (MAE).

## Repository Structure
```text
.
├── data/
│   ├── processed/          # Cleaned datasets and feature engineering outputs
│   └── raw/                # Raw historical data (ignored by git)
├── deploy/
│   └── app.py              # Cleaned Streamlit dashboard for deployment
├── reports/
│   └── figures/            # High-resolution charts and SHAP interpretation plots
├── results/                # Prediction outputs from all model scenarios
├── src/                    # Source code for data pipelines and modeling
├── requirements.txt        # Minimal dashboard dependencies
└── README.md
```

## How to Run the Dashboard Locally

1. Clone this repository to your local machine.
2. Install the required dependencies using pip:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```bash
   streamlit run deploy/app.py
   ```
4. Open your browser and navigate to `http://localhost:8501` to view the interactive dashboard.
