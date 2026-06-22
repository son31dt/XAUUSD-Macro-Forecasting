# PROJECT CONTEXT: XAU/USD INTRADAY FORECASTING (THESIS)

## 1. Project Overview
**Title:** Pengaruh Injeksi Fitur Makroekonomi pada XGBoost Dibandingkan Kemampuan Zero-Shot TimesFM dalam Peramalan Intraday XAU/USD.
**Objective:** Compare the forecasting performance (t+1, 15-minute resolution) of XGBoost (Endogenous only), XGBoost (Endogenous + Macroeconomic Exogenous), and TimesFM (Zero-Shot) for XAU/USD.

## 2. Dataset Requirements (Stored in `data/raw/`)
- **Data A (Price):** XAU/USD OHLCV, 15-minute timeframe (2021-2025). Timezone: UTC.
- **Data B (Macroeconomic):** US High-Impact News. Initial Timezone: WIB (GMT+7). Must be converted to UTC. Only 10 indicators are used (CPI, PPI, GDP, FedRate, NFP, Jobless, ADP, Earnings, ISM, Retail).

## 3. Methodology & Workflow
Please act as an Expert Data Scientist. We will execute this in phases:
- **Phase 1: Preprocessing & Time-Alignment.** - Macro Data: Convert WIB to UTC. Map event names using strictly defined Regex into 10 standard categories to avoid overlap (e.g., distinctly separate 'ADP Nonfarm' from 'Nonfarm Payrolls', and ONLY use 'Fed Funds Rate' or 'FOMC' for the 'FedRate' category, ignoring regional Fed surveys). Calculate Deviation = Actual - Forecast.
  - Alignment: Merge 15m OHLCV data with Macro data. Pivot the Macro data so each of the 10 indicators becomes its own Deviation column (e.g., `Dev_NFP`, `Dev_CPI`). If a news release falls during the XAU/USD daily market break (21:00-23:00 UTC), forward-fill the deviation to the first open candle (23:00 UTC). Fill non-release timestamps with `0.0`.
- **Phase 2: Feature Engineering.** Create lag features (t-1, t-2, t-3) for OHLCV.
- **Phase 3: Data Splitting.** Sequential split (NO RANDOM SPLIT). Train: Jan 2021 - Dec 2024. Test: Jan 2025 - Dec 2025.
- **Phase 4: Modeling Skenario A1 (Baseline XGBoost).** Train XGBoost using only OHLCV + Lags.
- **Phase 5: Modeling Skenario A2 (Injected XGBoost).** Train XGBoost using OHLCV + Lags + Macro Deviation features. Optimize using GridSearchCV.
- **Phase 6: Modeling Skenario B (TimesFM).** Use HuggingFace `timesfm-1.0-200m`. Implement Walk-Forward testing on the 2025 Test Set (Zero-Shot inference).
- **Phase 7: Evaluation & Explainability.** Calculate MAE, RMSE, MAPE, and MDA (Mean Directional Accuracy). Generate SHAP Summary Plot for Model A2.
- **Phase 8: Deployment.** Build a Streamlit dashboard to show actual vs predicted charts and evaluation metrics.

## 4. Coding Guidelines
- Write modular, clean, and well-commented Python code (PEP 8).
- Use `pandas`, `xgboost`, `scikit-learn`, `shap`, and `timesfm` libraries.
- Always handle temporal data carefully to avoid Data Leakage.
- When generating code, provide step-by-step reasoning in Indonesian.