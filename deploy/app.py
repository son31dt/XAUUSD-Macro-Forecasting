"""
app.py — XAU/USD Prediction Interactive Dashboard
=================================================
Comparison of three intraday gold price prediction model scenarios:
  A1 : XGBoost Baseline (Endogenous/Technical features only)
  A2 : XGBoost + Macroeconomic Features
  B  : TimesFM Zero-Shot (Foundation Model)

How to run:
  streamlit run deploy/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# =============================================================================
# PAGE CONFIGURATION
# =============================================================================
st.set_page_config(
    page_title="XAU/USD Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Thesis Dashboard: XGBoost vs TimesFM Comparison on Intraday XAU/USD."
    }
)

# Dark mode via custom CSS
st.markdown("""
<style>
    /* Sidebar title */
    .sidebar-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: var(--text-color);
        margin-bottom: 0.5rem;
        letter-spacing: 0.5px;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 10px;
        padding: 14px;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: var(--secondary-background-color);
        border-radius: 8px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        color: var(--text-color);
        padding: 8px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #f0c040 !important;
        color: #0e1117 !important;
    }

    /* Caption text */
    .caption-text { color: var(--text-color); font-size: 0.85rem; opacity: 0.8; }

    /* Event badge */
    .event-badge {
        background: rgba(240, 192, 64, 0.15);
        border: 1px solid #f0c040;
        border-radius: 4px;
        color: #f0c040;
        display: inline-block;
        font-size: 0.75rem;
        padding: 1px 6px;
        margin-left: 6px;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# PATHS
# =============================================================================
ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
RESULTS   = ROOT / "results"
FIGURES   = ROOT / "reports" / "figures"

# =============================================================================
# DATA LOADING (CACHED)
# =============================================================================
@st.cache_data(show_spinner="Loading prediction data...")
def load_data():
    df_feat = pd.read_csv(PROCESSED / "dataset_featured.csv",
                          parse_dates=["Datetime"]).set_index("Datetime")
    df_a1 = pd.read_csv(RESULTS / "predictions_a1.csv",
                        parse_dates=["Datetime"]).set_index("Datetime")
    df_a2 = pd.read_csv(RESULTS / "predictions_a2.csv",
                        parse_dates=["Datetime"]).set_index("Datetime")
    df_b  = pd.read_csv(RESULTS / "predictions_b.csv",
                        parse_dates=["Datetime"]).set_index("Datetime")

    # Select prediction columns from each model
    df_a1 = df_a1[["Target_Close_Actual", "Target_Close_Pred_a1",
                   "Absolute_Error_a1", "Direction_Correct_a1"]]
    df_a2 = df_a2[["Target_Close_Pred_a2",
                   "Absolute_Error_a2", "Direction_Correct_a2"]]
    df_b  = df_b[["Target_Close_Pred_b",
                  "Absolute_Error_b",  "Direction_Correct_b"]]

    # Inner join on Datetime
    df = df_a1.join([df_a2, df_b], how="inner")

    # Add is_event column from dataset_featured
    dev_cols = [c for c in df_feat.columns if c.startswith("Dev_")]
    is_event = (df_feat[dev_cols] != 0.0).any(axis=1)
    df["is_event"] = is_event.reindex(df.index, fill_value=False)

    return df

df_full = load_data()

# =============================================================================
# SIDEBAR: CONTROL PANEL
# =============================================================================
with st.sidebar:
    st.markdown('<p class="sidebar-title">🎛️ Control Panel</p>', unsafe_allow_html=True)
    st.divider()

    # Date Range
    st.subheader("📅 Date Range")
    min_date = df_full.index.min().date()
    max_date = df_full.index.max().date()

    date_range = st.date_input(
        "Select date range:",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        label_visibility="collapsed"
    )

    if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_dt = pd.Timestamp(date_range[0])
        end_dt = pd.Timestamp(date_range[1]).replace(hour=23, minute=59, second=59)
    else:
        start_dt = pd.Timestamp(min_date)
        end_dt = pd.Timestamp(max_date).replace(hour=23, minute=59, second=59)

    st.divider()

    # Model selection
    st.subheader("🤖 Displayed Models")
    model_options = {
        "XGBoost Baseline (A1)": ("Target_Close_Pred_a1", "Absolute_Error_a1", "Direction_Correct_a1", "#4C72B0"),
        "XGBoost + Macro (A2)":  ("Target_Close_Pred_a2", "Absolute_Error_a2", "Direction_Correct_a2", "#DD8452"),
        "TimesFM Zero-Shot (B)": ("Target_Close_Pred_b",  "Absolute_Error_b",  "Direction_Correct_b",  "#55A868"),
    }
    selected_models = st.multiselect(
        "Select models:",
        options=list(model_options.keys()),
        default=list(model_options.keys()),
        label_visibility="collapsed"
    )

    st.divider()

    # Event filter
    st.subheader("📰 Macro News Filter")
    event_only = st.checkbox("Focus on News Release Hours Only", value=False)
    if event_only:
        st.info(f"Active mode: evaluating only **{df_full['is_event'].sum()} candles** during news releases.")

    st.divider()
    st.caption("📁 Data: 2025 Test Set (XAU/USD 15-minute)")

# =============================================================================
# DATA FILTERING
# =============================================================================
mask = (df_full.index >= start_dt) & (df_full.index <= end_dt)
df = df_full.loc[mask].copy()
if event_only:
    df = df[df["is_event"]].copy()

n_rows   = len(df)
n_events = int(df["is_event"].sum())

# =============================================================================
# MAIN HEADER
# =============================================================================
st.markdown("## 📈 Gold Price Prediction Dashboard (XAU/USD)")
col_h1, col_h2, col_h3 = st.columns(3)
col_h1.metric("Displayed Candles", f"{n_rows:,}", help="Number of 15-min candles in selected range")
col_h2.metric("News Release Hours", f"{n_events}", help="Candles where macroeconomic news was released")
col_h3.metric("Active Models", f"{len(selected_models)}", help="Number of models being compared")

st.divider()

# =============================================================================
# MAIN TABS
# =============================================================================
tab1, tab2, tab3 = st.tabs(["📊 Prediction Chart", "📐 Evaluation Metrics", "🧠 SHAP Interpretation"])

# ---------------------------------------------------------------------------
# TAB 1: PREDICTION CHART
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Actual vs Predicted XAU/USD Price Comparison")

    if n_rows == 0:
        st.warning("No data available for the selected date range.")
    else:
        fig = go.Figure()

        # Actual price (always displayed)
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["Target_Close_Actual"],
            mode="lines",
            name="Actual Price",
            line=dict(color="#f0c040", width=1.5),
        ))

        # Prediction per model
        for model_name in selected_models:
            pred_col, _, _, color = model_options[model_name]
            if pred_col in df.columns:
                fig.add_trace(go.Scatter(
                    x=df.index,
                    y=df[pred_col],
                    mode="lines",
                    name=model_name,
                    line=dict(color=color, width=1, dash="dot"),
                    opacity=0.85
                ))

        # Highlight news release hours (vertical vrect)
        event_times = df[df["is_event"]].index
        for et in event_times:
            fig.add_vrect(
                x0=et - pd.Timedelta(minutes=7),
                x1=et + pd.Timedelta(minutes=7),
                fillcolor="#f0c040",
                opacity=0.15,
                line_width=0,
                annotation_text="📰",
                annotation_position="top left",
                annotation_font_size=10
            )

        fig.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(showgrid=True, gridcolor="rgba(128, 128, 128, 0.2)", title="Datetime (UTC)"),
            yaxis=dict(showgrid=True, gridcolor="rgba(128, 128, 128, 0.2)", title="XAU/USD Price"),
            height=500,
            margin=dict(l=10, r=10, t=20, b=10),
            hovermode="x unified",
            modebar=dict(
                bgcolor="rgba(0,0,0,0)",
                color="rgba(128,128,128,0.7)",
                activecolor="rgba(128,128,128,1)"
            )
        )
        st.plotly_chart(fig, on_select="ignore", selection_mode="points", width="stretch")
        st.caption(f"🟡 Yellow zone = macroeconomic news release hour ({n_events} points). Dashed line = model predictions.")

# ---------------------------------------------------------------------------
# TAB 2: EVALUATION METRICS
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Evaluation Metrics (calculated from filtered data)")

    if n_rows == 0:
        st.warning("No data available for the selected date range.")
    else:
        def compute_metrics(df_subset, ae_col, dc_col, pred_col, actual_col="Target_Close_Actual"):
            """Calculate MAE, RMSE, MAPE, MDA from subset data."""
            if len(df_subset) == 0 or ae_col not in df_subset.columns:
                return None, None, None, None
            actual = df_subset[actual_col].values
            pred   = df_subset[pred_col].values
            mae  = df_subset[ae_col].mean()
            rmse = np.sqrt(np.mean((actual - pred) ** 2))
            with np.errstate(divide='ignore', invalid='ignore'):
                mape_arr = np.abs((actual - pred) / np.where(actual != 0, actual, np.nan))
            mape = float(np.nanmean(mape_arr) * 100)
            mda  = df_subset[dc_col].astype(int).mean() * 100
            return mae, rmse, mape, mda

        # Global (within selected range)
        global_data   = df
        event_data    = df[df["is_event"]]

        for model_name in selected_models:
            pred_col, ae_col, dc_col, color = model_options[model_name]
            if pred_col not in df.columns:
                continue

            mae_g, rmse_g, mape_g, mda_g = compute_metrics(global_data,    ae_col, dc_col, pred_col)
            mae_e, rmse_e, mape_e, mda_e = compute_metrics(event_data,      ae_col, dc_col, pred_col)

            st.markdown(f"#### 🤖 {model_name}")
            col_label, col1, col2, col3, col4 = st.columns([1.2, 1, 1, 1, 1])

            with col_label:
                st.markdown(
                    f"<div style='padding-top:32px;color:{color};font-weight:700;font-size:0.9rem;'>"
                    f"Global<br><span style='color:var(--text-color); opacity:0.6;'>({n_rows:,} candles)</span></div>",
                    unsafe_allow_html=True
                )
            col1.metric("MAE",  f"{mae_g:.4f} USD"  if mae_g  is not None else "—", help="Mean Absolute Error (absolute price)")
            col2.metric("RMSE", f"{rmse_g:.4f} USD" if rmse_g is not None else "—", help="Root Mean Squared Error")
            col3.metric("MAPE", f"{mape_g:.4f}%"    if mape_g is not None else "—", help="Mean Absolute Percentage Error")
            col4.metric("MDA",  f"{mda_g:.2f}%"     if mda_g  is not None else "—",
                        delta=f"{'↑' if mda_g and mda_g > 50 else '↓'} from 50%",
                        help="Mean Directional Accuracy (>50% = better than random guess)")

            if len(event_data) > 0:
                col_label2, col1e, col2e, col3e, col4e = st.columns([1.2, 1, 1, 1, 1])
                with col_label2:
                    st.markdown(
                        f"<div style='padding-top:32px;color:{color};font-weight:700;font-size:0.9rem;'>"
                        f"News Event 📰<br><span style='color:var(--text-color); opacity:0.6;'>({len(event_data)} candles)</span></div>",
                        unsafe_allow_html=True
                    )
                col1e.metric("MAE",  f"{mae_e:.4f} USD"  if mae_e  is not None else "—")
                col2e.metric("RMSE", f"{rmse_e:.4f} USD" if rmse_e is not None else "—")
                col3e.metric("MAPE", f"{mape_e:.4f}%"    if mape_e is not None else "—")
                col4e.metric("MDA",  f"{mda_e:.2f}%"     if mda_e  is not None else "—",
                             delta=f"{'↑' if mda_e and mda_e > 50 else '↓'} from 50%")
            else:
                st.caption("_No event candles found in the selected range._")

            st.divider()

# ---------------------------------------------------------------------------
# TAB 3: SHAP INTERPRETATION
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Feature Interpretation with SHAP (SHapley Additive exPlanations)")
    st.markdown(
        "SHAP explains the contribution of each feature to the model's decision. "
        "The Y-axis sorts features from **most influential** (top) to least influential (bottom). "
        "Red color = high feature value; Blue color = low feature value."
    )
    st.divider()

    shap_global_path = FIGURES / "shap_summary.png"
    shap_event_path  = FIGURES / "shap_event_summary.png"

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.markdown("##### 🌍 Global SHAP (Entire 2025 Test Set)")
        st.markdown(
            "<p class='caption-text'>Technical features (OHLCV & lags) dominate because 99.5% of the data "
            "does not contain macroeconomic signals.</p>",
            unsafe_allow_html=True
        )
        if shap_global_path.exists():
            st.image(str(shap_global_path), use_container_width=True)
        else:
            st.warning(f"File not found: `{shap_global_path.name}`. Run `src/shap_analysis.py` first.")

    with col_s2:
        st.markdown("##### 📰 Event-Driven SHAP (102 News Release Candles)")
        st.markdown(
            "<p class='caption-text'>During news releases, <code>Dev_...</code> features jump to the top rankings "
            "— proving that XGBoost successfully captured the macroeconomic signals.</p>",
            unsafe_allow_html=True
        )
        if shap_event_path.exists():
            st.image(str(shap_event_path), use_container_width=True)
        else:
            st.warning(f"File not found: `{shap_event_path.name}`. Run `src/shap_analysis.py` first.")

    st.divider()
    st.info(
        "💡 **Key Insight**: The comparison between these two plots proves the research hypothesis — "
        "macroeconomic features (`Dev_NFP`, `Dev_CPI`, `Dev_FedRate`, etc.) provide significant predictive *alpha* "
        "**only during news release moments**, not overall. "
        "This explains why the MDA of Scenario A2 surged by **+8.82%** compared to the baseline during event hours."
    )

# =============================================================================
# FOOTER
# =============================================================================
st.divider()
st.markdown(
    "<div style='text-align:center; color:var(--text-color); opacity:0.6; font-size:0.8rem;'>"
    "📘 Thesis Dashboard — Intraday XAU/USD Price Prediction | XGBoost vs TimesFM | 2025 Test Set"
    "</div>",
    unsafe_allow_html=True
)
