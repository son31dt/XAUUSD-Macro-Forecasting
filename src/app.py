"""
app.py — Dashboard Interaktif XAU/USD Prediction
=================================================
Perbandingan tiga skenario model prediksi harga emas intraday:
  A1 : XGBoost Baseline (Endogen/Teknikal saja)
  A2 : XGBoost + Fitur Makroekonomi
  B  : TimesFM Zero-Shot (Foundation Model)

Cara menjalankan:
  streamlit run src/app.py

Kebutuhan:
  pip install streamlit plotly pandas numpy
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

# =============================================================================
# KONFIGURASI HALAMAN
# =============================================================================
st.set_page_config(
    page_title="Dashboard Prediksi XAU/USD",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": "Dashboard Skripsi: Perbandingan XGBoost vs TimesFM pada XAU/USD Intraday."
    }
)

# Dark mode via custom CSS
st.markdown("""
<style>
    /* Background utama */
    .stApp { background-color: #0e1117; }

    /* Sidebar */
    [data-testid="stSidebar"] { background-color: #161b22; }

    /* Judul sidebar */
    .sidebar-title {
        font-size: 1.4rem;
        font-weight: 700;
        color: #f0c040;
        margin-bottom: 0.5rem;
        letter-spacing: 0.5px;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: #1c2333;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 14px;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        background-color: #161b22;
        border-radius: 8px;
        padding: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        color: #8b949e;
        padding: 8px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #f0c040 !important;
        color: #0e1117 !important;
    }

    /* Caption teks */
    .caption-text { color: #8b949e; font-size: 0.85rem; }

    /* Event badge */
    .event-badge {
        background: #2d1b00;
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
@st.cache_data(show_spinner="Memuat data prediksi...")
def load_data():
    df_feat = pd.read_csv(PROCESSED / "dataset_featured.csv",
                          parse_dates=["Datetime"]).set_index("Datetime")
    df_a1 = pd.read_csv(RESULTS / "predictions_a1.csv",
                        parse_dates=["Datetime"]).set_index("Datetime")
    df_a2 = pd.read_csv(RESULTS / "predictions_a2.csv",
                        parse_dates=["Datetime"]).set_index("Datetime")
    df_b  = pd.read_csv(RESULTS / "predictions_b.csv",
                        parse_dates=["Datetime"]).set_index("Datetime")

    # Pilih kolom prediksi dari masing-masing model
    df_a1 = df_a1[["Target_Close_Actual", "Target_Close_Pred_a1",
                   "Absolute_Error_a1", "Direction_Correct_a1"]]
    df_a2 = df_a2[["Target_Close_Pred_a2",
                   "Absolute_Error_a2", "Direction_Correct_a2"]]
    df_b  = df_b[["Target_Close_Pred_b",
                  "Absolute_Error_b",  "Direction_Correct_b"]]

    # Inner join pada Datetime
    df = df_a1.join([df_a2, df_b], how="inner")

    # Tambahkan kolom is_event dari dataset_featured
    dev_cols = [c for c in df_feat.columns if c.startswith("Dev_")]
    is_event = (df_feat[dev_cols] != 0.0).any(axis=1)
    df["is_event"] = is_event.reindex(df.index, fill_value=False)

    return df

df_full = load_data()

# =============================================================================
# SIDEBAR: KONTROL PANEL
# =============================================================================
with st.sidebar:
    st.markdown('<p class="sidebar-title">🎛️ Kontrol Panel</p>', unsafe_allow_html=True)
    st.divider()

    # Rentang Tanggal
    st.subheader("📅 Rentang Tanggal")
    min_date = df_full.index.min().date()
    max_date = df_full.index.max().date()

    date_range = st.date_input(
        "Pilih rentang tanggal:",
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

    # Pilihan model
    st.subheader("🤖 Model yang Ditampilkan")
    model_options = {
        "XGBoost Baseline (A1)": ("Target_Close_Pred_a1", "Absolute_Error_a1", "Direction_Correct_a1", "#4C72B0"),
        "XGBoost + Makro (A2)":  ("Target_Close_Pred_a2", "Absolute_Error_a2", "Direction_Correct_a2", "#DD8452"),
        "TimesFM Zero-Shot (B)": ("Target_Close_Pred_b",  "Absolute_Error_b",  "Direction_Correct_b",  "#55A868"),
    }
    selected_models = st.multiselect(
        "Pilih model:",
        options=list(model_options.keys()),
        default=list(model_options.keys()),
        label_visibility="collapsed"
    )

    st.divider()

    # Filter event
    st.subheader("📰 Filter Berita Makro")
    event_only = st.checkbox("Fokus pada Jam Rilis Berita Saja", value=False)
    if event_only:
        st.info(f"Mode aktif: hanya **{df_full['is_event'].sum()} candle** saat rilis berita.")

    st.divider()
    st.caption("📁 Data: Test Set 2025 (XAU/USD 15-menit)")

# =============================================================================
# FILTER DATA
# =============================================================================
mask = (df_full.index >= start_dt) & (df_full.index <= end_dt)
df = df_full.loc[mask].copy()
if event_only:
    df = df[df["is_event"]].copy()

n_rows   = len(df)
n_events = int(df["is_event"].sum())

# =============================================================================
# HEADER UTAMA
# =============================================================================
st.markdown("## 📈 Dashboard Prediksi Harga Emas (XAU/USD)")
col_h1, col_h2, col_h3 = st.columns(3)
col_h1.metric("Candle Ditampilkan", f"{n_rows:,}", help="Jumlah candle 15-menit dalam rentang yang dipilih")
col_h2.metric("Jam Rilis Berita", f"{n_events}", help="Candle di mana ada rilis berita makroekonomi")
col_h3.metric("Model Aktif", f"{len(selected_models)}", help="Jumlah model yang sedang dibandingkan")

st.divider()

# =============================================================================
# TABS UTAMA
# =============================================================================
tab1, tab2, tab3 = st.tabs(["📊 Grafik Prediksi", "📐 Metrik Evaluasi", "🧠 Interpretasi SHAP"])

# ---------------------------------------------------------------------------
# TAB 1: GRAFIK PREDIKSI
# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Perbandingan Harga Aktual vs Prediksi XAU/USD")

    if n_rows == 0:
        st.warning("Tidak ada data pada rentang tanggal yang dipilih.")
    else:
        fig = go.Figure()

        # Harga aktual (selalu ditampilkan)
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["Target_Close_Actual"],
            mode="lines",
            name="Harga Aktual",
            line=dict(color="#f0c040", width=1.5),
        ))

        # Prediksi per model
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

        # Sorot jam rilis berita (vertical vrect)
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
            template="plotly_dark",
            paper_bgcolor="#0e1117",
            plot_bgcolor="#161b22",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            xaxis=dict(showgrid=True, gridcolor="#30363d", title="Datetime (UTC)"),
            yaxis=dict(showgrid=True, gridcolor="#30363d", title="Harga XAU/USD"),
            height=500,
            margin=dict(l=10, r=10, t=20, b=10),
            hovermode="x unified",
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"🟡 Zona kuning = jam rilis berita makroekonomi ({n_events} titik). Garis putus-putus = prediksi model.")

# ---------------------------------------------------------------------------
# TAB 2: METRIK EVALUASI
# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Metrik Evaluasi (dihitung dari data yang difilter)")

    if n_rows == 0:
        st.warning("Tidak ada data pada rentang tanggal yang dipilih.")
    else:
        def compute_metrics(df_subset, ae_col, dc_col, pred_col, actual_col="Target_Close_Actual"):
            """Hitung MAE, RMSE, MAPE, MDA dari subset data."""
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

        # Global (dalam rentang yang dipilih)
        global_data   = df
        event_data    = df[df["is_event"]]
        non_event_data = df[~df["is_event"]]

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
                    f"Global<br><span style='color:#8b949e;'>({n_rows:,} candle)</span></div>",
                    unsafe_allow_html=True
                )
            col1.metric("MAE",  f"{mae_g:.4f} USD"  if mae_g  is not None else "—", help="Mean Absolute Error harga absolut")
            col2.metric("RMSE", f"{rmse_g:.4f} USD" if rmse_g is not None else "—", help="Root Mean Squared Error")
            col3.metric("MAPE", f"{mape_g:.4f}%"    if mape_g is not None else "—", help="Mean Absolute Percentage Error")
            col4.metric("MDA",  f"{mda_g:.2f}%"     if mda_g  is not None else "—",
                        delta=f"{'↑' if mda_g and mda_g > 50 else '↓'} dari 50%",
                        help="Mean Directional Accuracy (>50% = lebih baik dari random)")

            if len(event_data) > 0:
                col_label2, col1e, col2e, col3e, col4e = st.columns([1.2, 1, 1, 1, 1])
                with col_label2:
                    st.markdown(
                        f"<div style='padding-top:32px;color:{color};font-weight:700;font-size:0.9rem;'>"
                        f"Jam Rilis 📰<br><span style='color:#8b949e;'>({len(event_data)} candle)</span></div>",
                        unsafe_allow_html=True
                    )
                col1e.metric("MAE",  f"{mae_e:.4f} USD"  if mae_e  is not None else "—")
                col2e.metric("RMSE", f"{rmse_e:.4f} USD" if rmse_e is not None else "—")
                col3e.metric("MAPE", f"{mape_e:.4f}%"    if mape_e is not None else "—")
                col4e.metric("MDA",  f"{mda_e:.2f}%"     if mda_e  is not None else "—",
                             delta=f"{'↑' if mda_e and mda_e > 50 else '↓'} dari 50%")
            else:
                st.caption("_Tidak ada candle event dalam rentang ini._")

            st.divider()

# ---------------------------------------------------------------------------
# TAB 3: INTERPRETASI SHAP
# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Interpretasi Fitur dengan SHAP (SHapley Additive exPlanations)")
    st.markdown(
        "SHAP menjelaskan kontribusi masing-masing fitur terhadap keputusan model. "
        "Sumbu Y mengurutkan fitur dari **paling berpengaruh** (atas) ke paling lemah (bawah). "
        "Warna merah = nilai fitur tinggi; biru = nilai fitur rendah."
    )
    st.divider()

    shap_global_path = FIGURES / "shap_summary.png"
    shap_event_path  = FIGURES / "shap_event_summary.png"

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.markdown("##### 🌍 SHAP Global (Seluruh Test Set 2025)")
        st.markdown(
            "<p class='caption-text'>Fitur teknikal (OHLCV & lag) mendominasi karena 99.5% data "
            "tidak memiliki sinyal makroekonomi.</p>",
            unsafe_allow_html=True
        )
        if shap_global_path.exists():
            st.image(str(shap_global_path), use_container_width=True)
        else:
            st.warning(f"File tidak ditemukan: `{shap_global_path.name}`. Jalankan `src/shap_analysis.py` terlebih dahulu.")

    with col_s2:
        st.markdown("##### 📰 SHAP Khusus Jam Rilis Berita (102 Candle)")
        st.markdown(
            "<p class='caption-text'>Saat berita dirilis, fitur <code>Dev_...</code> melonjak ke peringkat "
            "teratas — bukti nyata bahwa XGBoost berhasil menangkap sinyal makroekonomi.</p>",
            unsafe_allow_html=True
        )
        if shap_event_path.exists():
            st.image(str(shap_event_path), use_container_width=True)
        else:
            st.warning(f"File tidak ditemukan: `{shap_event_path.name}`. Jalankan `src/shap_analysis.py` terlebih dahulu.")

    st.divider()
    st.info(
        "💡 **Insight Kunci**: Perbandingan kedua plot ini membuktikan hipotesis penelitian — "
        "fitur makroekonomi (`Dev_NFP`, `Dev_CPI`, `Dev_FedRate`, dll.) memberikan *alpha* prediktif "
        "yang signifikan **hanya pada momen rilis berita**, bukan secara keseluruhan. "
        "Ini menjelaskan mengapa MDA Skenario A2 melonjak **+8.82%** dibanding baseline saat jam event."
    )

# =============================================================================
# FOOTER
# =============================================================================
st.divider()
st.markdown(
    "<div style='text-align:center; color:#8b949e; font-size:0.8rem;'>"
    "📘 Dashboard Skripsi — Prediksi Harga XAU/USD Intraday | XGBoost vs TimesFM | Test Set 2025"
    "</div>",
    unsafe_allow_html=True
)
