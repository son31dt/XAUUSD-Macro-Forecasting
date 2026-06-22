"""
evaluate_and_visualize.py
=========================
Fase Akhir: Mengevaluasi dan memvisualisasikan hasil ketiga skenario
(A1, A2, dan B) secara komprehensif, khususnya membandingkan performa
Global vs Event-Driven (jam rilis berita makro).

Output:
- Tabel ringkasan di terminal.
- Grafik mda_comparison.png dan mae_comparison.png di reports/figures/
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- Setup Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = PROJECT_ROOT / "reports" / "figures"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

def main():
    print("=" * 70)
    print("  FASE EVALUASI DAN VISUALISASI HASIL")
    print("=" * 70)
    
    # 1. Muat Keempat File CSV
    print("Memuat dataset dan file prediksi...")
    try:
        df_feat = pd.read_csv(PROCESSED_DIR / "dataset_featured.csv", index_col="Datetime", parse_dates=True)
        df_a1 = pd.read_csv(RESULTS_DIR / "predictions_a1.csv", index_col="Datetime", parse_dates=True)
        df_a2 = pd.read_csv(RESULTS_DIR / "predictions_a2.csv", index_col="Datetime", parse_dates=True)
        df_b = pd.read_csv(RESULTS_DIR / "predictions_b.csv", index_col="Datetime", parse_dates=True)
    except FileNotFoundError as e:
        print(f"ERROR: File tidak ditemukan - {e}")
        return

    # Ambil kolom metrik dan target price untuk RMSE & MAPE
    df_a1_metrics = df_a1[["Target_Close_Actual", "Target_Close_Pred_a1", "Absolute_Error_a1", "Direction_Correct_a1"]]
    df_a2_metrics = df_a2[["Target_Close_Pred_a2", "Absolute_Error_a2", "Direction_Correct_a2"]]
    df_b_metrics = df_b[["Target_Close_Pred_b", "Absolute_Error_b", "Direction_Correct_b"]]

    # Gabung (inner join) berdasarkan Datetime
    df_merged = df_feat.join([df_a1_metrics, df_a2_metrics, df_b_metrics], how="inner")
    
    print(f"Data berhasil digabung. Total baris dievaluasi: {len(df_merged):,} candle.")

    # 2. Filter / Mask is_event
    # Identifikasi kolom Dev_ (Fitur Makro)
    dev_cols = [c for c in df_merged.columns if c.startswith("Dev_")]
    is_event = (df_merged[dev_cols] != 0.0).any(axis=1)
    
    print(f"Total baris Global: {len(df_merged):,}")
    print(f"Total baris Event (rilis berita): {is_event.sum():,}\n")

    # 3. Kalkulasi Metrik (Global vs Event-Driven)
    def calculate_metrics(df, model_suffix):
        # Ambil kolom yang dibutuhkan dari nama kolom
        if model_suffix == "b":
            pred_col = "Target_Close_Pred_b"
        else:
            pred_col = f"Target_Close_Pred_{model_suffix}"
            
        actual_col = "Target_Close_Actual"
        
        # Ambil numpy array untuk performa lebih cepat
        actual = df[actual_col].values
        pred = df[pred_col].values
        
        # 1. MAE
        mae = df[f"Absolute_Error_{model_suffix}"].mean()
        
        # 2. MDA
        mda = df[f"Direction_Correct_{model_suffix}"].astype(int).mean() * 100
        
        # 3. RMSE
        rmse = np.sqrt(np.mean((actual - pred) ** 2))
        
        # 4. MAPE (Abaikan pembagian 0)
        mape_arr = np.abs((actual - pred) / np.where(actual != 0, actual, np.nan))
        mape = np.nanmean(mape_arr) * 100 # dalam persentase
        
        return mae, rmse, mape, mda

    models = [
        ("a1", "A1 (XGBoost Baseline)"),
        ("a2", "A2 (XGBoost + Makro)"),
        ("b", "B (TimesFM Zero-Shot)")
    ]

    metrics_data = []

    for suffix, name in models:
        # Global metrics
        mae_global, rmse_global, mape_global, mda_global = calculate_metrics(df_merged, suffix)
        
        # Event metrics
        df_event = df_merged[is_event]
        mae_event, rmse_event, mape_event, mda_event = calculate_metrics(df_event, suffix)
        
        metrics_data.append({
            "Model": name,
            "Global MAE": mae_global,
            "Global RMSE": rmse_global,
            "Global MAPE": mape_global,
            "Global MDA (%)": mda_global,
            "Event MAE": mae_event,
            "Event RMSE": rmse_event,
            "Event MAPE": mape_event,
            "Event MDA (%)": mda_event
        })

    df_metrics = pd.DataFrame(metrics_data)

    # Cetak tabel ringkasan
    print("=" * 145)
    print("  RINGKASAN METRIK EVALUASI (Global vs Event-Driven)")
    print("=" * 145)
    # Cetak header
    print(f"  {'Model':<25} | "
          f"{'G-MAE':>8} | {'G-RMSE':>8} | {'G-MAPE':>8} | {'G-MDA':>8} | "
          f"{'E-MAE':>8} | {'E-RMSE':>8} | {'E-MAPE':>8} | {'E-MDA':>8}")
    print("  " + "-" * 141)
    for idx, row in df_metrics.iterrows():
        print(f"  {row['Model']:<25} | "
              f"{row['Global MAE']:>8.4f} | {row['Global RMSE']:>8.4f} | {row['Global MAPE']:>7.4f}% | {row['Global MDA (%)']:>7.2f}% | "
              f"{row['Event MAE']:>8.4f} | {row['Event RMSE']:>8.4f} | {row['Event MAPE']:>7.4f}% | {row['Event MDA (%)']:>7.2f}%")
    print("=" * 145)

    # 4. Visualisasi Grafik Skripsi (300 dpi)
    print("\nMenghasilkan visualisasi grafik...")
    
    # Tema Seaborn untuk tampilan profesional
    sns.set_theme(style="whitegrid", context="paper")
    
    x = np.arange(len(df_metrics))
    bar_width = 0.35
    
    warna_global = "#4C72B0"  # Biru
    warna_event = "#DD8452"   # Oranye
    
    # --- Grafik 1: MDA Comparison ---
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    bars1_mda = ax1.bar(x - bar_width/2, df_metrics["Global MDA (%)"], bar_width, label='Global MDA', color=warna_global, edgecolor='black', linewidth=0.5)
    bars2_mda = ax1.bar(x + bar_width/2, df_metrics["Event MDA (%)"], bar_width, label='Event MDA', color=warna_event, edgecolor='black', linewidth=0.5)

    ax1.set_ylabel('Mean Directional Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Perbandingan Mean Directional Accuracy (MDA)\nGlobal vs Jam Rilis Berita', fontsize=14, fontweight='bold', pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(df_metrics["Model"], fontsize=11)
    
    # Garis threshold random (50%)
    ax1.axhline(50, color='red', linestyle='--', alpha=0.7, label='Random Guess (50%)')
    ax1.legend(loc='upper right', fontsize=10)
    
    # Y-limit agar ada ruang untuk teks di atas bar
    max_mda = max(df_metrics["Global MDA (%)"].max(), df_metrics["Event MDA (%)"].max())
    ax1.set_ylim(40, max_mda + 10) # Set minimal 40% agar perbedaan terlihat jelas

    # Anotasi teks
    def add_labels(bars, ax, fmt='{:.2f}%'):
        for bar in bars:
            height = bar.get_height()
            ax.annotate(fmt.format(height),
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3 point vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    add_labels(bars1_mda, ax1)
    add_labels(bars2_mda, ax1)

    fig1.tight_layout()
    mda_path = REPORTS_DIR / "mda_comparison.png"
    fig1.savefig(mda_path, dpi=300, bbox_inches='tight')
    plt.close(fig1)
    
    # --- Grafik 2: MAE Comparison ---
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    
    # Untuk MAE, makin kecil makin baik. Gunakan warna hijau dan merah pudar
    warna_global_mae = "#55A868"
    warna_event_mae = "#C44E52"
    
    bars1_mae = ax2.bar(x - bar_width/2, df_metrics["Global MAE"], bar_width, label='Global MAE', color=warna_global_mae, edgecolor='black', linewidth=0.5)
    bars2_mae = ax2.bar(x + bar_width/2, df_metrics["Event MAE"], bar_width, label='Event MAE', color=warna_event_mae, edgecolor='black', linewidth=0.5)

    ax2.set_ylabel('Mean Absolute Error (USD)', fontsize=12, fontweight='bold')
    ax2.set_title('Perbandingan Mean Absolute Error (MAE)\nGlobal vs Jam Rilis Berita', fontsize=14, fontweight='bold', pad=15)
    ax2.set_xticks(x)
    ax2.set_xticklabels(df_metrics["Model"], fontsize=11)
    ax2.legend(loc='upper left', fontsize=10)
    
    # Y-limit agar ada ruang untuk teks di atas bar
    max_mae = max(df_metrics["Global MAE"].max(), df_metrics["Event MAE"].max())
    ax2.set_ylim(0, max_mae * 1.2)

    add_labels(bars1_mae, ax2, fmt='{:.2f}')
    add_labels(bars2_mae, ax2, fmt='{:.2f}')

    fig2.tight_layout()
    mae_path = REPORTS_DIR / "mae_comparison.png"
    fig2.savefig(mae_path, dpi=300, bbox_inches='tight')
    plt.close(fig2)

    print(f"Visualisasi berhasil disimpan!")
    print(f"1. {mda_path}")
    print(f"2. {mae_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
