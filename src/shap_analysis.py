"""
shap_analysis.py
================
Fase 8: Interpretasi Model A2 dengan SHAP (SHapley Additive exPlanations)
Skrip ini melatih ulang model XGBoost (Skenario A2) dengan hyperparameter
terbaik dan menghitung SHAP values pada Test Set (2025).
Output berupa shap_summary.png untuk melihat seberapa besar pengaruh
fitur makroekonomi (Dev_...) dibandingkan fitur OHLCV teknikal.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
import shap
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# --- Setup Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports" / "figures"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Konstanta Split
TEST_START = "2025-01-01 00:00:00"

def main():
    print("=" * 60)
    print("  FASE 8: SHAP ANALYSIS (Model A2)")
    print("=" * 60)
    
    # 1. Load Data
    print("Memuat dataset_featured.csv...")
    try:
        df = pd.read_csv(PROCESSED_DIR / "dataset_featured.csv")
    except FileNotFoundError:
        print("ERROR: dataset_featured.csv tidak ditemukan.")
        return
        
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df.set_index("Datetime", inplace=True)
    
    # Buat Target_Diff
    df["Target_Diff"] = df["Target_Close"] - df["Close"]
    
    # Splitting Data
    train_mask = df.index < pd.to_datetime(TEST_START)
    test_mask = df.index >= pd.to_datetime(TEST_START)
    
    df_train = df.loc[train_mask].copy()
    df_test = df.loc[test_mask].copy()
    
    # X_A2: Semua fitur kecuali target
    drop_cols = ["Target_Close", "Target_Diff"]
    
    X_train = df_train.drop(columns=drop_cols)
    y_train = df_train["Target_Diff"]
    
    X_test = df_test.drop(columns=drop_cols)
    y_test = df_test["Target_Diff"]
    
    print(f"Dimensi X_train : {X_train.shape}")
    print(f"Dimensi X_test  : {X_test.shape}")
    
    # 2. Train Model dengan Best Params
    print("\nMelatih ulang Model XGBoost A2 dengan best params...")
    best_params = {
        "n_estimators": 100,
        "learning_rate": 0.05,
        "max_depth": 6,
        "colsample_bytree": 0.7,
        "min_child_weight": 1,
        "random_state": 42,
        "n_jobs": -1
    }
    
    model = xgb.XGBRegressor(**best_params)
    model.fit(X_train, y_train)
    print("Pelatihan selesai.")
    
    # 3. Hitung SHAP Values
    print("\nMenghitung SHAP Values pada Test Set (2025)...")
    # Gunakan TreeExplainer yang sangat dioptimalkan untuk XGBoost
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    
    # 4. Buat SHAP Summary Plot (Global)
    print("Membuat grafik SHAP Summary Plot (Global)...")
    
    # Konfigurasi ukuran plot agar nama fitur (y-axis) tidak terpotong
    plt.figure(figsize=(12, 8))
    
    # shap.summary_plot secara otomatis memanggil plt.gca() dan merender di figur aktif
    shap.summary_plot(
        shap_values, 
        X_test, 
        max_display=20, # Tampilkan top 20 fitur
        show=False,     # Jangan panggil plt.show()
        plot_type="dot" # Dot plot standar SHAP
    )
    
    # Kustomisasi plot
    plt.title("SHAP Summary Plot - XGBoost Skenario A2 (Global Test Set 2025)", fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()
    
    save_path_global = REPORTS_DIR / "shap_summary.png"
    plt.savefig(save_path_global, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nSukses! Grafik SHAP Global berhasil disimpan di:")
    print(f"-> {save_path_global}")

    # 5. Buat SHAP Event Summary Plot (Khusus Jam Rilis Berita)
    print("\nMengekstrak data khusus event (rilis berita makro)...")
    dev_cols = [c for c in X_test.columns if c.startswith("Dev_")]
    is_event = (X_test[dev_cols] != 0.0).any(axis=1)
    
    # Filter dataset uji dan nilai SHAP-nya
    X_test_events = X_test[is_event]
    shap_values_events = shap_values[is_event]
    
    print(f"Total baris event ditemukan: {is_event.sum()} dari {len(X_test)} baris.")
    print("Membuat grafik SHAP Event Summary Plot...")
    
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        shap_values_events, 
        X_test_events, 
        max_display=20,
        show=False,
        plot_type="dot"
    )
    
    plt.title("SHAP Event Summary Plot - XGBoost A2 (Hanya Jam Rilis Berita)", fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()
    
    save_path_event = REPORTS_DIR / "shap_event_summary.png"
    plt.savefig(save_path_event, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Sukses! Grafik SHAP khusus Event berhasil disimpan di:")
    print(f"-> {save_path_event}")

    print("=" * 60)

if __name__ == "__main__":
    main()
