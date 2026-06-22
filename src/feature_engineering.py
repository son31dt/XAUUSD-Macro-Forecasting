"""
feature_engineering.py
=======================
Modul untuk melakukan Feature Engineering (Fase 2) pada dataset hasil
Time-Alignment (dataset_final.csv).

Fase 2: Feature Engineering
Penulis : Jason Daniel Tanubrata
Tanggal : 2026

Pipeline (urutan eksekusi dalam build_features()):
  1. Load dataset_final.csv                   -> load_aligned_dataset()
  2. Buat Target: Target_Close = Close.shift(-1)
              & Target_Diff   = Target_Close - Close  (selisih harga, stasioner)
  3. Buat Lag Features (t-1, t-2, t-3) untuk OHLCV saja
     (BUKAN untuk kolom Dev_ -- sinyal makro harus point-in-time)
  4. Hapus baris NaN (akibat shift & lag di ujung dataset)
  5. Simpan dataset_featured.csv              -> build_features()

Mengapa Target_Diff Lebih Baik untuk XGBoost?
  XGBoost tidak dapat mengekstrapolasi di luar rentang nilai training.
  Jika training 2021-2024 (max harga ~$2700) dan test 2025 naik ke $4300,
  prediksi harga absolut akan 'flatline' di $2700.
  Target_Diff = Close[t+1] - Close[t] adalah selisih harga (return):
  - Distribusinya stasioner dan terpusat di sekitar 0
  - XGBoost bisa memprediksinya tanpa masalah ekstrapolasi
  - Harga absolut direkonstruksi: y_pred_price = Close[t] + y_pred_diff

Catatan Penting - Mengapa Dev_ TIDAK di-lag?
  Kolom Dev_ sudah bersifat point-in-time (hanya aktif di satu candle).
  Membuat lag untuk Dev_ justru melanggar prinsip point-in-time dan
  mengubah sinyal "kejutan berita saat ini" menjadi "memori berita lama",
  yang tidak relevan secara finansial untuk prediksi intraday.
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path

# --- Konfigurasi Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Konstanta Path ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

DATASET_FINAL_PATH    = PROCESSED_DATA_DIR / "dataset_final.csv"
DATASET_FEATURED_PATH = PROCESSED_DATA_DIR / "dataset_featured.csv"

# --- Konstanta Kolom ---
# Kolom OHLCV yang akan dibuatkan lag features
OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]

# Jumlah lag yang dibuat: t-1, t-2, t-3
LAG_STEPS = [1, 2, 3]

# Kolom Dev_ (10 indikator makro) — TIDAK akan dibuatkan lag
EVENT_CATEGORIES = [
    "CPI", "PPI", "GDP", "FedRate", "ADP",
    "NFP", "Jobless", "Earnings", "ISM", "Retail",
]
DEV_COLUMNS = [f"Dev_{cat}" for cat in EVENT_CATEGORIES]


# =============================================================================
# BAGIAN 1: LOAD DATASET TERALIGN
# =============================================================================

def load_aligned_dataset(
    filepath: str | Path = DATASET_FINAL_PATH,
) -> pd.DataFrame:
    """
    Membaca dataset_final.csv hasil Time-Alignment.

    Parameter
    ----------
    filepath : str | Path
        Path ke dataset_final.csv. Default ke DATASET_FINAL_PATH.

    Kembalian
    ---------
    pd.DataFrame
        DataFrame dengan kolom: Datetime, Open, High, Low, Close, Volume,
        Dev_CPI, ..., Dev_Retail.
    """
    filepath = Path(filepath)
    logger.info(f"Memuat dataset teralign dari: {filepath}")

    df = pd.read_csv(filepath, parse_dates=["Datetime"])

    # Urutkan berdasarkan Datetime (KRUSIAL untuk lag features)
    df = df.sort_values("Datetime").reset_index(drop=True)

    logger.info(
        f"  Dataset dimuat: {len(df):,} baris. "
        f"Rentang: {df['Datetime'].min()} s/d {df['Datetime'].max()}"
    )
    return df


# =============================================================================
# BAGIAN 2: BUAT KOLOM TARGET
# =============================================================================

def create_target_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membuat dua kolom target prediksi: Target_Close dan Target_Diff.

    Target_Close[t] = Close[t+1]
      -> Harga absolut candle berikutnya. Digunakan sebagai referensi
         evaluasi metrik akhir (MAE, RMSE dalam satuan USD).

    Target_Diff[t] = Target_Close[t] - Close[t] = Close[t+1] - Close[t]
      -> Selisih harga (price difference / 1-step return). Ini yang
         DIPREDIKSI oleh XGBoost untuk menghindari masalah ekstrapolasi.

    Masalah Ekstrapolasi XGBoost:
      XGBoost adalah model berbasis pohon yang tidak bisa mengekstrapolasi
      di luar rentang nilai yang dilihat saat training.
      Jika training 2021-2024 dan harga 2025 naik ke level baru (mis. $4300),
      prediksi harga absolut akan 'stuck' di sekitar max training ($2700).
      Solusi: prediksi Target_Diff (yang stasioner, terdistribusi normal
      di sekitar 0), lalu rekonstruksi harga absolut di inferensi:
        y_pred_price[t] = Close[t] + y_pred_diff[t]

    CATATAN ANTI-LEAKAGE:
      Close[t] adalah harga SAAT INI (bukan masa depan) — AMAN digunakan
      sebagai fitur. Target_Diff[t] = Close[t+1] - Close[t] adalah selisih
      dengan candle t+1 yang tidak ada dalam fitur X — AMAN.

    Parameter
    ----------
    df : pd.DataFrame
        DataFrame teralign dengan kolom Close.

    Kembalian
    ---------
    pd.DataFrame
        DataFrame dengan tambahan kolom 'Target_Close' dan 'Target_Diff'.
        Baris terakhir kedua kolom akan NaN (tidak ada candle t+1).
    """
    logger.info("Membuat kolom target: Target_Close dan Target_Diff...")

    df = df.copy()

    # Target_Close[t] = Close[t+1]  (harga absolut candle berikutnya)
    df["Target_Close"] = df["Close"].shift(-1)

    # Target_Diff[t] = Close[t+1] - Close[t]  (selisih stasioner)
    # Ini setara dengan: df["Target_Close"] - df["Close"]
    df["Target_Diff"] = df["Target_Close"] - df["Close"]

    n_nan = df["Target_Close"].isna().sum()
    diff_mean = df["Target_Diff"].mean()
    diff_std  = df["Target_Diff"].std()
    logger.info(
        f"  Target_Close dibuat. NaN baris terakhir: {n_nan}."
    )
    logger.info(
        f"  Target_Diff  dibuat. "
        f"mean={diff_mean:.4f}, std={diff_std:.4f} USD "
        f"(distribusi stasioner di sekitar 0)."
    )
    return df


# =============================================================================
# BAGIAN 3: BUAT LAG FEATURES
# =============================================================================

def create_lag_features(
    df: pd.DataFrame,
    cols: list[str] = OHLCV_COLS,
    lags: list[int] = LAG_STEPS,
) -> pd.DataFrame:
    """
    Membuat lag features untuk kolom OHLCV (t-1, t-2, t-3).

    Konvensi penamaan:
      Close_lag1 = Close pada candle t-1 (satu candle sebelumnya)
      Close_lag2 = Close pada candle t-2
      Close_lag3 = Close pada candle t-3

    Mengapa HANYA OHLCV yang di-lag?
    ----------------------------------
    1. Kolom Dev_ sudah point-in-time: sinyal makro hanya aktif di satu
       candle. Membuat lag Dev_CPI_lag1 berarti "kejutan CPI dari candle
       sebelumnya" — ini bukan informasi yang bermakna secara finansial
       untuk prediksi intraday 15 menit ke depan.
    2. Lag OHLCV sangat relevan: pola candlestick, momentum harga, dan
       volatilitas terakhir adalah informasi kunci untuk prediksi intraday.

    CATATAN ANTI-LEAKAGE:
    Lag features dibuat dengan shift(+n), sehingga:
      Close_lag1[t] = Close[t-1]  (data masa lalu — AMAN)
      Close_lag2[t] = Close[t-2]  (data masa lalu — AMAN)
    Tidak ada informasi dari masa depan yang bocor.

    Parameter
    ----------
    df : pd.DataFrame
        DataFrame dengan kolom OHLCV.
    cols : list[str]
        Kolom yang akan dibuatkan lag. Default: OHLCV_COLS.
    lags : list[int]
        Daftar lag step yang akan dibuat. Default: [1, 2, 3].

    Kembalian
    ---------
    pd.DataFrame
        DataFrame dengan tambahan kolom lag (misal: Close_lag1, Close_lag2, ...).
    """
    logger.info(f"Membuat lag features untuk {cols} dengan lag {lags}...")

    df = df.copy()
    lag_col_names = []

    for col in cols:
        if col not in df.columns:
            logger.warning(f"  Kolom '{col}' tidak ditemukan. Dilewati.")
            continue
        for lag in lags:
            col_name = f"{col}_lag{lag}"
            # shift(+lag): pindahkan nilai ke bawah sebanyak `lag` baris
            # sehingga lag_col[t] = col[t - lag] (data historis)
            df[col_name] = df[col].shift(lag)
            lag_col_names.append(col_name)

    logger.info(
        f"  {len(lag_col_names)} kolom lag dibuat: "
        f"{lag_col_names[:5]}{'...' if len(lag_col_names) > 5 else ''}"
    )
    return df


# =============================================================================
# BAGIAN 4: HAPUS BARIS NaN DAN FINALISASI
# =============================================================================

def drop_nan_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghapus baris yang mengandung NaN akibat operasi shift dan lag.

    Sumber NaN setelah feature engineering:
    1. Lag features: baris pertama (t=0 sampai t=max_lag-1) akan NaN
       karena tidak ada data historis yang cukup.
    2. Target_Close: baris terakhir akan NaN karena tidak ada candle t+1.

    Total baris yang hilang = max(lags) baris di awal + 1 baris di akhir.
    Untuk lag=[1,2,3] -> 3 baris pertama + 1 baris terakhir = 4 baris.
    Ini sangat kecil dibanding 117.000+ candle.

    Parameter
    ----------
    df : pd.DataFrame
        DataFrame dengan kemungkinan NaN di kolom lag dan target.

    Kembalian
    ---------
    pd.DataFrame
        DataFrame bersih tanpa NaN.
    """
    n_before = len(df)
    df = df.dropna().reset_index(drop=True)
    n_after  = len(df)
    n_removed = n_before - n_after

    logger.info(
        f"Menghapus baris NaN: {n_removed} baris dihapus "
        f"({n_before:,} -> {n_after:,})."
    )
    return df


# =============================================================================
# BAGIAN 5: PIPELINE UTAMA FEATURE ENGINEERING
# =============================================================================

def build_features(
    input_path: str | Path = DATASET_FINAL_PATH,
    output_path: str | Path = DATASET_FEATURED_PATH,
    save_output: bool = True,
) -> pd.DataFrame:
    """
    Pipeline utama Feature Engineering (Fase 2).

    Menghasilkan dataset lengkap yang siap untuk tahap modeling (Fase 3+).

    Kolom output dataset_featured.csv:
    -----------------------------------
    [Datetime]
    [OHLCV]         : Open, High, Low, Close, Volume
    [Lag OHLCV]     : Open_lag1..3, High_lag1..3, Low_lag1..3,
                      Close_lag1..3, Volume_lag1..3  (15 kolom)
    [Makro Dev_]    : Dev_CPI, Dev_PPI, ..., Dev_Retail  (10 kolom, point-in-time)
    [Target]        : Target_Close

    Total: 1 + 5 + 15 + 10 + 1 = 32 kolom

    Parameter
    ----------
    input_path : str | Path
        Path ke dataset_final.csv. Default ke DATASET_FINAL_PATH.
    output_path : str | Path
        Path output dataset_featured.csv. Default ke DATASET_FEATURED_PATH.
    save_output : bool
        Jika True, simpan hasil ke output_path. Default True.

    Kembalian
    ---------
    pd.DataFrame
        Dataset final siap modeling dengan semua fitur.
    """
    logger.info("=" * 60)
    logger.info("MEMULAI PIPELINE FEATURE ENGINEERING (FASE 2)")
    logger.info("=" * 60)

    # --- Langkah 1: Load dataset teralign ---
    df = load_aligned_dataset(input_path)

    # --- Langkah 2: Buat kolom target ---
    df = create_target_column(df)

    # --- Langkah 3: Buat lag features (HANYA OHLCV, bukan Dev_) ---
    df = create_lag_features(df, cols=OHLCV_COLS, lags=LAG_STEPS)

    # --- Langkah 4: Hapus baris NaN ---
    df = drop_nan_rows(df)

    # --- Langkah 5: Susun urutan kolom akhir ---
    # Urutan yang logis: Datetime, OHLCV, Lag, Dev_, Target
    lag_cols = [
        f"{col}_lag{lag}"
        for col in OHLCV_COLS
        for lag in LAG_STEPS
    ]
    cols_final = (
        ["Datetime"]
        + OHLCV_COLS          # Open, High, Low, Close, Volume (fitur t)
        + lag_cols             # Open_lag1..Close_lag3 (fitur t-n)
        + DEV_COLUMNS          # Dev_CPI, Dev_PPI, ... (sinyal makro, point-in-time)
        + ["Target_Close"]     # Target harga absolut: Close[t+1]
        + ["Target_Diff"]      # Target selisih stasioner: Close[t+1] - Close[t]
    )

    # Hanya ambil kolom yang ada (defensive)
    cols_available = [c for c in cols_final if c in df.columns]
    df = df[cols_available].copy()

    # Pastikan urutan kronologis
    df = df.sort_values("Datetime").reset_index(drop=True)

    # --- Simpan output ---
    if save_output:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Dataset featured disimpan ke: {output_path}")

    logger.info("=" * 60)
    logger.info(f"FEATURE ENGINEERING SELESAI.")
    logger.info(f"  Total baris  : {len(df):,}")
    logger.info(f"  Total kolom  : {len(df.columns)}")
    logger.info(
        f"  Kolom fitur  : {len(df.columns) - 3} "
        f"(tanpa Datetime, Target_Close, Target_Diff)"
    )
    logger.info("=" * 60)

    return df


# =============================================================================
# BAGIAN 6: ENTRY POINT (untuk pengujian langsung)
# =============================================================================

if __name__ == "__main__":
    """
    Jalankan seluruh pipeline end-to-end (Fase 1 + Fase 2):
        python src/feature_engineering.py

    Urutan eksekusi di sini:
      1. Regenerasi macro_news_clean.csv (data_loader dengan ISM yang sudah dikoreksi)
      2. Regenerasi dataset_final.csv     (time_alignment point-in-time)
      3. Buat dataset_featured.csv        (feature engineering Fase 2)
    """
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    # --- Import modul upstream ---
    from data_loader import load_and_process_macro
    from time_alignment import align_and_merge

    print("\n" + "=" * 60)
    print("PIPELINE END-TO-END: FASE 1 + FASE 2")
    print("=" * 60)

    # === FASE 1A: Regenerasi data makro bersih ===
    print("\n[FASE 1A] Meregenerasi macro_news_clean.csv...")
    df_macro = load_and_process_macro(save_processed=True)
    print(f"  Distribusi ISM setelah koreksi filter:")
    if "Event" in df_macro.columns:
        ism_rows = df_macro[df_macro["Event"] == "ISM"]
        print(f"  Baris ISM Manufacturing PMI saja: {len(ism_rows)}")

    # === FASE 1B: Regenerasi dataset final (time-alignment) ===
    print("\n[FASE 1B] Meregenerasi dataset_final.csv (time-alignment)...")
    df_final = align_and_merge(save_output=True, run_validation=True)

    # === FASE 2: Feature Engineering ===
    print("\n[FASE 2] Membangun fitur untuk Machine Learning...")
    df_featured = build_features(save_output=True)

    # === Laporan Final ===
    print("\n--- PRATINJAU DATASET FEATURED (5 baris pertama) ---")
    print(df_featured.head(5).to_string())

    print("\n--- INFO DATAFRAME ---")
    print(df_featured.info())

    print("\n--- DISTRIBUSI KOLOM Dev_ (nilai non-zero per kolom) ---")
    for col in DEV_COLUMNS:
        n_nonzero = (df_featured[col] != 0.0).sum()
        print(f"  {col:15s}: {n_nonzero:5d} candle aktif "
              f"({n_nonzero / len(df_featured) * 100:.3f}%)")

    print("\n--- CEK ANTI-CARRY-FORWARD: NFP 8 Jan 2021 ---")
    nfp_window = df_featured[
        (df_featured["Datetime"] >= "2021-01-08 13:15")
        & (df_featured["Datetime"] <= "2021-01-08 14:00")
    ][["Datetime", "Close", "Dev_NFP", "Dev_Earnings", "Target_Close"]]
    print(nfp_window.to_string())

    print("\n--- CEK ANTI-CARRY-FORWARD: ISM 5 Jan 2021 ---")
    ism_window = df_featured[
        (df_featured["Datetime"] >= "2021-01-05 14:45")
        & (df_featured["Datetime"] <= "2021-01-05 15:30")
    ][["Datetime", "Close", "Dev_ISM", "Target_Close"]]
    print(ism_window.to_string())

    print("\n--- RENTANG WAKTU ---")
    print(f"Candle pertama : {df_featured['Datetime'].min()}")
    print(f"Candle terakhir: {df_featured['Datetime'].max()}")
    print(f"Total fitur    : {len(df_featured.columns) - 2} kolom")
    print(f"Total sampel   : {len(df_featured):,}")
