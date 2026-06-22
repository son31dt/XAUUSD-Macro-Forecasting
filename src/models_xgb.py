"""
models_xgb.py
=============
Modul untuk melatih dan mengevaluasi model XGBoost (Fase 3 & 4).

Fase 3: Data Splitting (Sequential/Chronological)
Fase 4: Modeling Skenario A1 - XGBoost Baseline (Endogen Only)

Penulis : Jason Daniel Tanubrata
Tanggal : 2026

Skenario yang diimplementasi di file ini:
  A1 : XGBoost Baseline - hanya fitur OHLCV + Lag (tanpa makro)
       -> Fungsi: train_scenario_a1()

Skenario lain (A2 dengan GridSearch, B TimesFM) akan diimplementasi
di fase berikutnya sebagai perbandingan.

Catatan Splitting:
  Train : 1 Jan 2021 - 31 Des 2024  (~4 tahun, ~94.000 candle)
  Test  : 1 Jan 2025 - 31 Des 2025  (~1 tahun, ~23.000 candle)
  TIDAK menggunakan random split untuk menghindari Data Leakage temporal.
"""

import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore", category=FutureWarning)

# --- Konfigurasi Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Konstanta Path ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR   = PROJECT_ROOT / "results"

DATASET_FEATURED_PATH = PROCESSED_DIR / "dataset_featured.csv"
PREDICTIONS_A1_PATH   = RESULTS_DIR / "predictions_a1.csv"

# --- Konstanta Splitting ---
TRAIN_END = "2024-12-31 23:59:59"   # Akhir periode latih
TEST_START = "2025-01-01 00:00:00"  # Awal periode uji

# --- Kolom Dev_ yang akan dibuang untuk Skenario A1 ---
DEV_COLUMN_PREFIX = "Dev_"

# --- Parameter XGBoost Skenario A1 (Baseline) ---
XGB_PARAMS_A1 = {
    "n_estimators"  : 200,
    "learning_rate" : 0.05,
    "max_depth"     : 6,
    "subsample"     : 0.8,        # Subsample baris: mencegah overfitting
    "colsample_bytree": 0.8,      # Subsample fitur per pohon
    "random_state"  : 42,
    "n_jobs"        : -1,         # Gunakan semua CPU core
    "tree_method"   : "hist",     # Lebih cepat untuk dataset besar
    "verbosity"     : 0,          # Sembunyikan log XGBoost internal
}


# =============================================================================
# BAGIAN 1: LOAD DAN SPLITTING DATA
# =============================================================================

def load_featured_dataset(
    filepath: str | Path = DATASET_FEATURED_PATH,
) -> pd.DataFrame:
    """
    Membaca dataset_featured.csv dan menjadikan Datetime sebagai index.

    Parameter
    ----------
    filepath : str | Path
        Path ke dataset_featured.csv.

    Kembalian
    ---------
    pd.DataFrame
        DataFrame dengan DatetimeIndex, siap untuk splitting.
    """
    filepath = Path(filepath)
    logger.info(f"Memuat dataset fitur dari: {filepath}")

    df = pd.read_csv(filepath, parse_dates=["Datetime"], index_col="Datetime")
    df = df.sort_index()  # Pastikan urutan kronologis

    logger.info(
        f"  Dataset dimuat: {len(df):,} baris | "
        f"{df.index.min().date()} s/d {df.index.max().date()}"
    )
    return df


def split_train_test(
    df: pd.DataFrame,
    train_end: str = TRAIN_END,
    test_start: str = TEST_START,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Memotong dataset secara kronologis (sequential split).

    Tidak menggunakan random split untuk menghormati urutan temporal
    dan menghindari Data Leakage (data masa depan masuk ke training).

    Parameter
    ----------
    df : pd.DataFrame
        Dataset lengkap dengan DatetimeIndex.
    train_end : str
        Batas akhir data latih (inklusif).
    test_start : str
        Batas awal data uji (inklusif).

    Kembalian
    ---------
    tuple[pd.DataFrame, pd.DataFrame]
        (df_train, df_test)
    """
    df_train = df.loc[:train_end].copy()
    df_test  = df.loc[test_start:].copy()

    logger.info(f"Splitting kronologis:")
    logger.info(
        f"  Train: {df_train.index.min().date()} s/d "
        f"{df_train.index.max().date()} | {len(df_train):,} candle"
    )
    logger.info(
        f"  Test : {df_test.index.min().date()} s/d "
        f"{df_test.index.max().date()}  | {len(df_test):,} candle"
    )

    # Verifikasi tidak ada overlap
    assert df_train.index.max() < df_test.index.min(), (
        "ERROR: Ada overlap antara Train dan Test! "
        "Periksa parameter train_end dan test_start."
    )
    return df_train, df_test


# =============================================================================
# BAGIAN 2: FUNGSI EVALUASI METRIK
# =============================================================================

def calculate_metrics(
    y_true_price: np.ndarray | pd.Series,
    y_pred_price: np.ndarray | pd.Series,
    y_true_diff: np.ndarray | pd.Series | None = None,
    y_pred_diff: np.ndarray | pd.Series | None = None,
    scenario_name: str = "Model",
) -> dict:
    """
    Menghitung dan mencetak metrik evaluasi:
      - MAE, RMSE, MAPE : dihitung pada harga ABSOLUT (y_true_price vs y_pred_price)
      - MDA              : dihitung pada SELISIH harga (sign(y_true_diff) vs sign(y_pred_diff))

    Pemisahan ini penting karena:
      - MAE/RMSE/MAPE dalam USD harus dibandingkan pada harga absolut
        agar bermakna secara finansial.
      - MDA harus dibandingkan pada prediksi arah (naik/turun) yang langsung
        dihasilkan model diff — bukan diturunkan ulang dari harga rekonstruksi.

    Parameter
    ----------
    y_true_price : array-like
        Nilai harga aktual (Target_Close).
    y_pred_price : array-like
        Nilai harga prediksi yang sudah direkonstruksi (Close + y_pred_diff).
    y_true_diff : array-like, optional
        Selisih harga aktual (Target_Diff). Digunakan untuk MDA.
    y_pred_diff : array-like, optional
        Selisih harga prediksi dari model. Digunakan untuk MDA.
    scenario_name : str
        Nama skenario untuk label output.

    Kembalian
    ---------
    dict
        Dictionary berisi nilai MAE, RMSE, MAPE, MDA.
    """
    y_true_price = np.array(y_true_price, dtype=float)
    y_pred_price = np.array(y_pred_price, dtype=float)

    # --- MAE (harga absolut) ---
    mae = mean_absolute_error(y_true_price, y_pred_price)

    # --- RMSE (harga absolut) ---
    rmse = np.sqrt(mean_squared_error(y_true_price, y_pred_price))

    # --- MAPE (harga absolut, hindari div-by-zero) ---
    mape_arr = np.abs(
        (y_true_price - y_pred_price)
        / np.where(y_true_price != 0, y_true_price, np.nan)
    )
    mape = np.nanmean(mape_arr) * 100

    # --- MDA (pada selisih harga — langsung dari model) ---
    # sign(y_true_diff) = arah pergerakan aktual
    # sign(y_pred_diff) = arah pergerakan yang diprediksi model
    # Keduanya = +1 (naik), -1 (turun), atau 0 (flat)
    if y_true_diff is not None and y_pred_diff is not None:
        y_true_diff = np.array(y_true_diff, dtype=float)
        y_pred_diff = np.array(y_pred_diff, dtype=float)
        # Hanya hitung MDA pada candle yang benar-benar bergerak (diff != 0)
        # untuk menghindari denominasi yang mengandung banyak 0 (flat candle)
        mask_moving = y_true_diff != 0
        if mask_moving.sum() > 0:
            mda = np.mean(
                np.sign(y_true_diff[mask_moving])
                == np.sign(y_pred_diff[mask_moving])
            ) * 100
        else:
            mda = np.nan
    elif len(y_true_price) > 1:
        # Fallback: MDA dari harga absolut (kurang akurat)
        actual_dir = np.sign(y_true_price[1:] - y_true_price[:-1])
        pred_dir   = np.sign(y_pred_price[1:] - y_true_price[:-1])
        mda = np.mean(actual_dir == pred_dir) * 100
    else:
        mda = np.nan

    metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape, "MDA": mda}

    # Cetak hasil
    sep = "=" * 50
    print(f"\n{sep}")
    print(f"  HASIL EVALUASI: {scenario_name}")
    print(sep)
    print(f"  MAE  : {mae:>10.4f}  USD  (harga absolut)")
    print(f"  RMSE : {rmse:>10.4f}  USD  (harga absolut)")
    print(f"  MAPE : {mape:>10.6f}  %    (harga absolut)")
    print(f"  MDA  : {mda:>10.4f}  %    (arah diff | random = 50%)")
    print(sep)

    return metrics


# =============================================================================
# BAGIAN 3: SKENARIO A1 - XGBOOST BASELINE (ENDOGEN ONLY)
# =============================================================================

def train_scenario_a1(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    xgb_params: dict = XGB_PARAMS_A1,
    save_predictions: bool = True,
) -> tuple[XGBRegressor, pd.DataFrame, dict]:
    """
    Melatih XGBoost Baseline (Skenario A1) menggunakan HANYA fitur endogen
    (OHLCV + Lag). Kolom Dev_ makro dibuang sepenuhnya.

    Skenario A1 adalah benchmark/baseline:
    "Seberapa baik XGBoost memprediksi harga emas menggunakan
    hanya data harga historisnya sendiri, tanpa informasi eksternal?"

    Parameter
    ----------
    df_train : pd.DataFrame
        Data latih dengan DatetimeIndex.
    df_test : pd.DataFrame
        Data uji dengan DatetimeIndex.
    xgb_params : dict
        Parameter XGBRegressor. Default: XGB_PARAMS_A1.
    save_predictions : bool
        Jika True, simpan prediksi ke PREDICTIONS_A1_PATH.

    Kembalian
    ---------
    tuple[XGBRegressor, pd.DataFrame, dict]
        (model_a1, df_predictions, metrics_dict)
        - model_a1     : model XGBoost yang sudah dilatih
        - df_predictions: DataFrame dengan kolom y_true dan y_pred_a1
        - metrics_dict  : dictionary metrik evaluasi
    """
    logger.info("=" * 60)
    logger.info("SKENARIO A1: XGBoost Baseline (Endogen Only — Target: Diff)")
    logger.info("=" * 60)

    # --- Langkah 1: Tentukan kolom fitur dan target ---
    # Fitur A1: OHLCV + Lag saja. Buang Dev_ DAN kedua kolom target.
    dev_cols = [c for c in df_train.columns if c.startswith(DEV_COLUMN_PREFIX)]
    exclude  = set(dev_cols) | {"Target_Close", "Target_Diff"}
    feature_cols_a1 = [c for c in df_train.columns if c not in exclude]

    logger.info(
        f"  Membuang {len(dev_cols)} kolom Dev_ untuk Skenario A1."
    )
    logger.info(f"  Jumlah fitur A1    : {len(feature_cols_a1)}")
    logger.info(f"  Ukuran X_train     : {df_train[feature_cols_a1].shape}")
    logger.info(f"  Ukuran X_test      : {df_test[feature_cols_a1].shape}")

    X_train = df_train[feature_cols_a1]
    X_test  = df_test[feature_cols_a1]

    # Target: Target_Diff (selisih stasioner Close[t+1] - Close[t])
    y_train_diff = df_train["Target_Diff"]
    y_test_diff  = df_test["Target_Diff"]

    # Simpan juga Target_Close untuk evaluasi MAE/RMSE/MAPE dalam USD
    y_test_close = df_test["Target_Close"]

    logger.info(
        f"  Target training: Target_Diff "
        f"(mean={y_train_diff.mean():.4f}, std={y_train_diff.std():.4f} USD)"
    )

    # --- Langkah 2: Latih model pada Target_Diff ---
    logger.info(f"  Melatih XGBRegressor (target=Target_Diff)...")
    model_a1 = XGBRegressor(**xgb_params)
    model_a1.fit(
        X_train, y_train_diff,
        eval_set=[(X_test, y_test_diff)],
        verbose=False,
    )
    logger.info("  Pelatihan selesai.")

    # --- Langkah 3: Prediksi selisih harga ---
    y_pred_diff = model_a1.predict(X_test)

    # --- Langkah 4: Rekonstruksi harga absolut ---
    # y_pred_price[t] = Close[t] + y_pred_diff[t]
    # Close[t] ada di X_test (fitur saat ini, bukan masa depan — AMAN)
    close_test   = X_test["Close"].values
    y_pred_price = close_test + y_pred_diff

    logger.info(
        f"  Rekonstruksi harga selesai. "
        f"Rentang prediksi: [{y_pred_price.min():.2f}, {y_pred_price.max():.2f}] USD"
    )
    logger.info(
        f"  Rentang aktual  : [{y_test_close.min():.2f}, {y_test_close.max():.2f}] USD"
    )

    # --- Langkah 5: Evaluasi metrik ---
    metrics = calculate_metrics(
        y_true_price=y_test_close.values,
        y_pred_price=y_pred_price,
        y_true_diff=y_test_diff.values,
        y_pred_diff=y_pred_diff,
        scenario_name="A1 - XGBoost Baseline (Diff Target)",
    )

    # --- Langkah 6: Simpan prediksi ---
    # Format kolom yang mudah dibaca manusia:
    #   Current_Close        : harga Close saat ini (t), sebagai acuan
    #   y_true_diff          : selisih aktual Close[t+1] - Close[t]
    #   y_pred_diff_a1       : selisih prediksi model
    #   Target_Close_Actual  : harga aktual 15 menit ke depan = Close[t+1]
    #   Target_Close_Pred_a1 : harga prediksi = Current_Close + y_pred_diff_a1
    #   Absolute_Error_a1    : |Target_Close_Actual - Target_Close_Pred_a1|
    #   Direction_Correct_a1 : True jika sign(y_true_diff) == sign(y_pred_diff_a1)
    abs_error   = np.abs(y_test_close.values - y_pred_price)
    dir_correct = np.sign(y_test_diff.values) == np.sign(y_pred_diff)

    df_predictions = pd.DataFrame(
        {
            "Current_Close"       : close_test,         # Close[t] — harga acuan
            "y_true_diff"         : y_test_diff.values, # selisih aktual
            "y_pred_diff_a1"      : y_pred_diff,        # selisih prediksi model
            "Target_Close_Actual" : y_test_close.values,# Close[t+1] aktual
            "Target_Close_Pred_a1": y_pred_price,       # Close[t] + y_pred_diff
            "Absolute_Error_a1"   : abs_error,          # |aktual - prediksi|
            "Direction_Correct_a1": dir_correct,        # True/False arah benar
        },
        index=df_test.index,
    )

    if save_predictions:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        df_predictions.to_csv(PREDICTIONS_A1_PATH)
        logger.info(f"  Prediksi A1 disimpan ke: {PREDICTIONS_A1_PATH}")

    # --- Langkah 7: Feature Importance (Top 10) ---
    importance_series = pd.Series(
        model_a1.feature_importances_,
        index=feature_cols_a1,
    ).sort_values(ascending=False)

    print(f"\n  TOP 10 Feature Importance (Skenario A1):")
    print(f"  {'Fitur':<20} {'Importance':>10}")
    print(f"  {'-'*32}")
    for feat, imp in importance_series.head(10).items():
        print(f"  {feat:<20} {imp:>10.4f}")

    return model_a1, df_predictions, metrics


# =============================================================================
# BAGIAN 4: ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    """
    Jalankan Fase 3 & 4:
        python src/models_xgb.py
    """
    print("\n" + "=" * 60)
    print("  FASE 3 & 4: DATA SPLITTING + XGBOOST SKENARIO A1")
    print("=" * 60)

    # --- Fase 3: Load & Split ---
    df = load_featured_dataset()
    df_train, df_test = split_train_test(df)

    print(f"\n  Ringkasan splitting:")
    print(f"  Train: {len(df_train):,} candle "
          f"({df_train.index.min().date()} ~ {df_train.index.max().date()})")
    print(f"  Test : {len(df_test):,} candle "
          f"({df_test.index.min().date()} ~ {df_test.index.max().date()})")
    print(f"  Rasio Train:Test = "
          f"{len(df_train)/len(df)*100:.1f}%:{len(df_test)/len(df)*100:.1f}%")

    # --- Fase 4: Skenario A1 ---
    model_a1, df_pred_a1, metrics_a1 = train_scenario_a1(
        df_train, df_test, save_predictions=True
    )

    # --- Pratinjau Prediksi ---
    print(f"\n  Pratinjau 10 prediksi pertama (Test Set 2025):")
    print(
        f"  {'Datetime':<22} {'Cur Close':>10} "
        f"{'True Diff':>10} {'Pred Diff':>10} "
        f"{'Act Close':>10} {'Pred Close':>11} {'Error':>8}"
    )
    print(f"  {'-'*84}")
    for dt, row in df_pred_a1.head(10).iterrows():
        err = row["Target_Close_Pred_a1"] - row["Target_Close_Actual"]
        print(
            f"  {str(dt):<22} "
            f"{row['Current_Close']:>10.2f} "
            f"{row['y_true_diff']:>+10.4f} "
            f"{row['y_pred_diff_a1']:>+10.4f} "
            f"{row['Target_Close_Actual']:>10.2f} "
            f"{row['Target_Close_Pred_a1']:>11.2f} "
            f"{err:>+8.2f}"
        )

    print(f"\n{'=' * 60}")
    print(f"  Fase 3 & 4 selesai (target=Diff, rekonstruksi harga).")
    print(f"  Prediksi tersimpan di: results/predictions_a1.csv")
    print(f"  Model A1 siap dibandingkan dengan A2 (makro) dan B (TimesFM).")
    print(f"{'=' * 60}\n")
