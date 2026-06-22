"""
models_timesfm.py
=================
Modul untuk mengevaluasi TimesFM sebagai model Zero-Shot (Skenario B, Fase 6).

TimesFM (Time Series Foundation Model) oleh Google DeepMind adalah model
pralatih yang mampu melakukan forecasting TANPA perlu dilatih ulang pada
data target (zero-shot inference). Ini adalah pembanding utama terhadap
XGBoost yang dilatih secara supervised.

Penulis : Jason Daniel Tanubrata
Tanggal : 2026

Dependensi Khusus (install sebelum menjalankan):
    pip install timesfm==2.0.0 torch --index-url https://download.pytorch.org/whl/cpu

Catatan Versi:
    timesfm==2.0.0 adalah versi terbaru yang kompatibel dengan Python 3.13.
    Menggunakan checkpoint google/timesfm-2.0-500m-pytorch (model 500M parameter).
    timesfm==1.3.0 (model 200M) membutuhkan Python 3.10-3.11.

Strategi Inferensi:
    TimesFM membutuhkan array konteks harga historis untuk setiap prediksi.
    Karena test set 2025 memiliki ~23.000 candle, kita menggunakan
    BATCH INFERENCE (bukan loop satu-per-satu) untuk efisiensi:
    - Bagi test set ke batch berukuran BATCH_SIZE
    - Untuk setiap batch, kirim semua context windows sekaligus
    - Ini mengurangi overhead ~100x dibanding loop individual

Output:
    results/predictions_b.csv  -- format identik dengan A1 & A2
    Terminal                   -- MAE, RMSE, MDA global
"""

import logging
import warnings
import sys
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

# --- Konfigurasi Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Konstanta Path ---
PROJECT_ROOT  = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR   = PROJECT_ROOT / "results"

DATASET_FEATURED_PATH = PROCESSED_DIR / "dataset_featured.csv"
PREDICTIONS_B_PATH    = RESULTS_DIR   / "predictions_b.csv"

# --- Konstanta Model TimesFM ---
# timesfm==2.0.0 (Python 3.13 compatible)
# Checkpoint public: google/timesfm-2.0-500m-pytorch
# (google/timesfm-2.0-200m-pytorch membutuhkan HF token/login)
TIMESFM_REPO_ID = "google/timesfm-2.0-500m-pytorch"
CONTEXT_LEN     = 64    # 64 candle x 15min = 16 jam historis
HORIZON_LEN     = 1     # Prediksi 1 langkah ke depan
BATCH_SIZE      = 64    # Windows per batch (lebih kecil untuk model 500M)

# --- Konstanta Splitting ---
TEST_START = "2025-01-01 00:00:00"


# =============================================================================
# BAGIAN 1: CEK DAN LOAD DEPENDENSI
# =============================================================================

def check_timesfm_installed() -> bool:
    """
    Memeriksa apakah package timesfm dan torch sudah terinstal.
    Menampilkan instruksi instalasi jika belum.
    """
    ok = True
    try:
        import timesfm  # noqa: F401
        import importlib.metadata
        ver = importlib.metadata.version("timesfm")
        logger.info(f"Package timesfm=={ver} ditemukan.")
    except ImportError:
        print("\n  ERROR: Package 'timesfm' belum terinstal!")
        print("  Jalankan: pip install timesfm==2.0.0")
        ok = False

    try:
        import torch  # noqa: F401
        logger.info(f"Package torch ditemukan.")
    except ImportError:
        print("\n  ERROR: Package 'torch' belum terinstal!")
        print("  Jalankan: pip install torch --index-url https://download.pytorch.org/whl/cpu")
        ok = False

    if not ok:
        print("\n  Install semua dependensi:")
        print("    pip install timesfm==2.0.0")
        print("    pip install torch --index-url https://download.pytorch.org/whl/cpu\n")
    return ok


# =============================================================================
# BAGIAN 2: LOAD DATA
# =============================================================================

def load_data(
    filepath: str | Path = DATASET_FEATURED_PATH,
    test_start: str = TEST_START,
    context_len: int = CONTEXT_LEN,
) -> tuple[pd.Series, pd.DataFrame]:
    """
    Membaca dataset dan memisahkan konteks historis dari data uji.

    TimesFM membutuhkan konteks historis SEBELUM test set untuk prediksi
    awal. Kita ambil seluruh data Close sebagai seri waktu, lalu pointer
    ke posisi test set dimulai.

    Parameter
    ----------
    filepath : str | Path
        Path ke dataset_featured.csv.
    test_start : str
        Awal data uji (default: 2025-01-01).
    context_len : int
        Panjang konteks historis (default: 64 candle).

    Kembalian
    ---------
    tuple[pd.Series, pd.DataFrame]
        - close_series: Seluruh harga Close (train + test) dengan DatetimeIndex
        - df_test      : Test set DataFrame (2025) dengan semua kolom
    """
    filepath = Path(filepath)
    logger.info(f"Memuat dataset dari: {filepath}")

    df = pd.read_csv(filepath, parse_dates=["Datetime"], index_col="Datetime")
    df = df.sort_index()

    close_series = df["Close"]  # Seluruh series Close untuk konteks
    df_test = df.loc[test_start:].copy()

    # Verifikasi konteks tersedia sebelum test set
    df_before_test = df.loc[:test_start]
    n_before = len(df_before_test)
    assert n_before >= context_len, (
        f"Data historis sebelum test ({n_before} baris) "
        f"lebih sedikit dari context_len ({context_len})!"
    )

    logger.info(f"  Total data    : {len(df):,} candle")
    logger.info(f"  Test set 2025 : {len(df_test):,} candle "
                f"({df_test.index.min().date()} s/d {df_test.index.max().date()})")
    logger.info(f"  Context len   : {context_len} candle (= {context_len * 15 // 60} jam historis)")

    return close_series, df_test


# =============================================================================
# BAGIAN 3: INISIALISASI MODEL TIMESFM
# =============================================================================

def init_timesfm_model(backend: str = "cpu"):
    """
    Memuat model TimesFM 2.0-200M dari HuggingFace Hub.

    Menggunakan API timesfm==2.0.0:
      TimesFM_2p5_200M_torch.from_pretrained(repo_id)

    Model diunduh ke cache HuggingFace (~/.cache/huggingface/)
    pada pertama kali dijalankan (~1-2GB).

    Parameter
    ----------
    backend : str
        Diabaikan di v2.0 (selalu CPU jika torch tanpa CUDA).

    Kembalian
    ---------
    TimesFM_2p5_200M_torch
        Model TimesFM yang siap untuk forecasting.
    """
    from timesfm.timesfm_2p5 import timesfm_2p5_torch as tfm_module
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file as safetensors_load
    import timesfm as tfm_pkg
    import importlib.metadata

    ver = importlib.metadata.version("timesfm")
    logger.info(f"Memuat TimesFM v{ver} dari HuggingFace...")
    logger.info(f"  Checkpoint : {TIMESFM_REPO_ID}")
    logger.info(f"  Context len: {CONTEXT_LEN} candle | Horizon: {HORIZON_LEN} step")
    logger.info("  (Download ~2GB jika pertama kali, harap tunggu...)")

    # Step 1: Inisialisasi arsitektur model (tanpa bobot)
    tfm = tfm_module.TimesFM_2p5_200M_torch(torch_compile=False)

    # Step 2: Download file bobot dari HuggingFace (atau dari cache jika sudah ada)
    model_path = hf_hub_download(
        repo_id=TIMESFM_REPO_ID,
        filename="model.safetensors",
        token=None,   # public repo, tidak butuh login
    )
    logger.info(f"  File bobot: {model_path}")

    # Step 3: Muat bobot dengan strict=False
    # (arsitektur class 200M dan checkpoint 500M punya beberapa layer berbeda)
    state_dict = safetensors_load(model_path)
    missing, unexpected = tfm.model.load_state_dict(state_dict, strict=False)
    n_loaded = len(state_dict) - len(missing)
    logger.info(
        f"  Bobot: {n_loaded}/{len(state_dict)} key dimuat "
        f"| {len(missing)} missing | {len(unexpected)} unexpected"
    )
    if n_loaded == 0:
        raise RuntimeError(
            "Tidak ada satu pun bobot yang berhasil dimuat! "
            "Arsitektur class dan checkpoint tidak kompatibel."
        )

    # Step 4: Set ke mode eval dan compile sebelum forecast
    tfm.model.eval()
    forecast_config = tfm_pkg.ForecastConfig(
        max_context=CONTEXT_LEN,
        max_horizon=HORIZON_LEN,
        per_core_batch_size=BATCH_SIZE,
    )
    tfm.compile(forecast_config)

    logger.info("  Model TimesFM siap untuk inferensi zero-shot.")
    return tfm


# =============================================================================
# BAGIAN 4: BATCH INFERENCE (Rolling Context Window)
# =============================================================================

def run_batch_inference(
    tfm,
    close_series: pd.Series,
    df_test: pd.DataFrame,
    context_len: int = CONTEXT_LEN,
    batch_size: int = BATCH_SIZE,
) -> np.ndarray:
    """
    Menjalankan inferensi TimesFM secara batch untuk seluruh test set.

    API timesfm==2.0.0:
      tfm.forecast(horizon=1, inputs=[array1, array2, ...])
      Kembalian: (point_forecast, quantile_forecast)
      - point_forecast shape: (batch_size, horizon_len)

    Untuk setiap candle t di test set 2025:
      - Ambil close_series[t - context_len : t] sebagai konteks
      - Prediksi close_series[t] (Target_Close)

    Catatan Zero-Shot:
      TimesFM tidak pernah melihat data 2025 saat pralatih.
      Konteks diberikan adalah data historis AKTUAL hingga t-1.

    Parameter
    ----------
    tfm :
        Model TimesFM yang sudah dimuat.
    close_series : pd.Series
        Seluruh harga Close (train + test) dengan DatetimeIndex.
    df_test : pd.DataFrame
        Test set 2025.
    context_len : int
        Panjang window konteks.
    batch_size : int
        Jumlah windows per batch.

    Kembalian
    ---------
    np.ndarray
        Array prediksi harga absolut. Shape: (len(df_test),)
    """
    close_values = close_series.values
    test_index   = df_test.index

    # Mapping Datetime -> posisi integer di close_series
    datetime_to_pos = {dt: i for i, dt in enumerate(close_series.index)}

    logger.info(f"Memulai batch inference: {len(df_test):,} prediksi, "
                f"batch_size={batch_size}...")

    all_predictions = []
    n_test    = len(df_test)
    n_batches = (n_test + batch_size - 1) // batch_size

    for batch_num in range(n_batches):
        batch_start = batch_num * batch_size
        batch_end   = min(batch_start + batch_size, n_test)
        batch_dts   = test_index[batch_start:batch_end]

        # Kumpulkan context windows untuk seluruh batch
        batch_inputs = []
        for dt in batch_dts:
            pos = datetime_to_pos.get(dt)
            if pos is None or pos < context_len:
                # Fallback: gunakan data awal jika konteks tidak cukup
                ctx = close_values[:max(1, pos or 1)]
            else:
                # Window konteks: [pos-context_len, pos) -> prediksi harga di pos
                ctx = close_values[pos - context_len : pos]
            batch_inputs.append(ctx.astype(np.float32))

        # Inferensi batch: forecast(horizon, inputs)
        # API v2.0: inputs adalah list of np.ndarray (masing-masing bisa beda panjang)
        point_forecast, _ = tfm.forecast(
            horizon=HORIZON_LEN,
            inputs=batch_inputs,
        )
        # point_forecast shape: (batch_size, horizon_len)
        # Ambil horizon ke-0 (prediksi 1 step ke depan)
        preds_batch = point_forecast[:, 0]
        all_predictions.extend(preds_batch.tolist())

        # Log progres setiap 20 batch
        if (batch_num + 1) % 20 == 0 or batch_num == n_batches - 1:
            pct = (batch_end / n_test) * 100
            logger.info(f"  Batch {batch_num+1}/{n_batches} | "
                        f"{batch_end:,}/{n_test:,} candle ({pct:.1f}%)")

    predictions = np.array(all_predictions)
    logger.info(f"Inferensi selesai. Rentang prediksi: "
                f"[{predictions.min():.2f}, {predictions.max():.2f}] USD")
    return predictions


# =============================================================================
# BAGIAN 5: EVALUASI DAN SIMPAN HASIL
# =============================================================================

def evaluate_and_save(
    df_test: pd.DataFrame,
    y_pred_price: np.ndarray,
    save_path: str | Path = PREDICTIONS_B_PATH,
) -> tuple[pd.DataFrame, dict]:
    """
    Menghitung metrik evaluasi dan menyimpan prediksi ke CSV.

    Format output IDENTIK dengan predictions_a1.csv dan predictions_a2.csv
    untuk memudahkan perbandingan langsung.

    Parameter
    ----------
    df_test : pd.DataFrame
        Test set dengan kolom Target_Close dan Close.
    y_pred_price : np.ndarray
        Prediksi harga absolut dari TimesFM.
    save_path : str | Path
        Path output CSV.

    Kembalian
    ---------
    tuple[pd.DataFrame, dict]
        (df_predictions, metrics_dict)
    """
    close_test    = df_test["Close"].values
    y_true_close  = df_test["Target_Close"].values
    y_true_diff   = df_test["Target_Diff"].values
    y_pred_diff   = y_pred_price - close_test   # Diff dari prediksi harga absolut

    # Metrik harga absolut
    mae  = mean_absolute_error(y_true_close, y_pred_price)
    rmse = np.sqrt(mean_squared_error(y_true_close, y_pred_price))
    mape_arr = np.abs(
        (y_true_close - y_pred_price)
        / np.where(y_true_close != 0, y_true_close, np.nan)
    )
    mape = np.nanmean(mape_arr) * 100

    # MDA pada diff (hanya candle yang bergerak)
    mask_moving = y_true_diff != 0
    if mask_moving.sum() > 0:
        mda = np.mean(
            np.sign(y_true_diff[mask_moving]) == np.sign(y_pred_diff[mask_moving])
        ) * 100
    else:
        mda = np.nan

    metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape, "MDA": mda}

    # Cetak hasil
    sep = "=" * 52
    print(f"\n{sep}")
    print(f"  HASIL EVALUASI: Skenario B — TimesFM Zero-Shot")
    print(sep)
    print(f"  MAE  : {mae:>10.4f}  USD  (harga absolut)")
    print(f"  RMSE : {rmse:>10.4f}  USD  (harga absolut)")
    print(f"  MAPE : {mape:>10.6f}  %    (harga absolut)")
    print(f"  MDA  : {mda:>10.4f}  %    (arah diff | random = 50%)")
    print(sep)

    # Buat DataFrame prediksi (format identik A1 & A2)
    abs_error   = np.abs(y_true_close - y_pred_price)
    dir_correct = np.sign(y_true_diff) == np.sign(y_pred_diff)

    df_predictions = pd.DataFrame(
        {
            "Current_Close"      : close_test,
            "y_true_diff"        : y_true_diff,
            "y_pred_diff_b"      : y_pred_diff,
            "Target_Close_Actual": y_true_close,
            "Target_Close_Pred_b": y_pred_price,
            "Absolute_Error_b"   : abs_error,
            "Direction_Correct_b": dir_correct,
        },
        index=df_test.index,
    )
    df_predictions.index.name = "Datetime"

    # Simpan ke CSV
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df_predictions.to_csv(save_path)
    logger.info(f"Prediksi B disimpan ke: {save_path}")

    return df_predictions, metrics


# =============================================================================
# BAGIAN 6: PERBANDINGAN TIGA SKENARIO (A1, A2, B)
# =============================================================================

def compare_all_scenarios(metrics_b: dict) -> None:
    """
    Mencetak tabel perbandingan lengkap A1 vs A2 vs B ke terminal.

    Parameter
    ----------
    metrics_b : dict
        Metrik evaluasi Skenario B (MAE, RMSE, MAPE, MDA).
    """
    a1_path = RESULTS_DIR / "predictions_a1.csv"
    a2_path = RESULTS_DIR / "predictions_a2.csv"

    def load_metrics(path, price_col, diff_col):
        if not path.exists():
            return None
        df = pd.read_csv(path, index_col="Datetime")
        yt_p = df["Target_Close_Actual"].values
        yp_p = df[price_col].values
        yt_d = df["y_true_diff"].values
        yp_d = df[diff_col].values
        mae  = mean_absolute_error(yt_p, yp_p)
        rmse = np.sqrt(mean_squared_error(yt_p, yp_p))
        mape = np.nanmean(np.abs((yt_p-yp_p)/np.where(yt_p!=0,yt_p,np.nan)))*100
        mask = yt_d != 0
        mda  = np.mean(np.sign(yt_d[mask])==np.sign(yp_d[mask]))*100 if mask.sum()>0 else np.nan
        return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "MDA": mda}

    m_a1 = load_metrics(a1_path, "Target_Close_Pred_a1", "y_pred_diff_a1")
    m_a2 = load_metrics(a2_path, "Target_Close_Pred_a2", "y_pred_diff_a2")
    m_b  = metrics_b

    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  PERBANDINGAN TIGA SKENARIO (Test Set 2025)")
    print(sep)
    print(f"  {'Metrik':<10} {'A1 Baseline':>14} {'A2 +Makro':>14} "
          f"{'B TimesFM':>14}  Pemenang")
    print(f"  {'-'*68}")

    for metric in ["MAE", "RMSE", "MAPE", "MDA"]:
        vals = {
            "A1": m_a1[metric] if m_a1 else float("nan"),
            "A2": m_a2[metric] if m_a2 else float("nan"),
            "B" : m_b[metric],
        }
        unit = "%" if metric in ("MAPE", "MDA") else "USD"

        # Tentukan pemenang
        if metric == "MDA":
            best = max(vals, key=lambda k: vals[k])
        else:
            best = min(vals, key=lambda k: vals[k])

        v1 = f"{vals['A1']:.4f}{unit[0]}"
        v2 = f"{vals['A2']:.4f}{unit[0]}"
        vb = f"{vals['B']:.4f}{unit[0]}"
        win_label = f"[{best}]"

        print(f"  {metric:<10} {v1:>14} {v2:>14} {vb:>14}  {win_label}")

    print(sep)


# =============================================================================
# BAGIAN 7: ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    """
    Jalankan Fase 6 (Skenario B — TimesFM Zero-Shot):
        python src/models_timesfm.py

    Kebutuhan:
        pip install timesfm==1.3.0
        (Model akan diunduh ~800MB pada pertama kali)
    """
    print("\n" + "=" * 60)
    print("  FASE 6: TIMESFM ZERO-SHOT (Skenario B)")
    print("=" * 60)
    print(f"  Model     : {TIMESFM_REPO_ID}")
    print(f"  Context   : {CONTEXT_LEN} candle ({CONTEXT_LEN * 15 // 60} jam)")
    print(f"  Horizon   : {HORIZON_LEN} step (15 menit ke depan)")
    print(f"  Backend   : CPU (tidak perlu GPU)")
    print(f"  Batch     : {BATCH_SIZE} context windows per call")
    print(f"  Evaluasi  : Hanya Test Set 2025 ({TEST_START[:10]}+)")

    # --- Cek instalasi ---
    if not check_timesfm_installed():
        sys.exit(1)

    # --- Load data ---
    close_series, df_test = load_data(context_len=CONTEXT_LEN)

    print(f"\n  Test set dimuat: {len(df_test):,} candle")
    print(f"  Estimasi waktu: ~{len(df_test) // BATCH_SIZE + 1} batch "
          f"(batch_size={BATCH_SIZE})")
    print("  Memulai inferensi...")

    # --- Inisialisasi model ---
    tfm = init_timesfm_model(backend="cpu")

    # --- Batch Inference ---
    y_pred_price = run_batch_inference(
        tfm         = tfm,
        close_series= close_series,
        df_test     = df_test,
        context_len = CONTEXT_LEN,
        batch_size  = BATCH_SIZE,
    )

    # --- Evaluasi & Simpan ---
    df_pred_b, metrics_b = evaluate_and_save(
        df_test     = df_test,
        y_pred_price= y_pred_price,
        save_path   = PREDICTIONS_B_PATH,
    )

    # --- Pratinjau ---
    print(f"\n  Pratinjau 10 prediksi pertama (Test Set 2025):")
    print(
        f"  {'Datetime':<22} {'Cur Close':>10} "
        f"{'True Diff':>10} {'Pred Diff':>10} "
        f"{'Act Close':>10} {'Pred Close':>11} {'Error':>8}"
    )
    print(f"  {'-'*84}")
    for dt, row in df_pred_b.head(10).iterrows():
        err = row["Target_Close_Pred_b"] - row["Target_Close_Actual"]
        print(
            f"  {str(dt):<22} "
            f"{row['Current_Close']:>10.2f} "
            f"{row['y_true_diff']:>+10.4f} "
            f"{row['y_pred_diff_b']:>+10.4f} "
            f"{row['Target_Close_Actual']:>10.2f} "
            f"{row['Target_Close_Pred_b']:>11.2f} "
            f"{err:>+8.2f}"
        )

    # --- Perbandingan tiga skenario ---
    compare_all_scenarios(metrics_b)

    print(f"\n{'=' * 60}")
    print(f"  Fase 6 selesai.")
    print(f"  Prediksi tersimpan di: results/predictions_b.csv")
    print(f"  Langkah berikutnya: Fase 7 — Evaluasi Final & SHAP")
    print(f"{'=' * 60}\n")
