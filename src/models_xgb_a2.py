"""
models_xgb_a2.py
================
Modul untuk melatih dan mengevaluasi model XGBoost Skenario A2 (Fase 5).

Skenario A2 : XGBoost dengan Injeksi Fitur Makroekonomi
  - Fitur X  : OHLCV + Lag (t-1,2,3) + 10 sinyal Dev_ makro
  - Target y : Target_Diff (selisih stasioner, anti-ekstrapolasi)
  - Tuning   : GridSearchCV dengan TimeSeriesSplit(n_splits=3)
               agar tidak ada kebocoran waktu saat cross-validation

Penulis : Jason Daniel Tanubrata
Tanggal : 2026

Hipotesis yang diuji:
  Apakah injeksi fitur makro (Dev_CPI, Dev_NFP, ...) meningkatkan
  performa prediksi intraday XAU/USD dibanding model endogen saja (A1)?

Output:
  results/predictions_a2.csv  — format identik dengan predictions_a1.csv
  Terminal                    — metrik evaluasi + parameter terbaik GridSearch
"""

import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

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
PREDICTIONS_A2_PATH   = RESULTS_DIR   / "predictions_a2.csv"

# --- Konstanta Splitting ---
TRAIN_END  = "2024-12-31 23:59:59"
TEST_START = "2025-01-01 00:00:00"

# --- Grid Parameter untuk GridSearchCV (dioptimalkan untuk Sparse Features) ---
# Penambahan colsample_bytree dan min_child_weight mendorong model untuk
# sesekali menyembunyikan fitur OHLCV dominan sehingga fitur Dev_ yang
# jarang aktif (<0.05% candle) mendapat kesempatan dipelajari.
#
# Jumlah kombinasi: 2 x 2 x 2 x 2 x 2 = 32 kombinasi x 3 splits = 96 fit
PARAM_GRID = {
    "n_estimators"    : [100, 200],
    "learning_rate"   : [0.05, 0.1],
    "max_depth"       : [6, 8],        # Lebih dalam: tangkap pola langka makro
    "colsample_bytree": [0.7, 0.8],    # Sembunyikan beberapa OHLCV secara random
    "min_child_weight": [1, 3],        # Regularisasi untuk node dengan data sparse
}

# Parameter XGBoost yang TIDAK dimasukkan ke grid (tetap konstan)
XGB_BASE_PARAMS = {
    "subsample"  : 0.8,
    "random_state": 42,
    "n_jobs"     : -1,
    "tree_method": "hist",
    "verbosity"  : 0,
}

# Kolom yang DIBUANG dari X (bukan fitur — ini adalah target dan index)
EXCLUDE_FROM_FEATURES = {"Target_Close", "Target_Diff"}


# =============================================================================
# BAGIAN 1: LOAD DAN SPLITTING DATA
# =============================================================================

def load_and_split(
    filepath: str | Path = DATASET_FEATURED_PATH,
    train_end: str = TRAIN_END,
    test_start: str = TEST_START,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Membaca dataset_featured.csv dan memotongnya secara kronologis.

    Kembalian
    ---------
    tuple[pd.DataFrame, pd.DataFrame]
        (df_train, df_test) dengan DatetimeIndex.
    """
    filepath = Path(filepath)
    logger.info(f"Memuat dataset fitur dari: {filepath}")

    df = pd.read_csv(filepath, parse_dates=["Datetime"], index_col="Datetime")
    df = df.sort_index()

    df_train = df.loc[:train_end].copy()
    df_test  = df.loc[test_start:].copy()

    logger.info(f"  Total baris    : {len(df):,}")
    logger.info(
        f"  Train          : {df_train.index.min().date()} s/d "
        f"{df_train.index.max().date()} | {len(df_train):,} candle"
    )
    logger.info(
        f"  Test           : {df_test.index.min().date()} s/d "
        f"{df_test.index.max().date()}  | {len(df_test):,} candle"
    )

    assert df_train.index.max() < df_test.index.min(), (
        "ERROR: Ada overlap antara Train dan Test!"
    )
    return df_train, df_test


# =============================================================================
# BAGIAN 2: FUNGSI EVALUASI METRIK
# =============================================================================

def calculate_metrics(
    y_true_price: np.ndarray,
    y_pred_price: np.ndarray,
    y_true_diff : np.ndarray,
    y_pred_diff : np.ndarray,
    scenario_name: str = "Model",
) -> dict:
    """
    Hitung MAE, RMSE, MAPE (harga absolut) dan MDA (arah diff).

    MDA dihitung HANYA pada candle yang benar-benar bergerak (true_diff != 0)
    untuk menghindari bias dari candle flat.
    """
    mae  = mean_absolute_error(y_true_price, y_pred_price)
    rmse = np.sqrt(mean_squared_error(y_true_price, y_pred_price))

    mape_arr = np.abs(
        (y_true_price - y_pred_price)
        / np.where(y_true_price != 0, y_true_price, np.nan)
    )
    mape = np.nanmean(mape_arr) * 100

    mask_moving = y_true_diff != 0
    if mask_moving.sum() > 0:
        mda = np.mean(
            np.sign(y_true_diff[mask_moving])
            == np.sign(y_pred_diff[mask_moving])
        ) * 100
    else:
        mda = np.nan

    metrics = {"MAE": mae, "RMSE": rmse, "MAPE": mape, "MDA": mda}

    sep = "=" * 52
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
# BAGIAN 3: SKENARIO A2 — XGBOOST + FITUR MAKRO + GRIDSEARCHCV
# =============================================================================

def train_scenario_a2(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    param_grid: dict = PARAM_GRID,
    base_params: dict = XGB_BASE_PARAMS,
    save_predictions: bool = True,
) -> tuple[XGBRegressor, pd.DataFrame, dict]:
    """
    Melatih XGBoost Skenario A2 dengan SEMUA fitur (OHLCV + Lag + Dev_makro).

    Alur:
    1. X_A2 = semua kolom kecuali Target_Close & Target_Diff
    2. y    = Target_Diff (selisih stasioner)
    3. Tuning via GridSearchCV + TimeSeriesSplit(n_splits=3)
    4. Latih model terbaik pada seluruh data train
    5. Prediksi → rekonstruksi harga → evaluasi → simpan

    Mengapa TimeSeriesSplit?
      Cross-validation biasa (KFold) menggunakan data masa depan sebagai
      validation set, yang merupakan Data Leakage temporal. TimeSeriesSplit
      memastikan validation set selalu lebih baru dari training set,
      menjaga integritas urutan waktu.

    Parameter
    ----------
    df_train : pd.DataFrame
        Data latih dengan DatetimeIndex (Jan 2021 - Des 2024).
    df_test : pd.DataFrame
        Data uji dengan DatetimeIndex (Jan 2025 - Des 2025).
    param_grid : dict
        Grid parameter untuk GridSearchCV.
    base_params : dict
        Parameter XGBoost konstan (tidak di-tune).
    save_predictions : bool
        Jika True, simpan ke PREDICTIONS_A2_PATH.

    Kembalian
    ---------
    tuple[XGBRegressor, pd.DataFrame, dict]
        (best_model, df_predictions, metrics_dict)
    """
    logger.info("=" * 60)
    logger.info("SKENARIO A2: XGBoost + Fitur Makro (GridSearchCV)")
    logger.info("=" * 60)

    # --- Langkah 1: Tentukan kolom fitur (SEMUA, termasuk Dev_) ---
    feature_cols_a2 = [
        c for c in df_train.columns
        if c not in EXCLUDE_FROM_FEATURES
    ]

    dev_cols = [c for c in feature_cols_a2 if c.startswith("Dev_")]
    logger.info(
        f"  Jumlah fitur A2        : {len(feature_cols_a2)} "
        f"(termasuk {len(dev_cols)} kolom Dev_ makro)"
    )
    logger.info(f"  Kolom Dev_ yang dipakai: {dev_cols}")

    X_train = df_train[feature_cols_a2]
    X_test  = df_test[feature_cols_a2]

    y_train_diff  = df_train["Target_Diff"]
    y_test_diff   = df_test["Target_Diff"]
    y_test_close  = df_test["Target_Close"]
    close_test    = X_test["Close"].values

    logger.info(
        f"  Ukuran X_train         : {X_train.shape}"
    )
    logger.info(
        f"  Target training        : Target_Diff "
        f"(mean={y_train_diff.mean():.4f}, std={y_train_diff.std():.4f} USD)"
    )

    # --- Langkah 2: GridSearchCV dengan TimeSeriesSplit ---
    # TimeSeriesSplit(n_splits=3) membagi data train menjadi 3 fold berurutan:
    #   Fold 1: train=1/4 data, val=2/4 data
    #   Fold 2: train=1/2 data, val=3/4 data
    #   Fold 3: train=3/4 data, val=4/4 data
    # Total kombinasi: 2 x 2 x 2 = 8 grid x 3 splits = 24 fit
    tscv = TimeSeriesSplit(n_splits=3)

    total_combos = 1
    for v in param_grid.values():
        total_combos *= len(v)

    logger.info(
        f"  Memulai GridSearchCV: "
        f"{total_combos} kombinasi x {tscv.n_splits} splits = "
        f"{total_combos * tscv.n_splits} fit total..."
    )

    base_model = XGBRegressor(**base_params)
    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=tscv,
        scoring="neg_mean_absolute_error",  # minimasi MAE
        refit=True,                          # auto-refit pada best params
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train, y_train_diff)

    best_params = grid_search.best_params_
    best_score  = -grid_search.best_score_   # konversi dari neg_MAE ke MAE

    logger.info(f"  GridSearch selesai.")
    logger.info(f"  Parameter terbaik : {best_params}")
    logger.info(f"  Best CV MAE       : {best_score:.4f} USD (rata-rata 3 fold)")

    # Model terbaik sudah di-refit pada seluruh data train oleh GridSearchCV
    best_model = grid_search.best_estimator_

    # --- Langkah 3: Prediksi selisih harga pada test set ---
    y_pred_diff = best_model.predict(X_test)

    # --- Langkah 4: Rekonstruksi harga absolut ---
    y_pred_price = close_test + y_pred_diff

    logger.info(
        f"  Rekonstruksi harga: "
        f"[{y_pred_price.min():.2f}, {y_pred_price.max():.2f}] USD"
    )
    logger.info(
        f"  Rentang aktual    : "
        f"[{y_test_close.min():.2f}, {y_test_close.max():.2f}] USD"
    )

    # --- Langkah 5: Evaluasi metrik ---
    metrics = calculate_metrics(
        y_true_price=y_test_close.values,
        y_pred_price=y_pred_price,
        y_true_diff=y_test_diff.values,
        y_pred_diff=y_pred_diff,
        scenario_name="A2 - XGBoost + Makro (GridSearchCV)",
    )

    # --- Langkah 6: Feature Importance (Top 15 — termasuk Dev_) ---
    importance_series = pd.Series(
        best_model.feature_importances_,
        index=feature_cols_a2,
    ).sort_values(ascending=False)

    print(f"\n  TOP 15 Feature Importance (Skenario A2):")
    print(f"  {'Fitur':<22} {'Importance':>10}  {'Tipe':>12}")
    print(f"  {'-'*48}")
    for feat, imp in importance_series.head(15).items():
        tipe = "MAKRO" if feat.startswith("Dev_") else "endogen"
        print(f"  {feat:<22} {imp:>10.4f}  {tipe:>12}")

    # --- Langkah 7: Simpan prediksi (format identik A1) ---
    abs_error   = np.abs(y_test_close.values - y_pred_price)
    dir_correct = np.sign(y_test_diff.values) == np.sign(y_pred_diff)

    df_predictions = pd.DataFrame(
        {
            "Current_Close"       : close_test,
            "y_true_diff"         : y_test_diff.values,
            "y_pred_diff_a2"      : y_pred_diff,
            "Target_Close_Actual" : y_test_close.values,
            "Target_Close_Pred_a2": y_pred_price,
            "Absolute_Error_a2"   : abs_error,
            "Direction_Correct_a2": dir_correct,
        },
        index=df_test.index,
    )

    if save_predictions:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        df_predictions.to_csv(PREDICTIONS_A2_PATH)
        logger.info(f"  Prediksi A2 disimpan ke: {PREDICTIONS_A2_PATH}")

    return best_model, df_predictions, metrics


# =============================================================================
# BAGIAN 4: PERBANDINGAN A1 vs A2
# =============================================================================

def compare_with_a1(
    metrics_a2: dict,
    predictions_a1_path: str | Path | None = None,
) -> None:
    """
    Membandingkan metrik A2 dengan A1 dari file predictions_a1.csv.
    Mencetak tabel perbandingan ke terminal.

    Parameter
    ----------
    metrics_a2 : dict
        Metrik evaluasi Skenario A2 (MAE, RMSE, MAPE, MDA).
    predictions_a1_path : str | Path | None
        Path ke predictions_a1.csv. Jika None, gunakan default.
    """
    if predictions_a1_path is None:
        predictions_a1_path = RESULTS_DIR / "predictions_a1.csv"

    predictions_a1_path = Path(predictions_a1_path)
    if not predictions_a1_path.exists():
        print("\n  [INFO] predictions_a1.csv tidak ditemukan. Skip perbandingan.")
        return

    df_a1 = pd.read_csv(predictions_a1_path, index_col="Datetime")

    # Hitung ulang metrik A1 dari file CSV
    y_true_close = df_a1["Target_Close_Actual"].values
    y_pred_price_a1 = df_a1["Target_Close_Pred_a1"].values
    y_true_diff  = df_a1["y_true_diff"].values
    y_pred_diff_a1 = df_a1["y_pred_diff_a1"].values

    mae_a1  = mean_absolute_error(y_true_close, y_pred_price_a1)
    rmse_a1 = np.sqrt(mean_squared_error(y_true_close, y_pred_price_a1))
    mape_a1 = np.nanmean(np.abs(
        (y_true_close - y_pred_price_a1)
        / np.where(y_true_close != 0, y_true_close, np.nan)
    )) * 100
    mask = y_true_diff != 0
    mda_a1 = np.mean(np.sign(y_true_diff[mask]) == np.sign(y_pred_diff_a1[mask])) * 100

    metrics_a1 = {"MAE": mae_a1, "RMSE": rmse_a1, "MAPE": mape_a1, "MDA": mda_a1}

    # Tabel perbandingan
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  PERBANDINGAN A1 (Baseline) vs A2 (+ Makro)")
    print(sep)
    print(f"  {'Metrik':<8} {'A1 Baseline':>14} {'A2 + Makro':>14} {'Delta':>12} {'Lebih Baik':>10}")
    print(f"  {'-'*60}")

    for metric in ["MAE", "RMSE", "MAPE", "MDA"]:
        v1 = metrics_a1[metric]
        v2 = metrics_a2[metric]
        delta = v2 - v1

        # Untuk MAE/RMSE/MAPE: lebih kecil = lebih baik (delta negatif = A2 menang)
        # Untuk MDA: lebih besar = lebih baik (delta positif = A2 menang)
        if metric == "MDA":
            better = "[WIN] A2" if delta > 0 else ("= Sama" if delta == 0 else "[L]  A1")
        else:
            better = "[WIN] A2" if delta < 0 else ("= Sama" if delta == 0 else "[L]  A1")

        unit = "%" if metric in ("MAPE", "MDA") else "USD"
        print(
            f"  {metric:<8} {v1:>12.4f}{unit[0]:1}  "
            f"{v2:>12.4f}{unit[0]:1}  "
            f"{delta:>+10.4f}   {better:>8}"
        )

    print(sep)

    # Ringkasan
    mda_delta = metrics_a2["MDA"] - metrics_a1["MDA"]
    mae_delta_pct = (metrics_a2["MAE"] - metrics_a1["MAE"]) / metrics_a1["MAE"] * 100
    print(f"\n  Ringkasan:")
    print(f"  MDA   : {metrics_a1['MDA']:.2f}% (A1) -> {metrics_a2['MDA']:.2f}% (A2) | D = {mda_delta:+.2f}%")
    print(f"  MAE   : {metrics_a1['MAE']:.4f} (A1) -> {metrics_a2['MAE']:.4f} (A2) | D = {mae_delta_pct:+.1f}%")
    print(sep)


# =============================================================================
# BAGIAN 5: EVALUASI EVENT-DRIVEN (Khusus Jam Rilis Berita)
# =============================================================================

DEV_COLUMNS = [
    "Dev_CPI", "Dev_PPI", "Dev_GDP", "Dev_FedRate", "Dev_ADP",
    "Dev_NFP", "Dev_Jobless", "Dev_Earnings", "Dev_ISM", "Dev_Retail",
]


def evaluate_event_driven(
    df_test_full: pd.DataFrame,
    df_pred_a2: pd.DataFrame,
    predictions_a1_path: str | Path | None = None,
) -> None:
    """
    Evaluasi khusus pada candle yang memiliki rilis berita makro aktif.

    Definisi 'event candle': baris di mana setidaknya satu kolom Dev_...
    TIDAK bernilai 0.0. Ini adalah candle tepat saat rilis berita,
    di mana fitur makro seharusnya memberikan sinyal tambahan yang tidak
    dimiliki oleh Skenario A1 (baseline endogen).

    Membandingkan MDA dan MAE A1 vs A2 khusus pada subset ini untuk
    membuktikan apakah injeksi makro memberikan keunggulan saat market shock.

    Parameter
    ----------
    df_test_full : pd.DataFrame
        Data test set asli (dari dataset_featured.csv) dengan kolom Dev_.
        Index: DatetimeIndex identik dengan df_pred_a2.
    df_pred_a2 : pd.DataFrame
        DataFrame prediksi A2 (dari train_scenario_a2).
    predictions_a1_path : str | Path | None
        Path ke predictions_a1.csv. Jika None, gunakan default.
    """
    if predictions_a1_path is None:
        predictions_a1_path = RESULTS_DIR / "predictions_a1.csv"

    predictions_a1_path = Path(predictions_a1_path)
    if not predictions_a1_path.exists():
        print("\n  [SKIP] predictions_a1.csv tidak ditemukan. Skip event-driven eval.")
        return

    # --- Identifikasi event candle (setidaknya satu Dev_ != 0.0) ---
    # Gunakan kolom Dev_ dari df_test_full (data asli, bukan hasil prediksi)
    dev_cols_available = [c for c in DEV_COLUMNS if c in df_test_full.columns]
    mask_event = (df_test_full[dev_cols_available] != 0.0).any(axis=1)
    event_idx  = df_test_full.index[mask_event]

    n_total  = len(df_test_full)
    n_events = len(event_idx)
    print(f"\n  Event candle teridentifikasi: {n_events} dari {n_total:,} "
          f"({n_events/n_total*100:.2f}% dari total test set)")

    if n_events == 0:
        print("  [SKIP] Tidak ada event candle di test set.")
        return

    # --- Subset prediksi A2 pada event candle ---
    df_a2_ev = df_pred_a2.loc[df_pred_a2.index.isin(event_idx)]

    # --- Load dan subset prediksi A1 pada event candle ---
    df_a1 = pd.read_csv(predictions_a1_path, index_col="Datetime",
                        parse_dates=True)
    df_a1_ev = df_a1.loc[df_a1.index.isin(event_idx)]

    # Selaraskan index agar sama persis
    common_idx = df_a2_ev.index.intersection(df_a1_ev.index)
    df_a2_ev   = df_a2_ev.loc[common_idx]
    df_a1_ev   = df_a1_ev.loc[common_idx]

    print(f"  Candle dianalisis (A1 & A2 sama): {len(common_idx)}")

    # --- Hitung metrik pada event candle ---
    def _metrics_event(y_true_p, y_pred_p, y_true_d, y_pred_d):
        mae  = mean_absolute_error(y_true_p, y_pred_p)
        mask = y_true_d != 0
        mda  = np.mean(
            np.sign(y_true_d[mask]) == np.sign(y_pred_d[mask])
        ) * 100 if mask.sum() > 0 else np.nan
        return mae, mda

    mae_a1_ev, mda_a1_ev = _metrics_event(
        df_a1_ev["Target_Close_Actual"].values,
        df_a1_ev["Target_Close_Pred_a1"].values,
        df_a1_ev["y_true_diff"].values,
        df_a1_ev["y_pred_diff_a1"].values,
    )
    mae_a2_ev, mda_a2_ev = _metrics_event(
        df_a2_ev["Target_Close_Actual"].values,
        df_a2_ev["Target_Close_Pred_a2"].values,
        df_a2_ev["y_true_diff"].values,
        df_a2_ev["y_pred_diff_a2"].values,
    )

    # --- Cetak tabel evaluasi event-driven ---
    sep = "=" * 62
    print(f"\n{sep}")
    print(f"  EVALUASI EVENT-DRIVEN: Khusus Jam Rilis Berita Makro")
    print(f"  (Candle di mana setidaknya 1 kolom Dev_ != 0.0)")
    print(sep)
    print(f"  {'Metrik':<12} {'A1 Baseline':>14} {'A2 + Makro':>14} "
          f"{'Delta':>10}  {'Kesimpulan':<12}")
    print(f"  {'-'*58}")

    for label, v1, v2 in [
        ("MAE (USD)", mae_a1_ev, mae_a2_ev),
        ("MDA (%)" , mda_a1_ev, mda_a2_ev),
    ]:
        delta = v2 - v1
        is_mae = "MAE" in label
        if is_mae:
            wins = "A2 UNGGUL" if delta < 0 else ("= SAMA" if delta == 0 else "A1 lebih baik")
        else:
            wins = "A2 UNGGUL" if delta > 0 else ("= SAMA" if delta == 0 else "A1 lebih baik")
        print(f"  {label:<12} {v1:>14.4f} {v2:>14.4f} {delta:>+10.4f}  {wins}")

    print(sep)

    # Rincian per indikator
    print(f"\n  Rincian candle event per indikator makro:")
    print(f"  {'Indikator':<15} {'N Candle':>10} "
          f"{'MDA A1':>10} {'MDA A2':>10} {'Delta MDA':>10}")
    print(f"  {'-'*58}")
    for col in dev_cols_available:
        idx_col = df_test_full.index[(df_test_full[col] != 0.0)]
        idx_col = idx_col[idx_col.isin(common_idx)]
        if len(idx_col) == 0:
            continue
        sub_a1  = df_a1_ev.loc[df_a1_ev.index.isin(idx_col)]
        sub_a2  = df_a2_ev.loc[df_a2_ev.index.isin(idx_col)]
        if len(sub_a1) == 0:
            continue
        _, mda1 = _metrics_event(
            sub_a1["Target_Close_Actual"].values,
            sub_a1["Target_Close_Pred_a1"].values,
            sub_a1["y_true_diff"].values,
            sub_a1["y_pred_diff_a1"].values,
        )
        _, mda2 = _metrics_event(
            sub_a2["Target_Close_Actual"].values,
            sub_a2["Target_Close_Pred_a2"].values,
            sub_a2["y_true_diff"].values,
            sub_a2["y_pred_diff_a2"].values,
        )
        d = mda2 - mda1
        print(f"  {col:<15} {len(idx_col):>10} "
              f"{mda1:>9.1f}% {mda2:>9.1f}% {d:>+9.1f}%")
    print(sep)


# =============================================================================
# BAGIAN 6: ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    """
    Jalankan Fase 5 (Skenario A2):
        python src/models_xgb_a2.py
    """
    print("\n" + "=" * 60)
    print("  FASE 5: XGBOOST SKENARIO A2 — INJEKSI FITUR MAKRO")
    print("=" * 60)

    # --- Load & Split ---
    df_train, df_test = load_and_split()

    total_combos = 1
    for v in PARAM_GRID.values():
        total_combos *= len(v)
    print(f"\n  Grid yang diuji ({total_combos} kombinasi x 3 splits = {total_combos*3} fit):")
    for k, v in PARAM_GRID.items():
        print(f"    {k}: {v}")
    print(f"  TimeSeriesSplit: n_splits=3 (anti-leakage temporal)")

    # --- Train Skenario A2 ---
    best_model_a2, df_pred_a2, metrics_a2 = train_scenario_a2(
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
    for dt, row in df_pred_a2.head(10).iterrows():
        err = row["Target_Close_Pred_a2"] - row["Target_Close_Actual"]
        print(
            f"  {str(dt):<22} "
            f"{row['Current_Close']:>10.2f} "
            f"{row['y_true_diff']:>+10.4f} "
            f"{row['y_pred_diff_a2']:>+10.4f} "
            f"{row['Target_Close_Actual']:>10.2f} "
            f"{row['Target_Close_Pred_a2']:>11.2f} "
            f"{err:>+8.2f}"
        )

    # --- Perbandingan Global A1 vs A2 ---
    compare_with_a1(metrics_a2)

    # --- Evaluasi Event-Driven: khusus jam rilis berita ---
    evaluate_event_driven(df_test, df_pred_a2)

    print(f"\n{'=' * 60}")
    print(f"  Fase 5 selesai.")
    print(f"  Prediksi tersimpan di: results/predictions_a2.csv")
    print(f"  Langkah berikutnya: Fase 6 — Skenario B (TimesFM Zero-Shot)")
    print(f"{'=' * 60}\n")
