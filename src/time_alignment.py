"""
time_alignment.py
=================
Modul untuk melakukan Time-Alignment antara data harga XAU/USD (15-menit)
dan data makroekonomi yang sudah bersih (macro_news_clean.csv).

Fase 1 (Lanjutan): Time-Alignment & Finalisasi Dataset
Penulis : Jason Daniel Tanubrata
Tanggal : 2026

Pipeline (urutan eksekusi dalam align_and_merge()):
  1. Load XAUUSD.csv            -> load_ohlcv()
  2. Load macro_news_clean.csv  -> load_macro_clean()
  3. Pivot makro -> kolom Dev_  -> pivot_macro_to_deviation_columns()
  4. Floor timestamp ke 15 menit -> (internal di dalam pivot_macro...)
  5. Geser berita luar pasar -> candle pertama berikutnya (internal di merge)
  6. Exact Left Join pada Datetime -> point-in-time only (tanpa carry-forward)
  7. Fill NaN Dev_ dengan 0.0   -> (internal setelah merge)
  8. Simpan dataset_final.csv    -> align_and_merge()

Prinsip Point-in-Time:
  Sinyal Dev_ HANYA aktif tepat di candle rilis berita. Candle sesudahnya
  kembali ke 0.0. Ini mencegah XGBoost mempelajari 'stale features' (fitur
  basi) yang merupakan sumber Data Leakage intraday.

Catatan Penting - Jam Operasional XAU/USD (Dukascopy, UTC):
  - Data Dukascopy: candle ADA di 23:00 s/d 20:45 (UTC) tiap hari kerja.
  - Data Dukascopy: candle TIDAK ADA di 21:00-22:45 (UTC) — jeda harian.
  - Sehingga candle pembuka setelah jeda = 23:00 UTC hari yang sama.
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
RAW_DATA_DIR       = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

OHLCV_RAW_PATH       = RAW_DATA_DIR / "XAUUSD.csv"
MACRO_CLEAN_PATH     = PROCESSED_DATA_DIR / "macro_news_clean.csv"
DATASET_FINAL_PATH   = PROCESSED_DATA_DIR / "dataset_final.csv"

# --- Konstanta Kolom ---
# Nama semua kategori event yang akan menjadi kolom Dev_
EVENT_CATEGORIES = [
    "CPI", "PPI", "GDP", "FedRate", "ADP",
    "NFP", "Jobless", "Earnings", "ISM", "Retail",
]
# Nama kolom Dev_ yang akan muncul di dataset final
DEV_COLUMNS = [f"Dev_{cat}" for cat in EVENT_CATEGORIES]

# Interval 15-menit dalam timedelta
INTERVAL_15MIN = pd.Timedelta(minutes=15)

# Jam jeda pasar XAU/USD Dukascopy (UTC):
#   Candle terakhir sebelum jeda: 20:45 (tutup pukul 21:00)
#   Candle pertama setelah jeda : 23:00
MARKET_BREAK_START_HOUR = 21   # >= 21:00 UTC dianggap "di luar jam pasar"
MARKET_BREAK_END_HOUR   = 23   # < 23:00 UTC dianggap "di luar jam pasar"


# =============================================================================
# BAGIAN 1: FUNGSI LOAD DATA
# =============================================================================

def load_ohlcv(filepath: str | Path = OHLCV_RAW_PATH) -> pd.DataFrame:
    """
    Membaca file XAUUSD.csv dari Dukascopy dan menggabungkan kolom
    'Date' + 'Timestamp' menjadi satu kolom 'Datetime' bertipe UTC.

    Format file Dukascopy:
      Date,Timestamp,Open,High,Low,Close,Volume
      2021-01-03,23:00:00,1904.998,...

    Catatan:
    - Data Dukascopy sudah dalam UTC (tidak perlu konversi timezone).
    - Candle tidak ada saat jeda pasar (21:00-22:45 UTC), sehingga
      setelah 20:45 UTC langsung lompat ke 23:00 UTC.

    Parameter
    ----------
    filepath : str | Path
        Path ke file XAUUSD.csv. Default ke OHLCV_RAW_PATH.

    Kembalian
    ---------
    pd.DataFrame
        DataFrame dengan kolom: Datetime, Open, High, Low, Close, Volume.
        Diurutkan ascending berdasarkan Datetime.
    """
    filepath = Path(filepath)
    logger.info(f"Memuat data OHLCV dari: {filepath}")

    df = pd.read_csv(
        filepath,
        dtype={"Date": str, "Timestamp": str},   # Baca sebagai string dulu
        usecols=["Date", "Timestamp", "Open", "High", "Low", "Close", "Volume"],
    )

    # Gabungkan kolom Date + Timestamp menjadi Datetime
    # Format: "2021-01-03 23:00:00"
    df["Datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Timestamp"],
        format="%Y-%m-%d %H:%M:%S",
    )

    # Hapus kolom Date dan Timestamp yang sudah digabung
    df = df.drop(columns=["Date", "Timestamp"])

    # Urutkan berdasarkan Datetime (ascending)
    df = df.sort_values("Datetime").reset_index(drop=True)

    # Pastikan kolom numerik bertipe float
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    logger.info(
        f"  OHLCV dimuat: {len(df):,} baris. "
        f"Rentang: {df['Datetime'].min()} s/d {df['Datetime'].max()}"
    )
    return df


def load_macro_clean(filepath: str | Path = MACRO_CLEAN_PATH) -> pd.DataFrame:
    """
    Membaca file macro_news_clean.csv yang sudah diproses oleh data_loader.py.

    Kolom yang diharapkan: Datetime, Event, Actual, Forecast, Deviation.
    Datetime sudah dalam UTC (naive).

    Parameter
    ----------
    filepath : str | Path
        Path ke file macro_news_clean.csv. Default ke MACRO_CLEAN_PATH.

    Kembalian
    ---------
    pd.DataFrame
        DataFrame makroekonomi yang siap diproses.
    """
    filepath = Path(filepath)
    logger.info(f"Memuat data makro bersih dari: {filepath}")

    df = pd.read_csv(
        filepath,
        parse_dates=["Datetime"],   # Otomatis parse kolom Datetime
    )

    # Pastikan kolom Deviation bertipe float
    df["Deviation"] = pd.to_numeric(df["Deviation"], errors="coerce")

    logger.info(
        f"  Makro dimuat: {len(df):,} baris, {df['Event'].nunique()} kategori event. "
        f"Rentang: {df['Datetime'].min()} s/d {df['Datetime'].max()}"
    )
    return df


# =============================================================================
# BAGIAN 2: PIVOT MAKRO -> KOLOM Dev_
# =============================================================================

def pivot_macro_to_deviation_columns(df_macro: pd.DataFrame) -> pd.DataFrame:
    """
    Mengubah data makro dari format 'long' (satu baris per event) menjadi
    format 'wide' (setiap event menjadi kolom Dev_XXX tersendiri).

    Langkah internal:
      a) Floor timestamp ke interval 15 menit terdekat.
         (misal: 13:32 -> 13:30, 20:30 -> 20:30, 20:31 -> 20:30)
      b) Group by (Datetime_15min, Event) dan jumlahkan Deviation.
         (Jika ada 2 event CPI di menit yang sama, deviasinya dijumlah)
      c) Pivot: Event -> kolom, nilai -> Deviation.
      d) Tambahkan kolom Dev_ yang hilang (kategori yang tidak ada datanya
         di suatu timestamp -> NaN, nantinya diisi 0.0).

    Alasan floor (bukan round):
      Kita ingin sinyal makro berdampak pada candle YANG SEDANG BERJALAN saat
      berita rilis, bukan candle berikutnya. Contoh: berita rilis jam 13:30:00
      tepat masuk ke candle 13:30. Berita 13:31 juga masuk ke candle 13:30
      (karena belum tutup), bukan ke candle 13:45.

    Parameter
    ----------
    df_macro : pd.DataFrame
        DataFrame makro hasil load_macro_clean().

    Kembalian
    ---------
    pd.DataFrame
        DataFrame dengan kolom: [Datetime] + [Dev_CPI, Dev_PPI, ..., Dev_Retail].
        Datetime sudah dibulatkan ke 15 menit.
    """
    logger.info("Melakukan pivot data makro menjadi kolom Dev_...")

    df = df_macro.copy()

    # --- Langkah a: Floor timestamp ke interval 15 menit ---
    # pd.Timedelta("15min").value = jumlah nanosecond dalam 15 menit
    # np.floor(ts.value / interval_ns) * interval_ns menghasilkan timestamp
    # yang sudah di-floor ke 15 menit terdekat.
    interval_ns = INTERVAL_15MIN.value  # dalam nanosecond
    df["Datetime_15min"] = pd.to_datetime(
        (df["Datetime"].astype(np.int64) // interval_ns) * interval_ns
    )

    # Verifikasi: tampilkan contoh flooring untuk debugging
    sample_before = df["Datetime"].head(3).tolist()
    sample_after  = df["Datetime_15min"].head(3).tolist()
    for b, a in zip(sample_before, sample_after):
        logger.debug(f"  Floor: {b} -> {a}")

    # --- Langkah b: Group by (Datetime_15min, Event), jumlahkan Deviation ---
    # Menggunakan sum karena jika ada 2 berita CPI di 15 menit yang sama,
    # total kejutannya lebih relevan daripada hanya mengambil satu saja.
    grouped = (
        df.groupby(["Datetime_15min", "Event"], as_index=False)["Deviation"]
        .sum(min_count=1)   # min_count=1: jangan jadikan NaN jika ada NaN
    )

    # --- Langkah c: Pivot ---
    # index=Datetime_15min, columns=Event, values=Deviation
    pivoted = grouped.pivot(
        index="Datetime_15min",
        columns="Event",
        values="Deviation",
    )

    # Rename kolom: "CPI" -> "Dev_CPI", dll.
    pivoted.columns = [f"Dev_{col}" for col in pivoted.columns]
    pivoted = pivoted.reset_index().rename(columns={"Datetime_15min": "Datetime"})

    # --- Langkah d: Pastikan semua kolom Dev_ ada (tambahkan yang hilang) ---
    for dev_col in DEV_COLUMNS:
        if dev_col not in pivoted.columns:
            pivoted[dev_col] = np.nan
            logger.warning(f"  Kolom '{dev_col}' tidak ditemukan di data, diisi NaN.")

    # Urutkan kolom: Datetime dulu, lalu Dev_ sesuai urutan EVENT_CATEGORIES
    pivoted = pivoted[["Datetime"] + DEV_COLUMNS]

    logger.info(
        f"  Pivot selesai: {len(pivoted):,} timestamp unik dengan sinyal makro."
    )
    logger.info(f"  Kolom Dev_: {DEV_COLUMNS}")
    return pivoted


# =============================================================================
# BAGIAN 3: MERGE DENGAN PENANGANAN JEDA PASAR
# =============================================================================

def merge_with_market_break_handling(
    df_ohlcv: pd.DataFrame,
    df_macro_pivoted: pd.DataFrame,
) -> pd.DataFrame:
    """
    Menggabungkan data OHLCV dengan sinyal makro menggunakan strategi
    POINT-IN-TIME ONLY (exact merge), bukan carry-forward.

    Prinsip Utama:
    --------------
    Sinyal Dev_ hanya aktif di SATU candle saja (candle saat berita rilis).
    Candle sesudahnya kembali ke 0.0. Ini mencegah XGBoost mempelajari
    'stale features' yang menyebabkan kebocoran data intraday.

    Penanganan Jeda Pasar (Market Break):
    --------------------------------------
    XAU/USD tidak memiliki candle antara 21:00-22:45 UTC (jeda harian).
    Jika timestamp berita (setelah floor ke 15 menit) TIDAK ADA di index
    candle OHLCV, maka:
      -> Geser timestamp berita ke candle PERTAMA yang tersedia SETELAHNYA.
         Ini dilakukan dengan numpy.searchsorted pada array timestamp OHLCV.
    Setelah semua timestamp berita dipetakan ke timestamp candle yang valid,
    lakukan LEFT JOIN biasa (exact match) pada kolom Datetime.

    Mengapa tidak merge_asof?
      merge_asof dengan direction='backward' menyebabkan sinyal terbawa
      (carry-forward) ke semua candle berikutnya sampai ada sinyal baru,
      menghasilkan 'stale features' yang merusak model intraday.

    Parameter
    ----------
    df_ohlcv : pd.DataFrame
        DataFrame OHLCV hasil load_ohlcv(). Kolom: Datetime, O, H, L, C, V.
    df_macro_pivoted : pd.DataFrame
        DataFrame pivot hasil pivot_macro_to_deviation_columns().
        Kolom: Datetime, Dev_CPI, Dev_PPI, ..., Dev_Retail.

    Kembalian
    ---------
    pd.DataFrame
        DataFrame gabungan POINT-IN-TIME. Kolom: Datetime, Open, High, Low,
        Close, Volume, Dev_CPI, Dev_PPI, ..., Dev_Retail.
        Semua NaN pada kolom Dev_ sudah diisi 0.0.
    """
    logger.info("Menggabungkan OHLCV dengan sinyal makro (point-in-time only)...")

    # Urutkan kedua DataFrame berdasarkan Datetime
    df_ohlcv  = df_ohlcv.sort_values("Datetime").reset_index(drop=True)
    df_macro  = df_macro_pivoted.copy().sort_values("Datetime").reset_index(drop=True)

    # --- LANGKAH 1: Bangun index timestamp candle OHLCV yang valid ---
    # Ini adalah "peta" semua candle yang benar-benar ada di data harga.
    # np.array dari Datetime (int64 nanosecond) untuk searchsorted cepat.
    ohlcv_timestamps = df_ohlcv["Datetime"].values  # numpy array datetime64[ns]

    # --- LANGKAH 2: Petakan setiap timestamp berita ke candle yang valid ---
    # Untuk setiap timestamp berita makro:
    #   a) Jika timestamp ADA di OHLCV -> gunakan apa adanya (exact match).
    #   b) Jika timestamp TIDAK ADA (jeda pasar) -> geser ke candle PERTAMA
    #      yang timestamp-nya >= timestamp berita (np.searchsorted 'left').
    #
    # Contoh:
    #   Berita rilis 21:30 UTC -> tidak ada candle 21:30
    #   searchsorted menemukan posisi i di mana ohlcv[i] >= 21:30
    #   ohlcv[i] = 23:00 UTC -> timestamp berita diubah ke 23:00 UTC

    macro_ts_int    = df_macro["Datetime"].values.astype(np.int64)
    ohlcv_ts_int    = ohlcv_timestamps.astype(np.int64)

    # searchsorted 'left': cari indeks di mana macro_ts bisa disisipkan
    # agar array tetap terurut, artinya ohlcv[idx] >= macro_ts
    insert_idx = np.searchsorted(ohlcv_ts_int, macro_ts_int, side="left")

    # Kasus tepi: jika timestamp berita MELEWATI candle terakhir,
    # kunci ke indeks terakhir yang valid (jangan out-of-bounds)
    insert_idx = np.clip(insert_idx, 0, len(ohlcv_timestamps) - 1)

    # Petakan ke timestamp candle OHLCV yang sesuai
    mapped_ts = ohlcv_timestamps[insert_idx]

    # Hitung berapa baris yang perlu digeser (untuk logging)
    n_shifted = int((mapped_ts != df_macro["Datetime"].values).sum())
    if n_shifted > 0:
        logger.info(
            f"  {n_shifted} timestamp berita digeser ke candle pertama berikutnya "
            f"(jatuh saat jeda pasar / tidak ada di index OHLCV)."
        )
    else:
        logger.info("  Semua timestamp berita sudah sejajar dengan candle OHLCV.")

    # Update timestamp berita dengan hasil pemetaan
    df_macro["Datetime"] = mapped_ts

    # --- LANGKAH 3: Agregasi ulang setelah pergeseran timestamp ---
    # Setelah beberapa berita digeser ke timestamp yang sama (misal semua ke 23:00),
    # perlu diagregasi ulang agar tidak ada duplikat timestamp di df_macro.
    # Gunakan sum (konsisten dengan pivot awal): jika 2 berita CPI tergabung
    # di candle yang sama, jumlahkan deviasinya.
    df_macro = (
        df_macro.groupby("Datetime", as_index=False)[DEV_COLUMNS]
        .sum()
    )

    # --- LANGKAH 4: Exact LEFT JOIN ---
    # Sekarang semua timestamp berita dijamin ada di index OHLCV.
    # Left join biasa: setiap candle OHLCV hanya mendapat sinyal makro
    # TEPAT di candle itu saja (bukan carry-forward).
    df_merged = pd.merge(
        df_ohlcv,
        df_macro,
        on="Datetime",
        how="left",   # Left join: OHLCV sebagai acuan utama
    )

    # --- LANGKAH 5: Isi NaN -> 0.0 ---
    # Candle tanpa berita makro = Deviation = 0.0 (tidak ada sinyal)
    df_merged[DEV_COLUMNS] = df_merged[DEV_COLUMNS].fillna(0.0)

    # --- Statistik untuk validasi ---
    n_total = len(df_merged)
    for dev_col in DEV_COLUMNS:
        n_nonzero = (df_merged[dev_col] != 0.0).sum()
        logger.info(
            f"  {dev_col}: {n_nonzero:,} candle dengan sinyal aktif "
            f"({n_nonzero / n_total * 100:.2f}% dari total candle)"
        )

    logger.info(f"  Merge selesai. Total candle: {n_total:,}")
    return df_merged


# =============================================================================
# BAGIAN 4: VALIDASI PENANGANAN MARKET BREAK
# =============================================================================

def validate_market_break_handling(
    df_final: pd.DataFrame,
    df_macro_pivoted: pd.DataFrame,
    sample_n: int = 5,
) -> None:
    """
    Validasi dua hal sekaligus:
    1. Sinyal yang rilis saat jeda pasar berhasil digeser ke candle berikutnya.
    2. Sinyal bersifat POINT-IN-TIME ONLY: candle setelah rilis kembali ke 0.0.

    Fungsi ini bersifat informatif (tidak mengubah data).

    Parameter
    ----------
    df_final : pd.DataFrame
        Dataset final hasil merge_with_market_break_handling().
    df_macro_pivoted : pd.DataFrame
        Data makro setelah pivot (sebelum merge), timestamp ASLI (belum digeser).
    sample_n : int
        Jumlah sampel yang ditampilkan. Default 5.
    """
    logger.info("=" * 60)
    logger.info("VALIDASI 1: PENANGANAN MARKET BREAK")
    logger.info("=" * 60)

    # Cari baris makro yang JATUH DI LUAR jam pasar (21:00 - 22:59 UTC)
    jam_berita = df_macro_pivoted["Datetime"].dt.hour
    mask_luar_pasar = (
        (jam_berita >= MARKET_BREAK_START_HOUR) &
        (jam_berita < MARKET_BREAK_END_HOUR)
    )
    berita_luar_pasar = df_macro_pivoted[mask_luar_pasar].copy()

    if berita_luar_pasar.empty:
        logger.info("  Tidak ada berita yang rilis saat jeda pasar.")
    else:
        logger.info(
            f"  Ditemukan {len(berita_luar_pasar)} timestamp sinyal makro "
            f"yang jatuh saat jeda pasar (21:00-22:59 UTC)."
        )
        samples = berita_luar_pasar.head(sample_n)
        for _, row in samples.iterrows():
            waktu_berita = row["Datetime"]
            # Cari candle pertama setelah waktu berita di dataset final
            candle_setelah = df_final[df_final["Datetime"] > waktu_berita].head(1)
            if candle_setelah.empty:
                logger.info(f"  {waktu_berita}: tidak ada candle setelahnya.")
                continue
            target_dt = candle_setelah.iloc[0]["Datetime"]
            aktif_cols = [
                f"{c}={candle_setelah.iloc[0][c]:.4f}"
                for c in DEV_COLUMNS
                if candle_setelah.iloc[0][c] != 0.0
            ]
            status = "✓ TERTANGKAP" if aktif_cols else "✗ TIDAK TERTANGKAP"
            logger.info(
                f"  Berita {waktu_berita} -> Candle {target_dt}: "
                f"{status} | {aktif_cols}"
            )

    logger.info("=" * 60)
    logger.info("VALIDASI 2: POINT-IN-TIME ONLY (tidak ada carry-forward)")
    logger.info("=" * 60)

    # Cek beberapa sinyal aktif dan pastikan candle berikutnya = 0.0
    # Ambil baris di mana minimal satu Dev_ != 0
    mask_aktif = (df_final[DEV_COLUMNS] != 0.0).any(axis=1)
    baris_aktif = df_final[mask_aktif].head(sample_n)

    for _, row in baris_aktif.iterrows():
        candle_dt = row["Datetime"]
        # Cari candle TEPAT BERIKUTNYA (baris selanjutnya di df_final)
        idx_next = df_final.index[df_final["Datetime"] > candle_dt].min()
        if pd.isna(idx_next):
            continue
        row_next = df_final.loc[idx_next]
        # Kolom Dev_ aktif di candle ini
        aktif_sekarang = [
            f"{c}={row[c]:.4f}" for c in DEV_COLUMNS if row[c] != 0.0
        ]
        # Kolom Dev_ aktif di candle berikutnya (seharusnya 0.0)
        aktif_berikutnya = [
            f"{c}={row_next[c]:.4f}" for c in DEV_COLUMNS if row_next[c] != 0.0
        ]
        if aktif_berikutnya:
            status = "✗ CARRY-FORWARD TERDETEKSI!"
        else:
            status = "✓ BERSIH (candle +1 = 0.0)"
        logger.info(
            f"  Candle {candle_dt}: {aktif_sekarang} | "
            f"Candle +1 ({row_next['Datetime']}): {status}"
        )


# =============================================================================
# BAGIAN 5: PIPELINE UTAMA (TIME-ALIGNMENT LENGKAP)
# =============================================================================

def align_and_merge(
    ohlcv_path: str | Path = OHLCV_RAW_PATH,
    macro_path: str | Path = MACRO_CLEAN_PATH,
    output_path: str | Path = DATASET_FINAL_PATH,
    save_output: bool = True,
    run_validation: bool = True,
) -> pd.DataFrame:
    """
    Pipeline utama Time-Alignment: menggabungkan OHLCV dengan sinyal makro
    menjadi satu dataset final yang siap digunakan untuk Feature Engineering.

    Langkah-langkah:
      1. Load XAUUSD.csv                        -> load_ohlcv()
      2. Load macro_news_clean.csv              -> load_macro_clean()
      3. Pivot makro ke kolom Dev_              -> pivot_macro_to_deviation_columns()
      4. Merge + market break handling          -> merge_with_market_break_handling()
      5. (Opsional) Validasi market break       -> validate_market_break_handling()
      6. Simpan dataset_final.csv               -> (dalam fungsi ini)

    Kolom output dataset_final.csv:
      Datetime, Open, High, Low, Close, Volume,
      Dev_CPI, Dev_PPI, Dev_GDP, Dev_FedRate, Dev_ADP,
      Dev_NFP, Dev_Jobless, Dev_Earnings, Dev_ISM, Dev_Retail

    Parameter
    ----------
    ohlcv_path : str | Path
        Path ke XAUUSD.csv. Default ke OHLCV_RAW_PATH.
    macro_path : str | Path
        Path ke macro_news_clean.csv. Default ke MACRO_CLEAN_PATH.
    output_path : str | Path
        Path output dataset_final.csv. Default ke DATASET_FINAL_PATH.
    save_output : bool
        Jika True, simpan hasil ke output_path. Default True.
    run_validation : bool
        Jika True, jalankan validasi market break setelah merge. Default True.

    Kembalian
    ---------
    pd.DataFrame
        Dataset final yang siap untuk Fase 2 (Feature Engineering).
    """
    logger.info("=" * 60)
    logger.info("MEMULAI PIPELINE TIME-ALIGNMENT")
    logger.info("=" * 60)

    # --- Langkah 1: Load data OHLCV ---
    df_ohlcv = load_ohlcv(ohlcv_path)

    # --- Langkah 2: Load data makro bersih ---
    df_macro = load_macro_clean(macro_path)

    # --- Langkah 3: Pivot makro ke kolom Dev_ ---
    df_macro_pivoted = pivot_macro_to_deviation_columns(df_macro)

    # --- Langkah 4: Merge dengan penanganan market break ---
    df_final = merge_with_market_break_handling(df_ohlcv, df_macro_pivoted)

    # --- Langkah 5 (Opsional): Validasi ---
    if run_validation:
        validate_market_break_handling(df_final, df_macro_pivoted)

    # --- Langkah 6: Susun kolom final ---
    cols_final = ["Datetime", "Open", "High", "Low", "Close", "Volume"] + DEV_COLUMNS
    df_final = df_final[cols_final].copy()

    # Urutkan kembali berdasarkan Datetime (pastikan kronologis)
    df_final = df_final.sort_values("Datetime").reset_index(drop=True)

    # --- Simpan output ---
    if save_output:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        df_final.to_csv(output_path, index=False)
        logger.info(f"Dataset final disimpan ke: {output_path}")

    logger.info("=" * 60)
    logger.info(f"TIME-ALIGNMENT SELESAI. Total candle: {len(df_final):,}")
    logger.info("=" * 60)

    return df_final


# =============================================================================
# BAGIAN 6: ENTRY POINT (untuk pengujian langsung)
# =============================================================================

if __name__ == "__main__":
    """
    Jalankan skrip ini langsung untuk menguji pipeline time-alignment:
        python src/time_alignment.py
    """
    df = align_and_merge(save_output=True, run_validation=True)

    print("\n--- PRATINJAU DATASET FINAL (10 baris pertama) ---")
    print(df.head(10).to_string())

    print("\n--- INFO DATAFRAME ---")
    print(df.info())

    print("\n--- CEK RENTANG WAKTU ---")
    print(f"Candle pertama : {df['Datetime'].min()}")
    print(f"Candle terakhir: {df['Datetime'].max()}")

    print("\n--- STATISTIK KOLOM Dev_ ---")
    dev_stats = df[DEV_COLUMNS].describe().T
    dev_stats["pct_nonzero"] = (df[DEV_COLUMNS] != 0.0).mean() * 100
    print(dev_stats[["mean", "std", "min", "max", "pct_nonzero"]].to_string())

    print("\n--- CONTOH BARIS DENGAN SINYAL MAKRO AKTIF ---")
    # Tampilkan candle di mana minimal satu kolom Dev_ bernilai non-zero
    mask_aktif = (df[DEV_COLUMNS] != 0.0).any(axis=1)
    print(df[mask_aktif][["Datetime"] + DEV_COLUMNS].head(15).to_string())

    print("\n--- CEK CANDLE SEKITAR RILIS NFP (8 Jan 2021, 13:30 UTC) ---")
    # NFP Jan 2021 dirilis jam 13:30 UTC -> cek candle di sekitar itu
    target_dt = pd.Timestamp("2021-01-08 13:00:00")
    window = df[
        (df["Datetime"] >= target_dt) &
        (df["Datetime"] <= target_dt + pd.Timedelta(hours=1))
    ]
    print(window[["Datetime", "Open", "Close", "Dev_NFP", "Dev_Earnings"]].to_string())
