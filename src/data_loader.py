"""
data_loader.py
==============
Modul untuk memuat dan membersihkan data makroekonomi dari Investing.com.

Fase 1: Preprocessing & Time-Alignment
Penulis : Jason Daniel Tanubrata
Tanggal : 2026

Pipeline lengkap (urutan eksekusi):
  1. Baca CSV mentah                            -> load_raw_macro()
  2. Forward-fill tanggal                       -> clean_and_forwardfill_dates()
  3. Gabungkan Date+Time -> kolom Datetime       -> create_datetime_column()
  4. Bersihkan suffix K/M/B/% -> float          -> clean_numeric_columns()
  5. Hitung Deviation = Actual - Forecast        -> calculate_deviation()
  6. Hapus kolom yang tidak diperlukan           -> drop_unused_columns()
  7. Hapus baris tanpa Actual & Forecast         -> drop_empty_rows()
  8. Konversi timezone WIB (GMT+7) -> UTC        -> convert_timezone_to_utc()
  9. Mapping & filter kategori Event             -> map_and_filter_events()
 10. Simpan ke data/processed/macro_news_clean  -> (dalam load_and_process_macro)
"""

import re
import logging
import pandas as pd
import pytz
from pathlib import Path

# --- Konfigurasi Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# --- Konstanta Path ---
# Mendefinisikan path relatif dari lokasi proyek
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

MACRO_RAW_PATH = RAW_DATA_DIR / "macro_news_2021_2025.csv"
MACRO_PROCESSED_PATH = PROCESSED_DATA_DIR / "macro_news_clean.csv"

# --- Konstanta Timezone ---
TZ_WIB = pytz.timezone("Asia/Jakarta")   # GMT+7 (WIB)
TZ_UTC = pytz.utc                          # UTC

# --- Mapping Kategori Event ---
# Urutan SANGAT PENTING: pola yang lebih spesifik (ADP) harus didahulukan
# sebelum pola yang lebih umum (Nonfarm Payroll) untuk mencegah misklasifikasi.
#
# Contoh potensi tumpang-tindih:
#   "ADP Nonfarm Employment Change" -> mengandung 'ADP' DAN 'Nonfarm'
#   Jika 'Nonfarm' dicek lebih dulu, baris ADP akan salah masuk ke 'NFP'.
#   Solusi: taruh pola 'ADP' lebih atas dari pola 'Payroll/Nonfarm Payroll'.
#
# Format: (pola_regex, nama_standar)
EVENT_MAPPING: list[tuple[str, str]] = [
    # --- 1. CPI: Consumer Price Index ---
    (r"CPI",                         "CPI"),
    # --- 2. PPI: Producer Price Index ---
    (r"PPI",                         "PPI"),
    # --- 3. GDP: Gross Domestic Product ---
    (r"GDP",                         "GDP"),
    # --- 4. FedRate: Keputusan suku bunga acuan utama Fed / FOMC ---
    #    SENGAJA dibuat ketat agar HANYA menangkap:
    #      - "Fed Interest Rate Decision"   -> Interest Rate Decision
    #      - "FOMC Statement/Press Conference/Meeting Minutes/Economic Projections"
    #      - "Fed Funds Rate" (jika ada)
    #    Pola 'Fed' saja terlalu luas dan akan menangkap indeks regional seperti
    #    "Philadelphia Fed Manufacturing Index" atau "Richmond Fed Manufacturing Index".
    #    Solusi: gunakan pola spesifik yang mengharuskan kata kunci pendamping.
    (r"FOMC|Fed Funds Rate|Interest Rate Decision", "FedRate"),
    # --- 5. ADP: Laporan tenaga kerja ADP (HARUS sebelum NFP) ---
    #    Regex 'ADP' saja cukup spesifik karena nama eventnya selalu "ADP ..."
    (r"\bADP\b",                     "ADP"),
    # --- 6. NFP: Nonfarm Payrolls (bukan ADP) ---
    #    Gunakan 'Nonfarm Payroll' (bukan sekedar 'Nonfarm') untuk keamanan,
    #    karena 'ADP Nonfarm Employment Change' sudah tertangkap di atas.
    (r"Nonfarm Payroll|Payrolls",    "NFP"),
    # --- 7. Jobless: Initial Jobless Claims ---
    (r"Initial Jobless",             "Jobless"),
    # --- 8. Earnings: Average Hourly Earnings ---
    (r"Average Hourly Earnings",     "Earnings"),
    # --- 9. ISM: HANYA ISM Manufacturing PMI ---
    #    Pola 'ISM' saja terlalu luas: akan menangkap:
    #      - "ISM Manufacturing PMI"        -> INGIN ini ✓
    #      - "ISM Non-Manufacturing PMI"    -> TIDAK INGIN ✗
    #      - "ISM Manufacturing Prices"     -> TIDAK INGIN ✗
    #      - "ISM Non-Manufacturing Prices" -> TIDAK INGIN ✗
    #    Solusi: pola spesifik yang memerlukan kata "Manufacturing PMI" persis,
    #    sekaligus TIDAK mengandung "Non" (gunakan negative lookahead regex).
    #    (?!.*Non) = pastikan string setelah ini TIDAK mengandung kata "Non"
    (r"ISM(?!.*Non).*Manufacturing PMI",  "ISM"),
    # --- 10. Retail: Retail Sales ---
    (r"Retail Sales",                "Retail"),
]

# Nama-nama kategori valid yang akan dipertahankan setelah mapping
VALID_EVENT_CATEGORIES: set[str] = {t[1] for t in EVENT_MAPPING}


# =============================================================================
# BAGIAN 1: FUNGSI PEMBACAAN DAN PEMBERSIHAN STRUKTUR CSV
# =============================================================================

def _parse_suffix_value(value_str: str) -> float | None:
    """
    Mengonversi string nilai numerik dengan suffix menjadi float.

    Menangani karakter: 'K' (ribuan), 'M' (jutaan), 'B' (miliaran), '%' (persen).
    Juga menangani tanda negatif dan pemisah ribuan dengan koma (misal: "1,021K").

    Parameter
    ----------
    value_str : str
        String nilai yang akan dikonversi. Contoh: "-123K", "5.40%", "1,021K",
        "-1,837.0B", "60.7".

    Kembalian
    ---------
    float | None
        Nilai numerik dalam float, atau None jika string kosong/tidak dapat diproses.
    """
    if not isinstance(value_str, str):
        return None

    # Hapus spasi awal/akhir
    cleaned = value_str.strip()

    if cleaned == "" or cleaned == "-":
        return None

    # Peta suffix ke pengali
    multiplier_map = {
        "K": 1_000,
        "M": 1_000_000,
        "B": 1_000_000_000,
        "%": 1,          # Persentase dibiarkan apa adanya (misal: 5.40% -> 5.40)
    }

    # Cek apakah ada suffix di akhir string (case-insensitive)
    suffix = cleaned[-1].upper()
    # Gunakan default multiplier = 1 jika suffix tidak dikenali
    # (artinya angka polos tanpa suffix seperti "60.7")
    multiplier = multiplier_map.get(suffix, 1)

    if suffix in multiplier_map:
        # Lepas suffix dari string
        numeric_part = cleaned[:-1]
    else:
        # Tidak ada suffix, ambil seluruh string sebagai angka
        numeric_part = cleaned

    try:
        # Hapus pemisah ribuan dengan koma sebelum konversi
        # Contoh: "1,021" -> "1021", "-1,837.0" -> "-1837.0"
        numeric_part = numeric_part.replace(",", "")
        result = float(numeric_part)
        return result * multiplier
    except ValueError:
        logger.warning(f"  Tidak dapat mengonversi nilai: '{value_str}' -> None")
        return None


def _is_date_row(row: pd.Series) -> bool:
    """
    Mendeteksi apakah suatu baris adalah baris header tanggal.

    Baris tanggal dari Investing.com memiliki format seperti:
    "Friday, January 1, 2021" di kolom 'Time', dan semua kolom lainnya kosong.

    Parameter
    ----------
    row : pd.Series
        Satu baris dari DataFrame mentah.

    Kembalian
    ---------
    bool
        True jika baris tersebut adalah baris tanggal, False jika baris berita.
    """
    time_val = str(row.get("Time", "")).strip()
    # Pola baris tanggal: mengandung nama hari dalam bahasa Inggris diikuti koma
    date_pattern = re.compile(
        r"^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),",
        re.IGNORECASE,
    )
    return bool(date_pattern.match(time_val))


def load_raw_macro(filepath: str | Path = MACRO_RAW_PATH) -> pd.DataFrame:
    """
    Membaca file CSV mentah hasil copy-paste dari Investing.com.

    File ini memiliki struktur campuran:
    - Baris tanggal (misal: "Friday, January 1, 2021") sebagai penanda kelompok.
    - Baris berita di bawahnya (misal: "22:00,US,CPI (MoM),...").

    Parameter
    ----------
    filepath : str | Path
        Path ke file CSV mentah. Default ke MACRO_RAW_PATH.

    Kembalian
    ---------
    pd.DataFrame
        DataFrame mentah yang sudah dimuat dari CSV.
    """
    filepath = Path(filepath)
    logger.info(f"Memuat file CSV mentah dari: {filepath}")

    # Membaca CSV dengan semua kolom sebagai string agar tidak ada konversi otomatis
    # yang merusak nilai seperti "0.30%", "-123K", dll.
    df = pd.read_csv(
        filepath,
        dtype=str,              # Semua kolom dibaca sebagai string
        keep_default_na=False,  # Jangan konversi string kosong ke NaN secara otomatis
    )

    logger.info(f"  Berhasil memuat {len(df)} baris, {len(df.columns)} kolom.")
    logger.info(f"  Kolom yang ditemukan: {list(df.columns)}")
    return df


# =============================================================================
# BAGIAN 2: FUNGSI PEMBERSIHAN DAN FORWARD-FILL TANGGAL
# =============================================================================

def clean_and_forwardfill_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membersihkan DataFrame mentah dengan cara:
    1. Mengidentifikasi baris tanggal dan baris berita.
    2. Mengekstrak informasi tanggal dari baris tanggal.
    3. Melakukan forward-fill: setiap baris berita mendapatkan tanggal
       dari baris tanggal terdekat di atasnya.
    4. Menghapus baris tanggal (sudah tidak diperlukan setelah forward-fill).

    Parameter
    ----------
    df : pd.DataFrame
        DataFrame mentah hasil load_raw_macro().

    Kembalian
    ---------
    pd.DataFrame
        DataFrame yang sudah dibersihkan dengan kolom 'Date' terisi penuh.
    """
    logger.info("Memulai proses pembersihan dan forward-fill tanggal...")

    # Salin DataFrame agar tidak mengubah data asli (best practice)
    df = df.copy()

    # --- Langkah 1: Identifikasi baris tanggal dan ekstrak nilai tanggalnya ---
    # Buat kolom 'Date' yang awalnya kosong
    df["Date"] = None

    # Iterasi untuk mendeteksi baris tanggal dan mengisi kolom 'Date'
    date_rows_idx = []
    for idx, row in df.iterrows():
        if _is_date_row(row):
            # Nilai tanggal ada di kolom 'Time'
            df.at[idx, "Date"] = row["Time"].strip().strip('"')
            date_rows_idx.append(idx)

    logger.info(f"  Ditemukan {len(date_rows_idx)} baris tanggal (header grup).")

    # --- Langkah 2: Forward-fill kolom 'Date' ---
    # Ganti string kosong dengan NaN dulu agar ffill bekerja
    df["Date"] = df["Date"].replace("", None)
    df["Date"] = df["Date"].ffill()

    # --- Langkah 3: Hapus baris tanggal (sudah tidak diperlukan) ---
    df = df.drop(index=date_rows_idx).reset_index(drop=True)

    logger.info(f"  Tersisa {len(df)} baris berita setelah menghapus baris tanggal.")
    return df


# =============================================================================
# BAGIAN 3: FUNGSI PENGGABUNGAN DATETIME
# =============================================================================

def create_datetime_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menggabungkan kolom 'Date' dan 'Time' menjadi satu kolom 'Datetime' bertipe
    pandas Timestamp (naive, belum ada timezone info).

    Asumsi format tanggal dari Investing.com: "Friday, January 1, 2021"
    Asumsi format waktu: "HH:MM" (24-jam) atau "All Day"

    Catatan: Timezone akan ditambahkan di langkah terpisah (convert_timezone_to_utc).

    Parameter
    ----------
    df : pd.DataFrame
        DataFrame yang sudah melewati clean_and_forwardfill_dates().

    Kembalian
    ---------
    pd.DataFrame
        DataFrame dengan kolom 'Datetime' baru bertipe datetime64[ns].
        Baris dengan waktu "All Day" akan ditetapkan pukul 00:00.
    """
    logger.info("Menggabungkan kolom 'Date' dan 'Time' menjadi 'Datetime'...")

    df = df.copy()

    def _combine_datetime(row: pd.Series) -> pd.Timestamp | None:
        """Fungsi helper untuk menggabungkan satu baris Date + Time."""
        date_str = str(row.get("Date", "")).strip()
        time_str = str(row.get("Time", "")).strip()

        if not date_str or date_str == "nan":
            return None

        # Tangani kasus khusus "All Day" -> set ke 00:00
        if time_str.lower() == "all day" or time_str == "":
            time_str = "00:00"

        # Gabungkan string tanggal + waktu dan parse
        combined_str = f"{date_str} {time_str}"
        try:
            # Format: "Friday, January 1, 2021 22:00"
            return pd.to_datetime(combined_str, format="%A, %B %d, %Y %H:%M")
        except (ValueError, TypeError):
            logger.warning(f"  Gagal parse datetime: '{combined_str}' -> None")
            return None

    df["Datetime"] = df.apply(_combine_datetime, axis=1)

    # Hapus baris yang gagal parse Datetime
    n_failed = df["Datetime"].isna().sum()
    if n_failed > 0:
        logger.warning(f"  {n_failed} baris gagal diparse dan akan dihapus.")
        df = df.dropna(subset=["Datetime"]).reset_index(drop=True)

    logger.info(f"  Kolom 'Datetime' berhasil dibuat. Total baris: {len(df)}")
    return df


# =============================================================================
# BAGIAN 4: FUNGSI PEMBERSIHAN NILAI NUMERIK (K, M, B, %)
# =============================================================================

def clean_numeric_columns(
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """
    Membersihkan karakter suffix (K, M, B, %) pada kolom numerik dan
    mengonversinya menjadi tipe data float.

    Parameter
    ----------
    df : pd.DataFrame
        DataFrame input.
    columns : list[str] | None
        Daftar nama kolom yang akan dibersihkan.
        Default: ['Actual', 'Forecast', 'Previous'].

    Kembalian
    ---------
    pd.DataFrame
        DataFrame dengan kolom yang sudah dikonversi ke float.
    """
    if columns is None:
        columns = ["Actual", "Forecast", "Previous"]

    logger.info(f"Membersihkan nilai numerik pada kolom: {columns}")
    df = df.copy()

    for col in columns:
        if col not in df.columns:
            logger.warning(f"  Kolom '{col}' tidak ditemukan. Dilewati.")
            continue

        # Terapkan fungsi konversi ke setiap sel
        df[col] = df[col].apply(_parse_suffix_value)
        logger.info(
            f"  Kolom '{col}' -> float. "
            f"Non-null: {df[col].notna().sum()} / {len(df)}"
        )

    return df


# =============================================================================
# BAGIAN 5: FUNGSI PERHITUNGAN DEVIASI
# =============================================================================

def calculate_deviation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghitung kolom 'Deviation' sebagai selisih antara nilai Actual dan Forecast.

    Rumus: Deviation = Actual - Forecast

    Interpretasi:
    - Deviation > 0: Actual lebih baik/tinggi dari Forecast (Surprise Positif)
    - Deviation < 0: Actual lebih buruk/rendah dari Forecast (Surprise Negatif)
    - Deviation = 0: Actual sesuai ekspektasi
    - Deviation = NaN: Salah satu nilai tidak tersedia (misal: event tanpa forecast)

    Parameter
    ----------
    df : pd.DataFrame
        DataFrame dengan kolom 'Actual' dan 'Forecast' bertipe float.

    Kembalian
    ---------
    pd.DataFrame
        DataFrame dengan kolom 'Deviation' baru.
    """
    logger.info("Menghitung kolom 'Deviation' = Actual - Forecast...")

    df = df.copy()

    if "Actual" not in df.columns or "Forecast" not in df.columns:
        raise ValueError(
            "Kolom 'Actual' dan 'Forecast' harus ada sebelum menghitung Deviation."
        )

    df["Deviation"] = df["Actual"] - df["Forecast"]

    n_valid = df["Deviation"].notna().sum()
    n_total = len(df)
    logger.info(
        f"  Deviation berhasil dihitung. "
        f"Valid: {n_valid}/{n_total} baris "
        f"({n_valid / n_total * 100:.1f}%)."
    )
    return df


# =============================================================================
# BAGIAN 6: HAPUS KOLOM YANG TIDAK DIPERLUKAN
# =============================================================================

def drop_unused_columns(
    df: pd.DataFrame,
    cols_to_drop: list[str] | None = None,
) -> pd.DataFrame:
    """
    Menghapus kolom-kolom yang tidak akan digunakan dalam pemodelan.

    Kolom yang dibuang (default):
    - 'Previous' : nilai sebelumnya; tidak dipakai sebagai fitur langsung.
    - 'Cur.'     : semua data sudah difilter hanya US, redundan.
    - 'Imp.'     : tingkat importansi dari Investing.com; subyektif, tidak konsisten.

    Parameter
    ----------
    df : pd.DataFrame
        DataFrame input.
    cols_to_drop : list[str] | None
        Kolom yang ingin dihapus. Default: ['Previous', 'Cur.', 'Imp.'].

    Kembalian
    ---------
    pd.DataFrame
        DataFrame tanpa kolom yang sudah dihapus.
    """
    if cols_to_drop is None:
        cols_to_drop = ["Previous", "Cur.", "Imp."]

    # Hanya hapus kolom yang benar-benar ada di DataFrame (aman)
    existing_cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    missing = set(cols_to_drop) - set(existing_cols_to_drop)
    if missing:
        logger.warning(f"  Kolom berikut tidak ditemukan, dilewati: {missing}")

    df = df.drop(columns=existing_cols_to_drop)
    logger.info(
        f"Menghapus kolom tidak perlu: {existing_cols_to_drop}. "
        f"Kolom tersisa: {list(df.columns)}"
    )
    return df


# =============================================================================
# BAGIAN 7: HAPUS BARIS YANG TIDAK MEMILIKI DATA NUMERIK
# =============================================================================

def drop_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Menghapus baris yang tidak memiliki nilai pada KEDUA kolom Actual dan Forecast.

    Baris seperti ini muncul untuk:
    - Hari libur bank (misal: "United States - New Year's Day")
    - Acara pidato tanpa angka (misal: "Fed Chair Powell Speaks")
    - Rilis meeting minutes tanpa revisi angka

    Strategi: hapus baris yang nilainya NaN pada KEDUANYA (Actual DAN Forecast).
    Baris yang hanya salah satunya NaN (misal: event tanpa consensus forecast)
    tetap dipertahankan jika memiliki nilai Actual.

    Parameter
    ----------
    df : pd.DataFrame
        DataFrame input dengan kolom 'Actual' dan 'Forecast' bertipe float.

    Kembalian
    ---------
    pd.DataFrame
        DataFrame tanpa baris yang semua kolom numeriknya kosong.
    """
    n_before = len(df)

    # Hapus baris di mana Actual DAN Forecast keduanya NaN
    mask_both_empty = df["Actual"].isna() & df["Forecast"].isna()
    df = df[~mask_both_empty].reset_index(drop=True)

    n_after = len(df)
    n_removed = n_before - n_after
    logger.info(
        f"Menghapus baris tanpa data numerik: {n_removed} baris dihapus. "
        f"Tersisa: {n_after} baris."
    )
    return df


# =============================================================================
# BAGIAN 8: KONVERSI TIMEZONE WIB -> UTC
# =============================================================================

def convert_timezone_to_utc(df: pd.DataFrame) -> pd.DataFrame:
    """
    Mengonversi kolom 'Datetime' dari WIB (GMT+7 / Asia/Jakarta) ke UTC.

    Latar belakang:
    - Data dari Investing.com (versi Indonesia) ditampilkan dalam WIB.
    - Data harga XAU/USD dari Dukascopy menggunakan UTC.
    - Konversi ini wajib dilakukan agar time-alignment antar dataset akurat.

    Contoh konversi:
      WIB 22:00  ->  UTC 15:00  (selisih -7 jam)
      WIB 02:30  ->  UTC 19:30 (hari sebelumnya)

    Setelah konversi, kolom 'Datetime' akan bertipe datetime64[ns, UTC]
    (timezone-aware), lalu di-strip menjadi naive UTC (datetime64[ns])
    agar kompatibel dengan Pandas standar saat disimpan ke CSV.

    Parameter
    ----------
    df : pd.DataFrame
        DataFrame dengan kolom 'Datetime' bertipe datetime64[ns] (naive, WIB).

    Kembalian
    ---------
    pd.DataFrame
        DataFrame dengan kolom 'Datetime' sudah dalam UTC (naive).
    """
    logger.info("Mengonversi timezone Datetime: WIB (GMT+7) -> UTC...")

    df = df.copy()

    # Langkah 1: Localize (beri tahu Pandas bahwa waktu ini adalah WIB)
    # Menggunakan tz_localize karena Datetime saat ini masih 'naive' (tanpa tz info)
    df["Datetime"] = df["Datetime"].dt.tz_localize(TZ_WIB)

    # Langkah 2: Konversi ke UTC
    df["Datetime"] = df["Datetime"].dt.tz_convert(TZ_UTC)

    # Langkah 3: Hapus informasi timezone (jadikan naive UTC) agar kompatibel
    # saat disimpan ke CSV dan dibaca kembali tanpa ambiguitas
    df["Datetime"] = df["Datetime"].dt.tz_localize(None)

    # Verifikasi: tampilkan beberapa sampel sebelum dan sesudah (untuk debugging)
    logger.info(
        f"  Konversi selesai. "
        f"Rentang UTC: {df['Datetime'].min()} s/d {df['Datetime'].max()}"
    )
    return df


# =============================================================================
# BAGIAN 9: MAPPING DAN FILTER KATEGORI EVENT
# =============================================================================

def map_and_filter_events(
    df: pd.DataFrame,
    mapping: list[tuple[str, str]] = EVENT_MAPPING,
    valid_categories: set[str] = VALID_EVENT_CATEGORIES,
) -> pd.DataFrame:
    """
    Menstandarkan nama Event menggunakan peta Regex, lalu memfilter
    hanya kategori yang relevan untuk pemodelan.

    Urutan pengecekan regex SANGAT PENTING untuk mencegah misklasifikasi.
    Lihat konstanta EVENT_MAPPING di atas untuk penjelasan urutan.

    Contoh mapping:
      "ADP Nonfarm Employment Change (Dec)" -> "ADP"   (bukan "NFP"!)
      "Nonfarm Payrolls (Dec)"              -> "NFP"
      "CPI (MoM) (Dec)"                    -> "CPI"

    Baris yang tidak cocok dengan pola manapun akan di-set NaN dan kemudian
    dihapus dari dataset.

    Parameter
    ----------
    df : pd.DataFrame
        DataFrame dengan kolom 'Event' berisi nama event asli dari Investing.com.
    mapping : list[tuple[str, str]]
        Daftar (pola_regex, nama_standar) yang diurutkan dari paling spesifik
        ke paling umum. Default: EVENT_MAPPING.
    valid_categories : set[str]
        Kumpulan nama standar yang akan DIPERTAHANKAN. Baris lain dihapus.
        Default: VALID_EVENT_CATEGORIES.

    Kembalian
    ---------
    pd.DataFrame
        DataFrame yang hanya berisi baris dengan kategori Event yang valid.
        Kolom 'Event' berisi nama standar (bukan nama asli lagi).
    """
    logger.info("Melakukan mapping dan filter kategori Event...")

    df = df.copy()
    n_before = len(df)

    def _map_event(event_name: str) -> str | None:
        """
        Memeriksa nama event satu per satu terhadap daftar pola regex.
        Mengembalikan nama standar pertama yang cocok, atau None jika tidak ada.
        """
        if not isinstance(event_name, str):
            return None
        for pattern, standard_name in mapping:
            # re.search: mencari pola di mana saja dalam string (bukan hanya awal)
            # re.IGNORECASE: tidak peka huruf besar/kecil
            if re.search(pattern, event_name, re.IGNORECASE):
                return standard_name
        return None  # Tidak ada pola yang cocok -> akan dihapus

    # Terapkan fungsi mapping ke seluruh kolom Event
    df["Event"] = df["Event"].apply(_map_event)

    # Log distribusi mapping untuk verifikasi
    event_counts = df["Event"].value_counts(dropna=False)
    logger.info(f"  Distribusi event setelah mapping:\n{event_counts.to_string()}")

    # Hapus baris yang tidak cocok dengan kategori manapun (Event == None/NaN)
    df = df.dropna(subset=["Event"]).reset_index(drop=True)

    n_after = len(df)
    n_removed = n_before - n_after
    logger.info(
        f"  Filter selesai: {n_removed} baris dihapus (tidak masuk kategori). "
        f"Tersisa: {n_after} baris dengan {df['Event'].nunique()} kategori unik."
    )
    return df


# =============================================================================
# BAGIAN 10: FUNGSI UTAMA (PIPELINE LENGKAP)
# =============================================================================

def load_and_process_macro(
    filepath: str | Path = MACRO_RAW_PATH,
    save_processed: bool = True,
) -> pd.DataFrame:
    """
    Pipeline lengkap untuk memuat dan memproses data makroekonomi menjadi
    format yang siap untuk tahap Time-Alignment.

    Langkah-langkah yang dijalankan secara berurutan:
      1.  Membaca CSV mentah                         (load_raw_macro)
      2.  Forward-fill tanggal                       (clean_and_forwardfill_dates)
      3.  Membuat kolom Datetime                     (create_datetime_column)
      4.  Membersihkan kolom numerik (K/M/B/%)       (clean_numeric_columns)
      5.  Menghitung Deviation = Actual - Forecast   (calculate_deviation)
      6.  Hapus kolom Previous, Cur., Imp.           (drop_unused_columns)
      7.  Hapus baris tanpa Actual & Forecast        (drop_empty_rows)
      8.  Konversi timezone WIB -> UTC               (convert_timezone_to_utc)
      9.  Mapping & filter kategori Event            (map_and_filter_events)
      10. Simpan ke data/processed/                  (opsional)

    Parameter
    ----------
    filepath : str | Path
        Path ke file CSV mentah. Default ke MACRO_RAW_PATH.
    save_processed : bool
        Jika True, simpan hasil ke MACRO_PROCESSED_PATH (overwrite jika ada).
        Default True.

    Kembalian
    ---------
    pd.DataFrame
        DataFrame bersih siap pakai. Kolom akhir:
        ['Datetime', 'Event', 'Actual', 'Forecast', 'Deviation']
    """
    logger.info("=" * 60)
    logger.info("MEMULAI PIPELINE PREPROCESSING DATA MAKROEKONOMI")
    logger.info("=" * 60)

    # --- Langkah 1: Muat CSV mentah ---
    df = load_raw_macro(filepath)

    # --- Langkah 2: Bersihkan dan forward-fill tanggal ---
    df = clean_and_forwardfill_dates(df)

    # --- Langkah 3: Buat kolom Datetime (naive, WIB) ---
    df = create_datetime_column(df)

    # --- Langkah 4: Bersihkan kolom numerik ---
    df = clean_numeric_columns(df, columns=["Actual", "Forecast", "Previous"])

    # --- Langkah 5: Hitung Deviation ---
    df = calculate_deviation(df)

    # --- Langkah 6: Hapus kolom yang tidak diperlukan ---
    df = drop_unused_columns(df, cols_to_drop=["Previous", "Cur.", "Imp."])

    # --- Langkah 7: Hapus baris yang tidak memiliki data numerik ---
    df = drop_empty_rows(df)

    # --- Langkah 8: Konversi timezone WIB -> UTC ---
    df = convert_timezone_to_utc(df)

    # --- Langkah 9: Mapping dan filter kategori Event ---
    df = map_and_filter_events(df)

    # --- Langkah 10: Susun kolom final dan urutkan ---
    cols_final = ["Datetime", "Event", "Actual", "Forecast", "Deviation"]
    cols_available = [c for c in cols_final if c in df.columns]
    df = df[cols_available].copy()

    # Urutkan berdasarkan Datetime (krusial untuk analisis time-series)
    df = df.sort_values("Datetime").reset_index(drop=True)

    # --- Simpan hasil (overwrite file lama jika ada) ---
    if save_processed:
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        # mode='w' memastikan file lama akan ditimpa (overwrite)
        df.to_csv(MACRO_PROCESSED_PATH, index=False, mode="w")
        logger.info(f"Data terproses disimpan ke: {MACRO_PROCESSED_PATH}")

    logger.info("=" * 60)
    logger.info(f"PIPELINE SELESAI. Total baris bersih final: {len(df)}")
    logger.info("=" * 60)

    return df


# =============================================================================
# BAGIAN 11: ENTRY POINT (untuk pengujian langsung)
# =============================================================================

if __name__ == "__main__":
    """
    Jalankan skrip ini langsung untuk menguji pipeline lengkap:
        python src/data_loader.py
    """
    df_macro = load_and_process_macro(save_processed=True)

    print("\n--- PRATINJAU DATA HASIL PREPROCESSING (20 baris pertama) ---")
    print(df_macro.head(20).to_string())

    print("\n--- INFO DATAFRAME ---")
    print(df_macro.info())

    print("\n--- STATISTIK DESKRIPTIF ---")
    print(df_macro[["Actual", "Forecast", "Deviation"]].describe())

    print("\n--- CEK RENTANG WAKTU (UTC) ---")
    print(f"Tanggal paling awal : {df_macro['Datetime'].min()}")
    print(f"Tanggal paling akhir: {df_macro['Datetime'].max()}")

    print("\n--- DISTRIBUSI KATEGORI EVENT (FINAL) ---")
    print(df_macro["Event"].value_counts().to_string())

    print("\n--- SAMPLE PER KATEGORI (3 baris per event) ---")
    for event_name in sorted(df_macro["Event"].unique()):
        sample = df_macro[df_macro["Event"] == event_name].head(3)
        print(f"\n[{event_name}]")
        print(sample[["Datetime", "Event", "Actual", "Forecast", "Deviation"]].to_string())
