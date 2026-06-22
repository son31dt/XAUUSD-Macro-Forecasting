import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error

RESULTS   = Path("results")
PROCESSED = Path("data/processed")

a1      = pd.read_csv(RESULTS / "predictions_a1.csv", index_col="Datetime", parse_dates=True)
a2      = pd.read_csv(RESULTS / "predictions_a2.csv", index_col="Datetime", parse_dates=True)
df_test = pd.read_csv(PROCESSED / "dataset_featured.csv", index_col="Datetime", parse_dates=True)
df_test = df_test.loc["2025-01-01":]

DEV_COLS = ["Dev_CPI","Dev_PPI","Dev_GDP","Dev_FedRate","Dev_ADP",
            "Dev_NFP","Dev_Jobless","Dev_Earnings","Dev_ISM","Dev_Retail"]
dev_avail = [c for c in DEV_COLS if c in df_test.columns]

def get_metrics(yt_p, yp_p, yt_d, yp_d):
    mae  = mean_absolute_error(yt_p, yp_p)
    mask = yt_d != 0
    mda  = np.mean(np.sign(yt_d[mask]) == np.sign(yp_d[mask])) * 100 if mask.sum() > 0 else np.nan
    return mae, mda

# ─── GLOBAL ───────────────────────────────────────────────────────────────────
print("\n" + "="*62)
print("  PERBANDINGAN GLOBAL A1 vs A2")
print("="*62)

mae1, mda1 = get_metrics(a1["Target_Close_Actual"].values, a1["Target_Close_Pred_a1"].values,
                         a1["y_true_diff"].values,         a1["y_pred_diff_a1"].values)
mae2, mda2 = get_metrics(a2["Target_Close_Actual"].values, a2["Target_Close_Pred_a2"].values,
                         a2["y_true_diff"].values,         a2["y_pred_diff_a2"].values)

print(f"  A1 Baseline : MAE={mae1:.4f} USD | MDA={mda1:.4f}%")
print(f"  A2 +Makro   : MAE={mae2:.4f} USD | MDA={mda2:.4f}%")
print(f"  Delta       : MAE={mae2-mae1:+.4f} USD | MDA={mda2-mda1:+.4f}%")
print("="*62)

# ─── EVENT-DRIVEN ─────────────────────────────────────────────────────────────
mask_event = (df_test[dev_avail] != 0.0).any(axis=1)
event_idx  = df_test.index[mask_event]
common     = a1.index.intersection(a2.index).intersection(event_idx)

a1e = a1.loc[common]
a2e = a2.loc[common]

print(f"\n  Event candle: {len(event_idx)} (total) | {len(common)} (ada di kedua prediksi)")
mae1e, mda1e = get_metrics(a1e["Target_Close_Actual"].values, a1e["Target_Close_Pred_a1"].values,
                           a1e["y_true_diff"].values,         a1e["y_pred_diff_a1"].values)
mae2e, mda2e = get_metrics(a2e["Target_Close_Actual"].values, a2e["Target_Close_Pred_a2"].values,
                           a2e["y_true_diff"].values,         a2e["y_pred_diff_a2"].values)

print("\n" + "="*62)
print("  EVALUASI EVENT-DRIVEN: Khusus Jam Rilis Berita Makro")
print("="*62)
print(f"  {'Metrik':<12} {'A1 Baseline':>14} {'A2 +Makro':>12} {'Delta':>10}  Kesimpulan")
print("  " + "-"*58)
for label, v1, v2 in [("MAE (USD)", mae1e, mae2e), ("MDA (%)", mda1e, mda2e)]:
    d    = v2 - v1
    is_mae = "MAE" in label
    wins = "A2 UNGGUL" if (is_mae and d < 0) or (not is_mae and d > 0) else "A1 lebih baik"
    print(f"  {label:<12} {v1:>14.4f} {v2:>12.4f} {d:>+10.4f}  {wins}")
print("="*62)

# ─── PER INDIKATOR ────────────────────────────────────────────────────────────
print("\n  Rincian per indikator makro:")
print(f"  {'Indikator':<15} {'N':>5}  {'MDA A1':>8}  {'MDA A2':>8}  {'Delta':>8}")
print("  " + "-"*52)
for col in dev_avail:
    idx = df_test.index[(df_test[col] != 0.0)]
    idx = idx[idx.isin(common)]
    if len(idx) == 0:
        continue
    s1 = a1e.loc[a1e.index.isin(idx)]
    s2 = a2e.loc[a2e.index.isin(idx)]
    if len(s1) == 0:
        continue
    _, m1 = get_metrics(s1["Target_Close_Actual"].values, s1["Target_Close_Pred_a1"].values,
                        s1["y_true_diff"].values,         s1["y_pred_diff_a1"].values)
    _, m2 = get_metrics(s2["Target_Close_Actual"].values, s2["Target_Close_Pred_a2"].values,
                        s2["y_true_diff"].values,         s2["y_pred_diff_a2"].values)
    d   = m2 - m1
    tag = "<< A2 MENANG" if d > 0 else ""
    print(f"  {col:<15} {len(idx):>5}  {m1:>7.1f}%  {m2:>7.1f}%  {d:>+7.1f}%  {tag}")
print("="*62)
