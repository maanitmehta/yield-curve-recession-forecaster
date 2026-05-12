"""
Phase 1 — Data Acquisition (REAL DATA)
========================================
Pulls US Treasury constant-maturity yields from FRED
and builds the NBER recession indicator.

Run this file directly to download and save all data:
    python phase1_data.py

Outputs (written to data/):
    yields_monthly.csv      — dates × 11 maturities (yield %)
    recession_indicator.csv — monthly 0/1 NBER indicator + announcement date
    qc_report.png           — missingness heatmap + data coverage chart
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── API key ────────────────────────────────────────────────────────────────
FRED_API_KEY = "e0b05c797576a3db6d0a186713a70f4d"

# ── FRED series → maturity (years) ─────────────────────────────────────────
FRED_TICKERS = {
    "DGS1MO":  0.083,
    "DGS3MO":  0.25,
    "DGS6MO":  0.5,
    "DGS1":    1.0,
    "DGS2":    2.0,
    "DGS3":    3.0,
    "DGS5":    5.0,
    "DGS7":    7.0,
    "DGS10":   10.0,
    "DGS20":   20.0,
    "DGS30":   30.0,
}

START_DATE = "1970-01-01"
END_DATE   = "2024-12-31"

# ── NBER recessions with announcement dates ────────────────────────────────
# (peak, trough, date NBER publicly announced the trough)
# Source: nber.org/research/business-cycle-dating
NBER_ANNOUNCEMENTS = [
    ("1969-12-01", "1970-11-01", "1971-11-01"),
    ("1973-11-01", "1975-03-01", "1975-06-01"),
    ("1980-01-01", "1980-07-01", "1980-07-01"),
    ("1981-07-01", "1982-11-01", "1983-07-01"),
    ("1990-07-01", "1991-03-01", "1991-04-01"),
    ("2001-03-01", "2001-11-01", "2003-07-01"),
    ("2007-12-01", "2009-06-01", "2010-09-01"),
    ("2020-02-01", "2020-04-01", "2020-06-01"),
]


# ══════════════════════════════════════════════════════════════════════════
#  1.  FRED DATA PULL
# ══════════════════════════════════════════════════════════════════════════

def load_fred_yields(
    api_key: str = FRED_API_KEY,
    start:   str = START_DATE,
    end:     str = END_DATE,
) -> pd.DataFrame:
    """
    Pull all 11 Treasury CMT series from FRED, resample to month-start,
    return a clean DataFrame (dates x maturities).

    Note on the 30Y gap: the 30-year bond was suspended Feb 2002 - Feb 2006.
    Those cells are left as NaN — the NS fitter handles missing maturities.
    """
    from fredapi import Fred
    fred = Fred(api_key=api_key)

    frames = {}
    print("Pulling Treasury yields from FRED...")
    for ticker, maturity in FRED_TICKERS.items():
        try:
            s = fred.get_series(ticker, observation_start=start, observation_end=end)
            s.name = maturity
            frames[maturity] = s
            print(f"  {ticker:8s} ({maturity:5.3f}y)  {len(s):4d} obs"
                  f"  [{s.first_valid_index().date()} -> {s.last_valid_index().date()}]")
        except Exception as e:
            print(f"  {ticker:8s} -- FAILED: {e}")

    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)

    # Resample to month-start using last available reading each month
    df = df.resample("MS").last()
    df.index.name = "date"
    df.columns.name = "maturity"
    df = df.sort_index()

    # Remove stale/erroneous values
    df = df.where(df > 0)    # zero = missing sentinel
    df = df.where(df < 25)   # >25% is a data error

    # Forward-fill gaps of 2 months max (holiday/lag artefacts only)
    # The 30Y 2002-2006 gap (48m) is intentionally left as NaN
    df = df.apply(lambda col: col.ffill(limit=2))

    print(f"\nYield panel shape : {df.shape[0]} months x {df.shape[1]} maturities")
    print(f"Date range        : {df.index[0].date()} -> {df.index[-1].date()}")
    missing_pct = df.isnull().mean().mul(100).round(1)
    print("\nMissingness by maturity (%):")
    for mat, pct in missing_pct.items():
        bar = "#" * int(pct / 2)
        print(f"  {mat:5.3f}y  {pct:5.1f}%  {bar}")

    return df


# ══════════════════════════════════════════════════════════════════════════
#  2.  NBER RECESSION INDICATOR
# ══════════════════════════════════════════════════════════════════════════

def build_recession_indicator(date_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Returns a DataFrame with two columns:
      recession    -- 0/1 NBER indicator for each month
      announced_by -- date NBER publicly called the trough
                      (used in Phase 5 to prevent look-ahead bias)
    """
    rec       = pd.Series(0,      index=date_index, name="recession",    dtype=int)
    announced = pd.Series(pd.NaT, index=date_index, name="announced_by",
                          dtype="datetime64[ns]")

    for peak, trough, ann in NBER_ANNOUNCEMENTS:
        mask = (date_index >= peak) & (date_index <= trough)
        rec[mask]       = 1
        announced[mask] = pd.Timestamp(ann)

    df_rec = pd.concat([rec, announced], axis=1)
    df_rec.index.name = "date"

    n_rec = int(rec.sum())
    n_tot = len(rec)
    print(f"\nRecession indicator:")
    print(f"  {n_rec} recession months / {n_tot} total ({100*n_rec/n_tot:.1f}%)")
    print(f"  {len(NBER_ANNOUNCEMENTS)} NBER cycles with announcement dates stored")

    return df_rec


# ══════════════════════════════════════════════════════════════════════════
#  3.  QC REPORT
# ══════════════════════════════════════════════════════════════════════════

def run_qc(df_yields: pd.DataFrame, df_rec: pd.DataFrame,
           out_dir: str = "output") -> pd.Series:
    """
    Generate QC report:
      Panel A — missingness heatmap (maturities x time)
      Panel B — valid maturity count per month with recession shading
    Flags any month with < 6 valid maturities.
    """
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle("Phase 1 QC — US Treasury Yield Panel",
                 fontsize=13, fontweight="bold")

    # Panel A: missingness heatmap
    ax = axes[0]
    miss = df_yields.isnull().astype(int).T
    ax.imshow(miss.values, aspect="auto", cmap="RdYlGn_r",
              vmin=0, vmax=1, interpolation="none")
    ax.set_yticks(range(len(df_yields.columns)))
    ax.set_yticklabels([f"{m:.3g}y" for m in df_yields.columns], fontsize=9)
    n = len(df_yields)
    step = max(1, n // 12)
    ax.set_xticks(range(0, n, step))
    ax.set_xticklabels(
        [df_yields.index[i].strftime("%Y") for i in range(0, n, step)],
        fontsize=8, rotation=45
    )
    ax.set_title("Missingness heatmap (red = missing)", fontsize=10, loc="left")

    # Panel B: valid count + recession shading
    ax = axes[1]
    valid_count = df_yields.notnull().sum(axis=1)
    rec = df_rec["recession"]
    common = valid_count.index.intersection(rec.index)

    in_rec, start_rec = False, None
    for date in common:
        if rec[date] == 1 and not in_rec:
            start_rec, in_rec = date, True
        elif rec[date] == 0 and in_rec:
            ax.axvspan(start_rec, date, color="#E8E0F0", alpha=0.6, zorder=0)
            in_rec = False
    if in_rec:
        ax.axvspan(start_rec, common[-1], color="#E8E0F0", alpha=0.6, zorder=0)

    ax.plot(valid_count.index, valid_count.values, color="#185FA5", linewidth=1.2)
    ax.axhline(6, color="#DC2626", linewidth=0.8, linestyle="--", label="Min threshold (6)")
    ax.fill_between(valid_count.index, valid_count.values, 6,
                    where=valid_count.values < 6,
                    color="#FCEBEB", alpha=0.8, label="Below threshold")
    ax.set_ylabel("Valid maturities per month")
    ax.set_title("Data coverage (shading = NBER recession)", fontsize=10, loc="left")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.set_ylim(0, len(df_yields.columns) + 1)

    plt.tight_layout()
    path = os.path.join(out_dir, "qc_report.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[QC] Report saved -> {path}")

    flagged = valid_count[valid_count < 6]
    if len(flagged):
        print(f"[QC] WARNING: {len(flagged)} months with < 6 valid maturities:")
        print(flagged.to_string())
    else:
        print("[QC] All months have >= 6 valid maturities.")

    return valid_count


# ══════════════════════════════════════════════════════════════════════════
#  4.  MAIN
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.makedirs("data",   exist_ok=True)
    os.makedirs("output", exist_ok=True)

    print("=" * 60)
    print("PHASE 1 -- Data Acquisition")
    print("=" * 60)

    df_yields = load_fred_yields()
    df_yields.to_csv("data/yields_monthly.csv")
    print(f"\n[OK] data/yields_monthly.csv  {df_yields.shape}")

    df_rec = build_recession_indicator(df_yields.index)
    df_rec.to_csv("data/recession_indicator.csv")
    print(f"[OK] data/recession_indicator.csv")

    run_qc(df_yields, df_rec)

    print("\n[OK] Phase 1 complete -- run phase2_nelson_siegel.py next")
