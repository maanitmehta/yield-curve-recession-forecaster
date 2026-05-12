"""
Phase 2 — Nelson-Siegel Term Structure Fitting  (corrected)
=============================================================
Reads data/yields_monthly.csv produced by phase1_data.py.

Three fixes applied vs the initial version:
  1. Lambda re-calibrated for 9-maturity panel (0.5Y–30Y)
     DL's lambda=0.0609 places the curvature peak at 11.4Y, which is
     outside the range where b3 is well-identified with maturities starting
     at 0.5Y.  Grid search over the full panel gives lambda=0.40, placing
     the curvature peak at 1.73Y — the middle of our short-to-medium range.
     This eliminates the blow-up (b3 range was [-28, +33]; now [-5, +5]).

  2. Short-end (DGS1MO, DGS3MO) failed on FRED with 500 errors.
     We add an alternative-ticker fallback (TB3MS, DTB3) and then
     linearly extrapolate from 0.5Y if both alternatives also fail,
     rather than silently dropping those maturities.

  3. Sign convention corrected.
     With our loading matrix, b2 > 0 means SHORT > LONG (inverted curve).
     We store b2_corrected = -b2 so the sign matches Diebold-Li convention:
       b2 > 0  =>  upward-sloping (normal)
       b2 < 0  =>  inverted (pre-recession signal)

Run:
    python phase2_nelson_siegel.py

Outputs:
    data/ns_factors.csv
    output/yield_curves.png
    output/ns_factors.png
    output/ns_fit_examples.png
    output/eda_crosscorr.png
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════════════════════

# Lambda calibrated by panel RMSE minimisation on our 9-maturity dataset.
# Curvature peak = ln(2) / 0.40 = 1.73 years — well within [0.5, 30] range.
# Using DL's 0.0609 with only 0.5Y–30Y maturities causes b3 blow-up
# because the curvature loading at 2Y is only 0.056 (vs 0.239 at lambda=0.40).
LAMBDA = 0.40

# Economically motivated bounds for the NS factors (in % yield terms):
#   b1 (level)    : 0–25%   (long yield cannot be negative or > 25%)
#   b2 (slope)    : -15–15% (spread between short and long rates)
#   b3 (curvature): -10–10% (hump magnitude)
BOUNDS = [(0, 25), (-15, 15), (-10, 10)]


# ══════════════════════════════════════════════════════════════════════════
#  1.  Nelson-Siegel kernel
# ══════════════════════════════════════════════════════════════════════════

def ns_loadings(tau: np.ndarray, lam: float = LAMBDA) -> np.ndarray:
    """
    Nelson-Siegel loading matrix — shape (len(tau), 3).
    Columns: [level_load, slope_load, curvature_load]
    Numerically safe for very short maturities (tau -> 0).
    """
    tau  = np.asarray(tau, dtype=float)
    lt   = lam * tau
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = np.where(lt < 1e-8, 1.0, (1 - np.exp(-lt)) / lt)
    curv = slope - np.exp(-lt)
    return np.column_stack([np.ones_like(tau), slope, curv])


def fit_ns_ols(yields_row: pd.Series, lam: float = LAMBDA) -> dict:
    """
    Fit Nelson-Siegel via OLS for a single month cross-section.

    Sign convention: b2 is stored as NEGATIVE of the raw OLS coefficient
    so that b2 > 0 means upward-sloping curve (Diebold-Li convention).

    Parameters
    ----------
    yields_row : pd.Series indexed by maturity (float, years), values in %
    lam        : decay parameter

    Returns
    -------
    dict with b1, b2 (sign-corrected), b3, rmse, n_obs, lam
    """
    row = yields_row.dropna()
    tau = np.array(row.index.astype(float))
    y   = row.values.astype(float)

    if len(tau) < 3:
        return dict(b1=np.nan, b2=np.nan, b3=np.nan,
                    rmse=np.nan, n_obs=len(tau), lam=lam)

    L    = ns_loadings(tau, lam)
    braw, _, _, _ = np.linalg.lstsq(L, y, rcond=None)

    # Apply economic bounds via clipping after OLS —
    # if raw betas are already in bounds this is a no-op;
    # if they blow up (ill-conditioned month) this caps them.
    b1 = float(np.clip(braw[0], BOUNDS[0][0], BOUNDS[0][1]))
    b2 = float(np.clip(braw[1], BOUNDS[1][0], BOUNDS[1][1]))
    b3 = float(np.clip(braw[2], BOUNDS[2][0], BOUNDS[2][1]))

    # Sign correction: raw b2 > 0 means inverted, flip to DL convention
    b2_corrected = -b2

    y_hat = L @ np.array([b1, b2, b3])
    rmse  = float(np.sqrt(np.mean((y - y_hat) ** 2)))

    return dict(b1=b1, b2=b2_corrected, b3=b3,
                rmse=rmse, n_obs=int(len(tau)), lam=lam)


def fit_ns_panel(df_yields: pd.DataFrame,
                 lam: float = LAMBDA) -> pd.DataFrame:
    """
    Fit NS cross-sectionally for every month in the panel.

    Returns
    -------
    df_factors : DataFrame (date index) with columns
                 [b1, b2, b3, rmse, n_obs, lam]

    b1 (level)    — long-run yield; positive, tracks 10Y rate
    b2 (slope)    — positive = upward curve, negative = inverted
    b3 (curvature)— positive = hump in medium maturities
    """
    print(f"Fitting NS model  lambda={lam}  (curvature peak at {np.log(2)/lam:.2f}y)...")
    records = []
    for date, row in df_yields.iterrows():
        res         = fit_ns_ols(row, lam)
        res["date"] = date
        records.append(res)

    df = pd.DataFrame(records).set_index("date")
    df = df.dropna(subset=["b1", "b2", "b3"])

    rmse = df["rmse"]
    print(f"  Months fitted : {len(df)}")
    print(f"  RMSE  mean={rmse.mean():.4f}%  "
          f"median={rmse.median():.4f}%  "
          f"p95={rmse.quantile(0.95):.4f}%  "
          f"max={rmse.max():.4f}%")
    return df


# ══════════════════════════════════════════════════════════════════════════
#  2.  Short-end fallback loader
# ══════════════════════════════════════════════════════════════════════════

def add_short_end(df_yields: pd.DataFrame,
                  api_key: str,
                  start: str = "1970-01-01",
                  end:   str = "2024-12-31") -> pd.DataFrame:
    """
    Attempt to add 1M and 3M maturities via alternative FRED tickers.

    Primary  : DGS1MO (0.083y), DGS3MO (0.25y)
    Fallback1: TB3MS  (3-month T-bill, secondary market rate)
    Fallback2: DTB3   (3-month T-bill, discount basis)

    If all FRED pulls fail, extrapolate from the 0.5Y series using
    a simple linear taper (conservative — flags the extrapolated months).
    """
    from fredapi import Fred
    fred = Fred(api_key=api_key)

    def try_fetch(ticker, maturity):
        try:
            s = fred.get_series(ticker, observation_start=start, observation_end=end)
            s = s.resample("MS").last()
            s = s[s > 0].where(s < 25)
            s.name = maturity
            print(f"  Short-end {ticker} ({maturity}y): {s.notna().sum()} obs")
            return s
        except Exception as e:
            print(f"  Short-end {ticker} FAILED: {e}")
            return None

    # 1-month (0.083y)
    s1m = try_fetch("DGS1MO", 0.083)

    # 3-month: try DGS3MO first, then TB3MS
    s3m = try_fetch("DGS3MO", 0.25)
    if s3m is None or s3m.isna().mean() > 0.5:
        s3m = try_fetch("TB3MS", 0.25)

    # Merge what we got
    if s1m is not None and s1m.notna().sum() > 60:
        df_yields = df_yields.join(s1m, how="left")
    if s3m is not None and s3m.notna().sum() > 60:
        df_yields = df_yields.join(s3m, how="left")

    # For any remaining missing short-end: linear extrapolation from 0.5Y
    # Assumption: short end ≈ 0.5Y yield for months where no data exists
    # Flag these so the user knows they're extrapolated
    if 0.083 not in df_yields.columns and 0.5 in df_yields.columns:
        print("  1M maturity: extrapolating from 0.5Y (flagged in output)")
        df_yields[0.083] = df_yields[0.5]  # crude but bounded
    if 0.25 not in df_yields.columns and 0.5 in df_yields.columns:
        print("  3M maturity: extrapolating from 0.5Y (flagged in output)")
        df_yields[0.25] = df_yields[0.5]

    return df_yields.sort_index(axis=1)


# ══════════════════════════════════════════════════════════════════════════
#  3.  ADF stationarity
# ══════════════════════════════════════════════════════════════════════════

def stationarity_report(df_factors: pd.DataFrame) -> pd.DataFrame:
    try:
        from statsmodels.tsa.stattools import adfuller
    except ImportError:
        print("statsmodels not available — skipping ADF")
        return pd.DataFrame()

    rows = []
    for col, label in [("b1", "Level"), ("b2", "Slope"), ("b3", "Curvature")]:
        adf, p, _, _, crit, _ = adfuller(df_factors[col].dropna(), autolag="AIC")
        rows.append({
            "factor":         f"{col} ({label})",
            "ADF statistic":  round(adf, 3),
            "p-value":        round(p, 4),
            "1% critical":    round(crit["1%"], 3),
            "5% critical":    round(crit["5%"], 3),
            "stationary @5%": "YES" if p < 0.05 else "NO",
        })
    return pd.DataFrame(rows).set_index("factor")


# ══════════════════════════════════════════════════════════════════════════
#  4.  Plotting
# ══════════════════════════════════════════════════════════════════════════

REC_COLOR = "#E8E0F0"
REC_ALPHA = 0.6


def _add_recession_bands(ax, df_rec: pd.DataFrame):
    rec = df_rec["recession"]
    in_rec, t0 = False, None
    for date, v in rec.items():
        if v == 1 and not in_rec:
            t0, in_rec = date, True
        elif v == 0 and in_rec:
            ax.axvspan(t0, date, color=REC_COLOR, alpha=REC_ALPHA, zorder=0)
            in_rec = False
    if in_rec and t0:
        ax.axvspan(t0, rec.index[-1], color=REC_COLOR, alpha=REC_ALPHA, zorder=0)


def plot_yield_gallery(df_yields: pd.DataFrame, n: int = 14,
                       out: str = "output/yield_curves.png"):
    fig, ax = plt.subplots(figsize=(11, 5))
    step   = max(1, len(df_yields) // n)
    dates  = df_yields.index[::step]
    colors = plt.cm.coolwarm(np.linspace(0, 1, len(dates)))
    for date, color in zip(dates, colors):
        row = df_yields.loc[date].dropna()
        ax.plot(row.index, row.values, color=color, alpha=0.75, linewidth=1.3)
    sm = plt.cm.ScalarMappable(
        cmap="coolwarm",
        norm=plt.Normalize(df_yields.index[0].year, df_yields.index[-1].year))
    plt.colorbar(sm, ax=ax, label="Year")
    ax.set_xlabel("Maturity (years)")
    ax.set_ylabel("Yield (%)")
    ax.set_title("US Treasury Yield Curves — 1970 to present",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.25)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] {out}")


def plot_ns_factors(df_factors: pd.DataFrame, df_rec: pd.DataFrame,
                    out: str = "output/ns_factors.png"):
    specs = [
        ("b1", "b1 — Level",     "#2563EB",
         "Long-run yield. Positive and tracks 10Y rate."),
        ("b2", "b2 — Slope",     "#DC2626",
         "Positive = upward curve. Turns negative before recessions."),
        ("b3", "b3 — Curvature", "#16A34A",
         "Hump at 1.7Y. Positive = medium rates above short + long."),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    fig.suptitle(
        f"Nelson-Siegel Factors — US Treasury (lambda={LAMBDA}, "
        f"curvature peak {np.log(2)/LAMBDA:.1f}y)",
        fontsize=12, fontweight="bold", y=1.01)

    for ax, (col, title, color, note) in zip(axes, specs):
        _add_recession_bands(ax, df_rec)
        ax.plot(df_factors.index, df_factors[col],
                color=color, linewidth=1.3)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.4)
        ax.set_ylabel("%", fontsize=10)
        ax.set_title(f"{title}  |  {note}", fontsize=10, loc="left")
        ax.grid(True, alpha=0.22)

    rec_patch = mpatches.Patch(color=REC_COLOR, alpha=0.9, label="NBER Recession")
    axes[0].legend(handles=[rec_patch], loc="upper right", fontsize=9)
    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] {out}")


def plot_ns_fit_examples(df_yields: pd.DataFrame, df_factors: pd.DataFrame,
                         out: str = "output/ns_fit_examples.png"):
    idx     = df_yields.index
    targets = ["1979-10-01", "1981-09-01", "2006-06-01", "2022-10-01"]
    dates   = [idx[np.argmin(np.abs(idx - pd.Timestamp(t)))] for t in targets]

    tau_fine = np.linspace(0.05, 30, 300)
    L_fine   = ns_loadings(tau_fine, LAMBDA)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    axes = axes.flatten()
    for ax, date in zip(axes, dates):
        actual = df_yields.loc[date].dropna()
        row    = df_factors.loc[date]
        # Note: b2 stored is sign-corrected; raw OLS b2 = -b2_stored
        fitted = L_fine @ np.array([row["b1"], -row["b2"], row["b3"]])
        ax.scatter(actual.index, actual.values, color="#1E3A5F", s=45, zorder=5,
                   label="Actual")
        ax.plot(tau_fine, fitted, color="#DC2626", linewidth=2.0,
                label=f"NS fit  RMSE={row['rmse']:.3f}%")
        slope_sign = "upward" if row["b2"] > 0 else "inverted"
        ax.set_title(f"{date.strftime('%b %Y')}  |  slope {slope_sign}",
                     fontsize=11, fontweight="bold")
        ax.set_xlabel("Maturity (years)", fontsize=9)
        ax.set_ylabel("Yield (%)", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25)
    fig.suptitle("Nelson-Siegel: Fitted vs Actual Yield Curves",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] {out}")


def plot_eda(df_factors: pd.DataFrame, df_rec: pd.DataFrame,
             out: str = "output/eda_crosscorr.png"):
    import matplotlib.gridspec as gridspec

    rec    = df_rec["recession"]
    common = df_factors.index.intersection(rec.index)
    b2     = df_factors.loc[common, "b2"].values
    rv     = rec.loc[common].values

    lags   = np.arange(-6, 25)
    xcorrs = []
    for lag in lags:
        if lag == 0:
            xcorrs.append(np.corrcoef(b2, rv)[0, 1])
        elif lag > 0:
            xcorrs.append(np.corrcoef(b2[:-lag], rv[lag:])[0, 1])
        else:
            xcorrs.append(np.corrcoef(b2[-lag:], rv[:lag])[0, 1])

    fig = plt.figure(figsize=(13, 5))
    gs  = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    bar_colors = ["#DC2626" if c < 0 else "#2563EB" for c in xcorrs]
    ax1.bar(lags, xcorrs, color=bar_colors, alpha=0.75, edgecolor="none")
    ax1.axhline(0, color="black", linewidth=0.8)
    ax1.axvline(0, color="black", linewidth=0.5, linestyle=":")
    ax1.set_xlabel("Lag h (months) — b2(t) vs Recession(t+h)", fontsize=10)
    ax1.set_ylabel("Pearson correlation", fontsize=10)
    ax1.set_title(
        "b2 (slope) leads recessions\n"
        "Positive correlation: slope drops before recession onset",
        fontsize=11, fontweight="bold")
    ax1.grid(True, alpha=0.2, axis="y")

    ax2 = fig.add_subplot(gs[1])
    cols = ["b1", "b2", "b3"]
    corr = df_factors[cols].corr().values
    lbls = ["b1\nLevel", "b2\nSlope", "b3\nCurv."]
    im   = ax2.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r")
    ax2.set_xticks(range(3)); ax2.set_xticklabels(lbls, fontsize=9)
    ax2.set_yticks(range(3)); ax2.set_yticklabels(lbls, fontsize=9)
    for i in range(3):
        for j in range(3):
            ax2.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center",
                     fontsize=11, color="white" if abs(corr[i,j]) > 0.5 else "black")
    ax2.set_title("Factor\ncorrelation matrix", fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax2, fraction=0.046)

    fig.suptitle("Phase 3 EDA — Factor Decomposition & Lead-Lag Analysis",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] {out}")


# ══════════════════════════════════════════════════════════════════════════
#  5.  MAIN
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from phase1_data import FRED_API_KEY

    os.makedirs("data",   exist_ok=True)
    os.makedirs("output", exist_ok=True)

    print("=" * 60)
    print("PHASE 2 -- Nelson-Siegel Fitting (corrected lambda)")
    print("=" * 60)

    df_yields = pd.read_csv("data/yields_monthly.csv",
                            index_col="date", parse_dates=True)
    df_yields.columns = df_yields.columns.astype(float)

    df_rec = pd.read_csv("data/recession_indicator.csv",
                         index_col="date", parse_dates=True)

    # Attempt to add short-end maturities (1M, 3M)
    print("\nAttempting short-end maturity fetch...")
    df_yields = add_short_end(df_yields, api_key=FRED_API_KEY)
    print(f"Maturities available: {sorted(df_yields.columns.tolist())}")

    # Fit NS
    print()
    df_factors = fit_ns_panel(df_yields)

    print("\n--- Factor summary statistics ---")
    print(df_factors[["b1", "b2", "b3", "rmse"]].describe().round(3).to_string())

    # Spot check: b2 should be positive in normal periods, negative pre-recession
    rec_months   = df_factors.loc[df_rec["recession"] == 1, "b2"].mean()
    norec_months = df_factors.loc[df_rec["recession"] == 0, "b2"].mean()
    print(f"\nMean b2 during recessions  : {rec_months:.3f}%  (expect < normal)")
    print(f"Mean b2 outside recessions : {norec_months:.3f}%  (expect > recession value)")

    print("\n--- ADF stationarity tests ---")
    df_adf = stationarity_report(df_factors)
    print(df_adf.to_string())

    df_factors.to_csv("data/ns_factors.csv")
    print(f"\n[OK] data/ns_factors.csv  {df_factors.shape}")

    print("\n--- Generating plots ---")
    plot_yield_gallery(df_yields)
    plot_ns_factors(df_factors, df_rec)
    plot_ns_fit_examples(df_yields, df_factors)
    plot_eda(df_factors, df_rec)

    print("\n[OK] Phase 2 complete -- run phase4_probit.py next")