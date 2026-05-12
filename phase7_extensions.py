"""
Phase 7 — Extensions: Statistical Rigour + Live Data
======================================================
Three additions that differentiate the project:

  7A. Diebold-Mariano test
      Formally tests whether M1 (slope probit) beats the naive
      no-recession benchmark at statistical significance.
      Output: data/dm_test_results.csv + output/dm_test.png

  7B. Calibration analysis
      Reliability diagram: does a stated 30% probability correspond
      to a 30% historical recession frequency?
      Output: output/calibration.png

  7C. Live FRED yield pull
      Fetches today's actual Treasury yields using the API key.
      Output: data/live_yields.json (read automatically by dashboard)

Run:
    python3 phase7_extensions.py

Then restart the dashboard:
    streamlit run phase6_dashboard.py
"""

import os
import json
import sys
import warnings
import datetime
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from scipy import stats
from scipy.stats import norm
import statsmodels.api as sm

# ── Config ─────────────────────────────────────────────────────────────────
FRED_API_KEY = "e0b05c797576a3db6d0a186713a70f4d"
HORIZONS     = [3, 6, 12]

FRED_TICKERS = {
    "DGS1MO": 0.083, "DGS3MO": 0.25,  "DGS6MO": 0.5,
    "DGS1":   1.0,   "DGS2":   2.0,   "DGS3":   3.0,
    "DGS5":   5.0,   "DGS7":   7.0,   "DGS10":  10.0,
    "DGS20":  20.0,  "DGS30":  30.0,
}


# ══════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════

def _read(path):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.index.name = "date"
    return df


# ══════════════════════════════════════════════════════════════════════════
#  7A. Diebold-Mariano Test
# ══════════════════════════════════════════════════════════════════════════

def diebold_mariano(e1: np.ndarray, e2: np.ndarray, h: int = 1) -> dict:
    """
    Diebold-Mariano (1995) test for equal predictive accuracy.
    H0: E[d_t] = 0  where d_t = L(e1_t) - L(e2_t), L = quadratic loss.
    Harvey-Leybourne-Newbold (1997) small-sample correction applied.

    Negative DM stat => e1 has smaller loss => model 1 is better.
    """
    d     = e1**2 - e2**2
    n     = len(d)
    d_bar = np.mean(d)

    # Newey-West long-run variance with h-1 lags
    gamma0 = np.var(d, ddof=0)
    gammas = [np.mean((d[k:] - d_bar) * (d[:-k] - d_bar))
              for k in range(1, max(1, h))]
    var_d  = (gamma0 + 2 * sum(gammas)) / n

    if var_d <= 0:
        return dict(dm_stat=np.nan, p_value=np.nan,
                    significant=False,
                    conclusion="insufficient variance")

    dm_raw = d_bar / np.sqrt(var_d)

    # HLN small-sample correction factor
    hln    = np.sqrt((n + 1 - 2*h + h*(h-1)/n) / n)
    dm_adj = dm_raw * hln

    p_val  = 2 * stats.t.sf(abs(dm_adj), df=n - 1)

    if dm_adj < 0 and p_val < 0.05:
        conclusion = "M1 significantly better than naive (p<0.05)"
    elif p_val >= 0.05:
        conclusion = "Cannot reject equal accuracy (p>=0.05)"
    else:
        conclusion = "Naive significantly better than M1"

    return dict(
        dm_stat     = round(float(dm_adj), 4),
        p_value     = round(float(p_val),  4),
        loss_diff   = round(float(d_bar),  6),
        n_obs       = n,
        significant = bool(p_val < 0.05),
        conclusion  = conclusion,
    )


def run_dm_tests(df_factors, df_rec, df_oos) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("7A. Diebold-Mariano Tests  (M1 vs naive no-recession baseline)")
    print("=" * 60)

    common = df_factors.index.intersection(df_rec.index)
    df     = df_factors[["b2"]].loc[common].copy()
    df["recession"] = df_rec.loc[common, "recession"]
    df = df.dropna()

    rows = []
    for h in HORIZONS:
        col_fwd = f"rec_fwd_{h}"
        df[col_fwd] = df["recession"].shift(-h)

        oos_col = next(
            (c for c in df_oos.columns if "M1" in c and f"h{h}" in c),
            None)
        if oos_col is None:
            print(f"  h={h}m: OOS column not found, skipping")
            continue

        sub     = df[[col_fwd]].dropna()
        oos     = df_oos[oos_col].dropna()
        common2 = sub.index.intersection(oos.index)

        if len(common2) < 40:
            print(f"  h={h}m: only {len(common2)} obs, skipping")
            continue

        y_true  = sub.loc[common2, col_fwd].values.astype(float)
        y_m1    = oos.loc[common2].values.astype(float)
        y_naive = np.full_like(y_true, y_true.mean())

        res = diebold_mariano(y_true - y_m1, y_true - y_naive, h=h)

        print(f"\n  h={h}m  (n={res['n_obs']})")
        print(f"    Naive baseline         : {y_true.mean():.3f}")
        print(f"    DM statistic (HLN)     : {res['dm_stat']:.4f}")
        print(f"    p-value (two-sided)    : {res['p_value']:.4f}")
        print(f"    Significant @5%        : {res['significant']}")
        print(f"    Conclusion             : {res['conclusion']}")

        rows.append({"horizon": h,
                     "naive_base": round(float(y_true.mean()), 4),
                     **res})

    return pd.DataFrame(rows)


def plot_dm(df_dm: pd.DataFrame, out: str = "output/dm_test.png"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    fig.suptitle(
        "Diebold-Mariano Test — M1 Slope Probit vs Naive (Unconditional Mean)\n"
        "H₀: Equal predictive accuracy  |  Harvey-Leybourne-Newbold correction",
        fontsize=11, fontweight="bold")

    hz     = df_dm["horizon"].tolist()
    stats_ = df_dm["dm_stat"].tolist()
    pvals  = df_dm["p_value"].tolist()
    sigs   = df_dm["significant"].tolist()
    labels = [f"h={h}m" for h in hz]

    # Panel A: DM statistics
    ax = axes[0]
    bar_cols = ["#16A34A" if s else "#9CA3AF" for s in sigs]
    bars = ax.bar(labels, stats_, color=bar_cols, alpha=0.85,
                  edgecolor="white", width=0.45)
    ax.axhline(0,     color="black",   linewidth=0.8)
    ax.axhline(-1.96, color="#DC2626", linewidth=1.1, linestyle="--",
               alpha=0.8, label="±1.96 (5% critical value)")
    ax.axhline( 1.96, color="#DC2626", linewidth=1.1, linestyle="--",
               alpha=0.8)
    for bar, val in zip(bars, stats_):
        ypos = val - 0.12 if val < 0 else val + 0.05
        ax.text(bar.get_x() + bar.get_width()/2, ypos,
                f"{val:.3f}", ha="center", va="top" if val < 0 else "bottom",
                fontsize=10, fontweight="bold",
                color="white" if abs(val) > 0.5 else "black")
    ax.set_ylabel("DM statistic (HLN)", fontsize=10)
    ax.set_title("DM Statistic\n(negative = M1 better than naive)",
                 fontsize=10, loc="left")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")

    # Panel B: p-values
    ax = axes[1]
    p_cols = ["#16A34A" if p < 0.05 else "#9CA3AF" for p in pvals]
    ax.bar(labels, pvals, color=p_cols, alpha=0.85,
           edgecolor="white", width=0.45)
    ax.axhline(0.05, color="#DC2626", linewidth=1.2, linestyle="--",
               label="5% significance")
    ax.axhline(0.10, color="#D97706", linewidth=0.8, linestyle=":",
               label="10% significance")
    for i, (p, s) in enumerate(zip(pvals, sigs)):
        ax.text(i, p + 0.005, "✓ sig." if s else "n.s.",
                ha="center", fontsize=9,
                color="#16A34A" if s else "#6B7280",
                fontweight="bold")
    ax.set_ylabel("p-value", fontsize=10)
    ax.set_title("p-value\n(below red line = reject H₀)",
                 fontsize=10, loc="left")
    ax.set_ylim(0, max(0.6, max(pvals) * 1.4))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.2, axis="y")

    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] {out}")


# ══════════════════════════════════════════════════════════════════════════
#  7B. Calibration Analysis
# ══════════════════════════════════════════════════════════════════════════

def calibration_curve(y_true, y_prob, n_bins=8):
    """Bin predicted probs, compute mean predicted vs observed frequency."""
    bins    = np.linspace(0, 1, n_bins + 1)
    idx     = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    centers, mean_pred, obs_freq, counts = [], [], [], []
    for i in range(n_bins):
        m = idx == i
        if m.sum() < 3:
            continue
        centers.append((bins[i] + bins[i+1]) / 2)
        mean_pred.append(float(y_prob[m].mean()))
        obs_freq.append(float(y_true[m].mean()))
        counts.append(int(m.sum()))
    return (np.array(centers), np.array(mean_pred),
            np.array(obs_freq), np.array(counts))


def brier_skill_score(y_true, y_prob):
    clim = y_true.mean()
    return 1 - np.mean((y_true - y_prob)**2) / np.mean((y_true - clim)**2)


def run_calibration(df_factors, df_rec, df_oos):
    print("\n" + "=" * 60)
    print("7B. Calibration — Reliability Diagrams")
    print("=" * 60)

    common = df_factors.index.intersection(df_rec.index)
    df     = df_factors[["b2"]].loc[common].copy()
    df["recession"] = df_rec.loc[common, "recession"]
    df = df.dropna()

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle(
        "Calibration Reliability Diagrams — M1 Real-Time OOS\n"
        "Perfect calibration = diagonal  |  "
        "Dot size ∝ number of observations in bin",
        fontsize=11, fontweight="bold")

    for ax, h in zip(axes, HORIZONS):
        col_fwd     = f"rec_fwd_{h}"
        df[col_fwd] = df["recession"].shift(-h)

        oos_col = next(
            (c for c in df_oos.columns if "M1" in c and f"h{h}" in c),
            None)
        if oos_col is None:
            ax.text(0.5, 0.5, "No data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        sub     = df[[col_fwd]].dropna()
        oos     = df_oos[oos_col].dropna()
        common2 = sub.index.intersection(oos.index)
        if len(common2) < 30:
            continue

        y_true  = sub.loc[common2, col_fwd].values.astype(float)
        y_prob  = oos.loc[common2].values.astype(float)

        _, mean_pred, obs_freq, counts = calibration_curve(
            y_true, y_prob, n_bins=8)

        bss = brier_skill_score(y_true, y_prob)
        ece = float(np.sum(counts * np.abs(obs_freq - mean_pred))
                    / counts.sum())

        # Perfect calibration
        ax.plot([0, 1], [0, 1], "k--", linewidth=1.0,
                alpha=0.5, label="Perfect calibration")

        # Calibration scatter
        ax.scatter(mean_pred, obs_freq,
                   s=counts * 1.5, c=counts,
                   cmap="Blues", vmin=0,
                   edgecolors="#1E3A5F", linewidth=0.8,
                   zorder=5, label="Calibration bins")
        ax.plot(mean_pred, obs_freq,
                color="#2563EB", linewidth=1.6, alpha=0.75, zorder=4)

        # Gap shading
        ax.fill_between(mean_pred, mean_pred, obs_freq,
                        alpha=0.10, color="#DC2626",
                        label="Calibration gap")

        # Marginal histogram on secondary axis
        ax2 = ax.twinx()
        ax2.hist(y_prob, bins=16, color="#D1D5DB",
                 alpha=0.35, density=True)
        ax2.set_ylim(0, ax2.get_ylim()[1] * 4)
        ax2.set_yticks([])

        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("Mean predicted probability", fontsize=10)
        ax.set_ylabel("Observed recession frequency", fontsize=10)
        ax.set_title(
            f"h = {h} months ahead\n"
            f"BSS = {bss:.3f}   ECE = {ece:.3f}   n = {len(y_true)}",
            fontsize=10, fontweight="bold", loc="left")
        ax.legend(fontsize=7.5, loc="upper left")
        ax.grid(True, alpha=0.2)

        print(f"\n  h={h}m  (n={len(y_true)})")
        print(f"    Brier Skill Score   : {bss:.4f}  "
              f"(0=no skill, 1=perfect)")
        print(f"    Exp. Calib. Error   : {ece:.4f}  "
              f"(0=perfectly calibrated)")
        print(f"    Unconditional freq  : {y_true.mean():.3f}")

    plt.tight_layout()
    out = "output/calibration.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[Plot] {out}")


# ══════════════════════════════════════════════════════════════════════════
#  7C. Live FRED Yield Pull
# ══════════════════════════════════════════════════════════════════════════

def fetch_live_yields(api_key: str = FRED_API_KEY) -> dict:
    """
    Pull the latest observation for every Treasury maturity from FRED.
    Saves to data/live_yields.json. Dashboard reads this automatically
    on next load and pre-fills the sidebar with today's yields.
    """
    print("\n" + "=" * 60)
    print("7C. Live FRED Yield Pull")
    print("=" * 60)

    from fredapi import Fred
    fred     = Fred(api_key=api_key)
    yields   = {}
    failures = []

    for ticker, maturity in FRED_TICKERS.items():
        try:
            s = fred.get_series(ticker).dropna()
            s = s[s > 0]
            if len(s) == 0:
                raise ValueError("empty series")
            val  = float(s.iloc[-1])
            date = str(s.index[-1].date())
            yields[str(maturity)] = {"yield": val, "date": date}
            print(f"  {ticker:8s} ({maturity:5.3f}y)  {val:.3f}%  [{date}]")
        except Exception as e:
            print(f"  {ticker:8s} -- FAILED: {e}")
            failures.append(ticker)

    if not yields:
        print("[ERROR] No yields retrieved — check API key / connection")
        return {}

    payload = {
        "fetched_at": datetime.datetime.utcnow().isoformat() + "Z",
        "yields":     yields,
        "failures":   failures,
    }

    path = "data/live_yields.json"
    os.makedirs("data", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"\n[OK] {path}  ({len(yields)} maturities, "
          f"{len(failures)} failures)")

    # Summarise curve shape
    mats  = sorted(float(k) for k in yields)
    short = yields[str(mats[0])]["yield"]
    long_ = yields[str(mats[-1])]["yield"]
    spd   = long_ - short
    shape = ("upward sloping" if spd >  0.3 else
             "flat"           if abs(spd) <= 0.3 else
             "inverted")
    print(f"\n  Curve shape  : {shape}")
    print(f"  Short end    : {short:.3f}%  ({mats[0]:.3g}y)")
    print(f"  Long end     : {long_:.3f}%  ({mats[-1]:.0f}y)")
    print(f"  Spread       : {spd:+.3f}%")

    return yields


# ══════════════════════════════════════════════════════════════════════════
#  Dashboard patch — inject live yield support into phase6_dashboard.py
# ══════════════════════════════════════════════════════════════════════════

LIVE_LOADER_CODE = '''
# ── Live yield loader (Phase 7) ────────────────────────────────────────────
def _load_live_yields_from_json():
    """
    Read data/live_yields.json if it exists and was fetched today.
    Returns {maturity_float: yield_float} or None.
    """
    import json as _j, datetime as _dt, os as _os
    path = "data/live_yields.json"
    if not _os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = _j.load(f)
        fetched = _dt.datetime.fromisoformat(
            data["fetched_at"].replace("Z", ""))
        age_h = (_dt.datetime.utcnow() - fetched).total_seconds() / 3600
        if age_h > 24:
            return None
        return {float(k): v["yield"] for k, v in data["yields"].items()}
    except Exception:
        return None
'''


def patch_dashboard():
    """
    Patch phase6_dashboard.py to:
      1. Load data/live_yields.json and pre-fill sidebar inputs
      2. Add a 'Fetch live yields' button to the sidebar
    """
    path = "phase6_dashboard.py"
    if not os.path.exists(path):
        print(f"[Skip] {path} not found")
        return

    with open(path, "r") as f:
        content = f.read()

    if "_load_live_yields_from_json" in content:
        print("[Skip] Dashboard already patched with live yield loader")
        return

    # 1. Inject live loader after 'import io'
    if "import io\n" not in content:
        print("[Skip] Could not find injection anchor in dashboard")
        return
    content = content.replace(
        "import io\n",
        "import io\n" + LIVE_LOADER_CODE + "\n")

    # 2. Pre-fill sidebar defaults from live yields
    old_default = "        default = LATEST_YIELDS_FALLBACK.get(mat, 4.0)"
    new_default = ("        _live_y = _load_live_yields_from_json()\n"
                   "        default = (_live_y or "
                   "LATEST_YIELDS_FALLBACK).get(mat, 4.0)")
    content = content.replace(old_default, new_default)

    # 3. Add refresh button just after sidebar header
    old_hdr = '    st.sidebar.header("📊 Input Yield Curve")\n'
    new_hdr = (
        '    st.sidebar.header("📊 Input Yield Curve")\n'
        '    if st.sidebar.button("🔄 Fetch live FRED yields"):\n'
        '        import subprocess\n'
        '        subprocess.Popen(["python3", "phase7_extensions.py",'
        ' "--live-only"])\n'
        '        st.sidebar.success("Fetching... refresh page in ~10s")\n'
        '    _live_status = _load_live_yields_from_json()\n'
        '    if _live_status:\n'
        '        st.sidebar.info("📡 Live yields active (fetched today)")\n'
        '    else:\n'
        '        st.sidebar.caption("Using Dec 2024 defaults. '
        'Click above to fetch live.")\n'
    )
    content = content.replace(old_hdr, new_hdr)

    with open(path, "w") as f:
        f.write(content)
    print("[OK] phase6_dashboard.py patched — live yield support added")


# ══════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    live_only = "--live-only" in sys.argv

    os.makedirs("data",   exist_ok=True)
    os.makedirs("output", exist_ok=True)

    if live_only:
        fetch_live_yields()
        sys.exit(0)

    print("=" * 60)
    print("PHASE 7 — Extensions")
    print("=" * 60)

    # Load outputs from Phases 2, 4, 5
    print("\nLoading data...")
    try:
        df_factors = _read("data/ns_factors.csv")
        df_rec     = _read("data/recession_indicator.csv")
        df_oos     = _read("data/probit_oos_results.csv")
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}")
        print("Run the pipeline first:")
        print("  python3 phase1_data.py")
        print("  python3 phase2_nelson_siegel.py")
        print("  python3 phase4_probit.py")
        sys.exit(1)

    print(f"  Factors : {df_factors.shape}")
    print(f"  Recession: {df_rec.shape}")
    print(f"  OOS probs: {df_oos.shape}")
    print(f"  OOS cols : {df_oos.columns.tolist()}")

    # 7A: Diebold-Mariano
    df_dm = run_dm_tests(df_factors, df_rec, df_oos)
    if len(df_dm):
        df_dm.to_csv("data/dm_test_results.csv", index=False)
        print(f"\n[OK] data/dm_test_results.csv")
        plot_dm(df_dm)

    # 7B: Calibration
    run_calibration(df_factors, df_rec, df_oos)

    # 7C: Live yields
    fetch_live_yields()

    # Patch dashboard
    print()
    patch_dashboard()

    print("\n" + "=" * 60)
    print("Phase 7 complete.")
    print("  output/dm_test.png      — Diebold-Mariano visual")
    print("  output/calibration.png  — Reliability diagrams")
    print("  data/live_yields.json   — Today's live Treasury yields")
    print("  phase6_dashboard.py     — Patched (live yield button)")
    print()
    print("Restart the dashboard:")
    print("  streamlit run phase6_dashboard.py")
    print("=" * 60)
