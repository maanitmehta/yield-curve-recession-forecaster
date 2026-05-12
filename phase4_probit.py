"""
Phase 4 — Probit Recession Probability Forecasting
Phase 5 — Real-Time Pseudo Out-of-Sample Evaluation
=====================================================

Models
------
Three horizons: h in {3, 6, 12} months
Four specifications per horizon:

  M1 — Slope only (b2):
         P(rec_{t+h}) = Phi(a + g1*b2_t)
         Estrella-Mishkin (1998) benchmark

  M2 — Slope + Curvature (b2, b3):
         P(rec_{t+h}) = Phi(a + g1*b2_t + g2*b3_t)
         Base model — both factors stationary

  M3 — Full NS levels (b1, b2, b3):
         P(rec_{t+h}) = Phi(a + g1*b1_t + g2*b2_t + g3*b3_t)
         Robustness — b1 non-stationary, interpret with caution

  M4 — Slope + Delta-Level (Db1, b2, b3):
         Addresses b1 unit root by first-differencing

Phase 5 discipline
------------------
Expanding window from 1975-01. At each step t:
  - Re-fit probit on data available up to t
  - Apply NBER announcement lag: recession labels only used
    once officially announced (stored in recession_indicator.csv)
  - Generate h-month-ahead probability forecast
No future information enters any forecast. Zero look-ahead bias.

Run
---
    python phase4_probit.py

Outputs
-------
    output/probit_insample.png      -- in-sample fitted probabilities
    output/probit_oos.png           -- real-time OOS probabilities
    output/probit_roc.png           -- ROC curves all models x horizons
    output/probit_performance.png   -- Brier score / AUC summary table
    data/probit_results.csv         -- in-sample fitted probs
    data/probit_oos_results.csv     -- real-time OOS probs
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
warnings.filterwarnings("ignore")

from scipy.stats import norm
from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve
import statsmodels.api as sm


# ══════════════════════════════════════════════════════════════════════════
#  Configuration
# ══════════════════════════════════════════════════════════════════════════

HORIZONS       = [3, 6, 12]
OOS_START      = "1975-01-01"   # start of real-time evaluation window
MIN_OBS        = 60             # minimum months before fitting probit

REC_COLOR      = "#E8E0F0"
REC_ALPHA      = 0.6

MODEL_SPECS = {
    "M1_slope":       ["b2"],
    "M2_slope_curv":  ["b2", "b3"],
    "M3_full_ns":     ["b1", "b2", "b3"],
    "M4_db1_b2_b3":   ["db1", "b2", "b3"],
}

MODEL_LABELS = {
    "M1_slope":       "M1: Slope only (b2)",
    "M2_slope_curv":  "M2: Slope + Curvature",
    "M3_full_ns":     "M3: Full NS (b1+b2+b3)",
    "M4_db1_b2_b3":   "M4: Δb1 + b2 + b3",
}

MODEL_COLORS = {
    "M1_slope":       "#6B7280",
    "M2_slope_curv":  "#2563EB",
    "M3_full_ns":     "#DC2626",
    "M4_db1_b2_b3":   "#16A34A",
}


# ══════════════════════════════════════════════════════════════════════════
#  1.  Data preparation
# ══════════════════════════════════════════════════════════════════════════

def prepare_data(df_factors: pd.DataFrame,
                 df_rec:     pd.DataFrame) -> pd.DataFrame:
    """
    Merge NS factors with recession indicator, add derived variables.

    Returns a single DataFrame with columns:
        b1, b2, b3, db1        -- NS factors + first-diff of b1
        recession              -- contemporaneous NBER indicator
        announced_by           -- date NBER announced this period
        rec_fwd_3/6/12         -- recession indicator h months ahead
    """
    # Align on common dates
    common = df_factors.index.intersection(df_rec.index)
    df = df_factors[["b1", "b2", "b3"]].loc[common].copy()
    df = df.join(df_rec[["recession", "announced_by"]])

    # First difference of b1 (addresses non-stationarity)
    df["db1"] = df["b1"].diff()

    # Forward recession indicators for each horizon
    for h in HORIZONS:
        df[f"rec_fwd_{h}"] = df["recession"].shift(-h)

    df = df.dropna(subset=["db1"])  # drops first row (NaN diff)
    return df


def make_xy(df: pd.DataFrame,
            features: list,
            horizon: int,
            end_date: pd.Timestamp = None) -> tuple:
    """
    Build feature matrix X and binary target y for a given horizon,
    optionally restricted to data up to end_date.

    Returns (X, y, dates) — dates aligned to X/y rows.
    """
    col = f"rec_fwd_{horizon}"
    sub = df[features + [col]].dropna()
    if end_date is not None:
        sub = sub[sub.index <= end_date]
    X     = sub[features].values
    y     = sub[col].values.astype(int)
    dates = sub.index
    return X, y, dates


# ══════════════════════════════════════════════════════════════════════════
#  2.  Probit fit + predict
# ══════════════════════════════════════════════════════════════════════════

def fit_probit(X: np.ndarray, y: np.ndarray):
    """
    Fit probit via statsmodels MLE.
    Returns fitted result object (or None if estimation fails).
    """
    if len(np.unique(y)) < 2:
        return None          # can't fit if only one class in window
    try:
        Xc  = sm.add_constant(X, has_constant="add")
        res = sm.Probit(y, Xc).fit(disp=False, method="bfgs",
                                    maxiter=200, warn_convergence=False)
        return res
    except Exception:
        return None


def predict_probit(result, X: np.ndarray) -> np.ndarray:
    """Predict probabilities from a fitted probit result."""
    Xc = sm.add_constant(X, has_constant="add")
    return result.predict(Xc)


def probit_metrics(y_true: np.ndarray,
                   y_prob: np.ndarray) -> dict:
    """
    Compute Brier score, AUC-ROC, and log-loss.
    Returns dict of metrics (NaN if insufficient variation).
    """
    if len(np.unique(y_true)) < 2 or len(y_true) < 10:
        return dict(brier=np.nan, auc=np.nan)
    try:
        brier = brier_score_loss(y_true, y_prob)
        auc   = roc_auc_score(y_true, y_prob)
        return dict(brier=round(brier, 4), auc=round(auc, 4))
    except Exception:
        return dict(brier=np.nan, auc=np.nan)


# ══════════════════════════════════════════════════════════════════════════
#  3.  Phase 4 — In-sample fit
# ══════════════════════════════════════════════════════════════════════════

def run_insample(df: pd.DataFrame) -> tuple:
    """
    Fit each model spec on the full sample for each horizon.
    Returns:
        results_is  -- dict: (model, horizon) -> fitted probabilities Series
        metrics_is  -- DataFrame: rows=(model x horizon), cols=[brier, auc]
    """
    print("\n" + "=" * 60)
    print("PHASE 4 -- In-Sample Probit Estimation")
    print("=" * 60)

    results_is = {}
    metrics_rows = []

    for h in HORIZONS:
        print(f"\n  Horizon h={h}m")
        for mname, feats in MODEL_SPECS.items():
            X, y, dates = make_xy(df, feats, h)
            if len(y) < MIN_OBS:
                print(f"    {mname}: insufficient data")
                continue

            result = fit_probit(X, y)
            if result is None:
                print(f"    {mname}: estimation failed")
                continue

            probs  = predict_probit(result, X)
            m      = probit_metrics(y, probs)
            results_is[(mname, h)] = pd.Series(probs, index=dates,
                                                name=f"{mname}_h{h}")

            # Print coefficient summary
            coef_str = "  ".join([f"{feats[i]}={result.params[i+1]:.3f}"
                                  for i in range(len(feats))])
            print(f"    {mname:18s}  {coef_str}"
                  f"  Brier={m['brier']:.4f}  AUC={m['auc']:.4f}")

            metrics_rows.append({
                "model": mname, "horizon": h,
                "type": "in-sample",
                **m,
                "n_obs": len(y), "n_rec": int(y.sum())
            })

    df_metrics = pd.DataFrame(metrics_rows)
    return results_is, df_metrics


# ══════════════════════════════════════════════════════════════════════════
#  4.  Phase 5 — Real-time expanding window OOS
# ══════════════════════════════════════════════════════════════════════════

def run_realtime_oos(df: pd.DataFrame) -> tuple:
    """
    Expanding-window real-time out-of-sample forecasting.

    At each date t >= OOS_START:
      1. Build the information set: all data up to t
      2. Apply NBER announcement lag: mask out recession labels
         that had not been announced by date t
      3. Fit probit on the lagged-masked training set
      4. Predict the h-month-ahead recession probability at t

    No future data enters any forecast. Look-ahead bias = zero.

    Returns:
        results_oos  -- dict: (model, horizon) -> predicted prob Series
        metrics_oos  -- DataFrame: OOS evaluation metrics
    """
    print("\n" + "=" * 60)
    print("PHASE 5 -- Real-Time OOS Evaluation")
    print(f"  OOS window: {OOS_START} to {df.index[-1].date()}")
    print(f"  Expanding window, NBER announcement lag enforced")
    print("=" * 60)

    oos_dates  = df.index[df.index >= OOS_START]
    results_oos = {(m, h): {} for m in MODEL_SPECS for h in HORIZONS}

    n_total = len(oos_dates)
    for i, t in enumerate(oos_dates):
        if i % 60 == 0:
            print(f"  Progress: {t.date()}  ({i}/{n_total})")

        # ── Build real-time information set ──────────────────────────────
        # All factor data available at t (factors are observed contemporaneously)
        df_t = df[df.index <= t].copy()

        # Apply NBER announcement lag:
        # A recession label at date s is only usable at t if it was
        # announced (announced_by <= t). Otherwise mask as NaN.
        if "announced_by" in df_t.columns:
            ann = pd.to_datetime(df_t["announced_by"], errors="coerce")
            not_yet_announced = ann > t
            df_t.loc[not_yet_announced, "recession"] = np.nan
            # Rebuild forward recession labels from the masked series
            for h in HORIZONS:
                df_t[f"rec_fwd_{h}"] = df_t["recession"].shift(-h)

        for mname, feats in MODEL_SPECS.items():
            for h in HORIZONS:
                # Training set: everything strictly before t
                train = df_t[df_t.index < t]
                X_tr, y_tr, _ = make_xy(train, feats, h)
                if len(y_tr) < MIN_OBS or len(np.unique(y_tr)) < 2:
                    continue

                result = fit_probit(X_tr, y_tr)
                if result is None:
                    continue

                # Predict at t using current factor values
                x_now = df_t.loc[[t], feats].values
                if np.any(np.isnan(x_now)):
                    continue

                prob = predict_probit(result, x_now)[0]
                results_oos[(mname, h)][t] = prob

    # Convert nested dicts to Series
    results_oos_series = {}
    metrics_rows = []

    for mname in MODEL_SPECS:
        for h in HORIZONS:
            probs_dict = results_oos[(mname, h)]
            if not probs_dict:
                continue

            s = pd.Series(probs_dict, name=f"{mname}_h{h}")
            results_oos_series[(mname, h)] = s

            # Evaluate against realised outcomes (using full-info recession labels)
            col    = f"rec_fwd_{h}"
            y_true = df.loc[s.index, col].dropna()
            common = s.index.intersection(y_true.index)
            if len(common) < 20:
                continue

            m = probit_metrics(y_true.loc[common].values,
                               s.loc[common].values)
            metrics_rows.append({
                "model": mname, "horizon": h,
                "type": "real-time OOS",
                **m,
                "n_obs": len(common),
                "n_rec": int(y_true.loc[common].sum())
            })

    df_metrics_oos = pd.DataFrame(metrics_rows)
    return results_oos_series, df_metrics_oos


# ══════════════════════════════════════════════════════════════════════════
#  5.  Plotting
# ══════════════════════════════════════════════════════════════════════════

def _add_recession_bands(ax, df_rec):
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


def plot_insample_probs(results_is: dict, df_rec: pd.DataFrame,
                        out: str = "output/probit_insample.png"):
    """
    Three-row figure (one per horizon). Each row shows in-sample
    recession probabilities for all model specs.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    fig.suptitle("Phase 4 — In-Sample Probit Recession Probabilities",
                 fontsize=13, fontweight="bold", y=1.01)

    for ax, h in zip(axes, HORIZONS):
        _add_recession_bands(ax, df_rec)

        for mname in MODEL_SPECS:
            key = (mname, h)
            if key not in results_is:
                continue
            s = results_is[key]
            ax.plot(s.index, s.values,
                    color=MODEL_COLORS[mname],
                    linewidth=1.3,
                    alpha=0.85,
                    label=MODEL_LABELS[mname])

        ax.axhline(0.5, color="black", linewidth=0.6,
                   linestyle="--", alpha=0.5, label="50% threshold")
        ax.set_ylim(-0.02, 1.05)
        ax.set_ylabel("P(recession)", fontsize=10)
        ax.set_title(f"h = {h} months ahead", fontsize=11, loc="left",
                     fontweight="bold")
        ax.grid(True, alpha=0.2)
        if h == HORIZONS[0]:
            ax.legend(loc="upper right", fontsize=8, ncol=2)

    rec_patch = mpatches.Patch(color=REC_COLOR, alpha=0.8, label="NBER Recession")
    axes[0].legend(
        handles=axes[0].get_legend().legend_handles + [rec_patch],
        loc="upper right", fontsize=8, ncol=2
    )
    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] {out}")


def plot_oos_probs(results_oos: dict, df_rec: pd.DataFrame,
                  out: str = "output/probit_oos.png"):
    """
    Three-row figure showing real-time OOS probabilities.
    Focus on M2 (base model) to keep it readable.
    """
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    fig.suptitle(
        "Phase 5 — Real-Time OOS Recession Probabilities\n"
        "(expanding window, NBER announcement lag enforced)",
        fontsize=13, fontweight="bold", y=1.01)

    for ax, h in zip(axes, HORIZONS):
        _add_recession_bands(ax, df_rec)

        # Plot all models, highlight M2
        for mname in MODEL_SPECS:
            key = (mname, h)
            if key not in results_oos:
                continue
            s = results_oos[key]
            lw    = 2.0 if mname == "M2_slope_curv" else 1.0
            alpha = 0.9 if mname == "M2_slope_curv" else 0.55
            ax.plot(s.index, s.values,
                    color=MODEL_COLORS[mname],
                    linewidth=lw, alpha=alpha,
                    label=MODEL_LABELS[mname])

        ax.axhline(0.5, color="black", linewidth=0.6,
                   linestyle="--", alpha=0.5)
        ax.set_ylim(-0.02, 1.05)
        ax.set_ylabel("P(recession)", fontsize=10)
        ax.set_title(f"h = {h} months ahead", fontsize=11, loc="left",
                     fontweight="bold")
        ax.grid(True, alpha=0.2)

    rec_patch = mpatches.Patch(color=REC_COLOR, alpha=0.8, label="NBER Recession")
    handles = [mpatches.Patch(color=MODEL_COLORS[m], label=MODEL_LABELS[m])
               for m in MODEL_SPECS] + [rec_patch]
    axes[0].legend(handles=handles, loc="upper right", fontsize=8, ncol=2)
    axes[-1].set_xlabel("Date")
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] {out}")


def plot_roc_curves(results_is: dict, df: pd.DataFrame,
                    out: str = "output/probit_roc.png"):
    """
    ROC curves for each model at each horizon.
    3 columns (horizons) x 1 row, all models overlaid per panel.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("ROC Curves — In-Sample",
                 fontsize=13, fontweight="bold")

    for ax, h in zip(axes, HORIZONS):
        col = f"rec_fwd_{h}"
        for mname in MODEL_SPECS:
            key = (mname, h)
            if key not in results_is:
                continue
            s      = results_is[key]
            y_true = df.loc[s.index, col].dropna()
            common = s.index.intersection(y_true.index)
            if len(common) < 20 or len(np.unique(y_true.loc[common])) < 2:
                continue

            fpr, tpr, _ = roc_curve(y_true.loc[common].values,
                                     s.loc[common].values)
            auc = roc_auc_score(y_true.loc[common].values,
                                s.loc[common].values)
            ax.plot(fpr, tpr,
                    color=MODEL_COLORS[mname],
                    linewidth=1.8,
                    label=f"{MODEL_LABELS[mname]}  AUC={auc:.3f}")

        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("False Positive Rate", fontsize=10)
        ax.set_ylabel("True Positive Rate", fontsize=10)
        ax.set_title(f"h = {h}m ahead", fontsize=11, fontweight="bold")
        ax.legend(fontsize=7.5, loc="lower right")
        ax.grid(True, alpha=0.2)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] {out}")


def plot_performance_table(df_is: pd.DataFrame, df_oos: pd.DataFrame,
                           out: str = "output/probit_performance.png"):
    """
    Visual performance summary table: Brier score and AUC
    for in-sample vs real-time OOS, all models x horizons.
    """
    combined = pd.concat([df_is, df_oos], ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Phase 4/5 Performance Summary — Brier Score & AUC-ROC",
                 fontsize=13, fontweight="bold")

    for ax, metric, better in zip(axes, ["brier", "auc"], ["lower", "higher"]):
        pivot_is  = df_is.pivot(index="model",  columns="horizon", values=metric)
        pivot_oos = df_oos.pivot(index="model", columns="horizon", values=metric)

        x     = np.arange(len(HORIZONS))
        width = 0.12
        n_m   = len(MODEL_SPECS)

        for i, mname in enumerate(MODEL_SPECS):
            offset_is  = (i - n_m/2) * width * 1.1
            offset_oos = offset_is + width * 0.55

            vals_is = [pivot_is.loc[mname, h]
                       if mname in pivot_is.index and h in pivot_is.columns
                       else np.nan for h in HORIZONS]
            vals_oos = [pivot_oos.loc[mname, h]
                        if mname in pivot_oos.index and h in pivot_oos.columns
                        else np.nan for h in HORIZONS]

            ax.bar(x + offset_is,  vals_is,  width=width,
                   color=MODEL_COLORS[mname], alpha=0.85,
                   label=f"{MODEL_LABELS[mname]} (IS)" if metric == "brier" else None)
            ax.bar(x + offset_oos, vals_oos, width=width,
                   color=MODEL_COLORS[mname], alpha=0.45,
                   hatch="//", edgecolor="white",
                   label=f"{MODEL_LABELS[mname]} (OOS)" if metric == "brier" else None)

        ax.set_xticks(x)
        ax.set_xticklabels([f"h={h}m" for h in HORIZONS], fontsize=11)
        title = "Brier Score (lower = better)" if metric == "brier" \
                else "AUC-ROC (higher = better)"
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.2, axis="y")
        if metric == "brier":
            ax.legend(fontsize=7, ncol=2, loc="upper right")

    # Add solid vs hatched legend
    solid_patch  = mpatches.Patch(color="gray", alpha=0.85, label="In-sample (solid)")
    hatch_patch  = mpatches.Patch(color="gray", alpha=0.45,
                                   hatch="//", label="Real-time OOS (hatched)")
    axes[1].legend(handles=[solid_patch, hatch_patch], fontsize=9, loc="upper right")

    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[Plot] {out}")


def print_performance_table(df_is: pd.DataFrame, df_oos: pd.DataFrame):
    """Print a clean text performance comparison."""
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    print(f"{'Model':<22} {'Horizon':>8} {'IS Brier':>10} {'OOS Brier':>10}"
          f" {'IS AUC':>8} {'OOS AUC':>8}")
    print("-" * 70)

    for mname in MODEL_SPECS:
        for h in HORIZONS:
            is_row  = df_is[(df_is["model"] == mname) & (df_is["horizon"] == h)]
            oos_row = df_oos[(df_oos["model"] == mname) & (df_oos["horizon"] == h)]

            is_brier = f"{is_row['brier'].values[0]:.4f}"  if len(is_row)  else "  --  "
            oos_brier= f"{oos_row['brier'].values[0]:.4f}" if len(oos_row) else "  --  "
            is_auc   = f"{is_row['auc'].values[0]:.4f}"    if len(is_row)  else "  --  "
            oos_auc  = f"{oos_row['auc'].values[0]:.4f}"   if len(oos_row) else "  --  "

            print(f"{MODEL_LABELS[mname]:<22} {h:>5}m {is_brier:>10}"
                  f" {oos_brier:>10} {is_auc:>8} {oos_auc:>8}")
        print()


# ══════════════════════════════════════════════════════════════════════════
#  6.  MAIN
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    os.makedirs("data",   exist_ok=True)
    os.makedirs("output", exist_ok=True)

    # ── Load Phase 2 outputs ──────────────────────────────────────────────
    print("Loading data...")
    df_factors = pd.read_csv("data/ns_factors.csv",
                             index_col="date", parse_dates=True)
    df_factors.columns = [c.strip() for c in df_factors.columns]

    df_rec = pd.read_csv("data/recession_indicator.csv",
                         index_col="date", parse_dates=True)
    df_rec.index = pd.to_datetime(df_rec.index)

    # ── Prepare merged dataset ────────────────────────────────────────────
    df = prepare_data(df_factors, df_rec)
    print(f"Analysis dataset: {df.shape[0]} months  "
          f"({df.index[0].date()} to {df.index[-1].date()})")
    print(f"Recession months: {int(df['recession'].sum())} "
          f"({100*df['recession'].mean():.1f}%)")

    # ── Phase 4: In-sample ────────────────────────────────────────────────
    results_is, df_metrics_is = run_insample(df)

    # ── Phase 5: Real-time OOS ────────────────────────────────────────────
    results_oos, df_metrics_oos = run_realtime_oos(df)

    # ── Performance table ─────────────────────────────────────────────────
    print_performance_table(df_metrics_is, df_metrics_oos)

    # ── Save results ──────────────────────────────────────────────────────
    # In-sample probabilities
    is_frames = [s.rename(f"{m}_h{h}")
                 for (m, h), s in results_is.items()]
    if is_frames:
        pd.concat(is_frames, axis=1).to_csv("data/probit_results.csv")
        print("[OK] data/probit_results.csv")

    # OOS probabilities
    oos_frames = [s.rename(f"{m}_h{h}")
                  for (m, h), s in results_oos.items()]
    if oos_frames:
        pd.concat(oos_frames, axis=1).to_csv("data/probit_oos_results.csv")
        print("[OK] data/probit_oos_results.csv")

    # Metrics
    pd.concat([df_metrics_is, df_metrics_oos]).to_csv(
        "data/probit_metrics.csv", index=False)
    print("[OK] data/probit_metrics.csv")

    # ── Plots ─────────────────────────────────────────────────────────────
    print("\n--- Generating plots ---")
    plot_insample_probs(results_is, df_rec)
    plot_oos_probs(results_oos, df_rec)
    plot_roc_curves(results_is, df)
    plot_performance_table(df_metrics_is, df_metrics_oos)

    print("\n[OK] Phase 4 + 5 complete.")
    print("     Plots saved to output/")
    print("     Results saved to data/")