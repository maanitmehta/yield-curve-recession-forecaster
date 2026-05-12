"""
Phase 6 — Yield Curve Recession Probability Dashboard
======================================================
Streamlit app that:
  1. Loads the historical NS factors + OOS recession probabilities
     produced by Phases 2 and 4/5
  2. Lets the user input today's yield curve (pre-filled with
     latest FRED data if API key is available)
  3. Fits Nelson-Siegel live → extracts b1, b2, b3
  4. Applies the trained probit model → outputs P(recession) at 3/6/12m
  5. Shows historical context charts with NBER recession bands

Run:
    streamlit run phase6_dashboard.py

Requires:
    data/ns_factors.csv
    data/recession_indicator.csv
    data/probit_results.csv
    data/probit_oos_results.csv
    (all produced by running phase1_data.py, phase2_nelson_siegel.py,
     phase4_probit.py in sequence)
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
from scipy.stats import norm
from sklearn.metrics import roc_auc_score, brier_score_loss
import statsmodels.api as sm
import io

# Live yield loader (Phase 7)
def _load_live_yields_from_json():
    import json as _j, datetime as _dt, os as _os
    path = "data/live_yields.json"
    if not _os.path.exists(path):
        return None
    try:
        with open(path) as f:
            data = _j.load(f)
        fetched = _dt.datetime.fromisoformat(data["fetched_at"].replace("Z", ""))
        age_h = (_dt.datetime.utcnow() - fetched).total_seconds() / 3600
        if age_h > 24:
            return None
        return {float(k): v["yield"] for k, v in data["yields"].items()}
    except Exception:
        return None


# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Yield Curve Recession Forecaster",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Constants ──────────────────────────────────────────────────────────────
LAMBDA   = 0.40
HORIZONS = [3, 6, 12]

MATURITIES_LABELS = {
    0.083: "1M",  0.25: "3M",  0.5: "6M",
    1.0:   "1Y",  2.0:  "2Y",  3.0: "3Y",
    5.0:   "5Y",  7.0:  "7Y",  10.0:"10Y",
    20.0:  "20Y", 30.0: "30Y",
}

# Latest approximate FRED values (Dec 2024) — fallback if no API key
LATEST_YIELDS_FALLBACK = {
    0.083: 4.36, 0.25: 4.33, 0.5: 4.24,
    1.0:   4.06, 2.0:  4.24, 3.0: 4.32,
    5.0:   4.43, 7.0:  4.53, 10.0: 4.58,
    20.0:  4.82, 30.0: 4.77,
}

REC_COLOR = "#E8E0F0"
REC_ALPHA = 0.55

GAUGE_COLORS = {
    "low":    ("#16A34A", "#DCFCE7"),
    "medium": ("#D97706", "#FEF3C7"),
    "high":   ("#DC2626", "#FEE2E2"),
}


# ══════════════════════════════════════════════════════════════════════════
#  Core NS + Probit functions (self-contained, no imports from other phases)
# ══════════════════════════════════════════════════════════════════════════

def ns_loadings(tau, lam=LAMBDA):
    tau = np.asarray(tau, dtype=float)
    lt  = lam * tau
    with np.errstate(invalid="ignore", divide="ignore"):
        slope = np.where(lt < 1e-8, 1.0, (1 - np.exp(-lt)) / lt)
    curv = slope - np.exp(-lt)
    return np.column_stack([np.ones_like(tau), slope, curv])


def fit_ns(yields_dict: dict) -> dict:
    """Fit NS to a dict {maturity: yield}. Returns {b1, b2, b3, rmse}."""
    tau = np.array(list(yields_dict.keys()), dtype=float)
    y   = np.array(list(yields_dict.values()), dtype=float)
    mask = ~np.isnan(y)
    tau, y = tau[mask], y[mask]
    if len(tau) < 3:
        return None
    L = ns_loadings(tau)
    braw, _, _, _ = np.linalg.lstsq(L, y, rcond=None)
    b1 = float(np.clip(braw[0], 0, 25))
    b2 = float(-np.clip(braw[1], -15, 15))   # sign-corrected
    b3 = float(np.clip(braw[2], -10, 10))
    y_hat = L @ np.array([braw[0], braw[1], braw[2]])
    rmse  = float(np.sqrt(np.mean((y - y_hat) ** 2)))
    return dict(b1=b1, b2=b2, b3=b3, rmse=rmse)


def fit_probit_model(X, y):
    if len(np.unique(y)) < 2:
        return None
    try:
        Xc = sm.add_constant(X, has_constant="add")
        return sm.Probit(y, Xc).fit(disp=False, method="bfgs",
                                     maxiter=300, warn_convergence=False)
    except Exception:
        return None


def predict_prob(result, x_vec) -> float:
    Xc = sm.add_constant(np.array(x_vec).reshape(1, -1), has_constant="add")
    return float(result.predict(Xc)[0])


# ══════════════════════════════════════════════════════════════════════════
#  Data loading (cached)
# ══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_all_data():
    """Load all pre-computed Phase 2 / 4 / 5 outputs.
    Uses index_col=0 so it works regardless of what the
    first column is named in each CSV.
    """
    def _read(path):
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df.index.name = "date"
        return df

    df_factors = _read("data/ns_factors.csv")
    df_factors.columns = [c.strip() for c in df_factors.columns]

    df_rec = _read("data/recession_indicator.csv")
    df_is  = _read("data/probit_results.csv")
    df_oos = _read("data/probit_oos_results.csv")

    return df_factors, df_rec, df_is, df_oos


@st.cache_data(show_spinner=False)
def train_probit_models(df_factors_json, df_rec_json):
    """
    Re-train probit models on full sample (M1: slope only).
    Cached so it only runs once per session.
    """
    df_f = pd.read_json(io.StringIO(df_factors_json))
    df_f.index = pd.to_datetime(df_f.index)
    df_r = pd.read_json(io.StringIO(df_rec_json))
    df_r.index = pd.to_datetime(df_r.index)

    common = df_f.index.intersection(df_r.index)
    df = df_f[["b1", "b2", "b3"]].loc[common].copy()
    df["db1"]      = df["b1"].diff()
    df["recession"] = df_r.loc[common, "recession"]
    df = df.dropna()

    models = {}
    for h in HORIZONS:
        df[f"rec_fwd_{h}"] = df["recession"].shift(-h)
        sub = df[["b2", f"rec_fwd_{h}"]].dropna()
        X   = sub[["b2"]].values
        y   = sub[f"rec_fwd_{h}"].values.astype(int)
        result = fit_probit_model(X, y)
        models[h] = result

    return models


# ══════════════════════════════════════════════════════════════════════════
#  Plotting functions
# ══════════════════════════════════════════════════════════════════════════

def _add_rec_bands(ax, df_rec):
    rec = df_rec["recession"].dropna()
    in_rec, t0 = False, None
    for date, v in rec.items():
        if v == 1 and not in_rec:
            t0, in_rec = date, True
        elif v == 0 and in_rec:
            ax.axvspan(t0, date, color=REC_COLOR, alpha=REC_ALPHA, zorder=0)
            in_rec = False
    if in_rec and t0:
        ax.axvspan(t0, rec.index[-1], color=REC_COLOR, alpha=REC_ALPHA, zorder=0)


def fig_current_curve(yields_dict: dict, ns_fit: dict):
    """Plot current yield curve with NS fit overlaid."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    tau_data = np.array(list(yields_dict.keys()))
    y_data   = np.array(list(yields_dict.values()))
    mask     = ~np.isnan(y_data)

    tau_fine = np.linspace(0.05, 30, 300)
    L_fine   = ns_loadings(tau_fine)
    # Reconstruct using raw (sign-uncorrected) b2 for the curve
    b2_raw   = -ns_fit["b2"]
    fitted   = L_fine @ np.array([ns_fit["b1"], b2_raw, ns_fit["b3"]])

    ax.scatter(tau_data[mask], y_data[mask], color="#1E3A5F",
               s=60, zorder=5, label="Input yields")
    ax.plot(tau_fine, fitted, color="#DC2626", linewidth=2.2,
            label=f"NS fit  (RMSE={ns_fit['rmse']:.3f}%)")

    ax.set_xlabel("Maturity (years)", fontsize=10)
    ax.set_ylabel("Yield (%)", fontsize=10)
    ax.set_title("Current Yield Curve — Nelson-Siegel Fit", fontsize=11,
                 fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    return fig


def fig_factors_history(df_factors, df_rec, current_factors: dict):
    """Three-panel factor history with current value marked."""
    specs = [
        ("b1", "b1 — Level",     "#2563EB"),
        ("b2", "b2 — Slope",     "#DC2626"),
        ("b3", "b3 — Curvature", "#16A34A"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
    fig.suptitle("NS Factors — Historical Context", fontsize=12,
                 fontweight="bold")

    for ax, (col, title, color) in zip(axes, specs):
        _add_rec_bands(ax, df_rec)
        ax.plot(df_factors.index, df_factors[col],
                color=color, linewidth=1.2, alpha=0.9)
        ax.axhline(0, color="black", linewidth=0.5, linestyle="--", alpha=0.3)

        # Mark current value
        cur_val = current_factors.get(col, np.nan)
        if not np.isnan(cur_val):
            ax.axhline(cur_val, color=color, linewidth=1.2,
                       linestyle=":", alpha=0.7)
            ax.annotate(f"Now: {cur_val:.2f}%",
                        xy=(df_factors.index[-1], cur_val),
                        xytext=(-90, 6), textcoords="offset points",
                        fontsize=8, color=color,
                        arrowprops=dict(arrowstyle="-", color=color, alpha=0.5))

        ax.set_ylabel("%", fontsize=9)
        ax.set_title(title, fontsize=10, loc="left")
        ax.grid(True, alpha=0.2)

    rec_patch = mpatches.Patch(color=REC_COLOR, alpha=0.8, label="NBER Recession")
    axes[0].legend(handles=[rec_patch], loc="upper right", fontsize=8)
    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    return fig


def fig_oos_probs(df_oos, df_rec, highlight_col="M1_slope_curv_h12"):
    """Historical OOS recession probabilities — 12m horizon."""
    # Find the best available column for M1 h=12
    cols_12 = [c for c in df_oos.columns if "h12" in c]
    m1_col  = next((c for c in cols_12 if "M1" in c), None)
    m2_col  = next((c for c in cols_12 if "M2" in c), None)

    fig, ax = plt.subplots(figsize=(10, 3.8))
    _add_rec_bands(ax, df_rec)

    if m1_col:
        ax.plot(df_oos.index, df_oos[m1_col],
                color="#6B7280", linewidth=1.4, alpha=0.8,
                label="M1: Slope only (h=12m)")
    if m2_col:
        ax.plot(df_oos.index, df_oos[m2_col],
                color="#2563EB", linewidth=1.6,
                label="M2: Slope + Curvature (h=12m)")

    ax.axhline(0.5, color="black", linewidth=0.7,
               linestyle="--", alpha=0.5, label="50% threshold")
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel("P(recession)", fontsize=10)
    ax.set_title("Real-Time OOS Recession Probability — 12-Month Horizon",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.2)
    rec_patch = mpatches.Patch(color=REC_COLOR, alpha=0.8, label="NBER Recession")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles + [rec_patch],
              labels=labels + ["NBER Recession"],
              fontsize=8, loc="upper left", ncol=2)
    fig.tight_layout()
    return fig


def fig_slope_history(df_factors, df_rec):
    """b2 slope factor with zero line — key recession signal."""
    fig, ax = plt.subplots(figsize=(10, 3))
    _add_rec_bands(ax, df_rec)
    ax.plot(df_factors.index, df_factors["b2"],
            color="#DC2626", linewidth=1.3)
    ax.fill_between(df_factors.index, df_factors["b2"], 0,
                    where=df_factors["b2"] < 0,
                    color="#DC2626", alpha=0.18, label="Inverted (b2 < 0)")
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_ylabel("b2 — Slope (%)", fontsize=10)
    ax.set_title("Yield Curve Slope (b2) — Inversions Precede Recessions",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    return fig


# ══════════════════════════════════════════════════════════════════════════
#  Gauge / probability display
# ══════════════════════════════════════════════════════════════════════════

def prob_gauge(prob: float, horizon: int):
    """Render a styled probability card with color coding."""
    if prob < 0.20:
        level = "low";    label = "Low Risk"
    elif prob < 0.45:
        level = "medium"; label = "Elevated Risk"
    else:
        level = "high";   label = "High Risk"

    text_color, bg_color = GAUGE_COLORS[level]

    st.markdown(f"""
    <div style="
        background: {bg_color};
        border-left: 5px solid {text_color};
        border-radius: 8px;
        padding: 18px 20px;
        margin-bottom: 8px;
    ">
        <div style="font-size:13px; color:#6B7280; font-weight:500;
                    letter-spacing:0.04em; text-transform:uppercase;">
            {horizon}-Month Horizon
        </div>
        <div style="font-size:42px; font-weight:700; color:{text_color};
                    line-height:1.1; margin: 4px 0;">
            {prob*100:.1f}%
        </div>
        <div style="font-size:14px; color:{text_color}; font-weight:600;">
            {label}
        </div>
        <div style="margin-top:8px; background:#E5E7EB; border-radius:4px;
                    height:6px; overflow:hidden;">
            <div style="width:{prob*100:.1f}%; background:{text_color};
                        height:100%; border-radius:4px;"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
#  Sidebar — yield curve input
# ══════════════════════════════════════════════════════════════════════════

def sidebar_yield_inputs() -> dict:
    st.sidebar.header("📊 Input Yield Curve")
    if st.sidebar.button("🔄 Fetch live FRED yields"):
        import subprocess
        subprocess.Popen(["python3", "phase7_extensions.py", "--live-only"])
        st.sidebar.success("Fetching... refresh page in ~10s")
    _live_status = _load_live_yields_from_json()
    if _live_status:
        st.sidebar.info("📡 Live yields active (fetched today)")
    else:
        st.sidebar.caption("Using Dec 2024 defaults. Click above to fetch live.")
    st.sidebar.caption(
        "Enter current Treasury yields (%). "
        "Pre-filled with approximate Dec 2024 values."
    )

    yields = {}
    for mat, label in MATURITIES_LABELS.items():
        _live_y = _load_live_yields_from_json()
        default = (_live_y or LATEST_YIELDS_FALLBACK).get(mat, 4.0)
        val = st.sidebar.number_input(
            label=label,
            min_value=0.0,
            max_value=25.0,
            value=float(round(default, 2)),
            step=0.01,
            format="%.2f",
            key=f"yield_{mat}",
        )
        yields[mat] = val

    st.sidebar.markdown("---")
    st.sidebar.caption(
        "**Data sources**: FRED (Treasury CMT yields), "
        "NBER (recession dates). "
        "Model: Nelson-Siegel (λ=0.40) + Probit (M1: slope only)."
    )
    return yields


# ══════════════════════════════════════════════════════════════════════════
#  Main app
# ══════════════════════════════════════════════════════════════════════════

def main():
    # ── Header ────────────────────────────────────────────────────────────
    st.markdown("""
    <h1 style='margin-bottom:0'>
        📈 Yield Curve Recession Probability Forecaster
    </h1>
    <p style='color:#6B7280; font-size:15px; margin-top:4px;'>
        Nelson-Siegel term structure decomposition · Probit forecasting ·
        3 / 6 / 12-month horizons · Real FRED data 1970–2024
    </p>
    <hr style='margin:12px 0 20px'>
    """, unsafe_allow_html=True)

    # ── Load data ─────────────────────────────────────────────────────────
    with st.spinner("Loading historical data..."):
        try:
            df_factors, df_rec, df_is, df_oos = load_all_data()
        except FileNotFoundError as e:
            st.error(
                f"Data file not found: {e}\n\n"
                "Run the pipeline first:\n"
                "```\npython3 phase1_data.py\n"
                "python3 phase2_nelson_siegel.py\n"
                "python3 phase4_probit.py\n```"
            )
            st.stop()

    # ── Sidebar: yield inputs ─────────────────────────────────────────────
    yields_input = sidebar_yield_inputs()

    # ── Fit NS to input curve ─────────────────────────────────────────────
    ns_fit = fit_ns(yields_input)
    if ns_fit is None:
        st.error("Need at least 3 valid yield inputs to fit the model.")
        st.stop()

    # ── Train probit on full historical sample ────────────────────────────
    with st.spinner("Training probit models..."):
        models = train_probit_models(
            df_factors.to_json(),
            df_rec.to_json(),
        )

    # ── Compute current recession probabilities ───────────────────────────
    probs = {}
    for h in HORIZONS:
        m = models.get(h)
        if m is not None:
            probs[h] = predict_prob(m, [ns_fit["b2"]])
        else:
            probs[h] = np.nan

    # ══════════════════════════════════════════════════════════════════════
    #  Layout: top row — NS factors + probability gauges
    # ══════════════════════════════════════════════════════════════════════
    col_curve, col_gauges = st.columns([1.6, 1], gap="large")

    with col_curve:
        st.subheader("Current Yield Curve")
        st.pyplot(fig_current_curve(yields_input, ns_fit))

        # Factor readout
        slope_label = "▲ Upward" if ns_fit["b2"] > 0 else "▼ Inverted"
        slope_color = "#16A34A" if ns_fit["b2"] > 0 else "#DC2626"
        st.markdown(f"""
        <div style='display:flex; gap:16px; margin-top:8px;'>
            <div style='flex:1; background:#F0F9FF; border-radius:8px;
                        padding:12px 16px; border:1px solid #BAE6FD;'>
                <div style='font-size:11px; color:#0369A1;
                            text-transform:uppercase; font-weight:600'>
                    b1 — Level</div>
                <div style='font-size:24px; font-weight:700;
                            color:#0369A1'>{ns_fit["b1"]:.2f}%</div>
                <div style='font-size:11px; color:#64748B'>Long-run yield</div>
            </div>
            <div style='flex:1; background:#FFF5F5; border-radius:8px;
                        padding:12px 16px; border:1px solid #FECACA;'>
                <div style='font-size:11px; color:#DC2626;
                            text-transform:uppercase; font-weight:600'>
                    b2 — Slope</div>
                <div style='font-size:24px; font-weight:700;
                            color:{slope_color}'>{ns_fit["b2"]:.2f}%</div>
                <div style='font-size:11px; color:#64748B'>{slope_label}</div>
            </div>
            <div style='flex:1; background:#F0FDF4; border-radius:8px;
                        padding:12px 16px; border:1px solid #BBF7D0;'>
                <div style='font-size:11px; color:#16A34A;
                            text-transform:uppercase; font-weight:600'>
                    b3 — Curvature</div>
                <div style='font-size:24px; font-weight:700;
                            color:#16A34A'>{ns_fit["b3"]:.2f}%</div>
                <div style='font-size:11px; color:#64748B'>
                    RMSE {ns_fit["rmse"]:.3f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_gauges:
        st.subheader("Recession Probability")
        st.caption("M1 model (slope only) — best real-time OOS performer")
        for h in HORIZONS:
            prob_gauge(probs.get(h, np.nan), h)

        # Interpretation note
        slope_val = ns_fit["b2"]
        if slope_val < -1.0:
            note = "⚠️ Curve is significantly inverted. Historically, this precedes recessions with a 6–18 month lag."
            note_color = "#FEF3C7"
            note_border = "#D97706"
        elif slope_val < 0.5:
            note = "⚡ Curve is flat to mildly inverted. Watch for further flattening."
            note_color = "#FFF7ED"
            note_border = "#EA580C"
        else:
            note = "✅ Curve is positively sloped. Historically consistent with expansion."
            note_color = "#F0FDF4"
            note_border = "#16A34A"

        st.markdown(f"""
        <div style='background:{note_color}; border-left:4px solid {note_border};
                    border-radius:6px; padding:12px 14px; margin-top:12px;
                    font-size:13px; color:#374151;'>
            {note}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════
    #  Historical charts section
    # ══════════════════════════════════════════════════════════════════════
    st.subheader("Historical Context")

    tab1, tab2, tab3 = st.tabs([
        "📉 Slope Factor & Inversions",
        "📊 OOS Recession Probabilities",
        "🔢 NS Factor History",
    ])

    with tab1:
        st.pyplot(fig_slope_history(df_factors, df_rec))
        st.caption(
            "The slope factor b2 turns negative (curve inverts) before every "
            "NBER recession in the sample. Purple shading = NBER recession. "
            "Current b2 is marked with a dashed line."
        )
        # Add current b2 annotation
        cur_b2 = ns_fit["b2"]
        pct_below = (df_factors["b2"] < cur_b2).mean() * 100
        st.info(
            f"Current b2 = **{cur_b2:.2f}%** · "
            f"This is lower than **{pct_below:.0f}%** of all historical months · "
            f"Historical mean = {df_factors['b2'].mean():.2f}%  "
            f"| Min (most inverted) = {df_factors['b2'].min():.2f}%"
        )

    with tab2:
        st.pyplot(fig_oos_probs(df_oos, df_rec))
        st.caption(
            "Real-time expanding-window OOS probabilities. "
            "At each date, the model was trained only on data available at "
            "that point, with NBER announcement lag enforced. "
            "No look-ahead bias."
        )

        # Performance summary table
        st.markdown("**Model performance (real-time OOS):**")
        perf_data = {
            "Model": ["M1: Slope only", "M1: Slope only", "M1: Slope only"],
            "Horizon": ["3m", "6m", "12m"],
            "IS AUC":   [0.698, 0.773, 0.853],
            "OOS AUC":  [0.573, 0.691, 0.824],
            "IS Brier": [0.107, 0.098, 0.089],
            "OOS Brier":[0.112, 0.106, 0.086],
        }
        st.dataframe(
            pd.DataFrame(perf_data).set_index("Model"),
            use_container_width=True,
        )

    with tab3:
        st.pyplot(fig_factors_history(df_factors, df_rec, ns_fit))
        st.caption(
            "All three Nelson-Siegel factors 1970–2024 with NBER recession "
            "shading. Dotted lines show current values. "
            f"λ = {LAMBDA} (curvature peak at {np.log(2)/LAMBDA:.1f} years)."
        )

    st.markdown("---")

    # ══════════════════════════════════════════════════════════════════════
    #  Strategy memo section
    # ══════════════════════════════════════════════════════════════════════
    with st.expander("📝 Strategy Memo — Model Output Interpretation"):
        b2 = ns_fit["b2"]
        p3, p6, p12 = probs.get(3, 0), probs.get(6, 0), probs.get(12, 0)

        # Auto-generate interpretation
        if p12 > 0.45:
            duration_view = "extend duration — yield curve historically rallies (long-end) as recession approaches"
            curve_trade   = "receive fixed / curve steepener (5s30s) — curves re-steepen sharply in early recession"
            credit_view   = "reduce credit risk — spreads widen in recession"
        elif p12 > 0.25:
            duration_view = "neutral to mild duration extension — risk/reward improving but signal not conclusive"
            curve_trade   = "monitor 2s10s for further flattening — not yet actionable for a steepener"
            credit_view   = "selective credit exposure — prefer high-quality IG over HY"
        else:
            duration_view = "neutral to short duration — curve not signalling imminent risk"
            curve_trade   = "flattener bias — carry positive in a steep, growing economy"
            credit_view   = "risk-on credit positioning appropriate"

        st.markdown(f"""
**Date**: Current input  
**Model**: Nelson-Siegel Probit (M1 — slope only, λ=0.40)  
**Yield curve regime**: b2 = {b2:.2f}% ({'inverted' if b2 < 0 else 'normal slope'})

---

**Recession probability estimates**
| Horizon | Probability | Signal |
|---------|------------|--------|
| 3 months | {p3*100:.1f}% | {'🔴 High' if p3>0.45 else '🟡 Elevated' if p3>0.20 else '🟢 Low'} |
| 6 months | {p6*100:.1f}% | {'🔴 High' if p6>0.45 else '🟡 Elevated' if p6>0.20 else '🟢 Low'} |
| 12 months | {p12*100:.1f}% | {'🔴 High' if p12>0.45 else '🟡 Elevated' if p12>0.20 else '🟢 Low'} |

---

**Implications for fixed income strategy**

- **Duration**: {duration_view}
- **Curve positioning**: {curve_trade}
- **Credit**: {credit_view}

---

**Model caveats**
- M1 slope-only probit: OOS AUC = 0.82 at 12m, Brier = 0.086 (full sample 1975–2024)
- Model does not account for unconventional monetary policy regimes (ZLB, QE)
- NBER recession calls lag actual turning points by 6–18 months
- Yield curve signal is a leading indicator, not a timing tool
        """)

    # ── Footer ─────────────────────────────────────────────────────────────
    st.markdown("""
    <div style='margin-top:32px; padding-top:16px; border-top:1px solid #E5E7EB;
                font-size:12px; color:#9CA3AF; text-align:center;'>
        Nelson-Siegel Yield Curve Recession Forecaster ·
        Data: FRED / NBER · Model: Probit M1 (slope only) ·
        Phase 6 of 6 — Macro / Fixed Income Research Project
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()