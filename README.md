# Yield Curve Recession Probability Forecaster

**Nelson-Siegel term structure decomposition · Probit forecasting · Real-time out-of-sample evaluation**

A quantitative macro research project that fits the Nelson-Siegel (1987) model to US Treasury yield curves, decomposes them into level, slope, and curvature factors, and uses those factors to forecast NBER recession probabilities at 3, 6, and 12-month horizons — with rigorous real-time out-of-sample validation and a live Streamlit dashboard.

---

## The core idea

Yield curves contain forward-looking information about the macroeconomy. When short-term rates exceed long-term rates (inversion), the market is pricing in future rate cuts — historically a reliable precursor to recession. This project quantifies that signal systematically using the Nelson-Siegel decomposition and a probit model, then validates it under real-time conditions to ensure the results are genuine rather than the product of hindsight.

---

## Live dashboard

```bash
git clone https://github.com/YOUR_USERNAME/yield-curve-recession-forecaster
cd yield-curve-recession-forecaster
bash setup_env.sh
streamlit run phase6_dashboard.py
```

The dashboard fetches live Treasury yields from FRED, fits the NS model in real time, and outputs recession probabilities at three horizons — updated with a single button click.

![Dashboard screenshot showing yield curve fit and recession probability gauges]

---

## Methodology

### Phase 1 — Data

- **Source**: FRED (St. Louis Fed) via API — 11 constant-maturity Treasury series (1M to 30Y), January 1970 to present
- **Recession labels**: NBER Business Cycle Dating Committee — 8 complete cycles, 92 recession months (13.9% of the sample)
- **Announcement dates stored**: Each recession label carries the date NBER publicly announced it. Used in Phase 5 to prevent look-ahead bias.

### Phase 2 — Nelson-Siegel fitting

The NS model parameterises the entire yield curve with three factors:

```
y(τ) = β₁ + β₂·[(1 − e^{−λτ}) / (λτ)] + β₃·[(1 − e^{−λτ}) / (λτ) − e^{−λτ}]
```

| Factor | Economic interpretation | Recession behaviour |
|--------|------------------------|---------------------|
| β₁ (level) | Long-run yield | Follows secular rate trend |
| β₂ (slope) | Short minus long rate | Turns negative before every NBER recession |
| β₃ (curvature) | Medium-term hump | Idiosyncratic; weak recession signal |

**Implementation note**: The Diebold-Li (2006) standard of λ = 0.0609 places the curvature peak at 11.4 years — outside the range where β₃ is identified when data starts at 0.5Y. We calibrate λ = 0.40 by panel RMSE minimisation, placing the curvature peak at 1.73 years and reducing mean RMSE from 0.127% to **0.084%**.

### Phase 3 — Factor decomposition and EDA

- ADF stationarity tests: β₂ and β₃ stationary at 1% level; β₁ non-stationary (unit root) → excluded from base model
- Lead-lag cross-correlation confirms β₂ is most informative at 6–18 month horizons
- β₂ turned negative before all 8 NBER recessions in the 1970–2024 sample

### Phase 4 — Probit models

Four specifications estimated at each of three horizons:

```
P(Recession_{t+h} = 1 | Ωₜ) = Φ(α + γ · β₂ₜ)         [M1 — base]
P(Recession_{t+h} = 1 | Ωₜ) = Φ(α + γ₁β₂ₜ + γ₂β₃ₜ)  [M2]
P(Recession_{t+h} = 1 | Ωₜ) = Φ(α + γ₁β₁ₜ + γ₂β₂ₜ + γ₃β₃ₜ) [M3]
P(Recession_{t+h} = 1 | Ωₜ) = Φ(α + γ₁Δβ₁ₜ + γ₂β₂ₜ + γ₃β₃ₜ) [M4]
```

**In-sample results (full 1970–2024 sample):**

| Model | h=3m AUC | h=6m AUC | h=12m AUC |
|-------|----------|----------|-----------|
| M1: Slope only | 0.698 | 0.773 | 0.853 |
| M2: Slope + Curvature | 0.700 | 0.781 | 0.861 |
| M3: Full NS | 0.729 | 0.810 | 0.870 |
| M4: Δβ₁ + β₂ + β₃ | 0.703 | 0.780 | 0.861 |

### Phase 5 — Real-time out-of-sample evaluation

**The critical discipline**: at each forecast date t (January 1975 onward):
1. Build the information set — all yield data up to t
2. Apply NBER announcement lag — recession labels masked until their official announcement date
3. Re-fit NS + probit on the restricted training set
4. Generate the h-month-ahead probability forecast

Zero look-ahead bias. No future information enters any forecast.

**Real-time OOS results (expanding window, 1975–2024):**

| Model | h=3m AUC | h=6m AUC | h=12m AUC | IS→OOS gap (12m) |
|-------|----------|----------|-----------|-----------------|
| M1: Slope only | 0.573 | 0.691 | **0.824** | **0.029** |
| M2: Slope + Curvature | 0.515 | 0.620 | 0.747 | 0.114 |
| M3: Full NS | 0.353 | 0.462 | 0.524 | 0.346 |
| M4: Δβ₁ + β₂ + β₃ | 0.526 | 0.611 | 0.754 | 0.107 |

**Key finding**: M1 (slope only) wins out-of-sample. The IS→OOS gap of 0.029 AUC points indicates minimal overfitting. M3's collapse (gap = 0.346) confirms the β₁ unit root creates spurious in-sample fit. This replicates and extends the finding in Estrella & Mishkin (1998) on 54 years of data with real-time discipline.

### Phase 7 — Statistical extensions

**Diebold-Mariano test** (Harvey-Leybourne-Newbold correction):
Formally tests H₀: equal predictive accuracy between M1 and the naive unconditional-mean benchmark. A statistically significant negative DM statistic at h=12m confirms M1's OOS advantage is not due to chance.

**Calibration (reliability diagrams)**:
Measures whether stated probabilities are accurate — does a 30% forecast correspond to a 30% historical recession frequency? Reports Brier Skill Score (vs climatological baseline) and Expected Calibration Error per horizon.

---

## Results summary

The slope factor (β₂) is the yield curve's primary recession signal:

- Turned negative ahead of all 8 NBER recessions 1969–2020
- Slope-only probit achieves **OOS AUC = 0.82 at 12 months** in real-time evaluation
- OOS Brier score of 0.086 — lower than the unconditional recession frequency (0.139²) baseline
- DM test rejects equal accuracy vs naive at h=12m (p < 0.05)
- Adding curvature or level factors improves in-sample fit but degrades real-time performance — classic overfitting

---

## Portfolio / strategy implications

The model generates three outputs relevant to fixed income positioning:

| Signal | Implication |
|--------|-------------|
| P(recession) > 45% at 12m | Extend duration; 5s30s steepener; reduce credit risk |
| P(recession) 20–45% at 12m | Neutral duration; monitor 2s10s; selective IG credit |
| P(recession) < 20% at 12m | Neutral to short duration; flattener carry; risk-on credit |

The strategy memo in the dashboard auto-generates these implications from live model output.

---

## Project structure

```
yield-curve-recession-forecaster/
│
├── phase1_data.py              # FRED data pipeline + NBER recession indicator
├── phase2_nelson_siegel.py     # NS fitting + EDA + factor decomposition
├── phase4_probit.py            # Probit models (Phase 4) + real-time OOS (Phase 5)
├── phase6_dashboard.py         # Streamlit live dashboard
├── phase7_extensions.py        # DM test + calibration + live FRED pull
│
├── data/                       # Generated by pipeline (not committed)
│   ├── yields_monthly.csv
│   ├── recession_indicator.csv
│   ├── ns_factors.csv
│   ├── probit_results.csv
│   ├── probit_oos_results.csv
│   ├── dm_test_results.csv
│   └── live_yields.json
│
├── output/                     # Generated plots (not committed)
│   ├── yield_curves.png
│   ├── ns_factors.png
│   ├── ns_fit_examples.png
│   ├── eda_crosscorr.png
│   ├── probit_insample.png
│   ├── probit_oos.png
│   ├── probit_roc.png
│   ├── probit_performance.png
│   ├── dm_test.png
│   └── calibration.png
│
├── requirements.txt
└── setup_env.sh
```

---

## Quickstart

```bash
# 1. Clone and set up environment
git clone https://github.com/YOUR_USERNAME/yield-curve-recession-forecaster
cd yield-curve-recession-forecaster
bash setup_env.sh
source .venv/bin/activate

# 2. Add your FRED API key (free at fred.stlouisfed.org/docs/api/api_key.html)
#    Edit FRED_API_KEY in phase1_data.py

# 3. Run the pipeline
python3 phase1_data.py               # ~30 seconds
python3 phase2_nelson_siegel.py      # ~10 seconds
python3 phase4_probit.py             # ~5 minutes (real-time OOS loop)
python3 phase7_extensions.py         # ~30 seconds

# 4. Launch dashboard
streamlit run phase6_dashboard.py
```

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| fredapi | 0.5.1 | FRED data pull |
| pandas | 2.2.2 | Data manipulation |
| numpy | 1.26.4 | Numerical computing |
| scipy | 1.13.1 | NS curve fitting, DM test |
| statsmodels | 0.14.2 | Probit MLE estimation |
| scikit-learn | 1.5.0 | AUC-ROC, Brier score |
| matplotlib | 3.9.0 | All static plots |
| streamlit | 1.35.0 | Live dashboard |

---

## References

- Nelson, C.R. & Siegel, A.F. (1987). Parsimonious modeling of yield curves. *Journal of Business*, 60(4), 473–489.
- Diebold, F.X. & Li, C. (2006). Forecasting the term structure of government bond yields. *Journal of Econometrics*, 130(2), 337–364.
- Estrella, A. & Mishkin, F.S. (1998). Predicting U.S. recessions: Financial variables as leading indicators. *Review of Economics and Statistics*, 80(1), 45–61.
- Wright, J.H. (2006). The yield curve and predicting recessions. *Federal Reserve Board Working Paper 2006-07*.
- Diebold, F.X. & Mariano, R.S. (1995). Comparing predictive accuracy. *Journal of Business & Economic Statistics*, 13(3), 253–263.
- Harvey, D., Leybourne, S. & Newbold, P. (1997). Testing the equality of prediction mean squared errors. *International Journal of Forecasting*, 13(2), 281–291.

---

## Limitations and honest caveats

- The yield curve's recession signal weakened during QE/ZLB periods (2009–2015, 2020–2022). The model does not adjust for unconventional monetary policy regimes.
- NBER recession calls lag actual turning points by 6–18 months. The announcement lag correction is conservative but imperfect.
- The probit model assumes a stationary relationship between slope and recession risk. Structural breaks (e.g. post-GFC) are not modelled explicitly.
- 8 complete recession cycles in the sample limits statistical power at short horizons.

---

*Built with real FRED data · Real-time OOS discipline · No look-ahead bias*
