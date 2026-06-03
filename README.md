# Enhancing Time-Series Forecasting with ML through Structural-Break Identification — **GRU**

Research project, HSE FCS / DSBA. This repository is the **experimental engine** for the
GRU version of the project. It is the GRU counterpart of the LSTM coursework
(*Burlakov*) and uses the experimental constants (noise families, amplitudes, horizons,
epochs) from the group journal plan (*TimeSeries 2026, Journal of Forecasting*).

> **One-line story.** A plain GRU forecasts a non-stationary series with structural breaks
> and coloured noise; it is fair but imperfect. We detect the breaks with **PELT** (and a
> **Chow** test for comparison) and feed PELT's segment statistics into a **hybrid GRU**,
> which is consistently more accurate — most of all for *autocorrelated / structured*
> noise. We then separate the **offline** (oracle) hybrid from a **causal/online** one to
> motivate dynamic model adaptation.

---

## Quick start

```powershell
# from this folder, using the Windows py launcher
py -m pip install -r requirements.txt
```

Then open **`GRU_Structural_Breaks.ipynb`** in VS Code (or Jupyter) and **Run All**.
The notebook is *already executed* (figures and tables are embedded), so you can read it
without running anything.

* Full run ≈ **3 minutes** on CPU (3 seeds).
* For fast iteration set `QUICK = True` in the setup cell (1 seed, fewer epochs, ~35 s).
* To regenerate the notebook from source: `py _build_notebook.py`.

---

## What the project does (maps the assignment 1-to-1)

| Assignment requirement | Where it lives |
|---|---|
| Describe breaks & their impact on standard forecasters | Part 1–2 + single-series demo & **per-noise/per-horizon forecast galleries** (6.1) |
| Classical tests **(Chow, PELT)** + ML detection of *multiple* breakpoints | Part 4 (PELT, sliding-window Chow) + Part 5.4 (GRU residual monitor), scored by F1 **and Balanced Accuracy** |
| **How strong is a break?** detection power curve + strength distribution | Part 4.4 (δ = \|Δlevel\|/σ; recall rises 0.40→0.88 with strength) |
| Apply an **ML forecaster (GRU)** to a non-stationary series with breaks | Part 5 (plain GRU) |
| **Hybrid framework**: detect breaks, then adapt the model (breakpoint info as a feature) | Part 5 (feature hybrid) + **Part 6.6 de-levelled hybrid (−37% RMSE)** |
| **Empirically compare** standard vs hybrid | Part 6 (per-noise with **Wilcoxon significance + error bars**, horizon, α, global) |
| **Signed 3-class + factorial design** (TimeSeries-plan alignment) | Part 7 (GRU classifier; **6×3×2×10 = 360** factorial; binary vs signed; confusion matrix) |
| **Limitations & future work** in dynamic adaptation | Part 8 (offline-vs-causal hybrid + write-up bullets) |

## Files

```
GRU_Structural_Breaks.ipynb   <- MAIN deliverable (self-contained, pre-executed)
_build_notebook.py            <- regenerates the notebook from source
requirements.txt
figures/                      <- all .png figures (also embedded in the notebook)
results/                      <- all tables as .csv (ready for LaTeX)
```

## Method in one paragraph

The data-generating process is `x(t) = L(t) + α·N_β(t)`, where `L(t)` is a piecewise-constant
regime signal (generalised telegraph process, geometric segment lengths, exponential
relaxation `τ = max(4Δt, 0.2T)`) and `N_β` is unit-variance coloured noise
(white/pink/red/blue/violet via a `1/f^β` spectrum, plus a structured El-Niño process).
A one-layer **GRU** forecasts `x(t+h)` from a length-`L` window. **PELT** (own NumPy
implementation, `ℓ₂` cost, linear penalty) segments the observed series; the **hybrid GRU**
also receives the target's segment mean/std, in-segment position, and distances to the
nearest breaks. Forecasting is scored with **RMSE/MAE** (original units); break alignment
with **PR-AUC/ROC-AUC** of the residuals; detection with **precision/recall/F1** vs the
known switch points.

## Headline results (10 seeds, α = 1, h = 10 unless noted)

* **Hybrid beats plain on every noise, and it is statistically significant**: RMSE reduction
  white 4.3 %, blue 4.4 %, violet 3.9 %, **pink 9.8 %, red 10.7 %, El-Niño 13.1 %**, with a
  one-sided paired **Wilcoxon test p < 0.01 for all six noises** (p ≈ 1e-6 pooled). Largest
  gains are on autocorrelated/structured noise.
* **Advantage grows with horizon**: ~7 % (h=2) → ~18 % (h=25).
* **Advantage shrinks with noise amplitude**: ~12 % (α=1) → ~6 % (α=π) — once the
  irreducible noise floor dominates, no structure can help.
* **Detection — classical vs modern ML (Balanced Accuracy)**: Chow is strongest on
  white/blue/violet; the **GRU residual monitor is best on red**. PELT **over-segments** under
  autocorrelated noise (red, El-Niño) so its **specificity / Balanced Accuracy collapse** — a
  concrete "perils of misleading metrics" result (F1 alone would hide it).
* **Global**: GRU-hybrid (0.74) < naive ≈ GRU-plain (0.88) < GBM (1.21) on red noise.
* **Factorial classification (Part 7, 360 series)**: a GRU classifier detects breaks above
  chance but only modestly — **binary BA 0.57 / signed BA 0.50** overall — and detection
  **degrades with amplitude** (binary 0.62→0.53 as α: 1→π) and is **harder for the signed
  3-class task** (it must also tell up- from down-breaks). The signed confusion matrix shows
  most errors are *break-vs-no-break*, not up↔down.
* **De-levelled hybrid (Part 6.6)**: subtracting the PELT segment mean, forecasting the residual,
  and adding the level back lowers RMSE a further **−37% vs the feature hybrid** (−44% vs plain),
  biggest on structured noise (red, el-niño, pink). Same offline PELT info — better architecture.
* **Break strength (Part 4.4)**: ~**half the breaks are weak (δ<1)** — smaller than the noise — so
  detection recall climbs **0.40→0.88** with strength; this reframes the modest detection numbers as
  *weak breaks*, not a broken detector.

## Honesty / limitations (important — read before defending)

* The **headline hybrid uses PELT applied to the whole series (offline)**, so its segment
  features encode the target's regime *with hindsight*. This measures **the value of
  *accurate* break identification** — an informative **upper bound**, not a streaming result.
* The **causal/online** hybrid (Part 8.1) — running segment statistics using only past data —
  recovers only **~0–1 %** of that gain. The **offline→online gap is the central limitation**
  and the motivation for *future work in dynamic / online model adaptation* (online change-point
  detection, regime-specific fine-tuning / mixture-of-experts).
* The DGP is **synthetic, univariate, mean-shift-only**. Break-aware residual PR-AUC is
  similar for plain and hybrid: better forecasting does **not** automatically sharpen break
  *localisation*. PELT's penalty is **noise-specific** (no single global optimum).

## Reproducibility

Tested with: numpy 2.4.6, scipy 1.17.1, pandas 3.0.3, scikit-learn 1.8.0,
matplotlib 3.10.9, torch 2.12.0+cpu, Python 3.12.3 (Windows 10, CPU).
All randomness is seeded (NumPy + PyTorch); chronological 60/20/20 split.
