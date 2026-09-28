# Deep Hedging — Thesis Implementation

This is my implementation of the deep hedging paper by Buehler et al. (2019) as part of my thesis work. The project explores using neural networks to learn optimal hedging strategies for options under realistic market conditions (transaction costs, discrete trading).

The original repo is from [YuMan-Tam](https://github.com/YuMan-Tam/deep-hedging) — I've cleaned it up, fixed several bugs, and written a standalone runner script to make it easier to experiment with.

---

## What it does

A bank sells a call option and needs to hedge its exposure by trading the underlying stock over 30 days. Black-Scholes gives a formula for the optimal hedge — but it assumes zero transaction costs and continuous trading. In reality both assumptions break down.

Deep hedging trains a neural network to learn the hedging strategy directly from simulated market scenarios, with transaction costs included in the training objective. The result is a strategy that outperforms Black-Scholes when costs are present.

---

## Key results so far

| Transaction cost (ε) | BS mean P&L | DH mean P&L | DH advantage |
|---|---|---|---|
| 0.00 (no costs) | ≈ 0 | ≈ 0 | None — both identical ✓ |
| 0.01 (1% per trade) | ≈ −3.0 | ≈ −1.5 | **+£1.5 per option** |

With zero costs, the network correctly learns to replicate the Black-Scholes delta (sanity check). With 1% costs, it learns a near-constant hedge position — trading as little as possible to avoid accumulated transaction costs.

---

## Setup

**Requires Python 3.11** (not 3.12+, TensorFlow 2.13 has no wheel for newer versions).

```bash
# Create a virtual environment with Python 3.11
py -3.11 -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Then run:
```bash
python run_deep_hedging.py
```

Output saved to `results.png` in the same folder.

---

## Parameters (edit Section 1 of run_deep_hedging.py)

| Parameter | Default | What it controls |
|---|---|---|
| `epsilon` | 0.01 | Transaction cost per trade (0 = frictionless) |
| `sigma` | 0.2 | Stock volatility (20% annual) |
| `K` | 100 | Strike price |
| `N` | 30 | Number of trading days |
| `Ktrain` | 100,000 | Number of simulated paths for training |
| `loss_type` | "Entropy" | Risk measure: "Entropy" or "CVaR" |
| `loss_param` | 1.0 | λ for Entropy (risk aversion), α for CVaR |
| `epochs` | 200 | Training epochs |

---

## Bugs fixed from original repo

The original notebook was written in 2021 and had a few issues running on modern Python/TF:

- `is "string"` comparisons in `EuropeanCall.py` — always returned False in Python 3.11, causing silent wrong behaviour. Fixed to `== "string"`.
- `delta_constraint` was only applied for `sigmoid`/`tanh` output activations, silently ignored with `linear`. Changed to `sigmoid`.
- `@tf.function` caused indefinite hang on Windows with TF 2.13 and large functional models. Replaced with `model.compile` + `model.fit`.
- `from_tensors` passed entire dataset as one batch. Fixed to `from_tensor_slices`.
- `results.png` used a relative path that broke when VS Code changed the working directory. Fixed to absolute path via `__file__`.

---

## References

- Buehler, H., Gonon, L., Hyer, T., & Wood, B. (2019). [Deep Hedging](https://arxiv.org/abs/1802.03042). *Quantitative Finance*.
- Black, F., & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. *Journal of Political Economy*.
- Whalley, A. E., & Wilmott, P. (1997). An Asymptotic Analysis of an Optimal Hedging Model for Option Pricing with Transaction Costs.
