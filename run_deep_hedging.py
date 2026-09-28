# -*- coding: utf-8 -*-
# Force UTF-8 output on Windows (avoids cp1252 UnicodeEncodeError)
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Use non-interactive backend — saves to file instead of opening a window
import matplotlib
matplotlib.use('Agg')

"""
Deep Hedging - Local Runner
============================
Replicates the full pipeline from the Colab notebook.
Run with: python run_deep_hedging.py

Sections:
  1. Parameters          — edit this section to experiment
  2. Monte Carlo         — simulate stock price paths
  3. Payoff & BS Benchmark — compute option payoff and Black-Scholes delta
  4. Build Model         — construct the deep hedging Keras model
  5. Train               — run the training loop
  6. Results             — plot PnL histograms and learned delta curve
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Make all submodules importable regardless of working directory ──────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import QuantLib as ql
import tensorflow as tf
tf.random.set_seed(42)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau

from stochastic_processes import BlackScholesProcess
from instruments        import EuropeanCall
from deep_hedging       import Deep_Hedging_Model, Delta_SubModel
from loss_metrics       import Entropy, CVaR
from utilities          import train_test_split

print(f"TensorFlow {tf.__version__} | QuantLib {ql.__version__}")

# ════════════════════════════════════════════════════════════════════════════
# 1.  P A R A M E T E R S
#     ← Change anything here to experiment
# ════════════════════════════════════════════════════════════════════════════

# --- Option parameters ---
S0        = 100.0    # Initial stock price
K         = 100.0    # Strike price  (ATM)
sigma     = 0.2      # Implied volatility (20%)
risk_free = 0.0      # Risk-free rate
dividend  = 0.0      # Continuous dividend yield
N         = 30       # Number of trading days (= time steps)

# --- Transaction cost ---
epsilon = 0.01        # Proportional cost (e.g. 0.01 = 1%).
                     # Set > 0 to see deep hedging outperform BS.

# --- Monte Carlo ---
Ktrain      = 100_000   # Training paths
Ktest_ratio = 0.2       # Test size as fraction of training
seed        = 0         # 0 = deterministic; change for different random draw

# --- Deep Hedging network ---
d                          = 1          # Hidden layers per time step
m                          = 15         # Neurons per hidden layer
strategy_type              = "simple"   # "simple" or "recurrent"
activation_dense           = "relu"     # Hidden activation
activation_output          = "sigmoid"  # sigmoid ensures delta stays in [0,1] via delta_constraint
delta_constraint           = (0.0, 1.0)  # clips delta in [0,1] — physically correct for a call
use_batch_norm             = False
share_strategy_across_time = False
cost_structure             = "proportional"  # or "constant"
final_period_cost          = False

# --- Loss / risk measure ---
loss_type  = "Entropy"   # "Entropy" or "CVaR"
loss_param = 1.0         # λ for Entropy, α for CVaR (e.g. 0.95)

# --- Training ---
learning_rate = 1e-3
batch_size    = 256
epochs        = 200

# --- QuantLib calendar setup ---
calculation_date = ql.Date(1, 1, 2021)
maturity_date    = calculation_date + N
day_count        = ql.Actual365Fixed()

# ════════════════════════════════════════════════════════════════════════════
# 2.  M O N T E   C A R L O   S I M U L A T I O N
# ════════════════════════════════════════════════════════════════════════════
print("\n[1/5] Simulating stock price paths...")

nobs     = int(Ktrain * (1 + Ktest_ratio))
dt       = day_count.yearFraction(calculation_date, calculation_date + 1)
maturity = N * dt

stochastic_process = BlackScholesProcess(
    s0=S0, sigma=sigma, risk_free=risk_free,
    dividend=dividend, day_count=day_count, seed=seed
)

S = stochastic_process.gen_path(length=maturity, time_step=N, num_paths=nobs)
# S shape: (nobs, N+1)  — each row is one path, each column one time step

print(f"  S0={S0}, sigma={sigma}, risk_free={risk_free}")
print(f"  Time steps: {N}  |  dt = 1/365  |  Paths: {nobs}")
print("  Simulation complete.\n")

# ════════════════════════════════════════════════════════════════════════════
# 3.  O P T I O N   P A Y O F F   &   B L A C K - S C H O L E S   B E N C H M A R K
# ════════════════════════════════════════════════════════════════════════════
print("[2/5] Computing BS price, delta, and payoff...")

european_call = EuropeanCall()

# BS option price at every time step (shape: nobs × N+1)
price_BS = european_call.get_BS_price(
    S=S, sigma=sigma, risk_free=risk_free, dividend=dividend,
    K=K, exercise_date=N, calculation_date=0,
    day_count=day_count, dt=dt, evaluation_method="Numpy"
)

# Initial BS price (used as the "cost" of the option when entering the hedge)
initial_price_BS = price_BS[0, 0]

# BS delta at every time step (shape: nobs × N+1)
delta_BS = european_call.get_BS_delta(
    S=S, sigma=sigma, risk_free=risk_free, dividend=dividend,
    K=K, exercise_date=N, calculation_date=0,
    day_count=day_count, dt=dt, evaluation_method="Numpy"
)

# Terminal payoff for a SHORT call: the bank SOLD the option, so payoff is negative
payoff = -np.maximum(S[:, -1] - K, 0.0)   # shape: (nobs,)

# BS P&L (for benchmark comparison later)
PnL_BS = european_call.get_BS_PnL(
    S=S, payoff=payoff, delta=delta_BS, dt=dt,
    risk_free=risk_free, final_period_cost=final_period_cost,
    epsilon=epsilon, cost_structure=cost_structure
)

print(f"  Initial BS price: {initial_price_BS:.4f}")
print(f"  Mean BS PnL:  {PnL_BS.mean():.4f}  |  Std: {PnL_BS.std():.4f}\n")

# ════════════════════════════════════════════════════════════════════════════
# 4.  B U I L D   D E E P   H E D G I N G   M O D E L
# ════════════════════════════════════════════════════════════════════════════
print("[3/5] Building Keras model...")

# --- Prepare inputs ---
# The "information set" the network sees is the BS delta (a normalised price signal)
# Shape at each t: (nobs, 1) — note model expects a list of per-timestep arrays

# Build the list of inputs:
#   [prc_0, info_0, prc_1, info_1, ..., prc_{N-1}, info_{N-1}, prc_N, payoff]
model_inputs = []
for t in range(N):
    model_inputs.append(S[:, t:t+1])          # prc_t  (nobs, 1)
    if t < N:
        model_inputs.append(delta_BS[:, t:t+1])  # information_set_t  (nobs, 1)

model_inputs.append(S[:, N:N+1])              # prc_N
model_inputs.append(payoff[:, np.newaxis])    # payoff

# --- Train / test split ---
xtrain, xtest = train_test_split(model_inputs, test_size=Ktest_ratio)
Ktrain_actual = xtrain[0].shape[0]

# --- Build model ---
model = Deep_Hedging_Model(
    N=N, d=d, m=m,
    risk_free=risk_free, dt=dt,
    initial_wealth=initial_price_BS,  # Start with the option premium received
    epsilon=epsilon,
    final_period_cost=final_period_cost,
    strategy_type=strategy_type,
    use_batch_norm=use_batch_norm,
    kernel_initializer="he_uniform",
    activation_dense=activation_dense,
    activation_output=activation_output,
    delta_constraint=delta_constraint,
    share_stretegy_across_time=share_strategy_across_time,
    cost_structure=cost_structure
)

# Submodel: extracts delta at a given day (for plotting the learned delta curve)
submodel = Delta_SubModel(
    model=model,
    days_from_today=0,
    share_stretegy_across_time=share_strategy_across_time,
    strategy_type=strategy_type
)



# ════════════════════════════════════════════════════════════════════════════
# 5.  T R A I N I N G
# ════════════════════════════════════════════════════════════════════════════
print("[4/5] Training...")

certainty_equiv = tf.Variable(0.0, trainable=False, name="certainty_equiv")

# Custom loss: model.compile expects loss(y_true, y_pred).
# Our model outputs wealth; y_true is an ignored dummy.
def custom_loss(y_true, wealth):
    if loss_type == "Entropy":
        return Entropy(wealth=wealth, w=certainty_equiv, loss_param=loss_param)
    else:
        return CVaR(wealth=wealth, w=certainty_equiv, loss_param=loss_param)

model.compile(optimizer=Adam(learning_rate=learning_rate), loss=custom_loss)

# Plain numpy arrays — model.fit handles batching and shuffling
xtrain_np = [x.astype(np.float32) for x in xtrain]
xtest_np  = [x.astype(np.float32) for x in xtest]
ytrain_dummy = np.zeros(Ktrain_actual, dtype=np.float32)
ytest_dummy  = np.zeros(xtest[0].shape[0], dtype=np.float32)

print(f"  Strategy type: {strategy_type} | Layers: {d} | Neurons: {m}")
print(f"  Transaction cost (epsilon): {epsilon}  |  Loss: {loss_type}  |  param: {loss_param}\n")

history = model.fit(
    xtrain_np, ytrain_dummy,
    batch_size=batch_size,
    epochs=epochs,
    shuffle=True,
    validation_data=(xtest_np, ytest_dummy),
    callbacks=[tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=10, min_delta=1e-4,
        min_lr=1e-6, verbose=1
    )],
    verbose=1
)

loss_history = history.history["val_loss"]
xtest_tf  = [tf.constant(x, dtype=tf.float32) for x in xtest]
xtrain_tf = [tf.constant(x, dtype=tf.float32) for x in xtrain]

print("\nTraining complete.\n")

# ════════════════════════════════════════════════════════════════════════════
# 6.  R E S U L T S
# ════════════════════════════════════════════════════════════════════════════
print("[5/5] Plotting results...")

# Deep hedging P&L on test set
PnL_DH = model(xtest_tf, training=False).numpy().squeeze()

# BS P&L on test set (split same way)
_, PnL_BS_test = train_test_split([PnL_BS], test_size=Ktest_ratio)
PnL_BS_test = PnL_BS_test[0].squeeze()

# Learned delta curve (at day 0, as function of information_set = BS delta)
I_range = np.linspace(0, 1, 100)  # BS delta range
if strategy_type == "simple":
    delta_input = I_range[:, np.newaxis].astype(np.float32)
elif strategy_type == "recurrent":
    delta_input = [I_range[:, np.newaxis].astype(np.float32),
                   np.ones((100, 1), dtype=np.float32) * 0.5]

DH_delta = submodel(delta_input, training=False).numpy().squeeze()

# --- Figure ---
fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 2, figure=fig)

# Plot 1: PnL histogram comparison
ax1 = fig.add_subplot(gs[0, :])
x_min = min(PnL_DH.min(), (PnL_BS_test + initial_price_BS).min()) - 0.5
x_max = max(PnL_DH.max(), (PnL_BS_test + initial_price_BS).max()) + 0.5
bins  = np.linspace(x_min, x_max, 60)

ax1.hist(PnL_DH,                        bins=bins, alpha=0.6, label="Deep Hedging",  color="royalblue")
ax1.hist(PnL_BS_test + initial_price_BS, bins=bins, alpha=0.6, label="Black-Scholes", color="tomato")
ax1.axvline(0, color="black", linewidth=1.2, linestyle="--", label="Break-even")
ax1.set_xlabel("P&L")
ax1.set_ylabel("Count")
ax1.set_title(f"P&L Distribution — ε={epsilon}, Loss={loss_type}(λ={loss_param}), "
              f"Epochs={epochs}, d={d}, m={m}")
ax1.legend()
ax1.grid(alpha=0.3)

# Plot 2: Learned delta vs BS delta
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(I_range, I_range,   "r--",  label="Black-Scholes Δ", linewidth=1.5)
ax2.plot(I_range, DH_delta,  "b-",   label="Deep Hedging Δ",  linewidth=2)
ax2.set_xlabel("BS Delta (information set)")
ax2.set_ylabel("Hedge Position")
ax2.set_title("Learned Delta vs. Black-Scholes Delta  (Day 0)")
ax2.legend()
ax2.grid(alpha=0.3)

# Plot 3: Training loss curve
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(range(1, epochs + 1), loss_history, "g-o", markersize=3)
ax3.set_xlabel("Epoch")
ax3.set_ylabel(f"{loss_type} Loss (OOS)")
ax3.set_title("Training Loss")
ax3.grid(alpha=0.3)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")


# --- Summary stats ---
print("=" * 55)
print(f"  Initial BS option price:       {initial_price_BS:.4f}")
print(f"  Deep Hedging certainty equiv:  {-loss_history[-1]:.4f}")
print(f"  BS    PnL  mean / std:  {(PnL_BS_test + initial_price_BS).mean():.4f} / {PnL_BS_test.std():.4f}")
print(f"  DH    PnL  mean / std:  {PnL_DH.mean():.4f} / {PnL_DH.std():.4f}")
print("=" * 55)
print(f"\nDone! Results saved to {out_path}")
