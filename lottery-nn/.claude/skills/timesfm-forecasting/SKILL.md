---
name: timesfm-forecasting
description: Zero-shot time series forecasting with Google's TimesFM foundation model. Use for any univariate time series (sales, sensors, energy, vitals, weather) without training a custom model. Supports CSV/DataFrame/array inputs with point forecasts and prediction intervals. Includes a preflight system checker script to verify RAM/GPU before first use.
allowed-tools: Read Write Edit Bash
license: Apache-2.0 license
metadata:
  skill-author: Clayton Young / Superior Byte Works, LLC (@borealBytes)
  skill-version: "1.0.0"
---

# TimesFM Forecasting

## Overview

TimesFM (Time Series Foundation Model) is a pretrained decoder-only foundation model
developed by Google Research for time-series forecasting. It works **zero-shot** — feed it
any univariate time series and it returns point forecasts with calibrated quantile
prediction intervals, no training required.

This skill wraps TimesFM for safe, agent-friendly local inference. It includes a
**mandatory preflight system checker** that verifies RAM, GPU memory, and disk space
before the model is ever loaded so the agent never crashes a user's machine.

> **Key numbers**: TimesFM 2.5 uses 200M parameters (~800 MB on disk, ~1.5 GB in RAM on
> CPU, ~1 GB VRAM on GPU). The archived v1/v2 500M-parameter model needs ~32 GB RAM.
> Always run the system checker first.

## When to Use This Skill

Use this skill when:

- Forecasting **any univariate time series** (sales, demand, sensor, vitals, price, weather)
- You need **zero-shot forecasting** without training a custom model
- You want **probabilistic forecasts** with calibrated prediction intervals (quantiles)
- You have time series of **any length** (the model handles 1–16,384 context points)
- You need to **batch-forecast** hundreds or thousands of series efficiently
- You want a **foundation model** approach instead of hand-tuning ARIMA/ETS parameters

Do **not** use this skill when:

- Performing **multivariate forecasting** (multiple input series; use LSTM/VAR instead)
- Doing **classification, anomaly detection, or non-temporal prediction** (wrong task type)
- You need **interpretability** of feature importance (TimesFM is a black box)
- Forecasting **non-temporal, non-sequential data** (use tabular/statistical ML instead)

## Quick Start: Zero-Shot Forecasting in 5 Minutes

### Step 1: Verify Your System
Always run the preflight checker first:
```bash
python scripts/check_system.py
```

This checks RAM, GPU, disk space, Python version, and installed packages. **Do not skip this step** — it prevents crashes on resource-constrained machines.

### Step 2: Install (if not already done)
```bash
uv pip install timesfm[torch]
```

Install PyTorch separately if needed (CUDA/CPU/MPS variants available).

### Step 3: Basic Forecast
```python
import timesfm
import numpy as np

# Load pretrained model
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch"
)

# Compile with config
model.compile(timesfm.ForecastConfig(
    max_context=1024,
    max_horizon=256,
    normalize_inputs=True,
    use_continuous_quantile_head=True,
    fix_quantile_crossing=True
))

# Single series forecast (horizon=24 steps ahead)
data = np.array([100, 102, 105, 103, 107, 110, 108, 112, 115, 118, 120, 122])
point_forecast, quantile_forecast = model.forecast(
    horizon=24,
    inputs=[data]  # Note: list of 1-D arrays, not 2-D
)

print(f"Point forecast (median): {point_forecast[0]}")
print(f"Quantile 10th percentile: {quantile_forecast[0, :, 1]}")
print(f"Quantile 90th percentile: {quantile_forecast[0, :, 9]}")
```

**Output shapes:**
- `point_forecast`: (batch, horizon) → median predictions
- `quantile_forecast`: (batch, horizon, 10) → percentiles [mean, q10, q20, …, q90]

### Step 4: Interpret Quantiles
Index mapping for `quantile_forecast[:, :, i]`:
```
0 → mean
1 → q10  (10th percentile)
2 → q20
3 → q30
4 → q40
5 → median (50th percentile)
6 → q60
7 → q70
8 → q80
9 → q90  (90th percentile)
```

Access 90% prediction interval:
```python
lower = quantile_forecast[0, :, 1]  # q10
upper = quantile_forecast[0, :, 9]  # q90
```

## Core Configuration: ForecastConfig

All behavior is controlled by `ForecastConfig`. Three **critical** settings:

```python
config = timesfm.ForecastConfig(
    # Maximum input sequence length (1–16,384 points)
    max_context=1024,                   # Default
    
    # Maximum forecast horizon (1–256 steps)
    max_horizon=256,                    # Default
    
    # Normalize inputs to zero mean, unit variance
    normalize_inputs=True,              # MUST be True for stability
    
    # Return 10-quantile predictions (fixed: 10 quantiles)
    use_continuous_quantile_head=True,  # Use quantile head (recommended)
    
    # Fix quantile crossing (ensure q10 < q20 < … < q90)
    fix_quantile_crossing=True          # MUST be True for valid intervals
)
```

**Never change `use_continuous_quantile_head` or `fix_quantile_crossing` without good reason.**
Always set `normalize_inputs=True` — it prevents numerical instability on real data.

## Common Workflows

### Workflow 1: Single Series Forecasting

Forecast a single univariate series:
```python
import timesfm
import numpy as np

model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch"
)
model.compile(timesfm.ForecastConfig(
    max_context=1024, max_horizon=256, normalize_inputs=True,
    use_continuous_quantile_head=True, fix_quantile_crossing=True
))

# Example: Monthly sales
sales = np.array([100, 102, 105, 103, 107, 110, 108, 112, 115, 118, 120, 122])

# Forecast next 12 months
point, quantiles = model.forecast(
    horizon=12,
    inputs=[sales]  # Must be a list of 1-D arrays
)

print(f"Predicted sales (next 12 months): {point[0]}")
print(f"90% prediction interval: [{quantiles[0, :, 1]}, {quantiles[0, :, 9]}]")
```

### Workflow 2: Batch Forecasting (Multiple Series)

Efficiently forecast hundreds of series at once:
```python
# Example: 5 time series of different lengths
series_list = [
    np.array([100, 102, 105, 103, 107, 110, 108, 112, 115, 118, 120, 122]),
    np.array([50, 51, 52, 53, 54, 55, 56, 57, 58]),
    np.array([200, 198, 202, 200, 205, 203, 208, 210, 212, 215]),
    np.array([10, 11, 10, 12, 11, 13, 12, 14, 13, 15, 14, 16]),
    np.array([500, 510, 520, 515, 525, 530, 535, 540, 545, 550])
]

# Forecast all 5 series together (5 steps ahead)
point_forecasts, quantile_forecasts = model.forecast(
    horizon=5,
    inputs=series_list
)

# Extract results
for i, series in enumerate(series_list):
    print(f"Series {i+1}: median={point_forecasts[i]}, q90={quantile_forecasts[i, :, 9]}")
```

**Batch is the most efficient way to forecast many series.**

### Workflow 3: Anomaly Detection via Quantile Intervals

Detect anomalies by checking if actual values fall outside prediction intervals:
```python
import pandas as pd

# Training series
train = np.array([100, 102, 105, 103, 107, 110, 108, 112, 115, 118, 120, 122])

# Test series (with anomalies injected)
test = np.array([125, 126, 128, 130, 50, 51, 52, 135, 137, 140, 142, 145])

# Forecast each test point using previous history
anomalies = []
for t in range(len(test)):
    history = np.concatenate([train, test[:t]])
    point, quantiles = model.forecast(horizon=1, inputs=[history])
    
    lower = quantiles[0, 0, 1]  # q10
    upper = quantiles[0, 0, 9]  # q90
    actual = test[t]
    
    is_anomaly = actual < lower or actual > upper
    anomalies.append({
        't': t, 'actual': actual, 'lower': lower, 'upper': upper, 'is_anomaly': is_anomaly
    })

df = pd.DataFrame(anomalies)
print(df[df['is_anomaly']])  # Show anomalies
```

### Workflow 4: Trend + Seasonal Decomposition (Pre-Processing)

For strong seasonality, decompose before forecasting:
```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Decompose (e.g., monthly data with annual seasonality)
decomposed = seasonal_decompose(sales, model='additive', period=12)
trend = decomposed.trend
seasonal = decomposed.seasonal
residual = decomposed.resid

# Forecast trend
point_trend, _ = model.forecast(horizon=12, inputs=[trend.dropna().values])

# Add back seasonal pattern
future_seasonal = seasonal[-12:].values  # Repeat last year's pattern
final_forecast = point_trend[0] + future_seasonal

print(f"Final forecast (trend + seasonal): {final_forecast}")
```

## Key Concepts

### What TimesFM Does

**Input:** Univariate time series (any length from 1 to 16,384 points)

**Output:**
- **Point forecast:** Median prediction for each future step
- **Quantile forecast:** 10 calibrated quantiles (10th to 90th percentile) representing the full predictive distribution

**How it works:**
1. Encodes input series with a pretrained transformer
2. Decodes future time steps autoregressively
3. Returns both point predictions and uncertainty quantiles

**Training:** None required — the model is pretrained on diverse time series from multiple domains.

### Input Normalization

The model works best when inputs are normalized:
```python
normalize_inputs=True  # Automatically standardize to zero mean, unit variance
```

This prevents numerical instability on unnormalized data (e.g., very large values like stock prices, or very small like sensor readings).

### Quantile Interpretation

For a forecast at time `t`:
```
Quantile   Value
---------  -----
q10        10th percentile (confident lower bound)
q50        Median (best single estimate)
q90        90th percentile (confident upper bound)

90% Prediction Interval = [q10, q90]
50% Prediction Interval = [q30, q70]
```

### Maximum Context and Horizon

```python
max_context=1024    # Never use more than 1024 historical points
max_horizon=256     # Never forecast more than 256 steps ahead
```

If your input is longer than `max_context`, the model uses only the most recent 1024 points.
If you request `horizon > 256`, the API will cap it.

**Practical guidance:**
- Use `max_context=512` for shorter histories (< 2 years of daily data)
- Use `max_context=1024` for longer histories (multi-year data)
- Use `max_horizon=32` for short-term forecasts (1–4 weeks ahead for daily data)
- Use `max_horizon=256` for long-term forecasts

## Common Pitfalls and Solutions

### Pitfall 1: Forgetting model.compile()
```python
# WRONG
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(...)
point, quantiles = model.forecast(horizon=24, inputs=[data])  # FAILS!

# CORRECT
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(...)
model.compile(timesfm.ForecastConfig(...))  # Required!
point, quantiles = model.forecast(horizon=24, inputs=[data])
```

### Pitfall 2: Passing 2-D Array Instead of List of 1-D Arrays
```python
# WRONG
data = np.array([[100, 102, 105, 103]])  # 2-D (1, 4)
point, quantiles = model.forecast(horizon=24, inputs=data)  # FAILS!

# CORRECT
data = np.array([100, 102, 105, 103])    # 1-D (4,)
point, quantiles = model.forecast(horizon=24, inputs=[data])  # SUCCESS
```

### Pitfall 3: Disabling Input Normalization
```python
# Risky (may cause NaN/Inf in output)
model.compile(timesfm.ForecastConfig(normalize_inputs=False))

# Safe (always do this)
model.compile(timesfm.ForecastConfig(normalize_inputs=True))
```

### Pitfall 4: Misindexing Quantiles
```python
# WRONG
q90 = quantile_forecast[0, :, 90]  # Index out of bounds! Only 10 quantiles

# CORRECT
q90 = quantile_forecast[0, :, 9]   # Index 9 = 90th percentile
q10 = quantile_forecast[0, :, 1]   # Index 1 = 10th percentile
```

### Pitfall 5: Using Wrong TimesFM Version
```python
# Archived 500M model (requires ~32 GB RAM, very slow)
model = timesfm.TimesFM_500M_torch.from_pretrained(...)  # AVOID

# Current efficient 200M model (requires ~1.5 GB RAM, fast)
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(...)  # USE THIS
```

### Pitfall 6: Skipping System Check
```bash
# WRONG: Never skip the preflight check
uv pip install timesfm[torch]
python forecast.py  # May crash on resource-constrained machine!

# CORRECT: Always check system first
python scripts/check_system.py  # Verifies RAM, GPU, disk, Python
python forecast.py              # Safe to run
```

## Example: Complete End-to-End Workflow

```python
import timesfm
import numpy as np
import pandas as pd

# Step 1: Check system
print("Checking system...")
# (Assumes scripts/check_system.py has been run)

# Step 2: Load and compile model
print("Loading TimesFM model...")
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch"
)
model.compile(timesfm.ForecastConfig(
    max_context=512,
    max_horizon=32,
    normalize_inputs=True,
    use_continuous_quantile_head=True,
    fix_quantile_crossing=True
))

# Step 3: Prepare data
print("Loading time series data...")
df = pd.read_csv('sales.csv')
sales = df['sales'].values

# Step 4: Forecast
print("Forecasting...")
point_forecast, quantile_forecast = model.forecast(
    horizon=12,
    inputs=[sales]
)

# Step 5: Analyze results
print(f"Median forecast (next 12 months): {point_forecast[0]}")
print(f"90% prediction interval:")
for t in range(12):
    lower = quantile_forecast[0, t, 1]
    upper = quantile_forecast[0, t, 9]
    print(f"  Month {t+1}: [{lower:.2f}, {upper:.2f}]")

# Step 6: Visualize
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot(sales, label='Historical')
ax.plot(range(len(sales), len(sales)+12), point_forecast[0], label='Forecast', color='red')
ax.fill_between(
    range(len(sales), len(sales)+12),
    quantile_forecast[0, :, 1],
    quantile_forecast[0, :, 9],
    alpha=0.3, color='red', label='90% Prediction Interval'
)
ax.legend()
ax.set_title('Sales Forecast')
plt.savefig('forecast.png', dpi=150)
plt.show()
```

## Integration Examples

### With Pandas DataFrames
```python
import pandas as pd

df = pd.read_csv('time_series.csv', parse_dates=['date'])
df = df.set_index('date').sort_index()

series = df['value'].values  # Extract numpy array

point, quantiles = model.forecast(horizon=30, inputs=[series])

# Return as DataFrame for easy analysis
forecast_df = pd.DataFrame({
    'median': point[0],
    'q10': quantiles[0, :, 1],
    'q90': quantiles[0, :, 9]
})
```

### With CSV Input/Output
```python
import csv

# Read CSV
with open('input.csv') as f:
    reader = csv.reader(f)
    series = [float(val) for row in reader for val in row]

# Forecast
point, quantiles = model.forecast(horizon=30, inputs=[np.array(series)])

# Write CSV
with open('forecast.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    for t in range(len(point[0])):
        writer.writerow([point[0][t], quantiles[0, t, 1], quantiles[0, t, 9]])
```

## Reference

### Key Classes and Methods

**Load Model:**
```python
model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
    "google/timesfm-2.5-200m-pytorch"
)
```

**Compile with Config:**
```python
model.compile(timesfm.ForecastConfig(
    max_context=1024,
    max_horizon=256,
    normalize_inputs=True,
    use_continuous_quantile_head=True,
    fix_quantile_crossing=True
))
```

**Forecast:**
```python
point_forecast, quantile_forecast = model.forecast(
    horizon=int,           # Number of steps to forecast
    inputs=list_of_arrays  # List of 1-D numpy arrays
)
```

### Output Shapes

- `point_forecast`: (batch_size, horizon)
- `quantile_forecast`: (batch_size, horizon, 10)

## Troubleshooting

### Issue: "CUDA out of memory"
Solution: Set `CUDA_VISIBLE_DEVICES` to use CPU instead:
```bash
CUDA_VISIBLE_DEVICES="" python forecast.py
```

### Issue: "TimesFM model not found"
Solution: Ensure internet connection and retry. Model downloads on first load.

### Issue: "NaN or Inf in output"
Solution: Check input data for NaN/Inf values; enable `normalize_inputs=True`.

### Issue: "Quantiles not monotonic"
Solution: Already handled by `fix_quantile_crossing=True` (do not disable).

## Additional Resources

- Official docs: https://github.com/google-research/timesfm
- Paper: https://arxiv.org/abs/2310.10688
- Colab demo: https://colab.research.google.com/github/google-research/timesfm/blob/main/notebooks/getting_started.ipynb