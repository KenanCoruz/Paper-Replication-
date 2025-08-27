import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns


START = "2020-01-01"
END   = "2024-12-31"
INTERVAL = "1mo"

LOOKBACKS = [3, 6, 12, 18]
HOLDING_PERIODS = [1, 3, 6]

SKIP_RECENT   = 2
TARGET_VOL_M  = 0.01
VOL_METHOD    = "ewm"
VOL_SPAN      = 12
VOL_ROLL_WIN  = 12
MAX_LEVERAGE  = 5.0
TC            = 0.0005
TICKERS = {
    "SPY":  "Equities_US_SP500",
    "EFA":  "Equities_Developed_ExUS",
    "QQQ":  "Equities_US_NASDAQ100",
    "EEM":  "Equities_Emerging",
    "VGK":  "Equities_Europe",
    "EWJ":  "Equities_Japan",
    "TLT":  "Bonds_US_20Y",
    "GLD":  "Commodities_Gold",
    "USO":  "Commodities_Oil",
    "EURUSD=X": "FX_EURUSD",
    "JPYUSD=X": "FX_JPYUSD",
    "GBPUSD=X": "FX_GBPUSD",
}

# Methods
def download_prices(tickers, start, end, interval):
    df = yf.download(list(tickers), start=start, end=end, interval=interval)
    if "Adj Close" in df.columns:
        prices = df["Adj Close"]
    elif "Close" in df.columns:
        prices = df["Close"]
    else:
        raise ValueError("No 'Adj Close' or 'Close' column found in Yahoo Finance data.")
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    return prices.dropna(how="all").ffill()


def ts_signal_hp(returns_df, lookback, skip_recent, holding_period):
    base_signal = returns_df.shift(skip_recent).rolling(window=lookback).sum().apply(np.sign)

    # If HP=1 we just use base_signal (no forward-fill with limit=0!)
    if holding_period == 1:
        return base_signal

    # Otherwise, forward-fill each column up to holding_period-1 months
    signal_hp = base_signal.copy()
    for col in signal_hp.columns:
        # Forward-fill only up to HP-1 steps to simulate holding without re-forming
        signal_hp[col] = signal_hp[col].ffill(limit=holding_period - 1)

    return signal_hp


def estimate_vol(returns_df, method="ewm", span=12, roll_window=12):
    if method == "ewm":
        return returns_df.ewm(span=span, min_periods=span).std()
    else:
        return returns_df.rolling(window=roll_window, min_periods=roll_window).std()

def compute_gross_net_portfolio(signal_df, weights_df, returns_df, tc):
    sig_lag = signal_df.shift(1)
    w_lag   = weights_df
    gross_asset = sig_lag * returns_df * w_lag
    position    = sig_lag * w_lag
    position_l1 = position.shift(1)
    turnover    = (position - position_l1).abs()
    costs_asset = turnover * tc
    net_asset   = gross_asset - costs_asset
    gross_port  = gross_asset.mean(axis=1, skipna=True)
    net_port    = net_asset.mean(axis=1, skipna=True)
    total_cost  = costs_asset.sum(axis=1, skipna=True).sum()
    return gross_port, net_port, total_cost

def ann_return(x, periods=12):
    return x.mean(skipna=True) * periods

def ann_vol(x, periods=12):
    return x.std(skipna=True) * np.sqrt(periods)

def sharpe(x, periods=12):
    vol = x.std(skipna=True)
    return np.nan if vol == 0 or np.isnan(vol) else (x.mean(skipna=True) / vol) * np.sqrt(periods)

# Data
prices = download_prices(TICKERS.keys(), START, END, INTERVAL)
returns = prices.pct_change().dropna(how="all")

# Volatility (for scaled version)
vol = estimate_vol(returns, VOL_METHOD, VOL_SPAN, VOL_ROLL_WIN)
raw_weights = TARGET_VOL_M / vol
raw_weights = raw_weights.clip(upper=MAX_LEVERAGE).replace([np.inf, -np.inf], np.nan).fillna(0.0)
weights_scaled = raw_weights.shift(1)

# Volatility (for unscaled version)
weights_unscaled = pd.DataFrame(1.0, index=returns.index, columns=returns.columns)

# Buy & Hold benchmark
bh_returns = returns.mean(axis=1, skipna=True)

# Backtest
def run_backtest(weights_df, label):
    results = []
    for lb in LOOKBACKS:
        for hp in HOLDING_PERIODS:
            sig = ts_signal_hp(returns, lb, SKIP_RECENT, hp)
            g_port, n_port, cost_tot = compute_gross_net_portfolio(sig, weights_df, returns, TC)
            results.append({
                "Type": label,
                "Lookback": lb,
                "Holding_Period": hp,
                "Ann_Return": ann_return(n_port),
                "Ann_Vol": ann_vol(n_port),
                "Sharpe": sharpe(n_port),
                "Total_Cost": cost_tot,
                "CumRet_Series": (1 + n_port).cumprod()
            })
    return results

results_scaled   = run_backtest(weights_scaled, "Scaled")
results_unscaled = run_backtest(weights_unscaled, "Unscaled")

perf_df = pd.DataFrame(results_scaled + results_unscaled)[
    ["Type", "Lookback", "Holding_Period", "Ann_Return", "Ann_Vol", "Sharpe", "Total_Cost"]
]

print("\n=== Multi Lookback × Holding Period Performance ===")
print(perf_df.round(3).to_string(index=False))

# Heatmap
fig, axes = plt.subplots(1, 2, figsize=(14,6))
for ax, label in zip(axes, ["Scaled", "Unscaled"]):
    pivot_sharpe = perf_df[perf_df.Type == label].pivot(index="Lookback", columns="Holding_Period", values="Sharpe")
    sns.heatmap(pivot_sharpe, annot=True, fmt=".2f", cmap="RdYlGn", center=0, ax=ax)
    ax.set_title(f"{label} - Sharpe Ratio Heatmap")
plt.tight_layout()
plt.show()

# Plots of best performing strategies
top_n = 2
plt.figure(figsize=(12,6))
for dataset in [results_scaled, results_unscaled]:
    top_strats = sorted(dataset, key=lambda x: x["Sharpe"], reverse=True)[:top_n]
    for row in top_strats:
        label = f"{row['Type']} LB {row['Lookback']}M HP {row['Holding_Period']}M"
        plt.plot(row["CumRet_Series"], label=label)

plt.plot((1 + bh_returns).cumprod(), label="Buy & Hold", linestyle="--", color="black")
plt.title(f"Performance of Best TSMOM  TSMOM Strategies vs Buy & Hold")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
