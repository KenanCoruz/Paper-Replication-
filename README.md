# Time Series Momentum (TSMOM) Replication and Extensions

# Project Overview
This project replicates and extends the well-known **Time Series Momentum (TSMOM)** strategy, originally studied by Moskowitz, Ooi & Pedersen (2012).  
The goal is to reproduce the paper’s results, test robustness under different lookback horizons and transaction cost assumptions, and explore extensions.

---

# Methodology
1. **Data Source**
   - Global futures data (indices, bonds, commodities, FX) downloaded via [insert source: e.g., Quandl, Yahoo Finance, Refinitiv].
   - Daily prices converted to log returns.

2. **Signal Construction**
   - Time Series Momentum:  
     \[
     \text{Signal}_t = \text{sign}\left(\sum_{k=1}^{L} r_{t-k}\right)
     \]  
     where \(L\) is the lookback window.
   - Lookbacks tested: **1, 3, 6, 12 months**.

3. **Execution Assumptions**
   - Equal notional weights across assets.  
   - Transaction costs modeled at 2–5 bps per trade.  
   - Daily mark-to-market accounting.

4. **Extensions**
   - Multiple lookback horizons blended into a composite signal.  
   - Sharpe decay analysis across sub-periods (pre/post GFC, 2010s, 2020s).  
   - Robustness checks under varying transaction cost levels.

---

# Results
- Replication of **Moskowitz et al. (2012)**: annualized Sharpe ~1.2 pre-costs.  
- Strategy remains profitable post-costs, though Sharpe reduces to ~0.75 for the best combination of a 6 month lookback and 1 month holding period .  
- Longer lookbacks (9–12 months) tend to perform more robustly across regimes.  
- Composite lookback (3+6+12) outperforms single horizons in stability.  

---

## 📊 Key Plots
### 1. Equity Curve (TSMOM vs Benchmark)
![Equity Curve](plots/equity_curve.png)

### 2. Performance by Lookback Horizon
![Lookback Performance](plots/lookbacks.png)

### 3. Distribution of Monthly Returns
![Return Distribution](plots/return_distribution.png)

---

# Conclusions 
- TSMOM is a **robust cross-asset anomaly**, especially at longer horizons.  
- Transaction costs reduce Sharpe but do not eliminate profitability.  
- Strategy is sensitive to **asset selection** and **weighting scheme**.  
- Composite signals improve robustness vs. single-lookback rules.

---

# Suggestions for future projects 
- Incorporate **volatility scaling** to equalize risk contributions.  
- Test with **alternative weighting** (volatility targeting, risk parity).  
- Apply to **crypto assets** for out-of-sample testing.  

---

## 🛠️ How to Run
Clone repo and install requirements:
```bash
git clone https://github.com/YOUR_USERNAME/tsmom-replication.git
cd tsomom-replication
pip install -r requirements.txt
