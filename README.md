# Time Series Momentum (TSMOM) Replication and Extensions

# Project Overview
This project replicates and extends the well-known **Time Series Momentum (TSMOM)** strategy, originally studied by Moskowitz, Ooi & Pedersen (2012).  
The goal is to reproduce the paper’s results, test robustness under different lookback horizons and transaction cost assumptions, and explore extensions.

---

# Methodology
1. **Data**
   - Global futures data (indices, bonds, commodities, FX) downloaded Yahoo Finance API (n = 12)

2. **Signal Construction**
   - The performance of each assets is evaluated over the course of all possible lookback periods
   - If the cumulative return of this asset is > 0 the asset is bought and held determined by the respective holding periods
   - If the cumulative return of the asset is < 0, we go short for the respective holding period 
   - Lookbacks tested: **3, 6, 12, 18 months**.
   - Holding periods tested: **1, 3, 6**

3. **Execution Assumptions**
   - Equal weights across assets in one version of the strategy and a volatilty scaled weighted portfolio in the second version of the strategy.  
   - Transaction costs modeled at 5 bps per trade.  
   - Daily mark-to-market accounting.

4. **Extensions**
   - Multiple lookback horizons and holding periods applied.  
   - Heatmap comparison of different lookback and holding period combinations.  
   - Robustness checks under varying transaction cost levels.

---

# Results
- Strategy remains profitable post-costs, though Sharpe reduces to ~0.75 for the best combination of a 6 month lookback and 3 month holding period for the unscaled portfolio.
- This best startegy produced an annual return of 6.7% with annual volatility of 8.9%
- The 6 month lookback, 1 month hold strategy was the only startegy that managed to outperform the buy & hold benchmark strategy for the set of assets over the sample period 2020-2024
- Longer lookbacks (12–18 months) tend to perform less robustly across regimes.
- All strategies underperformed the buy & hold benchmark for longer time horizons 

---

# Conclusions 
- TSMOM is a robust startegies only in specific time periods.  
- Transaction costs reduce Sharpe but do not eliminate profitability in most scenario's.  
- Strategy is sensitive to asset selection and weighting scheme.  

---

# Suggestions for future projects 
- Test with **alternative weighting** (volatility targeting, risk parity).  
- Apply other assets such as crypto, other bond markets and equities from more different sectors.  

---

