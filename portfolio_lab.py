# pip install yfinance pandas numpy scipy matplotlib

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# -------------------------
# 1) PORTFOLIO INPUT
# -------------------------
holdings = pd.DataFrame([
    {"ticker": "KER.PA",   "name": "Kering",              "shares": 150},
    {"ticker": "STM.PA",   "name": "STMicroelectronics",  "shares": 685},  # on va fallback si bug
    {"ticker": "AF.PA",    "name": "Air France-KLM",      "shares": 620},
    {"ticker": "ALCRB.PA", "name": "Carbios",             "shares": 600},
    {"ticker": "AELIS.PA", "name": "Aelis Farma",         "shares": 3200},
    {"ticker": "MEDCL.PA", "name": "Medincell",           "shares": 170},
    {"ticker": "EMEIS.PA", "name": "Emeis",               "shares": 82},
])

benchmarks = {
    "CAC40": "^FCHI",
    "SP500": "^GSPC",
    "WORLD": "URTH"  # proxy MSCI World
}

start = "2015-01-01"
end = None

# -------------------------
# 2) HELPERS
# -------------------------
def safe_perf_stats(x, rf=0.0):
    """Return NaNs if x is empty."""
    ann = 252
    x = x.dropna()
    if len(x) == 0:
        return pd.Series({"CAGR": np.nan, "Vol": np.nan, "Sharpe": np.nan, "MaxDD": np.nan})

    cagr = (1 + x).prod()**(ann/len(x)) - 1
    vol = x.std() * np.sqrt(ann)
    sharpe = (cagr - rf) / vol if vol and vol > 0 else np.nan
    equity = (1 + x).cumprod()
    peak = equity.cummax()
    maxdd = ((equity / peak) - 1).min()
    return pd.Series({"CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "MaxDD": maxdd})

def beta(x, y):
    xy = pd.concat([x, y], axis=1).dropna()
    if len(xy) < 30:
        return np.nan
    x_ = xy.iloc[:,0]
    y_ = xy.iloc[:,1]
    return np.cov(x_, y_)[0,1] / np.var(y_)

def download_close(tickers, start, end):
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df = df["Close"]
    else:
        df = df[["Close"]].rename(columns={"Close": tickers[0]})
    return df

# -------------------------
# 3) DOWNLOAD (1st pass)
# -------------------------
all_tickers = holdings["ticker"].tolist() + list(benchmarks.values())
data = download_close(all_tickers, start, end)

# Drop columns that are ALL NaN
all_nan_cols = [c for c in data.columns if data[c].isna().all()]
if all_nan_cols:
    print("⚠️ Colonnes sans données (100% NaN) :", all_nan_cols)

data = data.drop(columns=all_nan_cols)

# -------------------------
# 4) FIX STM if needed: fallback to STM (NYSE) + FX conversion
# -------------------------
# If STM.PA missing or all NaN after download -> try "STM" in USD and convert to EUR using EURUSD=X
need_stm_fix = ("STM.PA" in holdings["ticker"].values) and ("STM.PA" not in data.columns)

if need_stm_fix:
    print("\n🔁 STM.PA indisponible -> fallback sur STM (NYSE, USD) + conversion en EUR via EURUSD=X")
    stm_usd = download_close(["STM"], start, end)["STM"]
    eurusd = download_close(["EURUSD=X"], start, end)["EURUSD=X"]

    # Convert USD price to EUR: EUR = USD / (USD per EUR) = USD / EURUSD
    stm_eur = (stm_usd / eurusd).rename("STM_EUR")

    data["STM_EUR"] = stm_eur

    # Replace ticker in holdings so weights/returns use STM_EUR
    holdings.loc[holdings["ticker"] == "STM.PA", "ticker"] = "STM_EUR"

# -------------------------
# 5) NOW BUILD PRICES TABLES
# -------------------------
available = [t for t in holdings["ticker"] if t in data.columns]
missing = [t for t in holdings["ticker"] if t not in data.columns]

print("\nTickers OK:", available)
print("Tickers manquants:", missing)

prices = data[available].dropna(how="all")
bench_cols = [v for v in benchmarks.values() if v in data.columns]
bench = data[bench_cols].dropna(how="all")

# -------------------------
# 6) WEIGHTS from last available price
# -------------------------
h = holdings[holdings["ticker"].isin(available)].copy()

# dernier prix non-NaN par ticker
last_px = prices.apply(lambda s: s.dropna().iloc[-1] if len(s.dropna()) else np.nan)

h["last_price"] = h["ticker"].map(last_px.to_dict())
h["value"] = h["shares"] * h["last_price"]

# debug utile
print("\n--- DEBUG last prices ---")
print(h[["ticker","shares","last_price","value"]])

# supprimer seulement ceux vraiment impossibles
h = h.dropna(subset=["value"])

h["weight"] = h["value"] / h["value"].sum()

print("\n--- Current Weights ---")
print(h[["ticker","name","shares","last_price","value","weight"]].sort_values("weight", ascending=False))

# -------------------------
# 7) RETURNS
# -------------------------
rets = prices.pct_change(fill_method=None).dropna(how="all")
rets_assets = rets[h["ticker"]].dropna()

# Align weights to returns columns
w = h.set_index("ticker")["weight"].reindex(rets_assets.columns).fillna(0.0).values

port_rets = (rets_assets * w).sum(axis=1)

print("\n--- Portfolio stats ---")
print(safe_perf_stats(port_rets))

# Bench stats
if not bench.empty:
    bench_rets = bench.pct_change(fill_method=None).dropna(how="all")
    print("\n--- Bench stats ---")
    print(bench_rets.apply(safe_perf_stats, axis=0))

    # Betas
    for k, t in benchmarks.items():
        if t in bench_rets.columns:
            print(f"Beta vs {k}: {beta(port_rets, bench_rets[t]):.2f}")

# -------------------------
# 8) Risk contribution
# -------------------------
cov = rets_assets.cov() * 252
port_vol = np.sqrt(w.T @ cov.values @ w) if len(w) > 0 else np.nan

if np.isfinite(port_vol) and port_vol > 0:
    marginal = (cov.values @ w) / port_vol
    contrib = w * marginal
    rc = pd.DataFrame({
        "ticker": rets_assets.columns,
        "weight": w,
        "risk_contrib": contrib,
        "risk_contrib_pct": contrib / contrib.sum()
    }).sort_values("risk_contrib_pct", ascending=False)

    print("\n--- Risk contribution ---")
    print(rc)

# -------------------------
# 9) VaR / CVaR
# -------------------------
alpha = 0.05
if len(port_rets.dropna()) > 0:
    var95 = np.quantile(port_rets.dropna(), alpha)
    cvar95 = port_rets[port_rets <= var95].mean()
    print(f"\nDaily VaR95: {var95:.2%} | Daily CVaR95: {cvar95:.2%}")

# -------------------------
# 10) Plot
# -------------------------
equity = (1 + port_rets.dropna()).cumprod()
equity.plot(title="Portfolio Equity Curve (normalized)")
plt.show()
