# pip install yfinance pandas numpy scipy

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

ANN = 252
START = "2015-01-01"
END = None

# -------------------------
# 1) Universe: stocks + ETFs (ajoute/enlève ce que tu veux)
# -------------------------
STOCKS_PA = ["KER.PA", "AF.PA", "ALCRB.PA", "AELIS.PA", "MEDCL.PA", "EMEIS.PA"]
CORE_ETF = ["URTH"]           # World
# Optionnel (fortement recommandé pour différencier les stratégies):
# - ETF dividendes: "VIG" (US) ou "SCHD" (US) ou équivalent UCITS si tu en as un ticker Yahoo
# - ETF obligations: "IEF" (US Treasuries 7-10y), "SHY" (1-3y), ou équivalent UCITS
OPTIONAL = ["VIG", "IEF"]     # si ces tickers ne passent pas, tu peux les enlever

FX = ["EURUSD=X"]
STM_US = ["STM"]              # ADR US (USD)

# On va créer STM_EUR nous-même
UNIVERSE = STOCKS_PA + ["STM_EUR"] + CORE_ETF + OPTIONAL

# -------------------------
# 2) Download helpers
# -------------------------
def download_close(tickers, start=START, end=END):
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        return df["Close"]
    return df[["Close"]].rename(columns={"Close": tickers[0]})

def build_prices():
    # Download EUR assets (Paris + ETFs)
    base_tickers = list(set(STOCKS_PA + CORE_ETF + OPTIONAL))
    prices_base = download_close(base_tickers, START, END)

    # STM in USD and FX
    stm_usd = download_close(STM_US, START, END)["STM"]
    eurusd = download_close(FX, START, END)["EURUSD=X"]
    stm_eur = (stm_usd / eurusd).rename("STM_EUR")

    prices = pd.concat([prices_base, stm_eur], axis=1).sort_index()
    prices = prices.dropna(how="all")
    return prices

prices = build_prices()

# returns
rets = prices.pct_change(fill_method=None)

# keep assets with enough observations
min_obs = int(ANN * 2)
valid_cols = [c for c in rets.columns if rets[c].dropna().shape[0] >= min_obs]
rets = rets[valid_cols].dropna()
UNIVERSE = valid_cols

mu = rets.mean() * ANN
cov = rets.cov() * ANN

# -------------------------
# 3) Risk functions
# -------------------------
def port_return(w):
    return float(w @ mu.values)

def port_vol(w):
    return float(np.sqrt(w @ cov.values @ w))

def port_cvar(w, alpha=0.05):
    pr = rets.values @ w
    var = np.quantile(pr, alpha)
    cvar = pr[pr <= var].mean()
    return float(-cvar)  # positive loss measure

# -------------------------
# 4) Optimizer with per-asset bounds + min allocations
# -------------------------
def make_bounds(default_max=0.25, overrides=None):
    """
    overrides: dict asset -> (min,max) or asset->max or asset->(None,max)
    """
    if overrides is None:
        overrides = {}
    b = []
    for a in UNIVERSE:
        if a in overrides:
            v = overrides[a]
            if isinstance(v, tuple):
                lo = 0.0 if v[0] is None else float(v[0])
                hi = float(v[1])
            else:
                lo, hi = 0.0, float(v)
        else:
            lo, hi = 0.0, float(default_max)
        b.append((lo, hi))
    return b

def solve(objective, bounds, cons, w0=None, name=""):
    n = len(UNIVERSE)
    if w0 is None:
        # start from equal weights projected into bounds
        w0 = np.repeat(1/n, n)
        hi = np.array([bb[1] for bb in bounds])
        lo = np.array([bb[0] for bb in bounds])
        w0 = np.minimum(w0, hi)
        w0 = np.maximum(w0, lo)
        w0 = w0 / w0.sum()

    res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 3000})
    if not res.success:
        raise RuntimeError(f"{name}: {res.message}")
    return res.x

def report(w, title):
    out = pd.DataFrame({"asset": UNIVERSE, "weight": w}).sort_values("weight", ascending=False)
    print(f"\n=== {title} ===")
    print(f"Expected Return(mu): {port_return(w):.2%} | Vol: {port_vol(w):.2%} | CVaR95(loss): {port_cvar(w):.2%}")
    print(out[out["weight"] > 1e-4].to_string(index=False))
    return out

# -------------------------
# 5) Strategy A: 10Y GROWTH (agressif mais structuré)
# -------------------------
def optimize_10y_growth():
    """
    Objectif: maximiser le rendement attendu sous contraintes de risque
    et forcer une vraie poche 'satellite'.
    """
    cons = [{"type":"eq", "fun": lambda w: np.sum(w) - 1.0}]

    # Caps pour éviter URTH 80% et forcer diversification
    overrides = {
        "URTH": (0.25, 0.55),     # core important mais pas dominant
        "STM_EUR": (0.05, 0.25),  # conviction techno
        "KER.PA": (0.05, 0.25),   # conviction value/luxe
    }
    bounds = make_bounds(default_max=0.20, overrides=overrides)

    # contrainte "poche satellites" = somme des actions hors ETFs >= 35%
    etfs = [a for a in UNIVERSE if a in ["URTH", "VIG", "IEF"]]
    idx_etf = [UNIVERSE.index(a) for a in etfs if a in UNIVERSE]
    cons.append({"type":"ineq", "fun": lambda w, idx=idx_etf: 0.35 - np.sum(w[idx]) * 1.0 * (-1)})  # <= -> rewrite
    # plus simple: sum(ETFs) <= 65%  => 0.65 - sum(ETFs) >= 0
    cons[-1] = {"type":"ineq", "fun": lambda w, idx=idx_etf: 0.65 - np.sum(w[idx])}

    # contrainte risque: CVaR95 <= 3.0% (ajuste)
    cons.append({"type":"ineq", "fun": lambda w: 0.030 - port_cvar(w, alpha=0.05)})

    # objectif: maximiser return (donc minimiser -return)
    def obj(w):
        return -port_return(w)

    return solve(obj, bounds, cons, name="10Y Growth")

# -------------------------
# 6) Strategy B: STABLE DEFENSIVE (min vol + structure)
# -------------------------
def optimize_stable_defensive():
    """
    Objectif: minimiser volatilité + limiter pertes extrêmes,
    avec un core ETF important et caps stricts sur titres risqués.
    """
    cons = [{"type":"eq", "fun": lambda w: np.sum(w) - 1.0}]

    overrides = {
        "URTH": (0.45, 0.70),
        "IEF": (0.10, 0.40) if "IEF" in UNIVERSE else (None, 0.0),  # si IEF absent, ignoré
        "VIG": (0.05, 0.25) if "VIG" in UNIVERSE else (None, 0.0),
        # caps durs sur small/biotech
        "AELIS.PA": (0.0, 0.04) if "AELIS.PA" in UNIVERSE else (None, 0.0),
        "ALCRB.PA": (0.0, 0.04) if "ALCRB.PA" in UNIVERSE else (None, 0.0),
        "EMEIS.PA": (0.0, 0.02) if "EMEIS.PA" in UNIVERSE else (None, 0.0),
    }
    bounds = make_bounds(default_max=0.10, overrides=overrides)

    # contrainte de rendement minimum (sinon min vol peut tomber trop bas)
    cons.append({"type":"ineq", "fun": lambda w: port_return(w) - 0.06})  # >= 6%

    # contrainte CVaR95 <= 2.6% (ajuste)
    cons.append({"type":"ineq", "fun": lambda w: 0.026 - port_cvar(w, 0.05)})

    def obj(w):
        return port_vol(w)

    return solve(obj, bounds, cons, name="Stable Defensive")

# -------------------------
# 7) Strategy C: CASHFLOW (income proxy + stabilité)
# -------------------------
def optimize_cashflow_income():
    """
    Objectif: portefeuille orienté income/quality
    - min allocation sur ETF income (VIG) si dispo
    - core URTH présent mais plafonné
    - limitation biotech/smallcaps
    """
    cons = [{"type":"eq", "fun": lambda w: np.sum(w) - 1.0}]

    overrides = {
        "URTH": (0.30, 0.60),
        "VIG": (0.20, 0.60) if "VIG" in UNIVERSE else (None, 0.0),
        "IEF": (0.00, 0.30) if "IEF" in UNIVERSE else (None, 0.0),
        "AELIS.PA": (0.0, 0.03) if "AELIS.PA" in UNIVERSE else (None, 0.0),
        "ALCRB.PA": (0.0, 0.03) if "ALCRB.PA" in UNIVERSE else (None, 0.0),
        "EMEIS.PA": (0.0, 0.02) if "EMEIS.PA" in UNIVERSE else (None, 0.0),
    }
    bounds = make_bounds(default_max=0.12, overrides=overrides)

    # objectif: compromis vol vs return (plus "income" = plus stable)
    lam = 0.15
    def obj(w):
        return port_vol(w) - lam * port_return(w)

    # rendement min et CVaR limit
    cons.append({"type":"ineq", "fun": lambda w: port_return(w) - 0.07})  # >= 7%
    cons.append({"type":"ineq", "fun": lambda w: 0.027 - port_cvar(w, 0.05)})

    return solve(obj, bounds, cons, name="Cashflow Income")

# -------------------------
# 8) Strategy D: TRADING (momentum selection + risk parity)
# -------------------------
def trading_momentum_risk_parity(lookback=126, top_k=4):
    """
    Allocation dynamique:
    1) calcule momentum sur lookback (~6 mois)
    2) retient top_k actifs au momentum positif
    3) fait risk parity sur ce sous-univers
    """
    px = prices[UNIVERSE].dropna()
    mom = px / px.shift(lookback) - 1
    last = mom.iloc[-1].dropna().sort_values(ascending=False)

    # select positive momentum
    sel = last[last > 0].head(top_k).index.tolist()
    if len(sel) < 2:
        sel = last.head(max(2, top_k)).index.tolist()

    sub = rets[sel].dropna()
    sub_cov = sub.cov() * ANN
    n = len(sel)

    def sub_vol(w):
        return float(np.sqrt(w @ sub_cov.values @ w))

    def obj(w):
        v = sub_vol(w)
        mrc = (sub_cov.values @ w) / (v + 1e-12)
        rc = w * mrc
        rc = rc / (rc.sum() + 1e-12)
        return ((rc - 1/n)**2).sum()

    bounds = [(0.0, 1.0)] * n
    cons = [{"type":"eq", "fun": lambda w: np.sum(w) - 1.0}]
    w0 = np.repeat(1/n, n)
    res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons)
    w_sub = res.x

    # map to full universe weights
    w_full = np.zeros(len(UNIVERSE))
    for i, a in enumerate(sel):
        w_full[UNIVERSE.index(a)] = w_sub[i]

    return w_full, last, sel

# -------------------------
# 9) Run all
# -------------------------
w_growth = optimize_10y_growth()
report(w_growth, "10Y Growth (distinct, URTH capped + satellites)")

w_stable = optimize_stable_defensive()
report(w_stable, "Stable Defensive (URTH+IEF/VIG, low tail risk)")

w_income = optimize_cashflow_income()
report(w_income, "Cashflow Income (income ETF min, low biotech)")

w_trade, mom_rank, selected = trading_momentum_risk_parity(lookback=126, top_k=4)
report(w_trade, f"Trading (Momentum top {len(selected)} + Risk Parity)")
print("\nSelected by momentum:", selected)
print("\nMomentum ranking (last):")
print(mom_rank.to_string())
