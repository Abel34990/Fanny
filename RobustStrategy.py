# pip install yfinance pandas numpy scipy scikit-learn

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

ANN = 252
START = "2015-01-01"
END = None

STOCKS_PA = ["KER.PA", "AF.PA", "ALCRB.PA", "AELIS.PA", "MEDCL.PA", "EMEIS.PA"]
CORE_ETF = ["URTH"]
OPTIONAL = ["VIG", "IEF"]   # enlève si Yahoo ne les trouve pas
FX = ["EURUSD=X"]
STM_US = ["STM"]

def download_close(tickers, start=START, end=END):
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        return df["Close"]
    return df[["Close"]].rename(columns={"Close": tickers[0]})

def build_prices():
    base = list(set(STOCKS_PA + CORE_ETF + OPTIONAL))
    prices_base = download_close(base, START, END)

    stm_usd = download_close(STM_US, START, END)["STM"]
    eurusd = download_close(FX, START, END)["EURUSD=X"]
    stm_eur = (stm_usd / eurusd).rename("STM_EUR")

    prices = pd.concat([prices_base, stm_eur], axis=1).sort_index()
    return prices.dropna(how="all")

prices = build_prices()
rets = prices.pct_change(fill_method=None)

min_obs = int(ANN * 2)
valid_cols = [c for c in rets.columns if rets[c].dropna().shape[0] >= min_obs]
rets = rets[valid_cols].dropna()
UNIVERSE = valid_cols

# -------------------------
# Helpers: stats
# -------------------------
def ann_return(x): return float(x.mean() * ANN)
def ann_vol(x): return float(x.std() * np.sqrt(ANN))
def max_dd(x):
    eq = (1 + x).cumprod()
    peak = eq.cummax()
    return float((eq/peak - 1).min())

def cvar95(x, alpha=0.05):
    q = np.quantile(x, alpha)
    return float(x[x <= q].mean())

def summarize_daily_returns(x):
    return {
        "CAGR(approx)": ann_return(x),
        "Vol": ann_vol(x),
        "MaxDD": max_dd(x),
        "CVaR95(daily)": cvar95(x),
    }

# -------------------------
# 2) Estimators (robust)
# -------------------------
def estimate_mu_cov_naive(train):
    mu = train.mean().values * ANN
    cov = train.cov().values * ANN
    return mu, cov

def estimate_mu_shrink(train, prior="zero", delta=0.7):
    """
    Shrink mean returns toward a stable prior.
    delta in [0,1]: higher => more shrink to prior.
    """
    mu_hat = train.mean().values * ANN
    if prior == "zero":
        mu0 = np.zeros_like(mu_hat)
    elif prior == "market":
        # simple market prior: use URTH mean if present
        if "URTH" in train.columns:
            m = float(train["URTH"].mean() * ANN)
        else:
            m = float(np.mean(mu_hat))
        mu0 = np.repeat(m, len(mu_hat))
    else:
        raise ValueError("prior must be 'zero' or 'market'")
    mu = (1 - delta) * mu_hat + delta * mu0
    return mu

def estimate_cov_ledoitwolf(train):
    """
    Ledoit-Wolf shrinkage covariance estimator.
    """
    lw = LedoitWolf().fit(train.values)
    cov = lw.covariance_ * ANN
    return cov

# -------------------------
# 3) Black–Litterman (simple & practical)
# -------------------------
def black_litterman_mu_cov(train, tau=0.05, delta=2.5, P=None, Q=None, omega=None):
    """
    BL with:
    - prior returns pi = delta * Sigma * w_mkt (use equal-weight as proxy for market)
    - views: P w = Q
    """
    # Use robust cov by Ledoit-Wolf
    Sigma = estimate_cov_ledoitwolf(train)

    n = train.shape[1]
    w_mkt = np.repeat(1/n, n)  # proxy market weights
    pi = delta * (Sigma @ w_mkt)

    # No views => return prior only
    if P is None or Q is None:
        return pi, Sigma

    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)

    if omega is None:
        # default: proportional to uncertainty in each view
        omega = np.diag(np.diag(P @ (tau * Sigma) @ P.T))

    # Posterior mean:
    # mu_bl = inv(inv(tauΣ) + P'Ω^-1 P) * (inv(tauΣ)π + P'Ω^-1 Q)
    inv = np.linalg.inv
    A = inv(tau * Sigma)
    B = P.T @ inv(omega) @ P
    mu_bl = inv(A + B) @ (A @ pi + P.T @ inv(omega) @ Q)

    return mu_bl, Sigma

# -------------------------
# 4) Optimizer: Max Sharpe with constraints
# -------------------------
def max_sharpe_opt(mu, cov, w_max=0.25, w_min=None, rf=0.0):
    n = len(mu)

    def port_ret(w): return float(w @ mu)
    def port_vol(w): return float(np.sqrt(w @ cov @ w))

    def obj(w):
        v = port_vol(w)
        r = port_ret(w) - rf
        return -(r / v) if v > 1e-12 else 1e6

    bounds = []
    for i in range(n):
        lo = 0.0 if w_min is None else float(w_min[i])
        bounds.append((lo, w_max))

    cons = [{"type":"eq", "fun": lambda w: np.sum(w) - 1.0}]
    w0 = np.repeat(1/n, n)
    w0 = np.minimum(w0, w_max)
    if w_min is not None:
        w0 = np.maximum(w0, w_min)
    w0 = w0 / w0.sum()

    res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 3000})
    if not res.success:
        # fallback
        w = w0
        return w
    return res.x

# -------------------------
# 5) Dynamic risk constraint: target volatility
# -------------------------
def scale_to_target_vol(w, cov, target_vol=0.12):
    """
    If you have cash asset: you can scale risky weights and allocate rest to cash.
    Here we mimic it by scaling weights and renormalizing (approx),
    or you can add a CASH column (recommended).
    """
    vol = float(np.sqrt(w @ cov @ w))
    if vol <= 1e-12:
        return w
    # scale risky exposure; remaining to "cash" would be (1-scale)
    scale = min(1.0, target_vol / vol)
    w_scaled = w * scale
    # renormalize back to 1 (no cash) -> keeps composition but doesn't truly reduce risk
    # Better: add CASH asset with 0 vol. See note below.
    w_scaled = w_scaled / w_scaled.sum()
    return w_scaled

# -------------------------
# 6) Walk-forward runner
# -------------------------
def walk_forward_backtest(rets, optimizer_fn, train_years=5, rebalance="QE", tc_bps=10):
    rets = rets.dropna().copy()
    idx = rets.index

    rebal_dates = rets.resample(rebalance).last().index.intersection(idx)

    w_prev = None
    port = []

    for d in rebal_dates:
        train_end = d
        train_start = train_end - pd.DateOffset(years=train_years)
        train = rets[(rets.index > train_start) & (rets.index <= train_end)]
        if train.shape[0] < ANN:
            continue

        w = optimizer_fn(train)

        if w_prev is None:
            cost = 0.0
        else:
            turnover = np.sum(np.abs(w - w_prev))
            cost = (tc_bps / 10000.0) * turnover

        next_dates = idx[idx > d]
        if len(next_dates) == 0:
            break

        start_apply = next_dates[0]
        future_rebal = rebal_dates[rebal_dates > d]
        end_apply = future_rebal[0] if len(future_rebal) else idx[-1]

        period = rets[(rets.index >= start_apply) & (rets.index <= end_apply)]
        if period.empty:
            continue

        pr = pd.Series(period.values @ w, index=period.index)
        pr.iloc[0] -= cost

        port.append(pr)
        w_prev = w

    port_rets = pd.concat(port).sort_index()
    port_rets = port_rets[~port_rets.index.duplicated(keep="first")]
    return port_rets

# -------------------------
# 7) Define 4 strategies to compare OOS
# -------------------------

# A) Naive Max Sharpe
def strat_naive(train):
    mu, cov = estimate_mu_cov_naive(train)
    return max_sharpe_opt(mu, cov, w_max=0.25)

# B) Robust: shrink mu + LedoitWolf cov
def strat_shrink(train, delta=0.7, prior="market"):
    mu = estimate_mu_shrink(train, prior=prior, delta=delta)
    cov = estimate_cov_ledoitwolf(train)
    return max_sharpe_opt(mu, cov, w_max=0.25)

# C) Black–Litterman (optional views)
def strat_black_litterman(train):
    # Example views:
    # "STM_EUR expected to outperform URTH by 2% annual"
    cols = list(train.columns)
    n = len(cols)

    if "STM_EUR" in cols and "URTH" in cols:
        P = np.zeros((1, n))
        P[0, cols.index("STM_EUR")] = 1.0
        P[0, cols.index("URTH")] = -1.0
        Q = np.array([0.02])  # +2% relative view
        mu_bl, cov = black_litterman_mu_cov(train, P=P, Q=Q)
    else:
        mu_bl, cov = black_litterman_mu_cov(train)

    return max_sharpe_opt(mu_bl, cov, w_max=0.25)

# D) Dynamic constraints: robust estimates + target vol (approx)
def strat_dynamic_risk(train, target_vol=0.12):
    mu = estimate_mu_shrink(train, prior="market", delta=0.7)
    cov = estimate_cov_ledoitwolf(train)
    w = max_sharpe_opt(mu, cov, w_max=0.25)
    # NOTE: best practice is add CASH column; this is an approximation
    w = scale_to_target_vol(w, cov, target_vol=target_vol)
    return w

# -------------------------
# 8) Run OOS comparisons
# -------------------------
print("Universe:", UNIVERSE)

oos_naive = walk_forward_backtest(rets, strat_naive, train_years=5, rebalance="QE", tc_bps=10)
print("\nOOS Naive MaxSharpe:", summarize_daily_returns(oos_naive))

oos_shrink = walk_forward_backtest(rets, lambda tr: strat_shrink(tr, delta=0.7, prior="market"),
                                   train_years=5, rebalance="QE", tc_bps=10)
print("\nOOS Shrink(mu)+LW(cov):", summarize_daily_returns(oos_shrink))

oos_bl = walk_forward_backtest(rets, strat_black_litterman, train_years=5, rebalance="QE", tc_bps=10)
print("\nOOS Black-Litterman:", summarize_daily_returns(oos_bl))

oos_dyn = walk_forward_backtest(rets, lambda tr: strat_dynamic_risk(tr, target_vol=0.12),
                                train_years=5, rebalance="QE", tc_bps=10)
print("\nOOS Dynamic Risk (target vol):", summarize_daily_returns(oos_dyn))

def walk_forward_with_weights(rets, optimizer_fn, train_years=5, rebalance="QE"):
    rets = rets.dropna().copy()
    idx = rets.index

    rebal_dates = rets.resample(rebalance).last().index.intersection(idx)

    weights_list = []
    dates_list = []

    for d in rebal_dates:
        train_end = d
        train_start = train_end - pd.DateOffset(years=train_years)
        train = rets[(rets.index > train_start) & (rets.index <= train_end)]
        if train.shape[0] < ANN:
            continue

        w = optimizer_fn(train)
        weights_list.append(w)
        dates_list.append(d)

    weights_df = pd.DataFrame(weights_list, index=dates_list, columns=rets.columns)
    return weights_df

weights_shrink = walk_forward_with_weights(
    rets,
    lambda tr: strat_shrink(tr, delta=0.7, prior="market"),
    train_years=5
)

print("\n=== Poids moyens Shrink (OOS) ===")
print(weights_shrink.mean().sort_values(ascending=False))
