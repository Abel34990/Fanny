# pip install yfinance pandas numpy scipy

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.optimize import minimize

ANN = 252
START = "2015-01-01"
END = None

# -------------------------
# 1) UNIVERSE (comme ton dernier script)
# -------------------------
STOCKS_PA = ["KER.PA", "AF.PA", "ALCRB.PA", "AELIS.PA", "MEDCL.PA", "EMEIS.PA"]
CORE_ETF = ["URTH"]
OPTIONAL = ["VIG", "IEF"]      # enlève si Yahoo ne les trouve pas
FX = ["EURUSD=X"]
STM_US = ["STM"]

# -------------------------
# 2) DOWNLOAD + BUILD STM_EUR
# -------------------------
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
    prices = prices.dropna(how="all")
    return prices

prices = build_prices()
rets = prices.pct_change(fill_method=None)

# garder colonnes avec assez d'historique
min_obs = int(ANN * 2)
valid_cols = [c for c in rets.columns if rets[c].dropna().shape[0] >= min_obs]
rets = rets[valid_cols].dropna()
UNIVERSE = valid_cols

print("Universe used:", UNIVERSE)

# -------------------------
# 3) PORTFOLIO METRICS
# -------------------------
def ann_return(x):  # x daily returns
    return float(x.mean() * ANN)

def ann_vol(x):
    return float(x.std() * np.sqrt(ANN))

def max_dd(x):
    eq = (1 + x).cumprod()
    peak = eq.cummax()
    dd = (eq / peak) - 1
    return float(dd.min())

def cvar95(x, alpha=0.05):
    q = np.quantile(x, alpha)
    return float(x[x <= q].mean())

def summarize_daily_returns(x):
    # returns a dict of annualized stats
    return {
        "CAGR(approx)": ann_return(x),   # approx, not geometric
        "Vol": ann_vol(x),
        "MaxDD": max_dd(x),
        "CVaR95(daily)": cvar95(x),
    }

# -------------------------
# 4) OPTIMIZERS (example: Max Sharpe)
#     Tu peux remplacer par tes optimizers/constraints si tu veux.
# -------------------------
def max_sharpe_weights(train_rets, w_max=0.25, rf=0.0):
    mu = train_rets.mean() * ANN
    cov = train_rets.cov() * ANN
    n = train_rets.shape[1]

    def port_ret(w): return float(w @ mu.values)
    def port_vol(w): return float(np.sqrt(w @ cov.values @ w))

    def obj(w):
        v = port_vol(w)
        r = port_ret(w) - rf
        return -(r / v) if v > 1e-12 else 1e6

    bounds = [(0.0, w_max)] * n
    cons = [{"type":"eq", "fun": lambda w: np.sum(w) - 1.0}]
    w0 = np.repeat(1/n, n)
    res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons)
    if not res.success:
        # fallback = equal weight projected into bounds
        w = np.minimum(w0, w_max)
        w = w / w.sum()
        return w
    return res.x

# -------------------------
# 5) TEST 1 — WALK-FORWARD (OUT-OF-SAMPLE)
# -------------------------
def walk_forward_backtest(
    rets,
    optimizer_fn,
    train_years=5,
    rebalance="Q",     # "M", "Q", "A"
    transaction_cost_bps=5
):
    """
    Walk-forward robuste et propre
    """

    rets = rets.dropna().copy()
    idx = rets.index

    # Génération propre des dates de rebalancing
    rebal_dates = (
        rets.resample(rebalance)
            .last()
            .index
    )

    # garder uniquement dates présentes
    rebal_dates = rebal_dates.intersection(idx)

    w_prev = None
    port = []

    for d in rebal_dates:

        train_end = d
        train_start = train_end - pd.DateOffset(years=train_years)

        train = rets[(rets.index > train_start) & (rets.index <= train_end)]

        if train.shape[0] < ANN:
            continue

        # optimisation sur période train
        w = optimizer_fn(train)

        # calcul turnover
        if w_prev is None:
            cost = 0.0
        else:
            turnover = np.sum(np.abs(w - w_prev))
            cost = (transaction_cost_bps / 10000.0) * turnover

        # période d'application
        next_dates = idx[idx > d]
        if len(next_dates) == 0:
            break

        start_apply = next_dates[0]

        future_rebal = rebal_dates[rebal_dates > d]
        end_apply = future_rebal[0] if len(future_rebal) else idx[-1]

        period = rets[(rets.index >= start_apply) & (rets.index <= end_apply)]

        if period.empty:
            continue

        pr = period.values @ w
        pr = pd.Series(pr, index=period.index)

        # appliquer coût au premier jour
        pr.iloc[0] -= cost

        port.append(pr)
        w_prev = w

    if len(port) == 0:
        raise RuntimeError("Aucune période walk-forward produite.")

    port_rets = pd.concat(port).sort_index()
    port_rets = port_rets[~port_rets.index.duplicated(keep="first")]

    return port_rets

# -------------------------
# 6) TEST 2 — BOOTSTRAP MONTE CARLO (10Y DISTRIBUTION)
# -------------------------
def bootstrap_monte_carlo(
    rets,
    w,
    horizon_years=10,
    n_sims=5000,
    block_size=20,     # block bootstrap to preserve autocorrelation a bit
    seed=42
):
    """
    Block bootstrap:
    - sample contiguous blocks of daily returns
    - build a 10-year path
    - compute CAGR and max drawdown distribution
    """
    rng = np.random.default_rng(seed)
    rets = rets.dropna()
    X = rets.values @ w  # daily portfolio returns
    T = horizon_years * ANN
    N = len(X)

    def one_path():
        path = []
        while len(path) < T:
            start = rng.integers(0, max(1, N - block_size))
            block = X[start:start+block_size]
            path.extend(block.tolist())
        path = np.array(path[:T])
        eq = np.cumprod(1 + path)
        cagr = eq[-1]**(1/horizon_years) - 1
        peak = np.maximum.accumulate(eq)
        dd = np.min(eq/peak - 1)
        return cagr, dd

    cagr_list, dd_list = [], []
    for _ in range(n_sims):
        c, d = one_path()
        cagr_list.append(c)
        dd_list.append(d)

    cagr_arr = np.array(cagr_list)
    dd_arr = np.array(dd_list)

    summary = {
        "CAGR_median": np.median(cagr_arr),
        "CAGR_p05": np.quantile(cagr_arr, 0.05),
        "CAGR_p95": np.quantile(cagr_arr, 0.95),
        "MaxDD_median": np.median(dd_arr),
        "MaxDD_p05(worst)": np.quantile(dd_arr, 0.05),
        "MaxDD_p95(best)": np.quantile(dd_arr, 0.95),
    }
    return summary, cagr_arr, dd_arr

# -------------------------
# 7) TEST 3 — SENSITIVITY (NOISE ON mu & cov)
# -------------------------
def perturbation_stability_test(
    rets,
    optimizer_template_fn,   # function(mu, cov)->weights
    n_sims=1000,
    mu_noise=0.20,          # 20% relative noise on mu
    cov_noise=0.10,         # 10% noise on cov
    seed=123
):
    """
    Add noise to mu and cov, recompute weights -> see stability.
    """
    rng = np.random.default_rng(seed)
    mu0 = rets.mean() * ANN
    cov0 = rets.cov() * ANN

    weights = []
    for _ in range(n_sims):
        mu_sim = mu0 * (1 + rng.normal(0, mu_noise, size=len(mu0)))
        cov_sim = cov0.values.copy()
        # multiplicative noise on covariance matrix (kept symmetric)
        eps = rng.normal(0, cov_noise, size=cov_sim.shape)
        cov_sim = cov_sim * (1 + eps)
        cov_sim = (cov_sim + cov_sim.T) / 2
        # fix to positive semi-definite
        eigvals, eigvecs = np.linalg.eigh(cov_sim)
        eigvals = np.clip(eigvals, 1e-8, None)
        cov_sim = eigvecs @ np.diag(eigvals) @ eigvecs.T

        w = optimizer_template_fn(mu_sim, cov_sim)
        weights.append(w)

    W = np.vstack(weights)
    stats = pd.DataFrame({
        "asset": UNIVERSE,
        "w_mean": W.mean(axis=0),
        "w_std": W.std(axis=0),
        "w_p05": np.quantile(W, 0.05, axis=0),
        "w_p95": np.quantile(W, 0.95, axis=0),
    }).sort_values("w_mean", ascending=False)

    return stats

def max_sharpe_template(mu_vec, cov_mat, w_max=0.25, rf=0.0):
    n = len(mu_vec)
    def port_ret(w): return float(w @ mu_vec)
    def port_vol(w): return float(np.sqrt(w @ cov_mat @ w))
    def obj(w):
        v = port_vol(w)
        r = port_ret(w) - rf
        return -(r / v) if v > 1e-12 else 1e6

    bounds = [(0.0, w_max)] * n
    cons = [{"type":"eq", "fun": lambda w: np.sum(w) - 1.0}]
    w0 = np.repeat(1/n, n)
    res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons)
    if not res.success:
        w = np.minimum(w0, w_max)
        w = w / w.sum()
        return w
    return res.x

# -------------------------
# 8) HOW TO USE (EXAMPLES)
# -------------------------

# Example A: Compute an in-sample Max Sharpe weight (for reference only)
w_insample = max_sharpe_weights(rets, w_max=0.25)
w_table = pd.DataFrame({"asset": UNIVERSE, "weight": w_insample}).sort_values("weight", ascending=False)
print("\nIn-sample MaxSharpe weights:")
print(w_table[w_table["weight"] > 1e-4].to_string(index=False))

# Example B: Walk-forward out-of-sample performance
oos = walk_forward_backtest(
    rets,
    optimizer_fn=lambda train: max_sharpe_weights(train, w_max=0.25),
    train_years=5,
    rebalance="Q",
    transaction_cost_bps=10  # 0.10% per turnover
)
print("\nWalk-forward OOS stats (quarterly rebal, 10bps cost):")
print(summarize_daily_returns(oos))

# Example C: Bootstrap Monte Carlo distribution for a FIXED allocation w_insample
mc_summary, mc_cagr, mc_dd = bootstrap_monte_carlo(
    rets=rets,
    w=w_insample,
    horizon_years=10,
    n_sims=3000,
    block_size=20
)
print("\nBootstrap Monte Carlo (10y) summary for fixed weights:")
for k,v in mc_summary.items():
    print(f"{k}: {v:.2%}")

# Example D: Weight stability under perturbations of mu/cov
stab = perturbation_stability_test(
    rets=rets,
    optimizer_template_fn=lambda mu_sim, cov_sim: max_sharpe_template(mu_sim, cov_sim, w_max=0.25),
    n_sims=500,
    mu_noise=0.25,
    cov_noise=0.10
)
print("\nStability test (mean/std/p05/p95 of weights under noisy mu/cov):")
print(stab.to_string(index=False))
