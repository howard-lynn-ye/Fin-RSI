"""Independent reference equations for the graded tasks. These never call fin-skills.

Each function takes the adjusted closes as a (rows x assets) array and the task parameters,
and returns the position on every row: row t is held over the return from t-1 to t and is
decided from closes through t-1. Warm-up rows are 0. The qualification tests compare every
reference with fin_skills.algorithms.positions(); a mismatch means one of the two is wrong.
"""
from __future__ import annotations

import numpy as np


def _returns(p):
    return p[1:] / p[:-1] - 1


def inverse_volatility(p, *, window):
    ret, w = _returns(p), np.zeros_like(p)
    for t in range(window + 1, len(p)):
        hist = ret[t - 1 - window:t - 1]          # the window returns ending at close t-1
        inv = 1 / hist.std(axis=0, ddof=1)
        w[t] = inv / inv.sum()
    return w


def cross_sectional_momentum(p, *, lookback, top_k, min_history=60):
    """Decisions start once min_history returns exist (the task text states this warm-up)."""
    ret, w = _returns(p), np.zeros_like(p)
    for t in range(max(lookback, min_history) + 1, len(p)):
        m = np.prod(1 + ret[t - 1 - lookback:t - 1], axis=0) - 1
        chosen = np.argsort(-m, kind="stable")[:top_k]
        w[t, chosen[m[chosen] > 0]] = 1 / top_k
    return w


def ma_crossover(p, *, fast, slow):
    n, w = p.shape[1], np.zeros_like(p)
    for t in range(slow, len(p)):
        w[t] = np.sign(p[t - fast:t].mean(axis=0) - p[t - slow:t].mean(axis=0)) / n
    return w


def vol_target_momentum(p, *, lookback, target_vol, periods_per_year=252, max_exposure=1.0):
    ret, n, w = _returns(p), p.shape[1], np.zeros_like(p)
    for t in range(lookback + 1, len(p)):
        vol = ret[t - 1 - lookback:t - 1].std(axis=0, ddof=1) * np.sqrt(periods_per_year)
        scale = np.where(vol == 0, 0.0, np.minimum(max_exposure, target_vol / np.where(
            vol == 0, 1.0, vol)))
        w[t] = np.sign(p[t - 1] / p[t - 1 - lookback] - 1) * scale / n
    return w


def momentum(p, *, lookback):
    n, w = p.shape[1], np.zeros_like(p)
    for t in range(lookback + 1, len(p)):
        w[t] = np.sign(p[t - 1] / p[t - 1 - lookback] - 1) / n
    return w


def _ema(x, span):
    """EMA started at the first value, defined from index span-1 (NaN before)."""
    alpha, out = 2 / (span + 1), np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
    out[:span - 1] = np.nan
    return out


def macd(p, *, fast, slow, signal_span):
    n, w = p.shape[1], np.zeros_like(p)
    for j in range(n):
        line = _ema(p[:, j], fast) - _ema(p[:, j], slow)       # defined from slow-1
        first = slow - 1
        base = np.full(len(p), np.nan)
        base[first:] = _ema(line[first:], signal_span)
        decision = np.sign(line - base)
        for t in range(1, len(p)):
            if np.isfinite(decision[t - 1]):
                w[t, j] = decision[t - 1] / n
    return w


def _walk(indicator, lower, upper, center):
    state, out = 0.0, np.full(len(indicator), np.nan)
    for t, value in enumerate(indicator):
        if not np.isfinite(value):
            continue
        if value < lower:
            state = 1.0
        elif value > upper:
            state = -1.0
        elif (state > 0 and value >= center) or (state < 0 and value <= center):
            state = 0.0
        out[t] = state
    return out


def _lagged(decisions, n):
    w = np.zeros_like(decisions)
    w[1:] = np.where(np.isfinite(decisions[:-1]), decisions[:-1], 0.0) / n
    return w


def bollinger_reversion(p, *, lookback, entry_z):
    n, d = p.shape[1], np.full(p.shape, np.nan)
    for j in range(n):
        z = np.full(len(p), np.nan)
        for t in range(lookback - 1, len(p)):
            x = p[t - lookback + 1:t + 1, j]
            s = x.std(ddof=1)
            z[t] = 0.0 if s == 0 else (p[t, j] - x.mean()) / s
        d[:, j] = _walk(z, -entry_z, entry_z, 0.0)
    return _lagged(d, n)


def rsi_reversion(p, *, lookback, lower, upper):
    n, d = p.shape[1], np.full(p.shape, np.nan)
    for j in range(n):
        delta = np.diff(p[:, j])                               # delta[t-1] = P_t - P_{t-1}
        rsi = np.full(len(p), np.nan)
        for t in range(lookback, len(p)):
            window = delta[t - lookback:t]
            gain, loss = np.clip(window, 0, None).mean(), np.clip(-window, 0, None).mean()
            rsi[t] = 50.0 if gain + loss == 0 else 100 * gain / (gain + loss)
        d[:, j] = _walk(rsi, lower, upper, 50.0)
    return _lagged(d, n)


def donchian_breakout(p, *, lookback):
    n, d = p.shape[1], np.full(p.shape, np.nan)
    for j in range(n):
        state = 0.0
        for t in range(lookback, len(p)):
            prior = p[t - lookback:t, j]
            if p[t, j] > prior.max():
                state = 1.0
            elif p[t, j] < prior.min():
                state = -1.0
            d[t, j] = state
    return _lagged(d, n)


def _min_variance_exact(cov):
    """Long-only, fully invested minimum variance, solved exactly: for every subset of assets
    take the equality-constrained solution and keep the feasible one with least variance."""
    from itertools import combinations
    n, best, best_var = len(cov), None, np.inf
    for k in range(1, n + 1):
        for subset in combinations(range(n), k):
            s = list(subset)
            x = np.linalg.solve(cov[np.ix_(s, s)], np.ones(k))
            w_s = x / x.sum()
            if (w_s < -1e-12).any():
                continue
            w = np.zeros(n)
            w[s] = np.clip(w_s, 0, None)
            var = w @ cov @ w
            if var < best_var - 1e-18:
                best, best_var = w, var
    return best


def min_variance(p, *, window, min_history=20):
    ret, w = _returns(p), np.zeros_like(p)
    for t in range(max(window, min_history) + 1, len(p)):
        w[t] = _min_variance_exact(np.cov(ret[t - 1 - window:t - 1], rowvar=False, ddof=1))
    return w


def _hrp(r, method):
    from scipy.cluster.hierarchy import leaves_list, linkage
    from scipy.spatial.distance import squareform
    corr, cov = np.corrcoef(r, rowvar=False), np.cov(r, rowvar=False, ddof=1)
    dist = np.sqrt(np.clip((1 - corr) / 2, 0, 1))
    order = list(leaves_list(linkage(squareform(dist, checks=False), method)))

    def cluster_var(items):
        c = cov[np.ix_(items, items)]
        ivp = 1 / np.diag(c)
        ivp /= ivp.sum()
        return ivp @ c @ ivp

    w = np.ones(r.shape[1])
    clusters = [order]
    while clusters:
        clusters = [c[a:b] for c in clusters if len(c) > 1
                    for a, b in ((0, len(c) // 2), (len(c) // 2, len(c)))]
        for i in range(0, len(clusters), 2):
            left, right = clusters[i], clusters[i + 1]
            v0, v1 = cluster_var(left), cluster_var(right)
            w[left] *= 1 - v0 / (v0 + v1)
            w[right] *= v0 / (v0 + v1)
    return w


def hrp(p, *, window, linkage, min_history=20):
    ret, w = _returns(p), np.zeros_like(p)
    for t in range(max(window, min_history) + 1, len(p)):
        w[t] = _hrp(ret[t - 1 - window:t - 1], linkage)
    return w


# Tables: one value per date and asset, known at close t-1 ----------------------------------

def _return_windows(p, window):
    """(t, the window returns ending at close t-1) for every row that has a full window."""
    ret = _returns(p)                        # ret[i] = P[i+1] / P[i] - 1
    for t in range(window + 1, len(p)):
        yield t, ret[t - 1 - window:t - 1]


def historical_volatility(p, *, window, periods_per_year):
    out = np.zeros_like(p)
    for t, r in _return_windows(p, window):
        out[t] = r.std(axis=0, ddof=1) * np.sqrt(periods_per_year)
    return out


def ewma_volatility(p, *, window, decay, periods_per_year):
    out = np.zeros_like(p)
    for t, r in _return_windows(p, window):
        v = r[0] ** 2
        for x in r[1:]:
            v = decay * v + (1 - decay) * x ** 2
        out[t] = np.sqrt(v * periods_per_year)
    return out


def historical_var_es(p, *, window, confidence, output):
    out = np.zeros_like(p)
    for t, r in _return_windows(p, window):
        for j in range(p.shape[1]):
            loss = -r[:, j]
            var = np.quantile(loss, confidence)          # linear interpolation
            out[t, j] = var if output == "var" else loss[loss >= var].mean()
    return out


def normal_var_es(p, *, window, confidence, output):
    from scipy.stats import norm
    z, out = norm.ppf(confidence), np.zeros_like(p)
    for t, r in _return_windows(p, window):
        mu, sd = (-r).mean(axis=0), r.std(axis=0, ddof=1)
        out[t] = mu + sd * z if output == "var" else mu + sd * norm.pdf(z) / (1 - confidence)
    return out


def drift(p, *, window):
    out = np.zeros_like(p)
    for t in range(window, len(p)):
        y = p[t - window:t]                               # closes through t-1
        out[t] = y[-1] + (y[-1] - y[0]) / (window - 1)
    return out


def mean(p, *, window):
    out = np.zeros_like(p)
    for t in range(window, len(p)):
        out[t] = p[t - window:t].mean(axis=0)
    return out


REFERENCES = {f.__name__: f for f in (
    inverse_volatility, cross_sectional_momentum, ma_crossover, vol_target_momentum, momentum,
    macd, bollinger_reversion, rsi_reversion, donchian_breakout, min_variance, hrp,
    historical_volatility, ewma_volatility, historical_var_es, normal_var_es, drift, mean)}
