"""Proxy-based history matching.

Workflow
  1. Global screening: 2^17 Sobol samples of the prior box -> proxy -> misfit.
  2. Local refinement: L-BFGS-B from the 20 best samples -> MAP estimate.
  3. Error-model calibration: the noise level is scaled so that the MAP has a reduced
     chi-square of 1. This absorbs model error (no parameter set reproduces the late
     water-cut trend exactly) and keeps the posterior from collapsing to a point.
  4. Posterior sampling: vectorised random-walk Metropolis (32 chains).
  5. Selection: the 10 lowest-misfit posterior samples that are at least
     HM_MIN_DISTANCE apart in normalised parameter space (non-uniqueness made visible).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import qmc

from . import config as C
from .proxy import to_unit

LO = np.array([C.PARAM_BOUNDS[p][0] for p in C.PARAM_SHORT])
HI = np.array([C.PARAM_BOUNDS[p][1] for p in C.PARAM_SHORT])


def from_cube(u: np.ndarray) -> np.ndarray:
    """Uniform prior: unit cube -> physical parameters (all linear, like the 100 cases)."""
    return LO + np.atleast_2d(u) * (HI - LO)


def to_cube(theta: np.ndarray) -> np.ndarray:
    return (np.atleast_2d(theta) - LO) / (HI - LO)


@dataclass
class Misfit:
    oil_obs: np.ndarray
    water_obs: np.ndarray
    scale: float = 1.0          # error-model inflation (calibrated)

    def __post_init__(self):
        self.sig_o = np.maximum(C.HM_OIL_REL_SIGMA * self.oil_obs, C.HM_ABS_SIGMA_FLOOR)
        self.sig_w = np.maximum(C.HM_WATER_REL_SIGMA * self.water_obs, C.HM_ABS_SIGMA_FLOOR)
        self.n_data = 2 * len(self.oil_obs)

    def chi2(self, oil, water, sd_o=None, sd_w=None):
        so2 = self.sig_o ** 2 + (0 if sd_o is None else sd_o ** 2)
        sw2 = self.sig_w ** 2 + (0 if sd_w is None else sd_w ** 2)
        return (np.sum((oil - self.oil_obs) ** 2 / so2, axis=-1)
                + np.sum((water - self.water_obs) ** 2 / sw2, axis=-1)) / self.scale ** 2

    def nrmse(self, oil, water):
        """Interpretable misfit: RMSE normalised by the max observed rate (per phase)."""
        return (np.sqrt(np.mean((oil - self.oil_obs) ** 2, -1)) / self.oil_obs.max(),
                np.sqrt(np.mean((water - self.water_obs) ** 2, -1)) / self.water_obs.max())


def _proxy_chi2(proxy, misfit, theta, batch=20000):
    out = np.empty(len(theta))
    for s in range(0, len(theta), batch):
        o, w, so, sw = proxy.predict(theta[s:s + batch], return_std=True)
        out[s:s + batch] = misfit.chi2(o, w, so, sw)
    return out


def global_search(proxy, misfit, n=C.HM_N_SAMPLES, seed=C.SEED):
    u = qmc.Sobol(4, scramble=True, seed=seed).random(n)
    theta = from_cube(u)
    return theta, _proxy_chi2(proxy, misfit, theta)


def refine(proxy, misfit, starts):
    best = []
    f = lambda u: float(_proxy_chi2(proxy, misfit, from_cube(u))[0])
    for th in starts:
        r = minimize(f, to_cube(th)[0], method="L-BFGS-B", bounds=[(0, 1)] * 4)
        best.append((r.fun, from_cube(r.x)[0]))
    best.sort(key=lambda t: t[0])
    return best[0][1], best[0][0]


def mcmc(proxy, misfit, starts, n_steps=1500, step=0.03, seed=C.SEED):
    """Vectorised random-walk Metropolis in the unit cube (uniform prior)."""
    rng = np.random.default_rng(seed)
    u = to_cube(starts).copy()
    lp = -0.5 * _proxy_chi2(proxy, misfit, from_cube(u))
    chain, chi = [], []
    acc = 0
    for it in range(n_steps):
        prop = u + rng.normal(0, step, u.shape)
        inside = np.all((prop >= 0) & (prop <= 1), axis=1)
        lp_prop = np.full(len(u), -np.inf)
        if inside.any():
            lp_prop[inside] = -0.5 * _proxy_chi2(proxy, misfit, from_cube(prop[inside]))
        accept = np.log(rng.random(len(u))) < lp_prop - lp
        u[accept], lp[accept] = prop[accept], lp_prop[accept]
        acc += accept.mean()
        if it >= n_steps // 3:                       # burn-in
            chain.append(u.copy())
            chi.append(-2 * lp.copy())
    return from_cube(np.vstack(chain)), np.concatenate(chi), acc / n_steps


def select_diverse(theta, chi2, k=C.HM_N_SELECT, min_dist=C.HM_MIN_DISTANCE):
    order = np.argsort(chi2)
    U = to_unit(theta)
    chosen = []
    for i in order:
        if all(np.linalg.norm(U[i] - U[j]) >= min_dist for j in chosen):
            chosen.append(i)
        if len(chosen) == k:
            break
    return np.array(chosen)


def run_history_match(proxy, oil_obs, water_obs, n_samples=C.HM_N_SAMPLES, mcmc_steps=1500,
                      seed=C.SEED, verbose=True):
    misfit = Misfit(oil_obs, water_obs)
    theta, chi = global_search(proxy, misfit, n_samples, seed)
    top = theta[np.argsort(chi)[:20]]
    th_map, chi_map = refine(proxy, misfit, top)
    # calibrate error model so that reduced chi2 of MAP == 1 (never shrink below nominal)
    misfit.scale = max(1.0, np.sqrt(chi_map / misfit.n_data))
    if verbose:
        print(f"  MAP chi2={chi_map:.1f} for {misfit.n_data} data -> sigma scale {misfit.scale:.2f}")
    starts = np.vstack([th_map, theta[np.argsort(chi)[:31]]])
    post, post_chi, acc = mcmc(proxy, misfit, starts, mcmc_steps, seed=seed)
    if verbose:
        print(f"  MCMC: {len(post)} samples, acceptance {acc:.2f}")
    # make sure MAP is a candidate
    post = np.vstack([th_map, post])
    post_chi = np.concatenate([[chi_map / misfit.scale ** 2], post_chi])
    idx = select_diverse(post, post_chi)
    return {
        "misfit": misfit, "map": th_map, "screen_theta": theta, "screen_chi2": chi / misfit.scale ** 2,
        "posterior": post, "posterior_chi2": post_chi, "selected": post[idx],
        "selected_chi2": post_chi[idx], "acceptance": acc,
    }


def summarise_posterior(post: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(post, columns=C.PARAM_SHORT)
    return df.describe(percentiles=[0.1, 0.5, 0.9]).T[["mean", "std", "10%", "50%", "90%", "min", "max"]]
