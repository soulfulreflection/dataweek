"""ML proxy of the reservoir simulator: 4 uncertainty parameters -> 10-year production.

Architecture ("PCA-GP"):
  1. Output curves (40 quarterly oil rates + 40 water rates) are standardised and
     compressed with PCA. 12 components keep 99.98 % of the variance; PCA here acts
     as a linear autoencoder of the production response.
  2. One Gaussian-process regressor per principal component (Matern 5/2, ARD length
     scales, white-noise term) maps the parameters to the component score.
  3. Prediction = GP means pushed back through the PCA decoder. The GP standard
     deviations give a per-time-step proxy uncertainty used by the history match.

Gas is not modelled separately: the produced GOR is constant (0.3632 MSCF/STB in all
100 cases, oil is undersaturated), so gas = GOR x oil.
"""
from __future__ import annotations

import warnings

import joblib
import numpy as np
from scipy.interpolate import RBFInterpolator
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

from . import config as C

N_T = 40  # quarterly history steps


def to_unit(theta: np.ndarray) -> np.ndarray:
    """Physical parameters -> model features in [0, 1] (permeability on a log scale)."""
    theta = np.atleast_2d(theta).astype(float)
    out = np.empty_like(theta)
    for j, name in enumerate(C.PARAM_SHORT):
        lo, hi = C.PARAM_BOUNDS[name]
        if name == "perm_mult":
            out[:, j] = (np.log(theta[:, j]) - np.log(lo)) / (np.log(hi) - np.log(lo))
        else:
            out[:, j] = (theta[:, j] - lo) / (hi - lo)
    return out


def stack_targets(oil: np.ndarray, water: np.ndarray) -> np.ndarray:
    return np.hstack([oil, water])


class PCAGPProxy:
    def __init__(self, n_components=C.PROXY_N_COMPONENTS, kernel=C.PROXY_KERNEL,
                 restarts=C.PROXY_RESTARTS, seed=C.SEED):
        self.n_components = n_components
        self.kernel = kernel
        self.restarts = restarts
        self.seed = seed

    def _kernel(self):
        base = (Matern([1.0] * 4, (1e-2, 1e2), nu=2.5) if self.kernel == "matern"
                else RBF([1.0] * 4, (1e-2, 1e2)))
        return ConstantKernel(1.0) * base + WhiteKernel(1e-4, (1e-8, 1e-1))

    def fit(self, theta: np.ndarray, oil: np.ndarray, water: np.ndarray):
        X = to_unit(theta)
        Y = stack_targets(oil, water)
        self.scaler_ = StandardScaler().fit(Y)
        Z = self.scaler_.transform(Y)
        self.pca_ = PCA(self.n_components, random_state=self.seed).fit(Z)
        S = self.pca_.transform(Z)
        self.gps_ = []
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for k in range(self.n_components):
                gp = GaussianProcessRegressor(self._kernel(), normalize_y=True,
                                              n_restarts_optimizer=self.restarts,
                                              random_state=self.seed + k)
                self.gps_.append(gp.fit(X, S[:, k]))
        # variance the PCA truncation cannot represent (per output, in original units)
        resid = Y - self.scaler_.inverse_transform(self.pca_.inverse_transform(S))
        self.trunc_std_ = resid.std(0)
        return self

    def predict(self, theta: np.ndarray, return_std: bool = False):
        X = to_unit(theta)
        if return_std:
            pairs = [gp.predict(X, return_std=True) for gp in self.gps_]
            S = np.column_stack([p[0] for p in pairs])
            Ssd = np.column_stack([p[1] for p in pairs])
        else:
            S = np.column_stack([gp.predict(X) for gp in self.gps_])
        Y = self.scaler_.inverse_transform(self.pca_.inverse_transform(S))
        oil, water = np.clip(Y[:, :N_T], 0, None), np.clip(Y[:, N_T:], 0, None)
        if not return_std:
            return oil, water
        load = self.pca_.components_ * self.scaler_.scale_          # (k, 80)
        var = (Ssd ** 2) @ (load ** 2) + self.trunc_std_ ** 2
        sd = np.sqrt(var)
        return oil, water, sd[:, :N_T], sd[:, N_T:]

    def explained_variance(self) -> float:
        return float(self.pca_.explained_variance_ratio_.sum())

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path) -> "PCAGPProxy":
        return joblib.load(path)


# --------------------------------------------------------------------------
# Benchmark models (used only to justify the choice of PCA-GP)
# --------------------------------------------------------------------------
class _ScaledRegressor:
    def __init__(self, make):
        self.make = make

    def fit(self, theta, oil, water):
        Y = stack_targets(oil, water)
        self.sc_ = StandardScaler().fit(Y)
        self.m_ = self.make(to_unit(theta), self.sc_.transform(Y))
        return self

    def predict(self, theta):
        Y = self.sc_.inverse_transform(self.m_(to_unit(theta)))
        return np.clip(Y[:, :N_T], 0, None), np.clip(Y[:, N_T:], 0, None)


def benchmark_models(seed=C.SEED):
    def rbf(X, Y):
        r = RBFInterpolator(X, Y, kernel="thin_plate_spline", smoothing=1e-3)
        return r

    def poly(deg):
        def make(X, Y):
            m = make_pipeline(PolynomialFeatures(deg), RidgeCV(np.logspace(-6, 2, 30))).fit(X, Y)
            return m.predict
        return make

    def mlp(X, Y):
        nets = [MLPRegressor(hidden_layer_sizes=(64, 64), activation="tanh", alpha=1e-4,
                             max_iter=20000, learning_rate_init=3e-3, tol=1e-7,
                             random_state=seed + s).fit(X, Y) for s in range(5)]
        return lambda Xn: np.mean([n.predict(Xn) for n in nets], axis=0)

    return {
        "PCA-GP (Matern, 12 PCs)": lambda: PCAGPProxy(),
        "PCA-GP (RBF, 8 PCs)": lambda: PCAGPProxy(n_components=8, kernel="rbf", restarts=2),
        "RBF interpolation (thin-plate)": lambda: _ScaledRegressor(rbf),
        "Polynomial ridge (deg 2)": lambda: _ScaledRegressor(poly(2)),
        "Polynomial ridge (deg 3)": lambda: _ScaledRegressor(poly(3)),
        "MLP ensemble (5x 64-64 tanh)": lambda: _ScaledRegressor(mlp),
    }


def score(oil_true, water_true, oil_pred, water_pred, days) -> dict:
    """Error metrics in engineering units."""
    nrmse_oil = np.sqrt(np.mean((oil_pred - oil_true) ** 2, 1)) / 15500.0
    nrmse_wat = np.sqrt(np.mean((water_pred - water_true) ** 2, 1)) / 8000.0
    Yt = stack_targets(oil_true, water_true)
    Yp = stack_targets(oil_pred, water_pred)
    r2 = 1 - np.sum((Yp - Yt) ** 2) / np.sum((Yt - Yt.mean(0)) ** 2)
    cum_o = (oil_pred @ days - oil_true @ days) / (oil_true @ days)
    cum_w = (water_pred @ days - water_true @ days) / (water_true @ days)
    return {
        "R2": float(r2),
        "NRMSE_oil_%": float(100 * nrmse_oil.mean()),
        "NRMSE_water_%": float(100 * nrmse_wat.mean()),
        "MAPE_cum_oil_2008_%": float(100 * np.abs(cum_o).mean()),
        "MAPE_cum_water_2008_%": float(100 * np.abs(cum_w).mean()),
        "max_abs_err_cum_oil_%": float(100 * np.abs(cum_o).max()),
    }
