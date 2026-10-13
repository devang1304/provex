"""Variational temporal explainer: VA-TGExplainer, made to do what v1 claimed (fixes C4).

A posterior over keep-gates for every item (neighbour edge or history event):

    zeta_k ~ q = N(mu_k, sigma_k^2)          m_k = clamp(1.2 * sigmoid(zeta_k) - 0.1, 0, 1)
    prior  N(mu0(dt_k), 1),  mu0(dt) = a - kappa * log1p(dt / 60 s),  dt_k = t_event - t_k
    likelihood  p(alert | m) = sigmoid((L(m) - tau) / T)

The temporal prior makes older events a priori less likely to be needed, and a/kappa can be
fitted on validation alerts. Training maximises the ELBO with reparameterised samples and a
closed-form Gaussian KL.

Exported per item:
- posterior inclusion probability P(m > 0.5) = Phi(mu / sigma);
- E[m] and Std[m] from Monte Carlo samples;
- mu, sigma and dt.

``provex.evaluate.calibration`` checks P(m > 0.5) against exact enumeration on small sets.
"""
from __future__ import annotations

import math
import time

import torch
import torch.nn.functional as F
from torch import Tensor

from .event_model import EventModel
from .joint_mask import Explanation


def stretched(zeta: Tensor) -> Tensor:
    return (1.2 * torch.sigmoid(zeta) - 0.1).clamp(0.0, 1.0)


def item_ages_seconds(em: EventModel, data_t: Tensor) -> Tensor:
    t_items = data_t[em.items].to(torch.float64)
    return ((em.cap.t - t_items) / 1e9).clamp_min(0.0).to(em.device)


class VariationalTemporalExplainer:
    def __init__(self, *, epochs: int = 300, lr: float = 0.05, samples: int = 8, T: float = 1.0,
                 prior_a: float = -1.0, prior_kappa: float = 0.5, mc_export: int = 256,
                 init_mu: float = 2.0, init_log_sigma: float = math.log(0.5), seed: int = 0):
        self.epochs, self.lr, self.samples, self.T = epochs, lr, samples, T
        self.a, self.kappa, self.mc_export = prior_a, prior_kappa, mc_export
        self.init_mu, self.init_log_sigma, self.seed = init_mu, init_log_sigma, seed

    def prior_mean(self, dt_s: Tensor) -> Tensor:
        return self.a - self.kappa * torch.log1p(dt_s / 60.0)

    def explain(self, em: EventModel, tau: float, data_t: Tensor) -> Explanation:
        t0 = time.perf_counter()
        torch.manual_seed(self.seed)
        n = em.num_items
        dt = item_ages_seconds(em, data_t)
        mu0 = self.prior_mean(dt).to(em.device, em.dtype)
        mu = torch.full((n,), self.init_mu, dtype=em.dtype, device=em.device, requires_grad=True)
        log_sigma = torch.full((n,), self.init_log_sigma, dtype=em.dtype, device=em.device, requires_grad=True)
        opt = torch.optim.Adam([mu, log_sigma], lr=self.lr)
        with torch.no_grad():
            L_full = float(em.loss(torch.ones(n, dtype=em.dtype, device=em.device), hard=True))
        for _ in range(self.epochs):
            opt.zero_grad()
            sigma = log_sigma.exp()
            ll = 0.0
            for _ in range(self.samples):
                zeta = mu + sigma * torch.randn_like(mu)
                L = em.loss(stretched(zeta))
                ll = ll + F.logsigmoid((L - tau) / self.T)
            ll = ll / self.samples
            kl = (torch.log(1.0 / sigma) + (sigma ** 2 + (mu - mu0) ** 2) / 2.0 - 0.5).sum()
            (-(ll - kl)).backward()
            opt.step()
        with torch.no_grad():
            sigma = log_sigma.exp()
            pi = 0.5 * (1 + torch.erf(mu / (sigma * math.sqrt(2.0))))
            draws = stretched(mu + sigma * torch.randn(self.mc_export, n, dtype=em.dtype, device=em.device))
            mean_m, std_m = draws.mean(0), draws.std(0)
            keep = pi > 0.5
            L_sel = float(em.loss(keep.to(em.dtype), hard=True))
        return Explanation(
            method="variational", mode="preserve", items=em.items.tolist(), scores=pi.tolist(),
            selected=em.items[keep.cpu()].tolist(), valid=L_sel >= tau, repaired=0, L_full=L_full,
            L_selected=L_sel, tau=tau, runtime_s=time.perf_counter() - t0,
            extra={"mean_m": mean_m.tolist(), "std_m": std_m.tolist(), "mu": mu.tolist(),
                   "sigma": sigma.tolist(), "dt_s": dt.tolist(), "prior_mu": mu0.tolist(),
                   "prior_a": self.a, "prior_kappa": self.kappa})