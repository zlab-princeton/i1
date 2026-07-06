from __future__ import annotations

import re
from typing import Iterable

import torch


def steps(prefix, config, data_size=None, batch_size=None, total_steps=None, default=ValueError):
    suffixes = {"steps", "examples", "epochs", "percent"}
    matches = {f"{prefix}_{s}" for s in suffixes if f"{prefix}_{s}" in config}
    assert len(matches) <= 1, f"Only one of '{matches}' should be defined."

    if f"{prefix}_steps" in config:
        return config[f"{prefix}_steps"]
    if batch_size and f"{prefix}_examples" in config:
        return max(round(config[f"{prefix}_examples"] / batch_size), 1)
    if batch_size and data_size and f"{prefix}_epochs" in config:
        steps_per_epoch = data_size / batch_size
        return max(round(config[f"{prefix}_epochs"] * steps_per_epoch), 1)
    if total_steps and f"{prefix}_percent" in config:
        pct = config[f"{prefix}_percent"]
        assert 0.0 <= pct <= 1.0, f"Percents should lie in [0.0, 1.0], but {prefix}_percent is {pct}"
        return max(round(pct * total_steps), 1)
    if default is ValueError:
        raise ValueError(f"Cannot convert {prefix} to steps.")
    return default


def _is_frozen(name: str, freeze_patterns: Iterable[str]) -> bool:
    return any(re.fullmatch(p, name) for p in freeze_patterns)


def _is_dtensor(x) -> bool:
    try:
        from torch.distributed.tensor import DTensor
    except Exception:
        return False
    return isinstance(x, DTensor)


@torch.no_grad()
def global_l2_norm(tensors) -> float:
    from collections import defaultdict

    groups = defaultdict(list)
    for t in tensors:
        key = (id(t.device_mesh), tuple(t.placements)) if _is_dtensor(t) else None
        groups[key].append(t)
    total = None
    for key, ts in groups.items():
        sq = torch.stack([t.detach().float().pow(2).sum() for t in ts])
        if key is not None:
            sq = sq.full_tensor()
        s = sq.sum()
        total = s if total is None else total + s
    return total.sqrt().item() if total is not None else 0.0


class Adam:
    def __init__(
        self,
        named_params,
        lr: float,
        b1: float = 0.9,
        b2: float = 0.95,
        eps: float = 1e-8,
        grad_clip_norm: float | None = 1.0,
        freeze_patterns: Iterable[str] = ("pos_embed",),
        mu_dtype: torch.dtype = torch.bfloat16,
        foreach: bool = True,
    ) -> None:
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.grad_clip_norm = grad_clip_norm
        self.mu_dtype = mu_dtype
        self.foreach = foreach
        self.count = 0
        self.params = []
        self.mu = {}
        self.nu = {}
        for name, p in named_params:
            if _is_frozen(name, freeze_patterns) or not p.requires_grad:
                continue
            self.params.append((name, p))
            self.mu[name] = torch.zeros_like(p, dtype=mu_dtype)
            self.nu[name] = torch.zeros_like(p, dtype=torch.float32)
        self._param_map = dict(self.params)
        self.last_grad_norm = None
        self.last_update_norm = None

    @torch.no_grad()
    def step(self, compute_update_norm: bool = False):
        grads = [(n, p.grad) for n, p in self.params if p.grad is not None]
        if not grads:
            self.last_grad_norm = 0.0
            self.last_update_norm = 0.0 if compute_update_norm else None
            return

        global_norm = global_l2_norm([g for _, g in grads])
        self.last_grad_norm = global_norm
        clip_coef = 1.0
        if self.grad_clip_norm:
            clip_coef = self.grad_clip_norm / max(global_norm, self.grad_clip_norm)

        self.count += 1
        t = self.count
        bc1 = 1.0 - self.b1 ** t
        bc2 = 1.0 - self.b2 ** t

        if self.foreach:
            updates = []
            for want_dtensor in (False, True):
                bucket = [(n, g) for n, g in grads if _is_dtensor(g) == want_dtensor]
                if bucket:
                    updates += self._step_foreach(bucket, clip_coef, bc1, bc2)
        else:
            updates = self._step_loop(grads, clip_coef, bc1, bc2)

        if compute_update_norm:
            self.last_update_norm = self.lr * global_l2_norm(updates)
        else:
            self.last_update_norm = None

    def _step_loop(self, grads, clip_coef, bc1, bc2):
        updates = []
        for name, g in grads:
            p = self._param_map[name]
            g = g.detach().float()
            if clip_coef != 1.0:
                g = g * clip_coef
            mu = (1.0 - self.b1) * g + self.b1 * self.mu[name].float()
            nu = (1.0 - self.b2) * g.square() + self.b2 * self.nu[name]
            mu_hat = mu / bc1
            nu_hat = nu / bc2
            update = mu_hat / (nu_hat.sqrt() + self.eps)
            p.add_(update, alpha=-self.lr)
            self.mu[name] = mu.to(self.mu_dtype)
            self.nu[name] = nu
            updates.append(update)
        return updates

    def _step_foreach(self, grads, clip_coef, bc1, bc2):
        names = [n for n, _ in grads]
        ps = [self._param_map[n] for n in names]
        gs = [g.detach().float() for _, g in grads]
        if clip_coef != 1.0:
            torch._foreach_mul_(gs, clip_coef)
        mu = [self.mu[n].float() for n in names]
        nu = [self.nu[n] for n in names]
        torch._foreach_mul_(mu, self.b1)
        torch._foreach_add_(mu, gs, alpha=1.0 - self.b1)
        g2 = torch._foreach_mul(gs, gs)
        torch._foreach_mul_(nu, self.b2)
        torch._foreach_add_(nu, g2, alpha=1.0 - self.b2)
        mu_hat = torch._foreach_div(mu, bc1)
        nu_hat = torch._foreach_div(nu, bc2)
        denom = torch._foreach_sqrt(nu_hat)
        torch._foreach_add_(denom, self.eps)
        updates = torch._foreach_div(mu_hat, denom)
        torch._foreach_add_(ps, updates, alpha=-self.lr)
        mu_bf16 = [m.to(self.mu_dtype) for m in mu]
        for n, m, v in zip(names, mu_bf16, nu):
            self.mu[n] = m
            self.nu[n] = v
        return updates


class EMA:
    def __init__(self, model: torch.nn.Module, decay_rate: float = 0.9999) -> None:
        self.decay = decay_rate
        self.shadow = {name: p.detach().float().clone() for name, p in model.named_parameters()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        shadows = [(self.shadow[n], p) for n, p in model.named_parameters()]
        for want_dtensor in (False, True):
            es = [e for e, _ in shadows if _is_dtensor(e) == want_dtensor]
            ps = [p.detach().float() for e, p in shadows if _is_dtensor(e) == want_dtensor]
            if es:
                torch._foreach_mul_(es, self.decay)
                torch._foreach_add_(es, ps, alpha=1.0 - self.decay)
