"""The student: a small MLP trained in numpy, and DAgger distillation.

Behaviour cloning alone trains on states the teacher visits; the student then
drifts into states it never saw and compounds its own errors. DAgger (Ross,
Gordon & Bagnell, AISTATS 2011) fixes this by rolling out the *student*,
asking the teacher what it would have done in the states the student actually
reached, and training on the union. The smaller the student, the more it
drifts, so this matters most exactly where microcontroller policies live.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .sim import ACT_DIM, PickPlace, State, observe
from .teacher import Teacher, rollout


@dataclass
class Norm:
    mu: np.ndarray
    sd: np.ndarray

    @classmethod
    def fit(cls, x: np.ndarray, floor: float = 1e-3) -> "Norm":
        return cls(x.mean(0), np.maximum(x.std(0), floor))

    def __call__(self, x):
        return (x - self.mu) / self.sd

    def inv(self, x):
        return x * self.sd + self.mu


@dataclass
class MLP:
    layers: list[tuple[np.ndarray, np.ndarray]]
    xn: Norm
    yn: Norm
    H: int
    _adam: dict = field(default_factory=dict, repr=False)

    @classmethod
    def init(cls, din: int, width: int, depth: int, dout: int, xn: Norm, yn: Norm, H: int,
             seed: int = 0) -> "MLP":
        rng = np.random.default_rng(seed)
        dims = [din] + [width] * depth + [dout]
        layers = [(rng.normal(0, np.sqrt(2 / i), (o, i)), np.zeros(o))
                  for i, o in zip(dims[:-1], dims[1:])]
        return cls(layers, xn, yn, H)

    def _fwd(self, xn: np.ndarray):
        acts = [xn]
        a = xn
        for i, (W, b) in enumerate(self.layers):
            a = a @ W.T + b
            if i < len(self.layers) - 1:
                a = np.maximum(a, 0)
            acts.append(a)
        return acts

    def predict_norm(self, obs: np.ndarray) -> np.ndarray:
        return self._fwd(self.xn(obs))[-1]

    def fit(self, X: np.ndarray, Y: np.ndarray, epochs: int, lr: float = 2e-3,
            batch: int = 256, seed: int = 0) -> float:
        """Adam on MSE in normalised target space; returns final epoch loss."""
        rng = np.random.default_rng(seed)
        Xn, Yn = self.xn(X), self.yn(Y)
        st = self._adam or {"t": 0, "m": [(np.zeros_like(W), np.zeros_like(b)) for W, b in self.layers],
                            "v": [(np.zeros_like(W), np.zeros_like(b)) for W, b in self.layers]}
        b1, b2, eps = 0.9, 0.999, 1e-8
        loss = np.inf
        for _ in range(epochs):
            idx = rng.permutation(len(Xn))
            tot = 0.0
            for s in range(0, len(idx), batch):
                j = idx[s:s + batch]
                acts = self._fwd(Xn[j])
                err = acts[-1] - Yn[j]
                tot += float((err**2).sum())
                g = 2 * err / len(j) / err.shape[1]
                st["t"] += 1
                for li in range(len(self.layers) - 1, -1, -1):
                    W, b = self.layers[li]
                    gW, gb = g.T @ acts[li], g.sum(0)
                    if li > 0:
                        g = (g @ W) * (acts[li] > 0)
                    (mW, mb), (vW, vb) = st["m"][li], st["v"][li]
                    mW[:] = b1 * mW + (1 - b1) * gW
                    mb[:] = b1 * mb + (1 - b1) * gb
                    vW[:] = b2 * vW + (1 - b2) * gW**2
                    vb[:] = b2 * vb + (1 - b2) * gb**2
                    c1, c2 = 1 - b1 ** st["t"], 1 - b2 ** st["t"]
                    W -= lr * (mW / c1) / (np.sqrt(vW / c2) + eps)
                    b -= lr * (mb / c1) / (np.sqrt(vb / c2) + eps)
            loss = tot / Yn.size
        self._adam = st
        return loss

    def policy(self, s: State) -> np.ndarray:
        """Chunk of absolute joint targets from the float model: [B, H, 6]."""
        d = self.yn.inv(self.predict_norm(observe(s))).reshape(s.B, self.H, ACT_DIM)
        return s.q[:, None, :] + d

    @property
    def n_params(self) -> int:
        return sum(W.size + b.size for W, b in self.layers)


def concat_states(states: list[State]) -> State:
    live = [s for s in states if (~s.done).any()]
    pick = [~s.done for s in live]
    cat = lambda f: np.concatenate([getattr(s, f)[m] for s, m in zip(live, pick)])
    B = sum(int(m.sum()) for m in pick)
    z = np.zeros(B, bool)
    return State(cat("q"), cat("obj"), cat("goal"), cat("held"), z, z.copy(), 0,
                 live[0].rng if live else np.random.default_rng())


def label(teacher: Teacher, s: State, H: int) -> tuple[np.ndarray, np.ndarray]:
    """Observations and teacher chunk deltas [N, H*6] for a batch of states."""
    chunk = teacher.chunk(s, H)
    return observe(s), (chunk - s.q[:, None, :]).reshape(s.B, -1)


def distill(env: PickPlace, teacher: Teacher, width: int, depth: int, H: int,
            dagger_rounds: int, episodes: int = 64, epochs: int = 40, seed: int = 0,
            exec_k: int | None = None) -> tuple[MLP, dict]:
    exec_k = exec_k or max(1, H // 2)
    teacher_policy = lambda s: teacher.chunk(s, 1)
    _, states = rollout(env, teacher_policy, episodes, seed, 1, record=True)
    X, Y = label(teacher, concat_states(states), H)
    net = MLP.init(X.shape[1], width, depth, Y.shape[1], Norm.fit(X), Norm.fit(Y), H, seed)
    losses = [net.fit(X, Y, epochs, seed=seed)]
    sizes = [len(X)]
    for r in range(dagger_rounds):
        beta = 0.5 ** (r + 1)  # teacher share of control decays each round
        rng = np.random.default_rng(seed + 1000 + r)

        def mixed(s, rng=rng, beta=beta):
            a = net.policy(s)
            use_t = rng.random(s.B) < beta
            if use_t.any():
                t = teacher.chunk(s, H)
                a[use_t] = t[use_t]
            return a

        _, st = rollout(env, mixed, episodes, seed + 100 + r, exec_k, record=True)
        Xr, Yr = label(teacher, concat_states(st), H)
        X, Y = np.concatenate([X, Xr]), np.concatenate([Y, Yr])
        losses.append(net.fit(X, Y, max(10, epochs // 2), seed=seed + r + 1))
        sizes.append(len(X))
    return net, {"loss": losses[-1], "samples": sizes[-1], "rounds": dagger_rounds}
