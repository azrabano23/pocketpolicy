"""Teachers: policies that are too big for the target and must be distilled.

`ScriptedTeacher` is a privileged inverse-kinematics expert. It stands in for
a foundation policy in every test and benchmark here, because it is
deterministic, runs anywhere and never needs a GPU.

`OpenPITeacher` wraps a Physical Intelligence openpi policy server
(pi0 / pi0-FAST / pi0.5) over its websocket client. Its checkpoints are
10-12 GB and inference needs a >8 GB GPU; it has not been run as part of this
repository's test suite. The distillation code does not care which teacher
it gets: both return action chunks of absolute joint targets.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from .sim import CLOSED, OPEN, PickPlace, State, _dls

HOVER = 0.06      # m above object/goal before descending
CARRY = 0.08      # m above table while carrying
STEP = 0.012      # m: max tool displacement commanded per tick (0.36 m/s)


class Teacher(Protocol):
    def chunk(self, s: State, H: int) -> np.ndarray:
        """Absolute joint targets for the next H ticks: [B, H, 6]."""


def _clip_norm(v: np.ndarray, m: float) -> np.ndarray:
    n = np.linalg.norm(v, axis=1, keepdims=True)
    return v * np.minimum(1.0, m / np.maximum(n, 1e-9))


class ScriptedTeacher:
    def __init__(self, env: PickPlace):
        self.env = env
        self.arm = env.arm
        self.q_nom = env.q0.copy()

    def act(self, s: State) -> np.ndarray:
        """One tick of absolute joint targets [B, 6], decided from state alone."""
        tcp = self.arm.tcp(s.q)
        B = s.B
        target = tcp.copy()
        grip = np.full(B, OPEN)

        free = ~s.held
        dxy_obj = np.linalg.norm(tcp[:, :2] - s.obj[:, :2], axis=1)
        d_obj = np.linalg.norm(tcp - s.obj, axis=1)
        over = dxy_obj < 0.012
        at = d_obj < 0.008
        m = free & ~over
        target[m] = s.obj[m] + [0, 0, HOVER]
        # keep altitude while travelling so the jaw never drags across the object
        target[m, 2] = np.maximum(target[m, 2], np.minimum(tcp[m, 2], s.obj[m, 2] + HOVER))
        m = free & over & ~at
        target[m] = s.obj[m]
        m = free & over & at
        target[m] = s.obj[m]
        grip[m] = CLOSED

        held = s.held
        dxy_goal = np.linalg.norm(tcp[:, :2] - s.goal[:, :2], axis=1)
        above_goal = dxy_goal < 0.012
        low = tcp[:, 2] < CARRY - 0.02
        m = held & ~above_goal & low
        target[m] = np.c_[tcp[m, :2], np.full(int(m.sum()), CARRY)]
        m = held & ~above_goal & ~low
        target[m] = s.goal[m] + [0, 0, CARRY]
        m = held & above_goal
        target[m] = s.goal[m] + [0, 0, 0.005]
        grip[held] = CLOSED
        release = held & above_goal & (tcp[:, 2] < s.goal[:, 2] + 0.015)
        grip[release] = OPEN

        dx = _clip_norm(target - tcp, STEP)
        q = s.q.copy()
        dq = _dls(self.arm, q, dx, 0.03)
        # null-space pull toward the rest posture keeps the solution unique
        q[:, :4] += dq + 0.02 * (self.q_nom[:4] - q[:, :4])
        q[:, 4] = 0.0
        q[:, 5] = grip
        return self.arm.clip(q)

    def chunk(self, s: State, H: int) -> np.ndarray:
        out = np.empty((s.B, H, 6))
        sim = s.copy()
        sim.done = np.zeros_like(sim.done)
        for h in range(H):
            a = self.act(sim)
            out[:, h] = a
            sim = self.env.step(sim, a, noise=False)
        return out


class OpenPITeacher:  # pragma: no cover - needs a running openpi server and GPU
    """Query an openpi policy server for action chunks.

    `obs_fn` maps simulator state to the observation dict the server's config
    expects (images, state, prompt); `act_fn` maps its action array back to
    SO-101 joint targets. openpi ships no SO-101 normalisation stats, so a
    fine-tuned checkpoint is required in practice.
    """

    def __init__(self, host: str, port: int, obs_fn, act_fn, prompt: str):
        from openpi_client import websocket_client_policy

        self.client = websocket_client_policy.WebsocketClientPolicy(host=host, port=port)
        self.obs_fn, self.act_fn, self.prompt = obs_fn, act_fn, prompt

    def chunk(self, s: State, H: int) -> np.ndarray:
        out = []
        for b in range(s.B):
            obs = self.obs_fn(s, b)
            obs["prompt"] = self.prompt
            actions = self.client.infer(obs)["actions"]  # [horizon, action_dim]
            out.append(self.act_fn(s, b, actions[:H]))
        return np.stack(out)


def rollout(env: PickPlace, policy, B: int, seed: int, exec_k: int = 1,
            record: bool = False):
    """Run a chunked policy closed-loop. policy(state) -> [B, H, 6] targets.

    The first `exec_k` actions of each chunk are executed open-loop before the
    policy is queried again, which is how chunked policies are run on robots
    and what sets the inference rate the microcontroller must sustain.
    """
    s = env.reset(B, seed)
    states = []
    plan = None
    while not s.done.all():
        if s.t % exec_k == 0 or plan is None:
            plan = policy(s)
            if record:
                states.append(s.copy())
        a = plan[:, s.t % exec_k if plan.shape[1] > 1 else 0]
        s = env.step(s, a)
    return (s, states) if record else s
