"""The distillation pipeline as a loopgraph graph, and the campaign around it.

    teacher_eval ─────────────────────────────┐
    student(width, depth, H, dagger) ─┬───────┤
    calib ──────────────── deployed ──┴─► metrics ─► ledger
                                          (float + int8 closed loop, C gate)

Every node is cached by content, so re-running a campaign after changing the
evaluation only re-evaluates; students already trained are reused.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np

from loopgraph import Graph, node
from loopgraph.cgate import CGateError, cc

from .deploy import DeployedPolicy, to_raw
from .gate import check_c
from .sim import HZ, PickPlace, observe
from .student import concat_states, distill
from .teacher import ScriptedTeacher, rollout

EVAL_SEED = 10_000        # held out: training rollouts use seeds < 2000
# Sizes are graph inputs, not module constants, so they enter the cache key:
# changing the evaluation can never silently reuse results computed under
# the old one.
DEFAULTS = {"eval_episodes": 200, "train_episodes": 128, "epochs": 60}
# Derived, not measured: a scalar int8 MAC in portable C on a Cortex-M4 costs
# a few cycles (two loads, a multiply-accumulate, loop overhead). 4 is a
# deliberately pessimistic figure at the Crazyflie's STM32F405 clock.
M4_CYCLES_PER_MAC = 4
M4_HZ = 168e6

SPACE = {
    "width": [8, 16, 32, 64, 128, 256],
    "depth": [1, 2, 3],
    "H": [1, 2, 4, 8],
    "dagger": [0, 2, 4],
}


@lru_cache(maxsize=1)
def _env() -> tuple[PickPlace, ScriptedTeacher]:
    env = PickPlace()
    return env, ScriptedTeacher(env)


def exec_k(H: int) -> int:
    """Execute half of each chunk before re-planning (at least one tick)."""
    return max(1, H // 2)


@node(version="2")
def teacher_eval(seed, eval_episodes):
    env, T = _env()
    s = rollout(env, lambda s: T.chunk(s, 1), eval_episodes, EVAL_SEED + seed, 1)
    return float(s.success.mean())


@node(version="1")
def student(width, depth, H, dagger, seed, train_episodes, epochs):
    env, T = _env()
    net, info = distill(env, T, width, depth, H, dagger, episodes=train_episodes,
                        epochs=epochs, seed=seed)
    return {"net": net, "info": info}


@node(version="1")
def calib(seed):
    env, T = _env()
    _, st = rollout(env, lambda s: T.chunk(s, 1), 64, 1500 + seed, 1, record=True)
    return observe(concat_states(st))


@node(version="2")
def deployed(student, calib):
    return DeployedPolicy.from_student(student["net"], calib)


@node(version="3")  # bumped: the C gate's emitter changed underneath this node
def metrics(student, deployed, teacher_eval, calib, H, seed, eval_episodes):
    env, _ = _env()
    net = student["net"]
    k = exec_k(H)
    sf = rollout(env, net.policy, eval_episodes, EVAL_SEED + seed, k)
    si = rollout(env, deployed.policy, eval_episodes, EVAL_SEED + seed, k)
    macs = deployed.q.macs
    m = {
        "teacher_success": teacher_eval,
        "success_float": float(sf.success.mean()),
        "success": float(si.success.mean()),
        "retention": float(si.success.mean()) / max(teacher_eval, 1e-9),
        "params": int(net.n_params),
        "flash_kb": deployed.flash_bytes / 1024,
        "ram_b": deployed.ram_bytes,
        "macs": int(macs),
        "infer_hz": HZ / k,
        "m4_ms_est": macs * M4_CYCLES_PER_MAC / M4_HZ * 1e3,
        "train_samples": int(student["info"]["samples"]),
        "eval_episodes": eval_episodes,
    }
    m["m4_duty_est"] = m["m4_ms_est"] * m["infer_hz"] / 1e3
    if cc() is not None:
        rng = np.random.default_rng(seed)
        raw = to_raw(calib[rng.choice(len(calib), min(256, len(calib)), replace=False)])
        try:
            m["bit_exact"] = int(check_c(deployed, raw).ok)
        except CGateError:
            m["bit_exact"] = 0
    return m


def graph(cache_dir: str | Path | None = ".loopgraph/cache") -> Graph:
    return Graph([teacher_eval, student, calib, deployed, metrics], cache_dir=cache_dir)


def execute(params: dict, seed: int = 0, cache_dir=".loopgraph/cache", **sizes):
    r = graph(cache_dir).run(["metrics"], {**DEFAULTS, **sizes, **params, "seed": seed})
    return r["metrics"], r.keys
