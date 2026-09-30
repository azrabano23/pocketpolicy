"""The sim with the errors a real SO-101 would add: the policy sees a corrupted
world, the physics runs on the true one.

What is perturbed, all assumed magnitudes, none measured on an arm:

  detector noise   per-tick xy noise on the object and goal positions
  detector bias    per-episode xy offset (camera extrinsics error)
  latency          the controller reads the world this many ticks late
  joint calib      per-episode encoder zero error on the five arm joints
  gravity sag      a loaded position servo settles short of its goal by a
                   steady error (shoulder_lift and elbow_flex)
  grasp radius     the jaw must close closer to the object than in the sim
  flag error       the grasp flag reads wrong this often, per tick

Commands live in the servo's own (encoder) frame: a goal c puts the joint at
c - calib - sag. The teacher and students command `measured q + step`. With
sag the joint lands `sag` short, and the next goal is built from that short
reading, so a loaded joint only moves when the policy's step exceeds the sag.
Near the object the steps shrink below it and the arm stalls, centimetres
short, until the episode times out. The fix
(`cmd_relative`) applies the policy's step to the command that produced the
reading instead of to the reading:

    cmd_t = cmd_{t-1-lat} + (target - q_read)

which cancels any constant error between goal and reading. It does not help
with calibration error: that corrupts where the policy thinks the tool is,
not how the servo tracks its goal. The
firmware does the same in `firmware/cmd_relative.c`; `CommandRing` is its
Python mirror, and a test holds the two to the same integers.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, fields

import numpy as np

from .robot import TICKS_PER_RAD
from .sim import GRASP_RADIUS, PickPlace, State, observe
from .student import concat_states

DEG = np.pi / 180
# Anti-windup: a blocked or speed-limited joint must not let the correction
# grow without bound. Same constant as CMD_MAX_CORR in the firmware.
MAX_CORR_TICKS = 114                     # ~10 degrees
MAX_CORR = MAX_CORR_TICKS / TICKS_PER_RAD
SAG_JOINTS = (1, 2)                      # shoulder_lift, elbow_flex carry the load


@dataclass(frozen=True)
class Perturb:
    """Error magnitudes. Each field is a scalar, or one value per episode [B]."""
    det_noise: float = 0.0               # m, sd of per-tick detector noise (xy)
    det_bias: float = 0.0                # m, sd of per-episode detector offset (xy)
    latency: int = 0                     # ticks
    joint_cal: float = 0.0               # rad, sd of per-episode encoder zero error
    sag: float = 0.0                     # rad, steady servo shortfall under load
    grasp_radius: float = GRASP_RADIUS   # m
    flag_err: float = 0.0                # probability per tick


PRESETS = {
    "ideal": Perturb(),
    "realistic": Perturb(det_noise=0.003, det_bias=0.005, latency=2, joint_cal=1.5 * DEG,
                         sag=1 * DEG, grasp_radius=0.015, flag_err=0.02),
    "pessimistic": Perturb(det_noise=0.005, det_bias=0.010, latency=3, joint_cal=3 * DEG,
                           sag=2 * DEG, grasp_radius=0.010, flag_err=0.05),
}


def sample(p: Perturb, B: int, rng: np.random.Generator, scale: float = 1.25) -> Perturb:
    """Per-episode magnitudes, each uniform between none and `scale` x p.

    Domain randomisation for training: every episode gets its own world, from
    perfect up to a little worse than `p`.
    """
    u = lambda: rng.uniform(0, scale, B)
    return Perturb(
        det_noise=u() * p.det_noise, det_bias=u() * p.det_bias,
        latency=rng.integers(0, int(np.ceil(scale * p.latency)) + 1, B),
        joint_cal=u() * p.joint_cal, sag=u() * p.sag,
        grasp_radius=np.maximum(GRASP_RADIUS - u() * (GRASP_RADIUS - p.grasp_radius), 0.002),
        flag_err=np.minimum(u() * p.flag_err, 0.5))


def _per_episode(x, B: int, dtype=float) -> np.ndarray:
    return np.broadcast_to(np.asarray(x, dtype), (B,)).copy()


class CommandRing:
    """The last lat+1 commands, and the latency-aware command-relative update.

    Mirrors `cmd_ring` in firmware/cmd_relative.c: `head` is the oldest slot,
    so with one latency L the command behind the current reading, cmd_{t-1-L},
    is buf[head]. Per-episode latencies below the ring's L read a newer slot.
    Works in any units: radians in the sim, encoder ticks in the C check.
    """

    def __init__(self, q0: np.ndarray, lat, max_corr: float = MAX_CORR):
        q0 = np.atleast_2d(q0)
        self.lat = _per_episode(lat, len(q0), int)
        self.n = int(self.lat.max()) + 1
        self.buf = np.repeat(q0[None], self.n, 0)
        self.head = 0
        self.max_corr = max_corr

    def base(self) -> np.ndarray:
        """cmd_{t-1-lat}: the command whose result the current reading shows."""
        slot = (self.head + self.n - 1 - self.lat) % self.n
        return self.buf[slot, np.arange(len(self.lat))]

    def correction(self, q_read: np.ndarray) -> np.ndarray:
        """base - reading, clamped; zero for the gripper, which is commanded absolutely."""
        corr = np.clip(self.base() - q_read, -self.max_corr, self.max_corr)
        corr[:, 5] = 0
        return corr

    def apply(self, target: np.ndarray, corr: np.ndarray) -> np.ndarray:
        """cmd = target + corr, pushed into the ring."""
        cmd = target + corr
        self.push(cmd)
        return cmd

    def push(self, cmd: np.ndarray) -> None:
        self.buf[self.head] = cmd
        self.head = (self.head + 1) % self.n

    def hold(self) -> np.ndarray:
        """A tick with no new command: the servo keeps the newest goal."""
        cmd = self.buf[(self.head - 1) % self.n].copy()
        self.push(cmd)
        return cmd


@dataclass
class Trace:
    """Planning ticks of a rollout: the true state, what the policy saw, and the
    true state that reading came from."""
    states: list
    observed: list
    read: list


class _Delay:
    """Readings of the true state, delayed per episode by `lat` ticks."""

    def __init__(self, lat: np.ndarray):
        self.lat = lat
        self.n = int(lat.max()) + 1
        self.hist: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    def push(self, s: State):
        self.hist.append((s.q, s.obj, s.held))
        del self.hist[:-self.n]

    def read(self):
        idx = np.maximum(len(self.hist) - 1 - self.lat, 0)
        b = np.arange(len(self.lat))
        return tuple(np.stack(f)[idx, b] for f in zip(*self.hist))


def rollout_real(env: PickPlace, policy, B: int, seed: int, exec_k: int = 1,
                 perturb: Perturb = PRESETS["ideal"], cmd_relative: bool = False,
                 record: bool = False, sees_truth: bool = False):
    """`teacher.rollout` with a corrupted observation and imperfect servos.

    policy(obs) -> [B, H, 6] absolute targets in the observed (encoder) frame,
    or policy(obs, true_state, read) when `sees_truth` (a privileged teacher),
    where `read` is the true state the (stale, corrupted) observation came
    from. The perturbation draws use their own generator, so
    under the `ideal` preset this reproduces `rollout` bit for bit. With
    `record`, returns the final state and a `Trace` of the planning ticks.
    """
    p = perturb
    rng = np.random.default_rng([seed, 7])
    env = copy.copy(env)
    env.grasp_radius = p.grasp_radius if np.ndim(p.grasp_radius) == 0 else _per_episode(
        p.grasp_radius, B)
    s = env.reset(B, seed)
    col = lambda x: _per_episode(x, B)[:, None]
    off = rng.normal(0, 1, (B, 6)) * col(p.joint_cal)
    off[:, 5] = 0
    xy = np.array([1.0, 1.0, 0.0])
    b_obj = rng.normal(0, 1, (B, 3)) * col(p.det_bias) * xy
    b_goal = rng.normal(0, 1, (B, 3)) * col(p.det_bias) * xy
    sag = np.zeros((B, 6))
    sag[:, SAG_JOINTS] = col(p.sag)
    noise, flag_err = col(p.det_noise), _per_episode(p.flag_err, B)
    lat = _per_episode(p.latency, B, int)
    delay = _Delay(lat)
    ring = plan = corr = None
    trace = Trace([], [], [])
    while not s.done.all():
        delay.push(s)
        if s.t % exec_k == 0 or plan is None:
            q, obj, held = delay.read()
            flip = rng.random(B) < flag_err
            r = State(q, obj, s.goal, held, s.done.copy(), s.success.copy(), s.t, s.rng)
            o = State(q + off, obj + b_obj + rng.normal(0, 1, (B, 3)) * noise * xy,
                      s.goal + b_goal + rng.normal(0, 1, (B, 3)) * noise * xy,
                      held ^ flip, s.done.copy(), s.success.copy(), s.t, s.rng)
            plan = policy(o, s, r) if sees_truth else policy(o)
            if record:
                trace.states.append(s.copy())
                trace.observed.append(o)
                trace.read.append(r)
            if cmd_relative:
                if ring is None:
                    ring = CommandRing(o.q, lat)
                corr = ring.correction(o.q)
        a = plan[:, s.t % exec_k if plan.shape[1] > 1 else 0]
        cmd = ring.apply(a, corr) if cmd_relative else a
        s = env.step(s, cmd - off - sag)
    return (s, trace) if record else s


TEACHER_VIEWS = ("now", "read", "observed")


def _view(view: str, o: State, s: State, r: State) -> tuple[State, np.ndarray]:
    """The state a teacher plans from, and the joints its step is measured from."""
    if view == "now":        # the true world at this tick
        return s, r.q
    if view == "read":       # the true world the reading came from (no latency privilege)
        return r, r.q
    if view == "observed":   # exactly what the student saw
        return o, o.q
    raise ValueError(view)


def privileged(teacher, H: int, view: str = "now"):
    """A teacher that plans from `view`, expressed as a step from the reading.

    Whichever way the goal is built, the joint ends up near `r.q + step` (the
    pose the reading came from, calibration error removed), so the step that
    lands on the teacher's target is `target - r.q`. That step is what the
    student is taught (see `labels`); the gripper stays absolute.
    """
    def act(o: State, s: State, r: State) -> np.ndarray:
        P, base = _view(view, o, s, r)
        c = teacher.chunk(P, H)
        out = o.q[:, None, :] + (c - base[:, None, :])
        out[..., 5] = c[..., 5]
        return out
    return act


def labels(teacher, trace: "Trace", H: int, view: str = "now"):
    """Student inputs (what it saw) and teacher steps [N, H*6] from a trace."""
    cat = {k: concat_states(getattr(trace, k)) for k in ("observed", "states", "read")}
    O = cat["observed"]
    P, base = _view(view, O, cat["states"], cat["read"])
    return observe(O), (teacher.chunk(P, H) - base[:, None, :]).reshape(O.B, -1)


def describe(p: Perturb) -> dict:
    return {f.name: float(getattr(p, f.name)) for f in fields(p)}
