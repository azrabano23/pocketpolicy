"""A batched kinematic pick-and-place task on the SO-101.

B episodes step in lockstep as arrays, so evaluating a policy on hundreds of
episodes is one vectorised loop, not hundreds.

What is modelled: the URDF kinematic chain and joint limits, a per-step joint
speed limit (STS3215-class servo), actuator noise, a grasp that succeeds only
if the gripper closes within reach of the object, and release onto the table.
What is not: contact dynamics, object mass, slip, collisions with the table.
A policy that succeeds here has learned the task's geometry and sequencing,
which is what distillation is being tested on; it has not been shown to
handle real contact.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .robot import Arm

HZ = 30.0                 # LeRobot records SO-101 datasets at 30 fps
DT = 1.0 / HZ
VMAX = 3.0                # rad/s joint speed cap
OPEN, CLOSED = 1.0, 0.0   # gripper joint targets, rad
GRASP_RADIUS = 0.02       # m: tcp must be this close to the object when the jaw closes
SUCCESS_RADIUS = 0.03     # m: object released within this xy distance of the goal
TABLE_Z = 0.015           # object centre height when resting on the table
MAX_STEPS = 300           # 10 s


@dataclass
class State:
    q: np.ndarray         # [B, 6] joint angles
    obj: np.ndarray       # [B, 3]
    goal: np.ndarray      # [B, 3]
    held: np.ndarray      # [B] bool
    done: np.ndarray      # [B] bool
    success: np.ndarray   # [B] bool
    t: int = 0
    rng: np.random.Generator = field(default_factory=np.random.default_rng)

    def copy(self) -> "State":
        return State(self.q.copy(), self.obj.copy(), self.goal.copy(), self.held.copy(),
                     self.done.copy(), self.success.copy(), self.t, self.rng)

    @property
    def B(self) -> int:
        return self.q.shape[0]


def _ring(rng: np.random.Generator, n: int) -> np.ndarray:
    r = rng.uniform(0.15, 0.27, n)
    a = rng.uniform(-0.8, 0.8, n)
    return np.stack([r * np.cos(a), r * np.sin(a), np.full(n, TABLE_Z)], 1)


class PickPlace:
    def __init__(self, arm: Arm | None = None, noise: float = 0.004):
        self.arm = arm or Arm()
        self.noise = noise
        self.q0 = self._rest_pose()

    def _rest_pose(self) -> np.ndarray:
        """Joint pose holding the tool 15 cm above the table, 20 cm out."""
        q = np.array([[0.0, -0.5, 0.8, 0.9, 0.0, OPEN]])
        target = np.array([[0.20, 0.0, 0.15]])
        for _ in range(200):
            q[:, :4] += _dls(self.arm, q, target - self.arm.tcp(q), 0.02)
            q = self.arm.clip(q)
        return q[0]

    def reset(self, B: int, seed: int) -> State:
        rng = np.random.default_rng(seed)
        obj = _ring(rng, B)
        goal = _ring(rng, B)
        for _ in range(100):  # goals at least 6 cm from the object
            bad = np.linalg.norm(goal[:, :2] - obj[:, :2], axis=1) < 0.06
            if not bad.any():
                break
            goal[bad] = _ring(rng, int(bad.sum()))
        q = np.tile(self.q0, (B, 1)) + rng.normal(0, 0.02, (B, 6))
        z = np.zeros(B, bool)
        return State(self.arm.clip(q), obj, goal, z.copy(), z.copy(), z.copy(), 0, rng)

    def step(self, s: State, target: np.ndarray, noise: bool = True) -> State:
        """Advance one control tick toward absolute joint targets [B, 6]."""
        live = ~s.done
        dq = np.clip(target - s.q, -VMAX * DT, VMAX * DT)
        if noise and self.noise:
            dq = dq + s.rng.normal(0, self.noise, dq.shape)
        q = np.where(live[:, None], self.arm.clip(s.q + dq), s.q)
        tcp = self.arm.tcp(q)

        grip = q[:, 5]
        near = np.linalg.norm(tcp - s.obj, axis=1) < GRASP_RADIUS
        grab = live & ~s.held & (grip < 0.4) & near
        drop = live & s.held & (grip > 0.7)
        held = (s.held | grab) & ~drop

        obj = s.obj.copy()
        obj[held] = tcp[held]
        obj[drop] = np.c_[tcp[drop, :2], np.full(int(drop.sum()), TABLE_Z)]

        placed = drop & (np.linalg.norm(obj[:, :2] - s.goal[:, :2], axis=1) < SUCCESS_RADIUS)
        success = s.success | placed
        done = s.done | placed | (s.t + 1 >= MAX_STEPS)
        return State(q, obj, s.goal, held, done, success, s.t + 1, s.rng)


def _dls(arm: Arm, q: np.ndarray, dx: np.ndarray, lam: float) -> np.ndarray:
    """Damped least squares on the first four joints: [B,3] -> [B,4]."""
    J = arm.jacobian(q)[:, :, :4]
    JJt = J @ J.transpose(0, 2, 1) + lam**2 * np.eye(3)
    return (J.transpose(0, 2, 1) @ np.linalg.solve(JJt, dx[..., None]))[..., 0]


def observe(s: State) -> np.ndarray:
    """What the controller is given: joints, object and goal positions, grasp flag.

    On hardware the joints come from the servo encoders and the positions from
    an upstream detector; the policy never sees images.
    """
    return np.concatenate([s.q, s.obj, s.goal, s.held[:, None].astype(float)], 1)


OBS_DIM = 13
ACT_DIM = 6
