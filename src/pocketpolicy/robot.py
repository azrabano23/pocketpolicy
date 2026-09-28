"""SO-101 kinematics from the robot's own URDF.

The URDF is vendored unmodified from TheRobotStudio/SO-ARM100 (Apache-2.0,
commit 5f6d2b8, Simulation/SO101/so101_new_calib.urdf). Only the kinematic
tree is used: joint origins, axes and limits. Meshes and inertias are ignored;
this is a kinematic model, not a dynamics simulation.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np

ARM = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
GRIPPER = "gripper"
JOINTS = ARM + (GRIPPER,)
TCP_LINK = "gripper_frame_link"

# Feetech STS3215: 12-bit magnetic encoder, 4096 counts per revolution.
TICKS_PER_RAD = 4096 / (2 * np.pi)


def _rpy(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def _vec(s: str | None, default=(0.0, 0.0, 0.0)) -> np.ndarray:
    return np.array([float(v) for v in s.split()]) if s else np.array(default, float)


@dataclass
class Joint:
    name: str
    type: str
    parent: str
    child: str
    T: np.ndarray            # 4x4 fixed transform parent -> joint frame
    axis: np.ndarray
    lower: float
    upper: float


def load_urdf(path: str | Path | None = None) -> dict[str, Joint]:
    if path is None:
        path = resources.files("pocketpolicy") / "assets" / "so101_new_calib.urdf"
    root = ET.parse(str(path)).getroot()
    joints = {}
    for j in root.findall("joint"):  # direct children; <transmission> also has <joint>
        o = j.find("origin")
        T = np.eye(4)
        if o is not None:
            T[:3, :3] = _rpy(*_vec(o.get("rpy")))
            T[:3, 3] = _vec(o.get("xyz"))
        ax = j.find("axis")
        lim = j.find("limit")
        joints[j.get("name")] = Joint(
            j.get("name"), j.get("type"), j.find("parent").get("link"),
            j.find("child").get("link"), T,
            _vec(ax.get("xyz") if ax is not None else None, (0, 0, 1)),
            float(lim.get("lower")) if lim is not None else 0.0,
            float(lim.get("upper")) if lim is not None else 0.0,
        )
    return joints


def _axis_angle(axis: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Batched rotation matrices about a fixed unit axis. q: [B] -> [B,3,3]."""
    a = axis / np.linalg.norm(axis)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    s, c = np.sin(q)[:, None, None], np.cos(q)[:, None, None]
    return np.eye(3)[None] + s * K[None] + (1 - c) * (K @ K)[None]


class Arm:
    """Batched forward kinematics for the chain base_link -> gripper frame."""

    def __init__(self, urdf: str | Path | None = None):
        joints = load_urdf(urdf)
        by_child = {j.child: j for j in joints.values()}
        chain = []
        link = TCP_LINK
        while link in by_child:
            j = by_child[link]
            chain.append(j)
            link = j.parent
        self.chain = chain[::-1]
        names = [j.name for j in self.chain if j.type == "revolute"]
        if tuple(names) != ARM:
            raise ValueError(f"unexpected chain {names}")
        self.joints = joints
        self.lower = np.array([joints[n].lower for n in JOINTS])
        self.upper = np.array([joints[n].upper for n in JOINTS])
        self.home = np.zeros(len(JOINTS))

    def tcp(self, q: np.ndarray) -> np.ndarray:
        """q: [B, >=5] joint angles -> [B, 3] tool-centre-point position in metres."""
        q = np.atleast_2d(q)
        B = q.shape[0]
        R = np.broadcast_to(np.eye(3), (B, 3, 3)).copy()
        p = np.zeros((B, 3))
        k = 0
        for j in self.chain:
            p = p + R @ j.T[:3, 3]
            R = R @ j.T[:3, :3]
            if j.type == "revolute":
                R = R @ _axis_angle(j.axis, q[:, k])
                k += 1
        return p

    def jacobian(self, q: np.ndarray, eps: float = 1e-5) -> np.ndarray:
        """Numerical position Jacobian w.r.t. the five arm joints: [B, 3, 5]."""
        q = np.atleast_2d(q).astype(float)
        J = np.empty((q.shape[0], 3, len(ARM)))
        for i in range(len(ARM)):
            dq = np.zeros(q.shape[1])
            dq[i] = eps
            J[:, :, i] = (self.tcp(q + dq) - self.tcp(q - dq)) / (2 * eps)
        return J

    def clip(self, q: np.ndarray) -> np.ndarray:
        return np.clip(q, self.lower, self.upper)
