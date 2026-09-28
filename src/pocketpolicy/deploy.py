"""Integer-only deployment: sensor units in, servo targets out.

The microcontroller never sees radians, metres or floats. Its contract is:

    raw[13]  int16   6 joint positions in STS3215 encoder ticks (4096/rev,
                     0 = URDF zero), object xyz and goal xyz in millimetres,
                     grasp flag 0/1
    out[H*6] int16   absolute joint targets in encoder ticks for the next H
                     control ticks

Three integer stages, all mirrored exactly by `DeployedPolicy.step_int`:
  prep:  x_i = clamp_int8((raw_i * g_i - c_i + 2^15) >> 16)
  net:   loopgraph.int8 dense layers
  post:  target_j = q_ticks_j + ((y_j * a_j + b_j + 2^15) >> 16)

Normalisation, input quantisation and output de-normalisation are folded into
the (c, g) and (a, b) constants, so every float in training disappears.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from loopgraph.int8 import QMLP, emit_c, quantize_mlp

from .robot import TICKS_PER_RAD
from .sim import ACT_DIM, OBS_DIM, State, observe
from .student import MLP

FRAC = 16
UNITS = np.array([1 / TICKS_PER_RAD] * 6 + [1e-3] * 6 + [1.0])  # SI per raw unit


def to_raw(obs: np.ndarray) -> np.ndarray:
    """What the sensors would report for a float observation, int16."""
    return np.clip(np.round(obs / UNITS), -32768, 32767).astype(np.int16)


@dataclass
class DeployedPolicy:
    q: QMLP
    c: np.ndarray   # int32 [13] Q16 input offsets
    g: np.ndarray   # int32 [13] Q16 input gains
    a: np.ndarray   # int32 [H*6] Q16 output gains (ticks per output lsb)
    b: np.ndarray   # int32 [H*6] Q16 output offsets (ticks)
    H: int

    @classmethod
    def from_student(cls, net: MLP, calib_obs: np.ndarray, pct: float = 99.99) -> "DeployedPolicy":
        """Fold per-feature ranges into the first and last layers, then quantize.

        One int8 scale shared by all inputs gives a joint angle about fifteen
        levels across its range, which is centimetres at the tool. Prep
        already has one gain per feature, so each input is rescaled to its own
        calibrated range and the first layer's columns absorb the factor;
        the last layer's rows and the post gains do the same for outputs.
        """
        xn = net.xn(calib_obs)
        r_in = np.maximum(np.percentile(np.abs(xn), pct, axis=0), 1e-3)
        yn = net._fwd(xn)[-1]
        r_out = np.maximum(np.percentile(np.abs(yn), pct, axis=0), 1e-3)
        layers = [(W.copy(), b.copy()) for W, b in net.layers]
        W0 = layers[0][0] * r_in[None, :]
        W0[:, r_in <= 1e-3] = 0.0
        layers[0] = (W0, layers[0][1])
        layers[-1] = (layers[-1][0] / r_out[:, None], layers[-1][1] / r_out)
        q = quantize_mlp(layers, xn / r_in, pct=pct)
        g = np.round(UNITS / (net.xn.sd * r_in * q.s_in) * 2**FRAC).astype(np.int64)
        g[r_in <= 1e-3] = 0  # constant in calibration (e.g. goal height): carries nothing
        # the offset lives in the Q16 domain: rounding it to whole raw units
        # would shift a 0/1 flag with mean 0.3 by most of its range
        c = np.round(net.xn.mu / UNITS * g).astype(np.int64)
        if np.abs(c).max() >= 2**31:
            raise OverflowError("input offset does not fit int32")
        if np.abs(g).max() >= 2**31:
            raise OverflowError("input gain does not fit int32")
        a = np.round(q.s_out * r_out * net.yn.sd * TICKS_PER_RAD * 2**FRAC).astype(np.int64)
        b = np.round(net.yn.mu * TICKS_PER_RAD * 2**FRAC).astype(np.int64)
        if max(np.abs(a).max(), np.abs(b).max()) >= 2**31:
            raise OverflowError("output constants do not fit int32")
        return cls(q, c.astype(np.int32), g.astype(np.int32), a.astype(np.int32), b.astype(np.int32), net.H)

    # -- exact integer reference --------------------------------------------
    def prep(self, raw: np.ndarray) -> np.ndarray:
        v = (raw.astype(np.int64) * self.g - self.c + (1 << (FRAC - 1))) >> FRAC
        return np.clip(v, -128, 127).astype(np.int8)

    def post(self, raw: np.ndarray, y: np.ndarray) -> np.ndarray:
        qt = np.tile(raw[:, :ACT_DIM].astype(np.int64), (1, self.H))
        d = (y.astype(np.int64) * self.a + self.b + (1 << (FRAC - 1))) >> FRAC
        return np.clip(qt + d, -32768, 32767).astype(np.int16)

    def step_int(self, raw: np.ndarray) -> np.ndarray:
        raw = np.atleast_2d(raw)
        return self.post(raw, self.q.forward_int(self.prep(raw)))

    def policy(self, s: State) -> np.ndarray:
        """Closed-loop use in simulation: quantise sensors, run ints, convert back."""
        out = self.step_int(to_raw(observe(s)))
        return out.reshape(s.B, self.H, ACT_DIM).astype(np.float64) / TICKS_PER_RAD

    @property
    def flash_bytes(self) -> int:
        return self.q.param_bytes + 4 * (self.c.size + self.g.size + self.a.size + self.b.size)

    @property
    def ram_bytes(self) -> int:
        return self.q.scratch_bytes + OBS_DIM  # activations + the int8 input vector

    # -- C ------------------------------------------------------------------
    def emit(self, name: str = "pocket") -> dict[str, str]:
        net = emit_c(self.q, f"{name}_net")
        up = name.upper()
        nout = self.H * ACT_DIM

        def arr(t, n, v):
            return f"static const {t} {n}[{len(v)}] = {{{', '.join(str(int(x)) for x in v)}}};\n"

        h = f"""/* Generated by pocketpolicy. Do not edit. */
#ifndef {up}_POLICY_H
#define {up}_POLICY_H
#include <stdint.h>

#define {up}_IN {OBS_DIM}      /* q[6] ticks, obj[3] mm, goal[3] mm, grasped */
#define {up}_H {self.H}         /* control ticks per inference */
#define {up}_OUT {nout}     /* H x 6 joint targets, ticks */

void {name}_step(const int16_t raw[{up}_IN], int16_t targets[{up}_OUT]);

#endif
"""
        c = (f"/* Generated by pocketpolicy. Do not edit. */\n#include \"{name}.h\"\n"
             f"#include \"{name}_net.h\"\n\n"
             + arr("int32_t", "C_OFF", self.c) + arr("int32_t", "C_GAIN", self.g)
             + arr("int32_t", "P_GAIN", self.a) + arr("int32_t", "P_OFF", self.b)
             + f"""
void {name}_step(const int16_t raw[{up}_IN], int16_t targets[{up}_OUT])
{{
    int8_t x[{up}_IN];
    int8_t y[{up}_OUT];
    for (int i = 0; i < {up}_IN; ++i) {{
        int64_t v = ((int64_t)raw[i] * C_GAIN[i] - C_OFF[i] + (1 << {FRAC - 1})) >> {FRAC};
        x[i] = (int8_t)(v > 127 ? 127 : (v < -128 ? -128 : v));
    }}
    {name}_net_forward(x, y);
    for (int j = 0; j < {up}_OUT; ++j) {{
        int64_t d = ((int64_t)y[j] * P_GAIN[j] + P_OFF[j] + (1 << {FRAC - 1})) >> {FRAC};
        int64_t t = (int64_t)raw[j % 6] + d;
        targets[j] = (int16_t)(t > 32767 ? 32767 : (t < -32768 ? -32768 : t));
    }}
}}
""")
        return {**net, f"{name}.h": h, f"{name}.c": c}
