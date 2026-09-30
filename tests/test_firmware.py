"""The servo protocol in C, checked against an independent Python encoding."""

from pathlib import Path

import pytest

from loopgraph.cgate import cc, compile_and_run

FW = Path(__file__).resolve().parents[1] / "firmware"


def chk(body):
    return (~sum(body)) & 0xFF


def sync_write(ids, pos):
    params = [42, 2]
    for i, p in zip(ids, pos):
        params += [i, p & 0xFF, p >> 8]
    body = [0xFE, len(params) + 2, 0x83] + params
    return bytes([0xFF, 0xFF] + body + [chk(body)]).hex().upper()


def read_pos(i):
    body = [i, 4, 0x02, 56, 2]
    return bytes([0xFF, 0xFF] + body + [chk(body)]).hex().upper()


@pytest.mark.skipif(cc() is None, reason="no C compiler")
def test_packets_match_protocol():
    srcs = {f: (FW / f).read_text() for f in ("sts3215.h", "sts3215.c")}
    out = compile_and_run(srcs, (FW / "test_host.c").read_text()).split("\n")
    assert out[0] == sync_write([1, 2], [2048, 1000])
    assert out[1] == read_pos(3)
    assert out[2] == "0 2100"      # 0x0834
    assert out[3] == "-1"          # corrupted checksum rejected


@pytest.mark.skipif(cc() is None, reason="no C compiler")
def test_control_loop_compiles_against_a_generated_policy(tmp_path):
    import numpy as np

    from pocketpolicy.deploy import DeployedPolicy
    from pocketpolicy.sim import PickPlace, observe
    from pocketpolicy.student import concat_states, distill
    from pocketpolicy.teacher import ScriptedTeacher, rollout

    env = PickPlace()
    T = ScriptedTeacher(env)
    net, _ = distill(env, T, 8, 1, 4, 0, episodes=4, epochs=1)
    _, st = rollout(env, lambda s: T.chunk(s, 1), 4, 0, 1, record=True)
    d = DeployedPolicy.from_student(net, observe(concat_states(st)))
    srcs = {**d.emit("pocket"), **{f: (FW / f).read_text()
                                    for f in ("sts3215.h", "sts3215.c", "cmd_relative.h",
                                              "cmd_relative.c", "so101_main.c")}}
    stub = """#include <stdint.h>
void so101_run(void);
void hal_uart_write(const uint8_t *b, int n) { (void)b; (void)n; }
int hal_uart_read(uint8_t *b, int n, int t) { (void)b; (void)n; (void)t; return 0; }
void hal_wait_tick(void) {}
int hal_detector(int16_t o[3], int16_t g[3]) { (void)o; (void)g; return 0; }
int hal_grasped(void) { return 0; }
int main(void) { return 0; }
"""
    compile_and_run(srcs, stub)  # links; the loop itself is not entered



def _ring_ops(lat, sag_ticks, steps=120, exec_k=2, seed=0):
    """A closed loop in encoder ticks: a servo that settles short of its goal,
    a reading `lat` ticks stale, a policy stepping toward a goal, and one
    sensing fault. Returns the ops for the C ring, the lines the Python
    CommandRing produced for them, and the joint trajectory."""
    import numpy as np

    from pocketpolicy.realworld import MAX_CORR_TICKS, CommandRing

    rng = np.random.default_rng(seed)
    goal = rng.integers(-300, 300, 6)
    sag = np.array([0, sag_ticks, sag_ticks, 0, 0, 0])
    pos = [rng.integers(-300, 300, 6)]                     # true joint ticks, per tick
    ring = CommandRing(pos[0][None], lat, MAX_CORR_TICKS)
    line = lambda tag, v: tag + "".join(f" {int(x)}" for x in np.ravel(v))
    ops, py = [line("i", pos[0])], []
    plan, k = None, exec_k
    for t in range(steps):
        if t == 17:                                        # sensing fault: hold
            ops.append("h")
            py.append(line("g", ring.hold()))
            k = exec_k                                     # re-plan next tick
        else:
            if k == exec_k:
                q = pos[max(0, len(pos) - 1 - lat)]
                step = (goal - q) // (2 + lat)             # slow enough for the delay
                plan = [q + step * (i + 1) for i in range(exec_k)]
                corr = ring.correction(q[None])
                ops.append(line("c", q))
                py.append(line("k", corr))
                k = 0
            ops.append(line("a", plan[k]))
            py.append(line("g", ring.apply(plan[k][None], corr)))
            k += 1
        cmd = np.array([int(x) for x in py[-1].split()[1:]])
        pos.append(cmd - sag)                              # the servo settles short
    return ops, py, pos, goal


@pytest.mark.skipif(cc() is None, reason="no C compiler")
@pytest.mark.parametrize("lat", [0, 2])
def test_cmd_relative_c_matches_python(lat):
    srcs = {f: (FW / f).read_text() for f in ("cmd_relative.h", "cmd_relative.c")}
    ops, py, pos, goal = _ring_ops(lat, sag_ticks=23)       # 2 degrees of sag
    out = compile_and_run(srcs, (FW / "test_cmd_relative.c").read_text(),
                          "\n".join(ops) + "\n", extra_flags=[f"-DCMD_LAT={lat}"])
    assert out.strip().splitlines() == py
    # the loop it drove reached the goal on every sagging joint
    assert abs(pos[-1][:5] - goal[:5]).max() <= 3


def test_naive_goals_fall_short_under_sag():
    """The same servo, sent `reading + step`: loaded joints stall short of the goal."""
    import numpy as np

    goal, sag = np.array([300, 300, 300]), np.array([0, 23, 23])
    q = np.zeros(3, int)
    for _ in range(60):
        q = q + (goal - q) // 2 - sag
    assert q[0] >= 299 and (q[1:] < 300 - 40).all()
