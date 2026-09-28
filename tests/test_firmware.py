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
                                    for f in ("sts3215.h", "sts3215.c", "so101_main.c")}}
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
