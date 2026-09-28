import numpy as np
import pytest

from loopgraph.cgate import cc
from pocketpolicy.deploy import DeployedPolicy, to_raw
from pocketpolicy.gate import check_c
from pocketpolicy.sim import PickPlace, observe
from pocketpolicy.student import concat_states, distill
from pocketpolicy.teacher import ScriptedTeacher, rollout


@pytest.fixture(scope="module")
def trained():
    env = PickPlace()
    T = ScriptedTeacher(env)
    net, _ = distill(env, T, 32, 2, 2, dagger_rounds=0, episodes=24, epochs=15)
    _, st = rollout(env, lambda s: T.chunk(s, 1), 16, 5, 1, record=True)
    S = concat_states(st)
    return env, net, S, DeployedPolicy.from_student(net, observe(S))


def test_integer_path_tracks_float(trained):
    env, net, S, d = trained
    err = np.abs(net.policy(S) - d.policy(S))
    assert err.mean() < 0.01  # rad


def test_flag_offset_is_not_rounded_away(trained):
    """Regression: a 0/1 feature's mean must survive into the Q16 offset."""
    env, net, S, d = trained
    x = d.prep(to_raw(observe(S))).astype(float)
    assert len(np.unique(x[:, 12])) == 2  # grasp flag stays two distinct levels
    # and the two levels straddle zero, as the normalised feature does
    assert x[:, 12].min() < 0 < x[:, 12].max()


def test_constant_features_get_zero_gain(trained):
    _, _, _, d = trained
    assert d.g[11] == 0  # goal z never varies


def test_emitted_source_is_integer_only(trained):
    body = trained[3].emit()["pocket.c"] + trained[3].emit()["pocket_net.c"]
    assert "float" not in body and "double" not in body and "malloc" not in body


@pytest.mark.skipif(cc() is None, reason="no C compiler")
def test_c_pipeline_is_bit_exact(trained):
    _, _, S, d = trained
    raw = to_raw(observe(S))
    rng = np.random.default_rng(0)
    extreme = rng.integers(-32768, 32767, (64, raw.shape[1])).astype(np.int16)
    rep = check_c(d, np.concatenate([raw[:200], extreme]))
    assert rep.ok, rep
