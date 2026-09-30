import numpy as np
import pytest

from pocketpolicy.realworld import (DEG, PRESETS, CommandRing, Perturb, _Delay, labels,
                                    privileged, rollout_real, sample)
from pocketpolicy.sim import PickPlace
from pocketpolicy.student import distill
from pocketpolicy.teacher import ScriptedTeacher, rollout


@pytest.fixture(scope="module")
def env_teacher():
    env = PickPlace()
    return env, ScriptedTeacher(env)


def test_ideal_preset_is_the_original_rollout_bit_for_bit(env_teacher):
    env, T = env_teacher
    for policy, k in ((lambda s: T.chunk(s, 1), 1), (lambda s: T.chunk(s, 4), 2)):
        a = rollout(env, policy, 12, 5, k)
        b = rollout_real(env, policy, 12, 5, k, PRESETS["ideal"])
        assert a.t == b.t
        assert np.array_equal(a.q, b.q) and np.array_equal(a.obj, b.obj)
        assert np.array_equal(a.success, b.success)


def test_grasp_radius_default_is_unchanged():
    from pocketpolicy.sim import GRASP_RADIUS
    assert PickPlace().grasp_radius == GRASP_RADIUS == PRESETS["ideal"].grasp_radius


def test_delay_reads_the_past_per_episode(env_teacher):
    env, _ = env_teacher
    s = env.reset(3, 0)
    d = _Delay(np.array([0, 1, 3]))
    qs = []
    for t in range(6):
        s.q = np.full((3, 6), float(t))
        qs.append(s.q)
        d.push(s)
        q, _, _ = d.read()
        assert q[:, 0].tolist() == [t, max(t - 1, 0), max(t - 3, 0)]


def test_ring_returns_the_command_behind_the_reading():
    q0 = np.zeros((2, 6))
    r = CommandRing(q0, [0, 2], max_corr=1e9)
    for t in range(1, 6):
        r.push(np.full((2, 6), float(t)))
        base = r.base()[:, 0]
        # after pushing cmd_t, the next reading shows cmd_t (lat 0) or cmd_{t-2}
        assert base.tolist() == [t, max(t - 2, 0)]


def test_correction_clamps_and_leaves_the_gripper_alone():
    r = CommandRing(np.zeros((1, 6)), 0, max_corr=0.1)
    c = r.correction(np.array([[1.0, -1.0, 0.05, 0, 0, 0.7]]))
    assert np.allclose(c, [[-0.1, 0.1, -0.05, 0, 0, 0]])


def test_cmd_relative_cancels_constant_sag(env_teacher):
    env, T = env_teacher
    pol = lambda s: T.chunk(s, 1)
    sag = Perturb(sag=1 * DEG)
    naive = rollout_real(env, pol, 24, 3, 1, sag).success.mean()
    fixed = rollout_real(env, pol, 24, 3, 1, sag, cmd_relative=True).success.mean()
    ideal = rollout(env, pol, 24, 3, 1).success.mean()
    assert naive < 0.2 and fixed >= ideal - 0.1


def test_sample_stays_within_scaled_preset():
    p = PRESETS["realistic"]
    s = sample(p, 500, np.random.default_rng(0), scale=1.25)
    assert (s.det_noise <= 1.25 * p.det_noise + 1e-12).all() and s.det_noise.min() >= 0
    assert set(np.unique(s.latency)) <= {0, 1, 2, 3}
    assert (s.grasp_radius <= PRESETS["ideal"].grasp_radius).all()
    assert s.grasp_radius.min() >= 0.02 - 1.25 * (0.02 - p.grasp_radius) - 1e-12


def test_per_episode_perturbations_run(env_teacher):
    env, T = env_teacher
    p = sample(PRESETS["realistic"], 8, np.random.default_rng(1))
    s, trace = rollout_real(env, privileged(T, 2), 8, 4, 1, p, cmd_relative=True,
                            record=True, sees_truth=True)
    X, Y = labels(T, trace, 2)
    assert X.shape[1] == 13 and Y.shape == (len(X), 12) and np.isfinite(Y).all()
    X2, Y2 = labels(T, trace, 2, "observed")
    assert np.array_equal(X, X2) and not np.allclose(Y, Y2)


def test_privileged_labels_match_plain_ones_without_errors(env_teacher):
    env, T = env_teacher
    from pocketpolicy.student import concat_states, label
    _, trace = rollout_real(env, privileged(T, 1), 6, 2, 1, record=True, sees_truth=True)
    X0, Y0 = label(T, concat_states(trace.states), 3)
    for view in ("now", "read", "observed"):
        X, Y = labels(T, trace, 3, view)
        assert np.array_equal(X, X0) and np.allclose(Y, Y0)


def test_robust_distill_is_off_by_default_and_runs(env_teacher):
    env, T = env_teacher
    a, _ = distill(env, T, 8, 1, 1, 0, episodes=4, epochs=1)
    b, _ = distill(env, T, 8, 1, 1, 0, episodes=4, epochs=1, robust=None)
    assert all(np.array_equal(x[0], y[0]) for x, y in zip(a.layers, b.layers))
    net, info = distill(env, T, 8, 1, 2, 1, episodes=4, epochs=1, robust=PRESETS["realistic"])
    assert info["robust"] and info["teacher_view"] == "now"
    assert np.isfinite(net.layers[0][0]).all()

