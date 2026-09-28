import numpy as np

from pocketpolicy.sim import MAX_STEPS, VMAX, DT, PickPlace, observe
from pocketpolicy.teacher import ScriptedTeacher, rollout


def test_reset_is_seeded_and_in_workspace():
    env = PickPlace()
    a, b = env.reset(16, 3), env.reset(16, 3)
    assert np.array_equal(a.obj, b.obj) and np.array_equal(a.goal, b.goal)
    r = np.hypot(a.obj[:, 0], a.obj[:, 1])
    assert (r > 0.14).all() and (r < 0.28).all()
    assert (np.linalg.norm(a.goal[:, :2] - a.obj[:, :2], axis=1) >= 0.06).all()


def test_speed_limit():
    env = PickPlace(noise=0)
    s = env.reset(4, 0)
    s2 = env.step(s, s.q + 5.0)
    assert np.all(np.abs(s2.q - s.q) <= VMAX * DT + 1e-12)


def test_doing_nothing_never_succeeds():
    env = PickPlace()
    s = rollout(env, lambda s: s.q[:, None].copy(), 8, 1)
    assert s.t == MAX_STEPS and not s.success.any()


def test_teacher_solves_the_task():
    env = PickPlace()
    T = ScriptedTeacher(env)
    s = rollout(env, lambda s: T.act(s)[:, None], 64, 7)
    assert s.success.mean() >= 0.95


def test_teacher_chunks_are_consistent_with_single_steps():
    env = PickPlace()
    T = ScriptedTeacher(env)
    s = env.reset(8, 2)
    c = T.chunk(s, 5)
    assert c.shape == (8, 5, 6)
    assert np.allclose(c[:, 0], T.act(s))
    assert observe(s).shape == (8, 13)
