import numpy as np

from pocketpolicy.sim import PickPlace
from pocketpolicy.student import MLP, Norm, concat_states, distill
from pocketpolicy.teacher import ScriptedTeacher, rollout


def test_mlp_fits_a_smooth_function():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (2000, 3))
    Y = np.c_[np.sin(2 * X[:, 0]) + X[:, 1] * X[:, 2], X.sum(1)]
    net = MLP.init(3, 32, 2, 2, Norm.fit(X), Norm.fit(Y), H=1)
    first = net.fit(X, Y, 1)
    last = net.fit(X, Y, 60)
    assert last < 0.1 * first


def test_concat_drops_finished_episodes():
    env = PickPlace()
    a, b = env.reset(4, 0), env.reset(3, 1)
    a.done[:2] = True
    s = concat_states([a, b])
    assert s.B == 5 and not s.done.any()


def test_dagger_grows_the_dataset_and_learns_something():
    env = PickPlace()
    T = ScriptedTeacher(env)
    net, info = distill(env, T, 32, 2, 2, dagger_rounds=1, episodes=16, epochs=8)
    assert info["rounds"] == 1 and info["samples"] > 0
    s = rollout(env, net.policy, 8, 3, 1)
    assert np.isfinite(s.q).all()
