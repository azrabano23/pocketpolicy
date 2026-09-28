import numpy as np

from pocketpolicy.robot import ARM, JOINTS, Arm


def test_chain_matches_urdf():
    a = Arm()
    assert tuple(j.name for j in a.chain if j.type == "revolute") == ARM
    assert len(a.lower) == len(JOINTS) == 6
    # limits read verbatim from so101_new_calib.urdf
    assert np.isclose(a.upper[0], 1.91986) and np.isclose(a.lower[5], -0.174533)


def test_zero_pose_reaches_forward():
    p = Arm().tcp(np.zeros(6))[0]
    assert 0.35 < p[0] < 0.45 and abs(p[1]) < 1e-3 and 0.15 < p[2] < 0.30


def test_pan_rotates_about_vertical():
    a = Arm()
    q = np.zeros((2, 6))
    q[1, 0] = 0.7
    p = a.tcp(q)
    assert np.isclose(np.hypot(*p[0, :2]), np.hypot(*p[1, :2]), atol=2e-3)
    assert np.isclose(p[0, 2], p[1, 2])


def test_jacobian_matches_finite_motion():
    a = Arm()
    q = np.array([[0.2, -0.8, 0.6, 1.0, 0.0, 0.5]])
    dq = np.array([0.01, -0.01, 0.02, 0.0, 0.0])
    pred = a.tcp(q)[0] + a.jacobian(q)[0] @ dq
    q2 = q.copy()
    q2[0, :5] += dq
    assert np.linalg.norm(pred - a.tcp(q2)[0]) < 5e-5
