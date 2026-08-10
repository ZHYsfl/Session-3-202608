"""运动学单元测试: FK 解析校验 + 雅可比解析校验 + IK 往返 + 限位/可达。"""
from __future__ import annotations

import numpy as np
import pytest

from ...robot.kinematics import (
    fk,
    is_reachable,
    numeric_jacobian,
    solve_ik,
)
from ...robot.kinematics import RobotModel, rot_matrix


@pytest.fixture(scope="module")
def model() -> RobotModel:
    from ...config import load_robot_model
    return load_robot_model()


def test_fk_identity_config(model: RobotModel) -> None:
    """全零关节角时, 末端应在 (0.36, 0, 0.06)。

    手算(见 robot.yaml 的 link 链; Phase 3 修正后 wrist_roll link 朝下 -0.06):
        pan(0)+[0,0,0.12] -> lift(0)+[0.11,0,0] -> elbow(0)+[0.14,0,0]
        -> wrist_flex(0)+[0.11,0,0] -> roll(0)+[0,0,-0.06]
    """
    p, _ = fk(model, [0.0] * model.dof)
    assert np.allclose(p, [0.36, 0.0, 0.06], atol=1e-9)


def test_fk_home_pose_is_sane(model: RobotModel) -> None:
    """home 姿态的末端应高于桌面且有限。"""
    p, R = fk(model, model.q_home())
    assert np.all(np.isfinite(p))
    assert np.all(np.isfinite(R))
    assert p[2] > 0.02, "home 末端高度应明显高于桌面"
    assert np.linalg.norm(p) <= model.max_reach_m + 1e-6


def test_fk_rot_z_about_world(model: RobotModel) -> None:
    """shoulder_pan(绕 z)旋转应只改变末端水平朝向, 不改变高度。"""
    p0, _ = fk(model, [0.0] * model.dof)
    p90, _ = fk(model, [np.pi / 2, 0.0, 0.0, 0.0, 0.0])
    assert abs(p90[2] - p0[2]) < 1e-9
    assert abs(np.linalg.norm(p90[:2]) - np.linalg.norm(p0[:2])) < 1e-9


def test_numeric_jacobian_matches_analytic(model: RobotModel) -> None:
    """全零构型下雅可比列可解析校验(旋转轴 × 半径向量)。

    pan 列 = ẑ × p = (0, 0.36, 0)
    lift 列 = ŷ × (p - 轴心[0,0,0.12]) = (-0.06, 0, -0.36)
    """
    q = np.zeros(model.dof)
    J = numeric_jacobian(model, q)
    pan_axis_origin = np.array([0.0, 0.0, 0.0])
    p, _ = fk(model, q)

    pan_col = np.cross([0, 0, 1], p)
    lift_col = np.cross([0, 1, 0], p - pan_axis_origin - np.array([0, 0, 0.12]))

    assert np.allclose(J[:, 0], pan_col, atol=1e-5)
    assert np.allclose(J[:, 1], lift_col, atol=1e-5)


def test_ik_roundtrip_reachable_points(model: RobotModel) -> None:
    """IK->FK 往返: 对若干可达目标, 逆解后正解应回到目标。"""
    targets = [
        (0.15, 0.05, 0.20),
        (0.25, -0.10, 0.12),
        (0.18, 0.00, 0.03),   # 桌面附近的低目标(物块高度)
        (0.05, 0.20, 0.25),
        (-0.20, 0.05, 0.18),
    ]
    for t in targets:
        q = solve_ik(model, t, q0=model.q_home(), tol=1e-3, max_iter=200)
        assert q is not None, f"目标 {t} 应可解"
        p, _ = fk(model, q)
        assert np.linalg.norm(p - np.array(t)) < 0.01, f"往返误差过大: {t} -> {p}"


def test_ik_unreachable_returns_none(model: RobotModel) -> None:
    """远超工作空间的目标应返回 None。"""
    q = solve_ik(model, (0.90, 0.0, 0.0), q0=model.q_home(), max_iter=100)
    assert q is None


def test_ik_respects_joint_limits(model: RobotModel) -> None:
    """IK 解必须在关节限位内。"""
    for t in [(0.15, 0.05, 0.20), (0.18, 0.0, 0.03)]:
        q = solve_ik(model, t, q0=model.q_home(), max_iter=200)
        assert q is not None
        assert np.all(q >= model.q_min() - 1e-9)
        assert np.all(q <= model.q_max() + 1e-9)


def test_reachable_precheck(model: RobotModel) -> None:
    assert is_reachable(model, (0.2, 0.0, 0.1))
    assert not is_reachable(model, (1.0, 0.0, 0.0))


def test_rot_matrix_orthonormal() -> None:
    """旋转矩阵必须是正交、行列式为 +1。"""
    for axis in ("z", "y"):
        for ang in (0.0, 0.7, -1.3, np.pi):
            R = rot_matrix(axis, ang)
            assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
            assert abs(np.linalg.det(R) - 1.0) < 1e-12
