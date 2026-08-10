"""取消机制测试: 令牌语义 + 控制循环可在约一个控制步内响应取消并停机制动。"""
from __future__ import annotations

import threading
import time

import numpy as np

from ...robot.controller import (
    FastPacing,
    MotionStatus,
    execute_joint_move,
)
from ...runtime.cancellation import CancellationToken
from ...runtime.state import state


def test_token_semantics() -> None:
    t = CancellationToken()
    assert not t.cancelled
    t.cancel()
    assert t.cancelled
    t.cancel()   # 幂等
    assert t.cancelled


def test_execute_joint_move_precancelled(env) -> None:
    """令牌预先取消: 循环应立即返回 CANCELLED 并停机制动(关节不再继续运动)。"""
    backend = env["backend"]
    model = env["model"]
    control = env["control"]
    backend.reset("all")
    q0 = backend.get_joint_positions()
    target = np.clip(model.q_home() + 0.6, model.q_min(), model.q_max())

    token = CancellationToken()
    token.cancel()
    outcome = execute_joint_move(
        backend=backend, model=model, q_target=target, token=token,
        dt=control.dt, timeout_s=1.0, joint_tol_rad=control.joint_convergence_tol_rad,
        pacing=FastPacing(), table_top_z_m=model.table_top_z_m,
    )
    assert outcome.status is MotionStatus.CANCELLED
    # 机械臂已被 stop: 关节位置几乎未动
    assert np.allclose(backend.get_joint_positions(), q0, atol=1e-9)


def test_execute_joint_move_cancel_midway(env) -> None:
    """运动中途取消: 控制循环应在约一个控制步(20ms)内感知并停止。"""
    backend = env["backend"]
    model = env["model"]
    control = env["control"]
    backend.reset("all")
    target = np.clip(model.q_home() + 1.2, model.q_min(), model.q_max())

    token = CancellationToken()

    def cancel_later() -> None:
        time.sleep(0.01)
        token.cancel()

    threading.Thread(target=cancel_later, daemon=True).start()

    # 用慢节奏(每步真实睡 20ms)让“墙钟”超过“仿真时间”, 便于中途取消
    class SlowPacing:
        def sleep(self, dt: float) -> None:
            time.sleep(0.02)

    outcome = execute_joint_move(
        backend=backend, model=model, q_target=target, token=token,
        dt=control.dt, timeout_s=10.0, joint_tol_rad=control.joint_convergence_tol_rad,
        pacing=SlowPacing(), table_top_z_m=model.table_top_z_m,
    )
    assert outcome.status is MotionStatus.CANCELLED


def test_state_request_cancel_plumbing(env) -> None:
    """request_cancel 只能取消“当前活动动作”; 无活动动作时返回 False。"""
    assert state.request_cancel() is False

    token = state.begin_action()
    try:
        assert state.request_cancel() is True
        assert token.cancelled
    finally:
        state.end_action(token)

    assert state.request_cancel() is False
