"""控制器：机械臂动作执行循环(50Hz)。

职责: 把「目标关节角」执行成一个可中断、可超时、带安全校验的闭环动作。
循环内每个控制步(约 20ms)至少检查一次:
    - 取消(cancellation token)   -> 立即 stop, 返回 CANCELLED
    - 超时(timeout)              -> stop, 返回 TIMEOUT
    - 关节收敛                    -> 返回 REACHED
    - NaN 检查                   -> stop, 返回 SAFETY
    - 工作空间/碰撞代理(末端低于桌面) -> stop, 返回 SAFETY

这保证了「外部同步 REST + 内部可取消控制循环」的架构。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import time
from typing import Protocol

import numpy as np

from .backend import RobotBackend
from .kinematics import RobotModel


class Pacing(Protocol):
    """执行节奏。realtime 按真实时间推进; fast 测试时尽快推进。"""

    def sleep(self, dt: float) -> None: ...


class RealtimePacing:
    """真实时间: 每个控制步 sleep(dt), 使墙钟时间 ≈ 仿真时间。"""

    def sleep(self, dt: float) -> None:
        time.sleep(dt)


class FastPacing:
    """尽快: 不 sleep, 仿真按固定 dt 快速推进(测试用)。"""

    def sleep(self, dt: float) -> None:
        pass


def make_pacing(mode: str) -> Pacing:
    if mode == "realtime":
        return RealtimePacing()
    if mode == "fast":
        return FastPacing()
    raise ValueError(f"未知 pacing 模式: {mode!r} (允许 realtime/fast)")


class MotionStatus(Enum):
    REACHED = "reached"      # 关节空间收敛
    TIMEOUT = "timeout"      # 超时未收敛
    CANCELLED = "cancelled"  # 被取消(用户打断)
    SAFETY = "safety"        # 安全违规(末端低于桌面 / NaN)


@dataclass
class MotionOutcome:
    status: MotionStatus
    residual_q_err: float    # 结束时关节空间最大误差 (rad)


def execute_joint_move(
    backend: RobotBackend,
    model: RobotModel,
    q_target: np.ndarray,
    token: "CancellationToken",
    dt: float,
    timeout_s: float,
    joint_tol_rad: float,
    pacing: Pacing,
    table_top_z_m: float,
) -> MotionOutcome:
    """执行一次关节动作, 返回 MotionOutcome。

    参数:
        backend: 机器人后端。
        model:   运动学模型(用于安全代理判定)。
        q_target: 目标关节角。
        token:   取消令牌(用户打断)。
        dt:      控制步长(秒)。
        timeout_s: 动作超时。
        joint_tol_rad: 关节收敛阈值。
        pacing:  执行节奏(realtime/fast)。
        table_top_z_m: 桌面高度, 用于安全代理判定。
    """
    q_target = np.clip(np.asarray(q_target, dtype=float), model.q_min(), model.q_max())
    backend.set_joint_targets(q_target)

    t0 = time.monotonic()
    while True:
        if token.cancelled:
            backend.stop()
            return MotionOutcome(MotionStatus.CANCELLED, _max_q_err(backend, model, q_target))

        elapsed = time.monotonic() - t0
        if elapsed > timeout_s:
            backend.stop()
            return MotionOutcome(MotionStatus.TIMEOUT, _max_q_err(backend, model, q_target))

        backend.step(dt)

        q = backend.get_joint_positions()
        if not np.all(np.isfinite(q)):
            backend.stop()
            return MotionOutcome(MotionStatus.SAFETY, float("nan"))

        # 安全代理: 末端不能低于桌面(碰撞代理)。Phase 2 起替换为真实碰撞检测。
        ee_pos = backend.get_end_effector_pose().position
        if ee_pos[2] < table_top_z_m - 0.005:
            backend.stop()
            return MotionOutcome(MotionStatus.SAFETY, _max_q_err(backend, model, q_target))

        err = float(np.max(np.abs(q - q_target)))
        if err < joint_tol_rad:
            # Phase 3(物理后端): 收敛瞬间的 ctrl 含 PD 残余项(容差内 err≠0),
            # 冻结它会导致停止后臂欠补偿下垂(释放物块时把物块压住, 见
            # mujoco_so101_backend._settle 注释)。stop() 把 ctrl 刷成本位形
            # 精确重力补偿, 臂停在真平衡点。运动学后端 stop() 只是状态快照,
            # 行为不变。
            backend.stop()
            return MotionOutcome(MotionStatus.REACHED, err)

        pacing.sleep(dt)


def _max_q_err(backend: RobotBackend, model: RobotModel, q_target: np.ndarray) -> float:
    q = backend.get_joint_positions()
    return float(np.max(np.abs(q - q_target)))
