"""动作执行辅助: 把「目标笛卡尔点」变成一次可取消的关节动作。

流程: 目标点 -> 工作空间投影 -> IK -> execute_joint_move(50Hz 闭环)。
被 move_to_coordinates 与 grab_the_block 复用。
"""
from __future__ import annotations

import numpy as np

from ..config import ControlConfig
from ..robot.backend import RobotBackend
from ..robot.controller import Pacing, execute_joint_move
from ..robot.controller import MotionStatus
from ..robot.kinematics import RobotModel, fk, pose_distance
from ..runtime.cancellation import CancellationToken
from ._common import UnreachableError, reachable_projection, solve_ik_robust


class GraspFailed(Exception):
    """抓取序列中断(移动未到/取消/未夹住)。"""


def move_to_point(
    target_xyz: np.ndarray | tuple[float, float, float],
    backend: RobotBackend,
    model: RobotModel,
    control: ControlConfig,
    token: CancellationToken,
    pacing: Pacing,
    force_wrist_roll: float | None = None,
) -> None:
    """把末端移动到 target_xyz。目标不可达则走到最近可达点。

    force_wrist_roll: 非 None 时把腕部自旋(q4)固定到该角度(rad)。
      物理后端抓取用它把夹爪转到侧向张开(见 backend.grasp_wrist_roll)。
      q4 绕工具自身轴旋转、不移动 tool tip(工具挂在旋转轴上), 强制后 FK
      位置不变, 故这是纯姿态约束。运动学后端传 None -> 行为完全不变。

    移动完成后不在这里判定“是否到达”——由调用方用真实测量误差判定
    (move_to_coordinates 需要协议字符串; grab 只要 REACHED)。
    中断(取消/超时/安全) -> 抛 GraspFailed。
    """
    target = np.asarray(target_xyz, dtype=float)
    proj = reachable_projection(model, target)
    q0 = backend.get_joint_positions()
    q_target = solve_ik_robust(model, proj, q0)
    if q_target is None:
        # IK 无解/目标不可达: 臂保持原位不移动, 误差 = 当前真实末端到目标点的
        # 欧氏距离(真实测量)。禁止用 max(dist,1e-3) 人为下界冒充真实误差
        # (那会让「在可达半径内但 IK 不可行」的目标报出 0.0010 的假误差)。
        ee = backend.get_end_effector_pose().position
        raise UnreachableError(pose_distance(ee, target))
    if force_wrist_roll is not None:
        q_target = q_target.copy()
        q_target[4] = float(np.clip(force_wrist_roll, model.q_min()[4], model.q_max()[4]))
        # 防御性复核: q4 不移动末端, 位置应与 IK 解一致。
        p_check, _ = fk(model, q_target)
        if np.linalg.norm(p_check - proj) > 1e-2:
            ee = backend.get_end_effector_pose().position
            raise UnreachableError(pose_distance(ee, target))

    outcome = execute_joint_move(
        backend=backend, model=model, q_target=q_target, token=token,
        dt=control.dt, timeout_s=control.move_timeout_s,
        joint_tol_rad=control.joint_convergence_tol_rad,
        pacing=pacing, table_top_z_m=model.table_top_z_m,
    )
    if outcome.status is not MotionStatus.REACHED:
        raise GraspFailed(f"移动未完成: {outcome.status.value}")
