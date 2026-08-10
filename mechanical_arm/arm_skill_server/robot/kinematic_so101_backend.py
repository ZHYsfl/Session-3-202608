"""KinematicSO101Backend：纯 Python 运动学仿真后端(Phase 1, Windows 原生)。

行为:
    - 关节: 梯形速度斜坡逼近目标(模拟舵机), 带限速/限位/防过冲。
    - 末端: FK 实时计算, 末端速度 = 雅可比 × 关节角速度。
    - 物块: 场景 ground truth(无视觉)。
    - 夹取: close 时若某物块中心距末端 < holding_proximity 则判定夹住,
      夹住后物块跟随末端; open 时物块落回桌面。
    - 安全: 末端低于桌面(碰撞代理) -> 安全停止, 不再运动。

这是「仿真 ground truth」实现, 不是真机动力学。真实动力学/物理交给
IsaacLab 或真机, 本类只保证 skill/API 全链路可运行、可测试。
"""
from __future__ import annotations

import numpy as np

from ..config import ControlConfig
from ..simulation.environment import Scene
from ..simulation.cameras import make_mock_frame
from .backend import BlockState, CameraFrame, EEPose, RobotBackend
from .kinematics import RobotModel, fk, numeric_jacobian
from .trajectory import next_ramped_velocity

# 安全代理: 末端允许低于桌面多少
_TABLE_MARGIN_M = 0.005


class KinematicSO101Backend(RobotBackend):
    def __init__(self, model: RobotModel, sim_cfg: dict, control: ControlConfig) -> None:
        self.model = model
        self.control = control
        self.scene = Scene(sim_cfg)

        self.q = model.q_home().copy()
        self.q_target = self.q.copy()
        self.q_dot = np.zeros(model.dof)
        self.gripper_open = True
        self._holding_block = None      # 当前夹住的 Block
        self._hold_offset = np.zeros(3) # 夹住瞬间 物块 - 末端 的相对偏移
        self.safety_stop_reason: str | None = None
        self._gripper_fault = False     # 测试用: 模拟夹爪卡死

        self.reset("no_blocks")

    # ------------------------------------------------------------------
    # 状态读取
    # ------------------------------------------------------------------
    def get_joint_positions(self) -> np.ndarray:
        return self.q.copy()

    def get_joint_velocities(self) -> np.ndarray:
        return self.q_dot.copy()

    def get_end_effector_pose(self) -> EEPose:
        p, R = fk(self.model, self.q)
        return EEPose(position=p, rotation=R)

    def get_end_effector_velocity(self) -> np.ndarray:
        J = numeric_jacobian(self.model, self.q)
        return J @ self.q_dot

    # ------------------------------------------------------------------
    # 运动指令
    # ------------------------------------------------------------------
    def set_joint_targets(self, q: np.ndarray) -> None:
        self.q_target = np.clip(np.asarray(q, dtype=float), self.model.q_min(), self.model.q_max())
        self.safety_stop_reason = None

    def step(self, dt: float) -> None:
        # 已安全停止: 冻结关节
        if self.safety_stop_reason is not None:
            self.q_dot[:] = 0.0
            return

        vmax = self.model.vmax()
        accel = self.model.accel()
        err = self.q_target - self.q
        new_dot = np.zeros(self.model.dof)
        for i in range(self.model.dof):
            new_dot[i] = next_ramped_velocity(
                error=err[i], v_prev=self.q_dot[i], dt=dt,
                vmax=float(vmax[i]), accel=float(accel[i]),
            )
        self.q_dot = new_dot
        self.q = self.q + self.q_dot * dt
        self.q = np.clip(self.q, self.model.q_min(), self.model.q_max())

        # 安全代理: 末端不能低于桌面
        ee = self.get_end_effector_pose().position
        if ee[2] < self.model.table_top_z_m - _TABLE_MARGIN_M:
            self.safety_stop_reason = "末端低于桌面(碰撞代理)"
            self.q_dot[:] = 0.0
            return

        # 夹住的物块跟随末端
        if self._holding_block is not None:
            self._move_held_block(ee)

    def stop(self) -> None:
        self.q_target = self.q.copy()
        self.q_dot[:] = 0.0

    # ------------------------------------------------------------------
    # 夹爪
    # ------------------------------------------------------------------
    def open_gripper(self) -> None:
        if self._gripper_fault:
            return                            # 故障: 夹爪卡住, 无法张开
        self.gripper_open = True
        if self._holding_block is not None:
            self._drop_held_block()
        self._holding_block = None

    def close_gripper(self) -> None:
        if self._gripper_fault:
            return
        self.gripper_open = False
        ee = self.get_end_effector_pose().position
        best = None
        best_d = float("inf")
        for b in self.scene.blocks:
            d = float(np.linalg.norm(ee - np.array([b.x, b.y, b.z])))
            if d < best_d:
                best, best_d = b, d
        if best is not None and best_d <= self.control.holding_proximity_m:
            self._holding_block = best
            self._hold_offset = np.array([best.x, best.y, best.z]) - ee

    def is_holding_object(self) -> bool:
        return self._holding_block is not None

    # ------------------------------------------------------------------
    # 感知
    # ------------------------------------------------------------------
    def get_block_states(self) -> list[BlockState]:
        return self.scene.get_block_states()

    def get_camera_frame(self, camera_id: str) -> CameraFrame:
        cam = self.scene.cameras.get(camera_id)
        if cam is None:
            raise KeyError(f"未知相机: {camera_id!r}, 可选 {sorted(self.scene.cameras)}")
        return make_mock_frame(cam, self.get_block_states())

    # ------------------------------------------------------------------
    # 场景
    # ------------------------------------------------------------------
    def reset(self, scene_preset: str | None = None) -> None:
        """复位: 回 home、张开夹爪、按预设重建物块。"""
        self.q = self.model.q_home().copy()
        self.q_target = self.q.copy()
        self.q_dot[:] = 0.0
        self.gripper_open = True
        self._holding_block = None
        self.safety_stop_reason = None
        self.scene.reset(scene_preset)

    # ---- 测试钩子(仅测试用) -------------------------------------------
    def set_gripper_fault(self, stuck: bool) -> None:
        """模拟夹爪卡死: 开/闭夹爪都不生效。仅用于测试失败路径。"""
        self._gripper_fault = stuck

    # ------------------------------------------------------------------
    # 内部: 物块跟随 / 释放
    # ------------------------------------------------------------------
    def _move_held_block(self, ee: np.ndarray) -> None:
        idx = self.scene.find_held_index(self._holding_block)
        if idx < 0:
            return
        pos = ee + self._hold_offset
        b = self._holding_block
        moved = type(b)(color=b.color, x=float(pos[0]), y=float(pos[1]),
                        z=float(pos[2]), yaw=b.yaw)
        self.scene.replace_block(idx, moved)
        self._holding_block = moved

    def _drop_held_block(self) -> None:
        idx = self.scene.find_held_index(self._holding_block)
        if idx < 0:
            return
        b = self._holding_block
        dropped = type(b)(color=b.color, x=b.x, y=b.y,
                          z=self.model.table_top_z_m + self.scene.block_half_height_m,
                          yaw=b.yaw)
        self.scene.replace_block(idx, dropped)
