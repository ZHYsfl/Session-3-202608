"""RobotBackend 抽象：skills/api 与机器人实现之间的唯一接口。

Sim → Real 只需替换实现类：
    KinematicSO101Backend  纯 Python 运动学仿真(Windows 原生, Phase 1)
    IsaacSO101Backend      IsaacLab/IsaacSim(WSL2/Ubuntu, 占位 stub)
    RealSO101Backend       真机(占位 stub)

所有方法都以 base frame 为参照, 长度单位 meter, 角度 radian。
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


class EEPose(NamedTuple):
    """末端执行器(tool tip)位姿, base frame。"""

    position: np.ndarray          # (3,) xyz, meter
    rotation: np.ndarray          # (3,3) 旋转矩阵


@dataclass(frozen=True)
class BlockState:
    """物块状态 (base frame)。Phase 1 为仿真 ground truth; 真机由视觉模块填充。"""

    color: str
    x: float
    y: float
    z: float
    yaw: float


@dataclass(frozen=True)
class CameraFrame:
    """相机帧。Phase 1 为占位(ground truth 摘要); Phase 2 起携带图像。"""

    camera_id: str
    visible_blocks: tuple[BlockState, ...]   # 该视角“可见”的物块


class RobotBackend(ABC):
    """机械臂后端接口。注意: 不要求线程安全, 由上层 GlobalState 保证单动作串行。"""

    # 抓取时是否把腕部自旋固定到某个角度(rad), None = 不固定。
    # 物理后端(Phase 3)需要让夹爪沿世界 Y 侧向张开以避开物块 -> 固定到 π/2。
    # 运动学后端(Phase 1/2)无真实夹爪几何, 保持 None, 行为完全不变。
    grasp_wrist_roll: float | None = None

    # ---- 状态读取 -------------------------------------------------------
    @abstractmethod
    def get_joint_positions(self) -> np.ndarray:
        """返回 5 个关节角 (rad)。"""

    @abstractmethod
    def get_joint_velocities(self) -> np.ndarray:
        """返回 5 个关节角速度 (rad/s)。"""

    @abstractmethod
    def get_end_effector_pose(self) -> EEPose:
        """返回末端位置 + 姿态。"""

    @abstractmethod
    def get_end_effector_velocity(self) -> np.ndarray:
        """返回末端速度向量 (m/s), base frame。"""

    # ---- 运动指令 -------------------------------------------------------
    @abstractmethod
    def set_joint_targets(self, q: np.ndarray) -> None:
        """下发关节目标。具体实现的内部会带限速/限位地逼近。"""

    @abstractmethod
    def step(self, dt: float) -> None:
        """推进一个控制步 dt 秒。"""

    @abstractmethod
    def stop(self) -> None:
        """紧急停止: 冻结关节, 清除目标。"""

    # ---- 夹爪 ------------------------------------------------------------
    @abstractmethod
    def open_gripper(self) -> None:
        """张开夹爪。"""

    @abstractmethod
    def close_gripper(self) -> None:
        """闭合夹爪。"""

    @abstractmethod
    def is_holding_object(self) -> bool:
        """是否真的夹着物块(交叉验证后的结论)。Phase 1 用仿真 ground truth。"""

    # ---- 感知 ------------------------------------------------------------
    @abstractmethod
    def get_block_states(self) -> list[BlockState]:
        """场景中存在的物块。Phase 1: ground truth; 真机: 视觉模块输出。"""

    @abstractmethod
    def get_camera_frame(self, camera_id: str) -> CameraFrame:
        """取相机帧。camera_id ∈ {"camera1", "camera2"}。"""

    # ---- 场景 ------------------------------------------------------------
    @abstractmethod
    def reset(self, scene_preset: str | None = None) -> None:
        """复位: 回到 home。仿真实现额外按预设重建场景物块; 真机实现仅回 home。"""
