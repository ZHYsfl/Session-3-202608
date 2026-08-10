"""全局运行时状态(单例)。

持有: 机器人后端、队列后端、控制配置、运动学模型。
并保证「同一时刻只有一个动作在执行」(单机械臂事实), 同时登记“当前活动动作”
的取消令牌, 使未来的取消信号(测试钩子 / 待批准的取消接口)能中断正在执行的动作。

Phase 1 没有公网取消接口。取消信号通路已经设计好:
    state.request_cancel()  -> 取消当前活动动作 -> 控制循环约 20ms 内 stop。
"""
from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Optional

from ..config import ControlConfig, VisionConfig
from ..perception.base import PerceptionBackend
from ..perception.camera import CameraProvider
from ..queue.backend import QueueBackend
from ..robot.backend import RobotBackend
from ..robot.kinematics import RobotModel
from .cancellation import CancellationToken


class _GlobalState:
    """线程安全的全局状态。方法名即用法, 不做多余抽象。"""

    def __init__(self) -> None:
        self.backend: Optional[RobotBackend] = None
        self.queue: Optional[QueueBackend] = None
        self.control: Optional[ControlConfig] = None
        self.model: Optional[RobotModel] = None
        self.vision: Optional[PerceptionBackend] = None      # Phase 2: HSV/Tiny 检测器
        self.camera: Optional[CameraProvider] = None         # Phase 2: 取帧 + 相机标定
        self.vision_cfg: Optional[VisionConfig] = None       # Phase 2: 视觉阈值/几何常量
        # Phase 6: 最近一次抓取成功的物块颜色(camera2 释放交叉验证用; 未抓过为 None)
        self.last_grabbed_color: Optional[str] = None

        self._action_lock = threading.Lock()   # 单机械臂: 同一时刻只允许一个动作
        self._active_lock = threading.Lock()
        self._active_token: Optional[CancellationToken] = None

    def configure(self, backend: RobotBackend, queue: QueueBackend,
                  control: ControlConfig, model: RobotModel,
                  vision: Optional[PerceptionBackend] = None,
                  camera: Optional[CameraProvider] = None,
                  vision_cfg: Optional[VisionConfig] = None) -> None:
        """初始化(启动时调用一次)。vision/camera/vision_cfg 为 Phase 2 可选注入。"""
        self.backend = backend
        self.queue = queue
        self.control = control
        self.model = model
        self.vision = vision
        self.camera = camera
        self.vision_cfg = vision_cfg

    # ---- 动作串行化 + 取消登记 -----------------------------------------
    def begin_action(self) -> CancellationToken:
        """获取新动作的执行权(阻塞直到上一个动作结束), 返回新动作的取消令牌。"""
        self._action_lock.acquire()
        token = CancellationToken()
        with self._active_lock:
            self._active_token = token
        return token

    def end_action(self, token: CancellationToken) -> None:
        """动作结束: 若该令牌仍是活动令牌则清除, 并释放执行权。"""
        with self._active_lock:
            if self._active_token is token:
                self._active_token = None
        self._action_lock.release()

    def request_cancel(self) -> bool:
        """取消当前正在执行的动作。没有活动动作时返回 False。"""
        with self._active_lock:
            token = self._active_token
        if token is None:
            return False
        token.cancel()
        return True

    def action_in_progress(self) -> bool:
        """是否正有动作在执行(技能线程持有执行权)。

        GUI/仿真主循环用: 动作执行期间由技能线程的控制器按 50Hz 推进物理,
        主循环必须**跳过**自己的 step() 以免双倍推进; 空闲时主循环负责推进
        (让释放的物块在重力下落下等)。
        """
        with self._active_lock:
            return self._active_token is not None

    # ---- 便捷属性(调用前必须 configure) --------------------------------
    def require_backend(self) -> RobotBackend:
        if self.backend is None:
            raise RuntimeError("GlobalState 尚未 configure")
        return self.backend

    def require_model(self) -> RobotModel:
        if self.model is None:
            raise RuntimeError("GlobalState 尚未 configure")
        return self.model

    def require_control(self) -> ControlConfig:
        if self.control is None:
            raise RuntimeError("GlobalState 尚未 configure")
        return self.control

    def require_queue(self) -> QueueBackend:
        if self.queue is None:
            raise RuntimeError("GlobalState 尚未 configure")
        return self.queue

    def require_vision(self) -> PerceptionBackend:
        """hsv 感知模式需要。Ground truth 模式不需要。"""
        if self.vision is None:
            raise RuntimeError("GlobalState 未配置 vision(perception_mode 需要)")
        return self.vision

    def require_camera(self) -> CameraProvider:
        """hsv 感知模式需要。"""
        if self.camera is None:
            raise RuntimeError("GlobalState 未配置 camera(perception_mode 需要)")
        return self.camera

    def require_vision_cfg(self) -> VisionConfig:
        """hsv 感知模式需要(读取桌面平面 z / 物块半高等几何常量)。"""
        if self.vision_cfg is None:
            raise RuntimeError("GlobalState 未配置 vision_cfg(perception_mode 需要)")
        return self.vision_cfg


# 进程内单例。run_api.py / 测试在启动时 configure。
state = _GlobalState()
