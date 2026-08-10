"""测试公共 fixture: 加载真实配置 + 构造 KinematicSO101Backend + 配置全局状态。

perception_mode 说明:
    控制配置文件里默认是 "hsv"(Phase 2 交付状态)。但 Phase 1 测试的抓取语义
    (如 test_grab_unreachable_block_fails: 场景里有不可达物块 -> 应报"夹取失败")
    依赖 ground truth 定位。为避免改测试来迁就新模式, env fixture 显式钉死
    "ground_truth", 保持 Phase 1 原语义; Phase 2 视觉测试用 env_hsv 切换。
"""
from __future__ import annotations

import pytest

from ..config import ControlConfig, load_all, load_vision_config
from ..queue.in_memory_queue import InMemoryQueue
from ..robot.kinematic_so101_backend import KinematicSO101Backend
from ..runtime.state import state  # 单例实例(不是 runtime 包内的 state 模块)


@pytest.fixture
def env():
    """一次性装载: 真实配置 + 仿真后端 + 进程内队列, 并配置全局状态。

    统一用 fast pacing, 测试不做真实时间等待。
    """
    model, control, sim_cfg = load_all()
    control = ControlConfig(**{**control.__dict__, "pacing": "fast",
                               "perception_mode": "ground_truth"})
    backend = KinematicSO101Backend(model, sim_cfg, control)
    queue = InMemoryQueue()
    state.configure(backend=backend, queue=queue, control=control, model=model)
    return {
        "model": model,
        "control": control,
        "sim_cfg": sim_cfg,
        "backend": backend,
        "queue": queue,
        "vision": None,
        "camera": None,
        "vision_cfg": None,
    }


@pytest.fixture
def app(env):
    """FastAPI 应用(用同一份 backend/queue/control/model)。"""
    from ..api.server import create_app

    return create_app(env["backend"], env["queue"], env["control"], env["model"])


@pytest.fixture
def vision(env):
    """构造视觉组件(合成相机 + HSV 检测器), 不改动 state 的 perception_mode。"""
    from ..perception.hsv_color_detector import HSVColorDetector
    from ..perception.synthetic_camera import SyntheticCamera

    vision_cfg = load_vision_config()
    camera = SyntheticCamera(env["backend"].scene, vision_cfg)
    detector = HSVColorDetector(vision_cfg, camera_model=camera.camera_model)
    return {"vision_cfg": vision_cfg, "camera": camera, "vision": detector}


@pytest.fixture
def env_hsv(env, vision):
    """hsv 感知模式: 物块定位走 相机->HSV->投影。返回 env 并注入视觉到全局状态。"""
    control = ControlConfig(**{**env["control"].__dict__, "perception_mode": "hsv"})
    state.configure(backend=env["backend"], queue=env["queue"], control=control,
                    model=env["model"],
                    vision=vision["vision"], camera=vision["camera"],
                    vision_cfg=vision["vision_cfg"])
    env["control"] = control
    env["vision"] = vision["vision"]
    env["camera"] = vision["camera"]
    env["vision_cfg"] = vision["vision_cfg"]
    return env


@pytest.fixture
def mujoco_env():
    """MuJoCo 真 3D 物理后端(Phase 3)。只被 simulator 标记测试使用。

    需要 mujoco 包; 未安装时跳过(不污染快速回归)。
    """
    try:
        from ..robot.mujoco_so101_backend import MuJoCoSO101Backend  # noqa: F401
    except ImportError:
        pytest.skip("需要 mujoco: python -m pip install mujoco")
    model, control, sim_cfg = load_all()
    control = ControlConfig(**{**control.__dict__, "pacing": "fast",
                               "perception_mode": "ground_truth"})
    backend = MuJoCoSO101Backend(model, sim_cfg, control)
    queue = InMemoryQueue()
    state.configure(backend=backend, queue=queue, control=control, model=model)
    return {
        "model": model,
        "control": control,
        "sim_cfg": sim_cfg,
        "backend": backend,
        "queue": queue,
        "vision": None,
        "camera": None,
        "vision_cfg": None,
    }
