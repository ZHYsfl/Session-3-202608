"""运行期组装: 按 control.backend_mode 构造 (backend, camera, vision) 三元组。

Phase 3 起有两条执行路径, 上层 skills/api 零改动:
    kinematic : KinematicSO101Backend + SyntheticCamera + HSVColorDetector(Phase 1/2)
    mujoco    : MuJoCoSO101Backend  + SimulatorCamera  + HSVColorDetector(Phase 3 真 3D)

由 scripts/run_api.py 与 scripts/run_3d_sim.py 复用, 避免三处重复构造逻辑。
"""
from __future__ import annotations

from ..config import ControlConfig, VisionConfig, load_vision_config
from ..perception.camera import CameraProvider
from ..perception.hsv_color_detector import HSVColorDetector
from ..robot.backend import RobotBackend
from ..robot.kinematics import RobotModel


def build_runtime(model: RobotModel, control: ControlConfig, sim_cfg: dict):
    """构造并返回 (backend, camera, vision, vision_cfg)。

    camera/vision 为 None 时对应 perception_mode == "ground_truth" 场景
    (不需要视觉; create_app 里传 None 即可)。
    """
    vision_cfg = load_vision_config()
    mode = control.backend_mode

    if mode == "kinematic":
        from ..perception.synthetic_camera import SyntheticCamera
        from ..robot.kinematic_so101_backend import KinematicSO101Backend

        backend = KinematicSO101Backend(model, sim_cfg, control)
        camera: CameraProvider = SyntheticCamera(backend.scene, vision_cfg)
    elif mode == "mujoco":
        from ..perception.simulator_camera import SimulatorCamera
        from ..robot.mujoco_so101_backend import MuJoCoSO101Backend

        backend = MuJoCoSO101Backend(model, sim_cfg, control, vision_cfg)
        camera = SimulatorCamera(backend, vision_cfg)
    else:
        raise ValueError(
            f"未知 backend_mode: {mode!r} (允许 kinematic/mujoco)")

    # Phase 4: perception_mode 决定用哪个感知后端(HSV 或自训练 TinyDetector)。
    # 两者都是 PerceptionBackend, grab_the_block 的视觉路径零改动。
    if control.perception_mode == "tiny_detector":
        from ..perception.tiny_detector import TinyDetector

        vision = TinyDetector(vision_cfg, camera_model=camera.camera_model)
    else:
        vision = HSVColorDetector(vision_cfg, camera_model=camera.camera_model)
    return backend, camera, vision, vision_cfg


def backend_is_mujoco(control: ControlConfig) -> bool:
    return control.backend_mode == "mujoco"
