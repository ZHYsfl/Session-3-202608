"""配置加载：把 configs/*.yaml 解析为类型化对象。

所有阈值/参数都来自配置文件, 业务代码禁止 hardcode。
坐标单位 meter, 角度 radian。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .robot.kinematics import JointModel, RobotModel

# 项目根目录(F:\pr2)。运行脚本时以 F:\pr2 为 cwd。
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_DIR = _PROJECT_ROOT / "configs"


# ---------------------------------------------------------------------------
# 控制配置
# ---------------------------------------------------------------------------

@dataclass
class ControlConfig:
    frequency_hz: float = 50.0
    dt: float = 0.02
    position_threshold_m: float = 0.02
    joint_convergence_tol_rad: float = 0.005
    move_timeout_s: float = 10.0
    grasp_timeout_s: float = 10.0
    velocity_threshold_mps: float = 0.01
    approach_height_m: float = 0.08
    descent_offset_m: float = 0.015
    lift_height_m: float = 0.10
    holding_proximity_m: float = 0.04
    pacing: str = "fast"
    # 物块定位方式: "ground_truth"(Phase 1, 直接读仿真状态) | "hsv"(Phase 2, 相机+HSV)
    #              | "tiny_detector"(Phase 4, 自训练 TinyDetector)
    perception_mode: str = "ground_truth"
    # 后端选择(Phase 3): "kinematic"(Phase 1/2, 运动学+合成相机) | "mujoco"(真 3D 物理)
    backend_mode: str = "kinematic"
    # Phase 5 抓取策略: "scripted"(脚本 IK 专家, 确定性默认) | "bc"(自训练 BC 策略闭环)
    grasp_mode: str = "scripted"
    # BC 策略推理参数(grasp_mode="bc" 时读; 见 control.yaml grasp_policy 注释)
    grasp_policy_checkpoint: str = "models/grasp_policy.pt"
    grasp_policy_grip_threshold: float = 0.5    # 闭夹爪触发: grip 概率超过该值
    grasp_policy_phase_threshold: float = 0.3   # 相位切换(approach->descend): grip ≥ 该值
    grasp_policy_close_xy_m: float = 0.04       # 闭夹爪前与物块的水平距离上限(守卫, m)
    grasp_policy_close_z_tol_m: float = 0.015   # 闭夹爪前与下降点的高度容差(守卫, m)
    grasp_policy_close_settle_ticks: int = 3    # 闭夹爪前须连续 N tick 停在下降点附近
    # 策略目标点相对当前末端的漂移(|wp - ee|)超该值即重新规划移动。用于在
    # approach 末端(A 处相位过渡的微小下移, ~4-18mm)触发 descend, 同时不
    # 被收敛位姿的策略噪声(~2mm)误触发。见 run_bc_grasp docstring。
    grasp_policy_waypoint_re_tol_m: float = 0.003
    # Phase 6 camera2 第三视角交叉验证(协议 §2.3/§2.4「第二机位摄像头判断」)
    grasp_verify_camera2: bool = True     # 抓取/释放结果用 camera2 图像交叉验证
    grasp_verify_camera2_px_tol: float = 40.0   # 物块质心与末端投影的像素距离阈值(px)
    grasp_verify_lift_z_m: float = 0.03   # 末端须离桌面超过该高度才算"已抬起"(m)
    # Phase 5 数据/训练参数(脚本读; 见 control.yaml grasp_training 注释)
    grasp_dataset_path: str = "data/grasp_dataset.npz"
    grasp_n_episodes: int = 60
    grasp_seed: int = 0
    grasp_x_range: tuple = (0.24, 0.32)
    grasp_y_range: tuple = (-0.08, 0.08)
    grasp_epochs: int = 200
    grasp_lr: float = 1.0e-3
    grasp_val_frac: float = 0.1
    grasp_grip_loss_weight: float = 0.5

    @classmethod
    def from_mapping(cls, d: dict) -> "ControlConfig":
        c = d.get("control", {})
        g = c.get("grasp", {})
        gp = c.get("grasp_policy", {})
        gt = c.get("grasp_training", {})

        _r = lambda v: (float(v[0]), float(v[1]))    # noqa: E731 -- 范围元组 [a,b] -> (a,b)
        return cls(
            frequency_hz=float(c.get("frequency_hz", 50.0)),
            dt=float(c.get("dt", 0.02)),
            position_threshold_m=float(c.get("position_threshold_m", 0.02)),
            joint_convergence_tol_rad=float(c.get("joint_convergence_tol_rad", 0.005)),
            move_timeout_s=float(c.get("move_timeout_s", 10.0)),
            grasp_timeout_s=float(c.get("grasp_timeout_s", 10.0)),
            velocity_threshold_mps=float(c.get("velocity_threshold_mps", 0.01)),
            approach_height_m=float(g.get("approach_height_m", 0.08)),
            descent_offset_m=float(g.get("descent_offset_m", 0.015)),
            lift_height_m=float(g.get("lift_height_m", 0.10)),
            holding_proximity_m=float(g.get("holding_proximity_m", 0.04)),
            pacing=str(c.get("pacing", "fast")),
            perception_mode=str(c.get("perception_mode", "ground_truth")),
            backend_mode=str(c.get("backend_mode", "kinematic")),
            grasp_mode=str(c.get("grasp_mode", "scripted")),
            grasp_policy_checkpoint=str(gp.get("checkpoint", "models/grasp_policy.pt")),
            grasp_policy_grip_threshold=float(gp.get("grip_threshold", 0.5)),
            grasp_policy_phase_threshold=float(gp.get("phase_threshold", 0.3)),
            grasp_policy_close_xy_m=float(gp.get("close_xy_m", 0.04)),
            grasp_policy_close_z_tol_m=float(gp.get("close_z_tol_m", 0.015)),
            grasp_policy_close_settle_ticks=int(gp.get("close_settle_ticks", 3)),
            grasp_policy_waypoint_re_tol_m=float(gp.get("waypoint_re_tol_m", 0.003)),
            grasp_verify_camera2=bool(c.get("grasp_verify_camera2", True)),
            grasp_verify_camera2_px_tol=float(c.get("grasp_verify_camera2_px_tol", 40.0)),
            grasp_verify_lift_z_m=float(c.get("grasp_verify_lift_z_m", 0.03)),
            grasp_dataset_path=str(gt.get("dataset_path", "data/grasp_dataset.npz")),
            grasp_n_episodes=int(gt.get("n_episodes", 60)),
            grasp_seed=int(gt.get("seed", 0)),
            grasp_x_range=_r(gt.get("x_range", [0.24, 0.32])),
            grasp_y_range=_r(gt.get("y_range", [-0.08, 0.08])),
            grasp_epochs=int(gt.get("epochs", 200)),
            grasp_lr=float(gt.get("lr", 1.0e-3)),
            grasp_val_frac=float(gt.get("val_frac", 0.1)),
            grasp_grip_loss_weight=float(gt.get("grip_loss_weight", 0.5)),
        )


# ---------------------------------------------------------------------------
# 机器人模型
# ---------------------------------------------------------------------------

def load_robot_model(path: Path | str | None = None) -> RobotModel:
    """解析 robot.yaml -> RobotModel(运动学 + 关节限位)。"""
    p = Path(path) if path else DEFAULT_CONFIG_DIR / "robot.yaml"
    d = yaml.safe_load(p.read_text(encoding="utf-8"))
    robot = d["robot"]
    links = {item["name"]: item for item in robot["links"]}
    joints = {item["name"]: item for item in robot["joints"]}
    if set(links) != set(joints):
        raise ValueError(f"links 与 joints 名称不一致: {set(links)} vs {set(joints)}")

    order = [item["name"] for item in robot["links"]]
    jm: list[JointModel] = []
    for name in order:
        lk, jt = links[name], joints[name]
        lim = jt["limit"]
        jm.append(JointModel(
            name=name,
            axis=str(lk["axis"]),
            link=[float(v) for v in lk["link"]],
            q_home=float(jt["q_home"]),
            limit_min=float(lim[0]),
            limit_max=float(lim[1]),
            vmax=float(jt["vmax"]),
            accel=float(jt["accel"]),
        ))
    ph = robot.get("collision_guard", {})
    return RobotModel(
        name=str(robot.get("name", "SO-101")),
        joints=jm,
        max_reach_m=float(robot.get("max_reach_m", 0.44)),
        table_top_z_m=float(robot.get("table_top_z_m", 0.0)),
        collision_guard=bool(ph.get("enabled", False)),
        base_radius_m=float(ph.get("base_radius_m", 0.014)),
        base_z_hi_m=float(ph.get("base_z_hi_m", 0.090)),
        link_radius_m=float(ph.get("link_radius_m", 0.022)),
    )


def load_control_config(path: Path | str | None = None) -> ControlConfig:
    """解析 control.yaml -> ControlConfig。"""
    p = Path(path) if path else DEFAULT_CONFIG_DIR / "control.yaml"
    return ControlConfig.from_mapping(yaml.safe_load(p.read_text(encoding="utf-8")))


def load_simulation_config(path: Path | str | None = None) -> dict:
    """解析 simulation.yaml -> 原始 dict(场景结构简单, 由 Scene 直接消费)。"""
    p = Path(path) if path else DEFAULT_CONFIG_DIR / "simulation.yaml"
    return yaml.safe_load(p.read_text(encoding="utf-8"))["simulation"]


# ---------------------------------------------------------------------------
# 视觉配置(Phase 2)
# ---------------------------------------------------------------------------

@dataclass
class ColorRange:
    """HSV 阈值区间(OpenCV 约定: H∈[0,180], S/V∈[0,255])。"""

    lower: tuple[int, int, int]
    upper: tuple[int, int, int]


@dataclass
class VisionConfig:
    """configs/vision.yaml 的类型化对象。所有视觉阈值都在这里, 禁止 hardcode。"""

    image_width: int = 640
    image_height: int = 480
    roi: tuple[int, int, int, int] = (0, 0, 640, 480)   # (x0, y0, x1, y1) 图像坐标
    camera_id: str = "camera1"
    camera_position: tuple[float, float, float] = (0.28, 0.0, 0.55)   # base frame, m
    camera_look_at: tuple[float, float, float] = (0.18, 0.0, 0.0)
    camera_up: tuple[float, float, float] = (0.0, 0.0, 1.0)
    fx: float = 600.0
    fy: float = 600.0
    block_side_m: float = 0.04           # 物块底面边长(合成渲染用)
    block_half_height_m: float = 0.015   # 与 simulation.yaml block.half_height_m 一致
    table_top_z_m: float = 0.0           # 与 robot.yaml table_top_z_m 一致
    table_color_bgr: tuple[int, int, int] = (110, 110, 110)
    block_colors_bgr: dict = field(default_factory=lambda: {
        "red": (0, 0, 255),
        "yellow": (0, 255, 255),
        "white": (255, 255, 255),
    })
    hsv_ranges: dict = field(default_factory=lambda: {
        "red": [ColorRange((0, 80, 80), (10, 255, 255)),
                ColorRange((170, 80, 80), (180, 255, 255))],   # 红色跨 0 度边界 -> 双区间
        "yellow": [ColorRange((20, 80, 80), (35, 255, 255))],
        "white": [ColorRange((0, 0, 180), (180, 50, 255))],     # 低饱和 + 高亮度
    })
    morphology_kernel: int = 3
    min_area_px: float = 120.0          # 见 vision.yaml 注释(经 1000 随机场景标定)
    confidence_threshold: float = 0.2   # 见 vision.yaml 注释(经 1000 随机场景标定)

    # Phase 4 TinyDetector: 自训练微型 CNN 逐 cell 分类(见 perception/tiny_detector.py)
    tiny_grid_width: int = 40            # 网格列数(小图宽)
    tiny_grid_height: int = 30           # 网格行数(小图高)
    tiny_checkpoint: str = "models/tiny_detector.pt"   # 项目根相对路径
    tiny_min_area_cells: float = 4.0     # 连通域最小 cell 数(滤噪点)
    tiny_confidence_threshold: float = 0.5   # cell 归属某色所需最小概率
    tiny_centroid_prob: float = 0.8      # 质心只统计 >= 该概率的高置信 cell

    @classmethod
    def from_mapping(cls, d: dict) -> "VisionConfig":
        v = d.get("vision", {})
        img = v.get("image", {})
        cam = v.get("camera", {})
        blk = v.get("block", {})
        det = v.get("detection", {})
        tiny = v.get("tiny", {})

        def t3(val) -> tuple[int, int, int]:
            return (int(val[0]), int(val[1]), int(val[2]))

        def t3f(val) -> tuple[float, float, float]:
            return (float(val[0]), float(val[1]), float(val[2]))

        colors_bgr = {
            c: t3(rgb) for c, rgb in v.get("block_colors_bgr", {}).items()
        }
        ranges: dict = {}
        for color, rngs in v.get("hsv_ranges", {}).items():
            ranges[color] = [
                ColorRange(lower=t3(r["lower"]), upper=t3(r["upper"]))
                for r in rngs
            ]
        return cls(
            image_width=int(img.get("width", 640)),
            image_height=int(img.get("height", 480)),
            roi=tuple(int(x) for x in img.get("roi", [0, 0, 640, 480])),
            camera_id=str(cam.get("camera_id", "camera1")),
            camera_position=t3f(cam.get("position", [0.28, 0.0, 0.55])),
            camera_look_at=t3f(cam.get("look_at", [0.18, 0.0, 0.0])),
            camera_up=t3f(cam.get("up", [0.0, 0.0, 1.0])),
            fx=float(cam.get("fx", 600.0)),
            fy=float(cam.get("fy", 600.0)),
            block_side_m=float(blk.get("side_m", 0.04)),
            block_half_height_m=float(blk.get("half_height_m", 0.015)),
            table_top_z_m=float(blk.get("table_top_z_m", 0.0)),
            table_color_bgr=t3(v.get("table_color_bgr", [110, 110, 110])),
            block_colors_bgr=colors_bgr,
            hsv_ranges=ranges,
            morphology_kernel=int(det.get("morphology_kernel", 3)),
            min_area_px=float(det.get("min_area_px", 150.0)),
            confidence_threshold=float(det.get("confidence_threshold", 0.4)),
            tiny_grid_width=int(tiny.get("grid_width", 40)),
            tiny_grid_height=int(tiny.get("grid_height", 30)),
            tiny_checkpoint=str(tiny.get("checkpoint", "models/tiny_detector.pt")),
            tiny_min_area_cells=float(tiny.get("min_area_cells", 4.0)),
            tiny_confidence_threshold=float(tiny.get("confidence_threshold", 0.5)),
            tiny_centroid_prob=float(tiny.get("centroid_prob", 0.8)),
        )


def load_vision_config(path: Path | str | None = None) -> VisionConfig:
    """解析 vision.yaml -> VisionConfig。"""
    p = Path(path) if path else DEFAULT_CONFIG_DIR / "vision.yaml"
    return VisionConfig.from_mapping(yaml.safe_load(p.read_text(encoding="utf-8")))


# 供脚本使用的默认加载
def load_all(
    config_dir: Path | str | None = None,
) -> tuple[RobotModel, ControlConfig, dict]:
    """一次性加载三个配置: (robot_model, control_config, simulation_cfg)。"""
    d = Path(config_dir) if config_dir else DEFAULT_CONFIG_DIR
    return (
        load_robot_model(d / "robot.yaml"),
        load_control_config(d / "control.yaml"),
        load_simulation_config(d / "simulation.yaml"),
    )
