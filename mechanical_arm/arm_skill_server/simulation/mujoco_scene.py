"""SO-101 MuJoCo 场景构建(Phase 3): 从配置生成真正的 3D 物理仿真模型。

职责:
    - 用 configs/robot.yaml 的关节链/限位/几何构建与 Phase 1 运动学模型
      FK 完全一致的 articulated robot(逐关节验证见 test_mujoco_backend.py)。
    - 添加 table、floor、三色 cube(free joint, 带重力/碰撞/接触)、
      camera1/camera2、光源。
    - 返回 MujocoSceneInfo: MjModel/MjData + 名称->索引映射。

约束:
    - 本文件不渲染、不推理, 只负责「模型即事实」。
    - 所有数值来自配置文件, 禁止 hardcode。
    - 相机位姿: camera1 用 vision.yaml(position/look_at, 与 perception 约定一致),
      camera2 用 simulation.yaml。运行时代码直接写 data.cam_xpos/cam_xmat
      (渲染器读取的就是这两个, 见 test_camera_convention)。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

try:
    import mujoco
except ImportError as e:  # pragma: no cover - 依赖未装时给出明确错误
    raise ImportError("Phase 3 需要 mujoco: python -m pip install mujoco") from e

from ..config import VisionConfig
from ..perception.projection import build_camera_pose
from ..robot.kinematics import RobotModel

# 夹爪指片几何(半尺寸, meter)
FINGER_HALF_THICK = 0.004      # x 方向半厚 -> 指片 8mm
FINGER_HALF_DEPTH = 0.012      # y 方向半深 -> 指片 24mm
FINGER_HALF_HEIGHT = 0.012     # z 方向半高 -> 指片 24mm(从 tool 原点向下挂)

# 物块几何(来自 vision.yaml)
_CUBE_COLORS = ("red", "yellow", "white")


@dataclass
class MujocoSceneInfo:
    """MuJoCo 场景的编译结果与名称映射。"""

    model: "mujoco.MjModel"
    data: "mujoco.MjData"
    arm_joint_names: list[str]                 # 5 个臂关节(顺序与 robot.yaml 一致)
    qpos_addr: dict[str, int]                  # 关节名 -> qpos 地址
    qvel_addr: dict[str, int]                  # 关节名 -> qvel 地址
    actuator_id: dict[str, int]                # 执行器名 -> id
    cube_free_addr: dict[str, int]             # 颜色 -> free joint qpos 地址
    cube_free_dof_addr: dict[str, int]         # 颜色 -> free joint qvel(dof) 地址
    cube_geom_names: dict[str, str]            # 颜色 -> geom 名
    finger_joint_names: list[str]              # ["finger_L", "finger_R"]
    finger_qpos_addr: dict[str, int]           # 指关节名 -> qpos 地址
    finger_dof_addr: dict[str, int]            # 指关节名 -> qvel(dof) 地址
    ee_site_id: int                            # ee_tip site id
    camera_names: list[str]                    # ["camera1", "camera2"]
    camera_id: dict[str, int]                  # 相机名 -> id
    camera_world_T: dict[str, np.ndarray]      # 相机名 -> 4x4 (perception 约定)
    gripper_ctrl_open: tuple[float, float]     # (finger_L, finger_R) 张开 ctrl
    gripper_ctrl_closed: tuple[float, float]   # 闭合 ctrl
    block_half_height_m: float = 0.015
    table_top_z_m: float = 0.0
    physics: dict = field(default_factory=dict)


def _yaw_quat(yaw: float) -> np.ndarray:
    """绕世界 z 轴旋转 yaw 的四元数 [w,x,y,z]。"""
    return np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)])


def _fovy_deg(height_px: int, fy: float) -> float:
    """由竖直像素焦距 fy 反推 MuJoCo camera.fovy(度)。fy=(H/2)/tan(fovy/2)。"""
    return float(np.degrees(2.0 * np.arctan((height_px / 2.0) / fy)))


def build_mujoco_scene(
    robot_model: RobotModel,
    sim_cfg: dict,
    vision_cfg: VisionConfig,
) -> MujocoSceneInfo:
    """从配置构建 MuJoCo 3D 物理场景, 返回映射信息。"""
    ph = sim_cfg.get("physics", {})
    timestep = float(ph.get("timestep", 0.002))
    gravity = float(ph.get("gravity", -9.81))
    solver = str(ph.get("solver", "Newton"))
    iterations = int(ph.get("iterations", 50))
    cube_mass = float(ph.get("cube_mass_kg", 0.03))
    friction_block = float(ph.get("friction_block", 0.6))
    friction_finger = float(ph.get("friction_finger", 1.0))
    arm_density = float(ph.get("arm_density", 1500.0))
    arm_kp = float(ph.get("arm_kp", 200.0))
    arm_kv = float(ph.get("arm_kv", 10.0))
    arm_force = float(ph.get("arm_force_limit", 8.0))
    # 关节 armature(电机转子等效惯量)。0.0005 太小 -> 手腕/手腕滚转关节
    # I≈7e-5, kp=200 时 ω·dt=3.4>2, 半隐式欧拉积分不稳定, 伺服挂不住姿势
    # (集成测试证明)。0.01 把所有臂关节拉回 ω·dt<1, 位置伺服稳定。
    arm_armature = float(ph.get("arm_armature", 0.01))
    grip_kp = float(ph.get("gripper_kp", 200.0))
    grip_kv = float(ph.get("gripper_kv", 10.0))
    grip_force = float(ph.get("gripper_force_limit", 4.0))

    # 渲染光照(Phase 3 视觉标定用)。默认值经 renderer 帧实测: specular=0 +
    # 中低亮度让桌面/机械臂不饱和成白色(否则 HSV 白物块范围会抓到整张桌面)。
    lit = ph.get("lighting", {})
    amb_main = float(lit.get("ambient_main", 0.35))
    dif_main = float(lit.get("diffuse_main", 0.35))
    amb_fill = float(lit.get("ambient_fill", 0.14))
    dif_fill = float(lit.get("diffuse_fill", 0.14))

    # 材质颜色(Phase 3 渲染标定, 见 docs/progress/phase3_report.md 视觉章节):
    #   机身/桌面必须渲染出「不是白色」的像素(S>50 蓝灰或 V<180), 否则白物块 HSV
    #   范围 [0,0,180]-[180,50,255] 会抓到浅灰桌面/浅灰机械臂整片误检
    #   (实测: rgba 0.55 灰 -> 桌面 V≈247 落入白范围, 白掩码 28 万像素全屏;
    #   深蓝灰 -> 白掩码只剩白物块 ~2200px, 桌面/机身 S=59..79 全部排除)。
    #   数值由 _mj_light_probe.py 就地标定, 光照不变(见 physics.lighting)。
    ARM_RGBA = ph.get("visual", {}).get("arm_rgba", "0.20 0.22 0.32 1")
    BASE_RGBA = ph.get("visual", {}).get("base_rgba", "0.16 0.17 0.24 1")
    TABLE_RGBA = ph.get("visual", {}).get("table_rgba", "0.26 0.29 0.38 1")
    FLOOR_RGBA = ph.get("visual", {}).get("floor_rgba", "0.14 0.15 0.19 1")

    half_side = vision_cfg.block_side_m / 2.0
    half_height = vision_cfg.block_half_height_m
    table_top_z = vision_cfg.table_top_z_m

    # 夹爪口径(来自 configs/robot.yaml gripper)。RobotModel 不含口径字段, 直接读 yaml。
    aperture_open = float(sim_cfg["_gripper_aperture_open"]) if "_gripper_aperture_open" in sim_cfg else 0.055
    aperture_closed = float(sim_cfg["_gripper_aperture_closed"]) if "_gripper_aperture_closed" in sim_cfg else 0.012
    # 滑块行程: 内侧面 = q + 半厚(F_L) / q - 半厚(F_R); 开口处内侧面 = ±aperture/2
    fL_open = -(aperture_open / 2.0) - FINGER_HALF_THICK
    fL_closed = -(aperture_closed / 2.0) - FINGER_HALF_THICK
    fR_open = (aperture_open / 2.0) + FINGER_HALF_THICK
    fR_closed = (aperture_closed / 2.0) + FINGER_HALF_THICK

    # ---- 生成 MJCF -------------------------------------------------------
    parts: list[str] = []
    parts.append(f'''<mujoco model="SO-101">
  <compiler angle="radian" autolimits="true" coordinate="local"/>
  <option timestep="{timestep}" gravity="0 0 {gravity}"
          solver="{solver}" iterations="{iterations}"/>
  <default>
    <geom density="{arm_density}" friction="{friction_block} 0.005 0.0001"/>
    <material specular="0" shininess="0"/>
    <joint damping="0.1" armature="{arm_armature}"/>
  </default>
  <worldbody>
    <light name="light_main" pos="0.4 0.2 0.9" dir="-0.4 -0.2 -0.9"
           diffuse="{dif_main} {dif_main} {dif_main}" ambient="{amb_main} {amb_main} {amb_main}"
           directional="true"/>
    <light name="light_fill" pos="-0.5 -0.4 0.7" dir="0.5 0.4 -0.7"
           diffuse="{dif_fill} {dif_fill} {dif_fill}" ambient="{amb_fill} {amb_fill} {amb_fill}"
           directional="true"/>
    <geom name="table" type="box" size="0.5 0.5 0.05" pos="0 0 -0.05"
          rgba="{TABLE_RGBA}" friction="{friction_block} 0.005 0.0001"/>
    <geom name="floor" type="plane" size="2 2 0.01" pos="0 0 -0.105"
          rgba="{FLOOR_RGBA}" friction="0.6 0.005 0.0001"/>
''')

    # 机械臂: base(固定) + 5 个关节链(顺序与 robot.yaml 一致)
    # Phase 3 底座几何(integration-test 证明的 bug):
    #   bug1: 旧立柱 r=0.028 顶到 shoulder_lift 关节面(z=0.12), 折叠式 IK 全被撞住。
    #   bug2(本行下方修): base_pedestal(z∈[0,0.020], r=0.025)与 base_column(r=0.012,
    #         z∈[0,0.090])空间重叠 -> MuJoCo 把它们当碰撞体以 ~1e17N 推开, 力传入
    #         整条机械臂, 位置伺服根本无法挂住 home。修复: 立柱整体抬到 pedestal 之上,
    #         两者不再重叠(底座 z∈[0,0.010], 立柱 z∈[0.014,0.090])。
    #   - base_pedestal: 宽矮底座 z∈[0,0.010], r=0.025。
    #   - base_column:   细高立柱 z∈[0.014,0.090], r=0.012。
    # 与 robot.yaml collision_guard(base_radius_m=0.027 覆盖 pedestal 包络) 保持一致。
    parts.append(f'''    <body name="base" pos="0 0 0">
      <geom name="base_pedestal" type="cylinder" size="0.025 0.005" pos="0 0 0.005"
            rgba="{BASE_RGBA}"/>
''')
    # 首关节: shoulder_pan 在 base 原点。立柱(0.014→0.090)挂在 pan 上(随 pan 旋转)。
    parts.append(f'''      <body name="shoulder_pan" pos="0 0 0">
        <joint name="shoulder_pan" type="hinge" axis="0 0 1"
               range="{robot_model.joints[0].limit_min} {robot_model.joints[0].limit_max}"/>
        <geom name="base_column" type="cylinder" size="0.012 0.038" pos="0 0 0.052"
              rgba="{ARM_RGBA}"/>
''')
    # 子关节 body pos = 上一关节的 link 向量(在上一关节坐标系中)。
    prev_link = robot_model.joints[0].link          # [0,0,0.095] -> 下一关节(lift)位置
    parts.append(f'''        <body name="shoulder_lift" pos="{prev_link[0]} {prev_link[1]} {prev_link[2]}">
          <joint name="shoulder_lift" type="hinge" axis="0 1 0"
                 range="{robot_model.joints[1].limit_min} {robot_model.joints[1].limit_max}"/>
          <geom name="link2" type="box" size="0.05 0.022 0.022" pos="0.05 0 0"
                rgba="{ARM_RGBA}"/>
''')
    prev_link = robot_model.joints[1].link          # [0.100,0,0] -> 下一关节(elbow)位置
    parts.append(f'''          <body name="elbow_flex" pos="{prev_link[0]} {prev_link[1]} {prev_link[2]}">
            <joint name="elbow_flex" type="hinge" axis="0 1 0"
                   range="{robot_model.joints[2].limit_min} {robot_model.joints[2].limit_max}"/>
            <geom name="link3" type="box" size="0.07 0.02 0.02" pos="0.07 0 0"
                  rgba="{ARM_RGBA}"/>
''')
    prev_link = robot_model.joints[2].link          # [0.140,0,0] -> 下一关节(wrist_flex)位置
    parts.append(f'''            <body name="wrist_flex" pos="{prev_link[0]} {prev_link[1]} {prev_link[2]}">
              <joint name="wrist_flex" type="hinge" axis="0 1 0"
                     range="{robot_model.joints[3].limit_min} {robot_model.joints[3].limit_max}"/>
              <geom name="link4" type="box" size="0.065 0.018 0.018" pos="0.065 0 0"
                    rgba="{ARM_RGBA}"/>
''')
    prev_link = robot_model.joints[3].link          # [0.130,0,0] -> 下一关节(wrist_roll)位置
    parts.append(f'''              <body name="wrist_roll" pos="{prev_link[0]} {prev_link[1]} {prev_link[2]}">
                <joint name="wrist_roll" type="hinge" axis="0 0 1"
                       range="{robot_model.joints[4].limit_min} {robot_model.joints[4].limit_max}"/>
                <geom name="link5" type="box" size="0.018 0.018 0.0375" pos="0 0 0.0375"
                      rgba="{ARM_RGBA}"/>
''')
    # tool(手部): wrist_roll link(robot.yaml) 之后, 挂 ee_tip + 两指
    prev_link = robot_model.joints[4].link
    parts.append(f'''                <body name="tool" pos="{prev_link[0]} {prev_link[1]} {prev_link[2]}">
                  <site name="ee_tip" pos="0 0 0"/>
''')
    # 指关节: body pos = 0, 滑块 qpos 直接等于内侧面偏移(见 fL_open/fL_closed)。
    # 这样 qpos=0 落在 joint range 内, 执行器 ctrl(=qpos 目标)不会与限位打架。
    parts.append(f'''                  <body name="finger_L" pos="0 0 0">
                    <joint name="finger_L" type="slide" axis="1 0 0"
                           range="{fL_open} {fL_closed}"/>
                    <geom name="finger_L_geom" type="box"
                          size="{FINGER_HALF_THICK} {FINGER_HALF_DEPTH} {FINGER_HALF_HEIGHT}"
                          pos="0 0 -{FINGER_HALF_HEIGHT}" rgba="0.12 0.12 0.16 1"
                          friction="{friction_finger} 0.005 0.0001"/>
                  </body>
''')
    parts.append(f'''                  <body name="finger_R" pos="0 0 0">
                    <joint name="finger_R" type="slide" axis="1 0 0"
                           range="{fR_closed} {fR_open}"/>
                    <geom name="finger_R_geom" type="box"
                          size="{FINGER_HALF_THICK} {FINGER_HALF_DEPTH} {FINGER_HALF_HEIGHT}"
                          pos="0 0 -{FINGER_HALF_HEIGHT}" rgba="0.12 0.12 0.16 1"
                          friction="{friction_finger} 0.005 0.0001"/>
                  </body>
''')
    # 收尾闭合标签
    parts.append('''                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
''')

    # 三色 cube(free joint)
    cube_rgba = {"red": "1 0 0 1", "yellow": "1 1 0 1", "white": "0.95 0.95 0.95 1"}
    for c in _CUBE_COLORS:
        parts.append(f'''    <body name="cube_{c}" pos="0 0 0">
      <freejoint name="free_cube_{c}"/>
      <geom name="cube_{c}_geom" type="box" size="{half_side} {half_side} {half_height}"
            pos="0 0 0" rgba="{cube_rgba[c]}" mass="{cube_mass}"
            friction="{friction_block} 0.005 0.0001"/>
    </body>
''')

    # 相机(位姿在运行时写入 data.cam_xpos/cam_xmat)
    fovy = _fovy_deg(vision_cfg.image_height, vision_cfg.fy)
    parts.append(f'''    <camera name="camera1" fovy="{fovy}"/>
    <camera name="camera2" fovy="{fovy}"/>
  </worldbody>
''')

    # 执行器: 5 臂关节 -> 力矩电机(PD + 重力补偿在 backend.step() 里做);
    # 2 指 -> 位置伺服(夹爪开合, 无重力问题, 保持 position)。
    # Phase 3 集成测试证明: 位置伺服(<position kp=..>)的稳态误差 = 重力矩/kp,
    #   夹取姿态下肩关节需 1.8 N·m -> kp=200 时下垂 0.009 rad > 收敛容差 0.005,
    #   提高 kp 又触发 forcerange 饱和下的伪平衡(臂折向远处), 无法收紧到 0.005。
    #   改用力矩电机: ctrl 即扭矩, 控制器显式加 qfrc_bias(重力/科氏)补偿,
    #   稳态误差 ~0(无下垂), 收敛只受积分/饱和限制。
    arm_names = [j.name for j in robot_model.joints]
    # 每关节力矩上限(按该关节所需保持力矩的 2-3 倍取; 手腕惯量小,
    # 过大力矩会让 50Hz 外层控制一步内就把手腕甩出可控范围 -> bang-bang 极限环)。
    force_lims = [float(x) for x in ph.get(
        "arm_force_limits", [arm_force] * len(arm_names))]
    if len(force_lims) != len(arm_names):
        raise ValueError(f"arm_force_limits 长度须等于关节数 {len(arm_names)}, 实际 {len(force_lims)}")
    acts = []
    for nm, F_i in zip(arm_names, force_lims):
        acts.append(f'''    <motor name="act_{nm}" joint="{nm}" gear="1"
              ctrlrange="-{F_i} {F_i}" forcerange="-{F_i} {F_i}"/>''')
    acts.append(f'''    <position name="act_finger_L" joint="finger_L" kp="{grip_kp}" kv="{grip_kv}"
              ctrlrange="{fL_open} {fL_closed}" forcerange="-{grip_force} {grip_force}"/>''')
    acts.append(f'''    <position name="act_finger_R" joint="finger_R" kp="{grip_kp}" kv="{grip_kv}"
              ctrlrange="{fR_closed} {fR_open}" forcerange="-{grip_force} {grip_force}"/>''')
    parts.append("<actuator>\n" + "\n".join(acts) + "\n</actuator>\n</mujoco>\n")
    xml = "\n".join(parts)

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    # ---- 名称映射 ---------------------------------------------------------
    def jid(name: str) -> int:
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        assert i >= 0, f"关节 {name} 未找到"
        return i

    def aid(name: str) -> int:
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        assert i >= 0, f"执行器 {name} 未找到"
        return i

    def cid(name: str) -> int:
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, name)
        assert i >= 0, f"相机 {name} 未找到"
        return i

    qpos_addr = {nm: int(model.jnt_qposadr[jid(nm)]) for nm in arm_names}
    qvel_addr = {nm: int(model.jnt_dofadr[jid(nm)]) for nm in arm_names}
    actuator_id = {f"act_{nm}": aid(f"act_{nm}") for nm in arm_names}
    actuator_id["act_finger_L"] = aid("act_finger_L")
    actuator_id["act_finger_R"] = aid("act_finger_R")
    cube_free_addr = {c: int(model.jnt_qposadr[jid(f"free_cube_{c}")]) for c in _CUBE_COLORS}
    cube_free_dof_addr = {c: int(model.jnt_dofadr[jid(f"free_cube_{c}")]) for c in _CUBE_COLORS}
    cube_geom_names = {c: f"cube_{c}_geom" for c in _CUBE_COLORS}
    finger_joint_names = ["finger_L", "finger_R"]
    finger_qpos_addr = {nm: int(model.jnt_qposadr[jid(nm)]) for nm in finger_joint_names}
    finger_dof_addr = {nm: int(model.jnt_dofadr[jid(nm)]) for nm in finger_joint_names}
    ee_site = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee_tip")
    assert ee_site >= 0, "ee_tip site 未找到"

    # 相机位姿(perception 约定): camera1 来自 vision.yaml; camera2 来自 sim_cfg
    cam1_T = build_camera_pose(
        vision_cfg.camera_position, vision_cfg.camera_look_at, vision_cfg.camera_up)
    cam2_pose = sim_cfg["camera2"]["pose"]
    cam2_pos = np.array([float(cam2_pose["x"]), float(cam2_pose["y"]), float(cam2_pose["z"])])
    cam2_T = build_camera_pose(cam2_pos, (0.15, 0.0, 0.0), (0.0, 0.0, 1.0))

    camera_names = ["camera1", "camera2"]
    camera_id = {nm: cid(nm) for nm in camera_names}
    camera_world_T = {"camera1": cam1_T, "camera2": cam2_T}

    info = MujocoSceneInfo(
        model=model, data=data,
        arm_joint_names=arm_names, qpos_addr=qpos_addr, qvel_addr=qvel_addr,
        actuator_id=actuator_id, cube_free_addr=cube_free_addr,
        cube_free_dof_addr=cube_free_dof_addr,
        cube_geom_names=cube_geom_names,
        finger_joint_names=finger_joint_names,
        finger_qpos_addr=finger_qpos_addr, finger_dof_addr=finger_dof_addr,
        ee_site_id=ee_site,
        camera_names=camera_names, camera_id=camera_id, camera_world_T=camera_world_T,
        gripper_ctrl_open=(fL_open, fR_open),
        gripper_ctrl_closed=(fL_closed, fR_closed),
        block_half_height_m=half_height, table_top_z_m=table_top_z,
        physics=dict(ph),
    )
    return info
