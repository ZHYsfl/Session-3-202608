"""grab_the_block(color) -> str：抓取指定颜色的物块。

合法颜色仅 yellow/red/white(协议 §2.3)。一次任务同颜色最多一个物块。

物块定位方式(control.perception_mode):
    "ground_truth" 直接读仿真物块状态(Phase 1 行为, 测试默认保持)。
    "hsv"          相机取帧 -> HSV 检测 -> 像素投影到世界(Phase 2)。
    "tiny_detector" 相机取帧 -> 自训练 TinyDetector(Phase 4) -> 像素投影到世界。
    (hsv 与 tiny_detector 走同一条 _locate_block_by_vision 视觉路径)

抓取执行(Phase 5, control.grasp_mode):
    "scripted" 脚本 IK 专家序列(_grasp_sequence, 确定性默认)。
    "bc"       自训练 BC 策略闭环抓取(grasp_policy.run_bc_grasp, 逐 tick 决策)。

两条路径都只负责「目标物块在哪」, 协议返回字符串完全一致, 业务语义不变。
"""
from __future__ import annotations

from ..perception.camera2_verify import camera2_check_enabled, verify_grasp
from ..robot.controller import make_pacing
from ..runtime.state import state
from ._common import UnreachableError
from ._motion import GraspFailed, move_to_point

VALID_COLORS = ("yellow", "red", "white")

_NO_COLOR = "没有这种颜色的物块，无法夹取"
_GRASP_OK = "有这种颜色的物块，且夹取物块成功"
_GRASP_FAIL = "有这种颜色的物块，但夹取物块失败"


def grab_the_block(color: str) -> str:
    backend = state.require_backend()

    # 1) 颜色合法性(协议 §2.3 第 1 步; 不合法时**不调用相机/检测器**)
    if color not in VALID_COLORS:
        return _NO_COLOR

    # 2) 定位目标物块(协议 §2.3 第 2 步; 按 perception_mode 走视觉或 ground truth)
    target = _locate_block_target(color)      # (x, y, z) base frame, meter
    if target is None:
        return _NO_COLOR

    # 3) 尝试抓取(协议 §2.3 第 3 步; Phase 5: grasp_mode 选择执行路径)
    model = state.require_model()
    control = state.require_control()
    token = state.begin_action()
    try:
        pacing = make_pacing(control.pacing)
        if control.grasp_mode == "bc":
            from .grasp_policy import run_bc_grasp
            run_bc_grasp(target, backend, model, control, token, pacing)
        else:
            _grasp_sequence(target, pacing, token)
    except GraspFailed:
        # 覆盖: 不可达 / 移动未到 / 未夹住 / 用户取消
        # 取消视为“未成功”, 返回协议允许的失败字符串(诚实)。
        return _GRASP_FAIL
    finally:
        state.end_action(token)

    # 4) 夹取验证(协议 §2.3: 第二机位交叉验证; Phase 1/2 为仿真物理,
    #    Phase 6: 再加 camera2 第三视角视觉交叉验证, 见 _grab_verified_by_camera2)
    if backend.is_holding_object() and _grab_verified_by_camera2(color):
        state.last_grabbed_color = color      # Phase 6: 供 release 的 camera2 交叉验证
        return _GRASP_OK
    return _GRASP_FAIL


def _locate_block_target(color: str):
    """定位目标颜色物块, 返回 (x, y, z); 未发现返回 None。

    视觉模式(hsv / tiny_detector)只依赖 图像 + 相机标定, 不读后端物块
    ground truth。
    """
    control = state.require_control()
    if control.perception_mode != "ground_truth":
        return _locate_block_by_vision(color)

    backend = state.require_backend()
    block = next((b for b in backend.get_block_states() if b.color == color), None)
    if block is None:
        return None
    return (block.x, block.y, block.z)


def _locate_block_by_vision(color: str):
    """视觉模式(hsv / tiny_detector): 相机取帧 -> 检测 -> 该颜色最佳候选 ->
    像素投影到桌面平面。

    绝不调用 backend.get_block_states(); 物块坐标完全来自图像感知。
    """
    vision = state.require_vision()
    camera = state.require_camera()
    vision_cfg = state.require_vision_cfg()

    frame = camera.capture()
    detections = vision.detect_blocks(frame.bgr)

    best = None
    for d in detections:
        if d.color != color:
            continue
        if best is None or d.confidence > best.confidence:
            best = d
    if best is None:
        return None

    # 世界投影: 优先用检测器已标定的 world_position; 否则用相机标定现算。
    if best.world_position is not None:
        return best.world_position
    xy = camera.camera_model.pixel_to_world_on_plane(
        best.center_pixel[0], best.center_pixel[1], vision_cfg.table_top_z_m)
    if xy is None:
        return None
    z_block = vision_cfg.table_top_z_m + vision_cfg.block_half_height_m
    return (xy[0], xy[1], z_block)


def _grab_verified_by_camera2(color: str) -> bool:
    """物理验证已通过后, 再跑 camera2 视觉交叉验证(协议 §2.3「第二机位」)。

    camera2 未启用 / 未注入视觉 / 无 camera2 / 取帧失败 -> 返回 True(不阻断,
    物理验证已足够); camera2 正常时视觉判定参与最终结果(诚实交叉验证)。
    """
    control = state.require_control()
    if not camera2_check_enabled(control):
        return True
    r = verify_grasp(color, state.require_backend(), state.require_camera(),
                     state.require_vision(), state.require_vision_cfg(), control)
    return True if r is None else r


def _grasp_sequence(target_xyz: tuple[float, float, float], pacing, token) -> None:
    """home 上方 -> 下降到物块 -> 闭夹爪 -> 抬升。

    失败(不可达/超时/取消/未夹住)抛 GraspFailed。
    target_xyz: 目标物块中心(base frame, meter)。
    """
    backend = state.require_backend()
    model = state.require_model()
    control = state.require_control()

    bx, by, bz = target_xyz
    # 物理后端: 抓取时把腕部自旋固定到侧向(沿世界 Y 张开夹爪), 前倾工具的
    # 前指才不会在下降时扫进物块(Phase 3 集成测试证明)。运动学后端为 None。
    wrist = backend.grasp_wrist_roll

    try:
        # 1) 移到物块上方 approach_height
        move_to_point((bx, by, bz + control.approach_height_m),
                      backend, model, control, token, pacing,
                      force_wrist_roll=wrist)
        # 2) 下降到物块中心附近
        move_to_point((bx, by, bz + control.descent_offset_m),
                      backend, model, control, token, pacing,
                      force_wrist_roll=wrist)
        # 3) 闭合夹爪
        backend.close_gripper()
        if not backend.is_holding_object():
            raise GraspFailed("夹爪闭合后未检测到物块")
        # 4) 抬升
        move_to_point((bx, by, bz + control.lift_height_m),
                      backend, model, control, token, pacing,
                      force_wrist_roll=wrist)
    except UnreachableError as e:
        # 物块在机械臂工作空间外: 统一归为抓取失败(协议返回失败字符串)
        raise GraspFailed(f"物块不可达: {e}") from e

    # 说明: 夹住后物块跟随末端; 抬升后的真实物块位置可经 get_block_states 查询。
