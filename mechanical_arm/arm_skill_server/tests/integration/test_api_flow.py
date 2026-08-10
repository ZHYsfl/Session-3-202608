"""API 集成测试: HTTP -> 精确 JSON 全流程(对应「成功定义」)。

重点: 返回字符串必须逐字匹配协议文档, 不能偷换表达。
"""
from __future__ import annotations

import threading
import time

from fastapi.testclient import TestClient

from ...config import ControlConfig
from ...runtime.state import state


def _body(resp) -> dict:
    return resp.json()


# ---------------------------------------------------------------------------
# 全流程: 我的坐标 -> 移动 -> 抓红色 -> 夹住后再查坐标 -> 释放
# ---------------------------------------------------------------------------

def test_full_flow(app, env) -> None:
    env["backend"].reset("all")
    c = TestClient(app)

    r = c.get("/api/v1/get_current_coordinates")
    assert r.status_code == 200
    assert _body(r)["code"] == 0
    assert _body(r)["result"].startswith("我的坐标是")
    assert _body(r)["error"] is None

    r = c.post("/api/v1/move_to_coordinates", json={"x": "0.20", "y": "-0.05", "z": "0.15"})
    assert r.status_code == 200
    assert _body(r)["result"] == "成功到达0.200,-0.050,0.150"

    # 抓 red: 必须先确认场景有 red(预设 all)
    r = c.post("/api/v1/grab_the_block", json={"color": "red"})
    assert r.status_code == 200
    assert _body(r)["result"] == "有这种颜色的物块，且夹取物块成功"

    # 夹住后末端被抬升(> 物块 z + 抬升高度)
    p = env["backend"].get_end_effector_pose().position
    assert p[2] > 0.05, "夹住后应抬升到桌面之上"

    r = c.post("/api/v1/release_the_block")
    assert r.status_code == 200
    assert _body(r)["result"] == "成功释放物块"
    assert not env["backend"].is_holding_object()


def test_grab_color_not_present_via_api(app, env) -> None:
    env["backend"].reset("red_only")
    c = TestClient(app)
    r = c.post("/api/v1/grab_the_block", json={"color": "yellow"})
    assert _body(r)["result"] == "没有这种颜色的物块，无法夹取"


def test_unknown_color_via_api(app, env) -> None:
    env["backend"].reset("all")
    c = TestClient(app)
    r = c.post("/api/v1/grab_the_block", json={"color": "blue"})
    assert _body(r)["result"] == "没有这种颜色的物块，无法夹取"


# ---------------------------------------------------------------------------
# 参数错误 -> HTTP 400
# ---------------------------------------------------------------------------

def test_move_missing_field(app) -> None:
    c = TestClient(app)
    r = c.post("/api/v1/move_to_coordinates", json={"x": "0.1", "y": "0.2"})
    assert r.status_code == 400
    assert _body(r) == {"code": 400, "result": None, "error": "缺少必填字段: z"}


def test_move_non_numeric(app) -> None:
    c = TestClient(app)
    r = c.post("/api/v1/move_to_coordinates", json={"x": "abc", "y": "0", "z": "0"})
    assert r.status_code == 400
    assert _body(r)["code"] == 400
    assert "必须是数值字符串" in _body(r)["error"]


def test_grab_missing_color(app) -> None:
    c = TestClient(app)
    r = c.post("/api/v1/grab_the_block", json={})
    assert r.status_code == 400
    assert _body(r)["error"] == "缺少必填字段: color"


def test_send_to_voice_missing_content(app) -> None:
    c = TestClient(app)
    r = c.post("/api/v1/send_to_voice_agent", json={})
    assert r.status_code == 400
    assert _body(r)["error"] == "缺少必填字段: content"


# ---------------------------------------------------------------------------
# 队列接口
# ---------------------------------------------------------------------------

def test_queue_api_flow(app, env) -> None:
    c = TestClient(app)
    env["queue"].push_voice_to_arm("第一条")
    env["queue"].push_voice_to_arm("第二条")
    r = c.post("/api/v1/get_message_from_voice_agent")
    assert _body(r)["result"] == "all_messages_from_voice_agent:第一条;第二条"

    r = c.post("/api/v1/send_to_voice_agent", json={"content": "已到达目标位置"})
    assert _body(r)["result"] == "发送成功"
    assert env["queue"].pop_arm_to_voice_all() == ["已到达目标位置"]


# ---------------------------------------------------------------------------
# 中断: 同步 REST 调用执行中, 取消信号能让机械臂尽快停下
# ---------------------------------------------------------------------------

def test_api_move_can_be_cancelled(env) -> None:
    """用 realtime 节奏发起一个需要数秒的移动, 中途取消 -> 返回 未到达 + 误差。"""
    model, sim_cfg = env["model"], env["sim_cfg"]
    control = ControlConfig(**{**env["control"].__dict__, "pacing": "realtime"})
    from ...api.server import create_app
    from ...robot.kinematic_so101_backend import KinematicSO101Backend

    backend = KinematicSO101Backend(model, sim_cfg, control)
    backend.reset("all")
    app = create_app(backend, env["queue"], control, model)
    c = TestClient(app)

    result_holder: dict = {}

    def do_move() -> None:
        # 一个需要较大关节摆动的目标, realtime 下约数秒完成
        result_holder["r"] = c.post("/api/v1/move_to_coordinates",
                                    json={"x": "0.35", "y": "0.05", "z": "0.28"})

    t = threading.Thread(target=do_move, daemon=True)
    t.start()
    time.sleep(0.3)                       # 让移动开始执行
    assert state.request_cancel() is True # 取消当前动作
    t.join(timeout=10)

    r = result_holder["r"]
    assert r.status_code == 200
    assert _body(r)["result"].startswith("未到达")
    assert "，误差是" in _body(r)["result"]
    # 机械臂已停止, 之后不再自行运动
    q1 = backend.get_joint_positions().copy()
    time.sleep(0.1)
    assert all(abs(a - b) < 1e-6 for a, b in zip(q1, backend.get_joint_positions()))
