"""Phase 6 双 Agent 集成测试。

覆盖(对应 Task #25/#26):
    - voice 网关 :8001 队列转发(与 arm 网关 :8000 共享同一队列)
    - Voice Agent 与 Arm Agent 全双工循环(进程内传输)
    - 实时中断: 动作执行中收到新指令 -> 取消当前 -> 如实上报中断 -> 执行新指令
    - HTTP 全链路: 两个真实 uvicorn 服务 + HttpTransport 走 REST 路径
"""
from __future__ import annotations

import threading
import time

from fastapi.testclient import TestClient

from ...config import ControlConfig
from ...queue.in_memory_queue import InMemoryQueue
from ...runtime.state import state
from ...voice import ArmAgent, HttpTransport, InProcessTransport, VoiceAgent
from ...voice.voice_agent import SimulatedTTS


class _RecTTS(SimulatedTTS):
    """录制播报内容, 便于断言全双工播报流。"""

    def __init__(self) -> None:
        self.said: list[str] = []

    def speak(self, text: str) -> None:
        self.said.append(text)


def _wait_for(pred, timeout: float = 10.0, interval: float = 0.02) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return False


# ---------------------------------------------------------------------------
# voice 网关 :8001(与 arm 网关共享队列)
# ---------------------------------------------------------------------------

def test_voice_gateway_forwards_between_agents(app, env) -> None:
    from ...api.voice_server import create_voice_app
    vc = TestClient(create_voice_app(env["queue"]))
    ac = TestClient(app)                    # 同一个队列实例

    # Voice 发命令 -> arm 网关能消费到
    r = vc.post("/api/v1/send_to_arm_agent", json={"content": "抓红色的物块"})
    assert r.status_code == 200 and r.json()["result"] == "发送成功"
    r = ac.post("/api/v1/get_message_from_voice_agent")
    assert r.json()["result"] == "all_messages_from_voice_agent:抓红色的物块"

    # Arm 上报结果 -> voice 网关能消费到
    r = ac.post("/api/v1/send_to_voice_agent",
                json={"content": "有这种颜色的物块，且夹取物块成功"})
    assert r.status_code == 200 and r.json()["result"] == "发送成功"
    r = vc.post("/api/v1/get_message_from_arm_agent")
    assert r.json()["result"] == "all_messages_from_arm_agent:有这种颜色的物块，且夹取物块成功"

    # 队列状态(Arm Agent 中断监控用)
    r = vc.get("/api/v1/queue_status")
    assert r.json()["result"] == "voice_to_arm:0;arm_to_voice:0"


def test_voice_gateway_empty(env) -> None:
    from ...api.voice_server import create_voice_app
    vc = TestClient(create_voice_app(env["queue"]))
    r = vc.post("/api/v1/get_message_from_arm_agent")
    assert r.json()["result"] == "当前没有新消息"
    r = vc.get("/api/v1/queue_status")
    assert "voice_to_arm:0" in r.json()["result"]


# ---------------------------------------------------------------------------
# 双 Agent 全双工循环(进程内传输, fast pacing)
# ---------------------------------------------------------------------------

def test_dual_agent_full_duplex_voice_pipeline(app, env) -> None:
    """语音 -> Voice Agent -> voice→arm 队列 -> Arm Agent -> 抓取/释放工具 ->
    arm→voice 队列 -> Voice Agent 播报。"""
    env["backend"].reset("all")
    transport = InProcessTransport(env["queue"])
    tts = _RecTTS()
    voice = VoiceAgent(transport, tts=tts)
    arm = ArmAgent(transport, poll_s=0.01, monitor_s=0.005)
    voice.start()
    arm.start()
    thread = threading.Thread(target=arm.run, daemon=True)
    thread.start()
    try:
        # 1) 语音: 抓红色
        assert voice.submit_voice("抓红色的物块") == "grab"
        assert _wait_for(lambda: any("夹取物块成功" in s for s in tts.said), timeout=10), tts.said
        assert env["backend"].is_holding_object()

        # 2) 语音: 释放
        assert voice.submit_voice("放下物块") == "release"
        assert _wait_for(lambda: any("成功释放物块" in s for s in tts.said), timeout=10), tts.said
        assert not env["backend"].is_holding_object()

        # 3) 语音: 坐标(命令被理解并执行, 播报含"我的坐标是")
        assert voice.submit_voice("你的坐标在哪") == "coordinates"
        assert _wait_for(lambda: any("我的坐标是" in s for s in tts.said), timeout=10), tts.said
    finally:
        arm.stop()
        voice.stop()


def test_dual_agent_grab_color_not_present(app, env) -> None:
    """语音抓不存在的颜色 -> 诚实上报"没有这种颜色的物块"。"""
    env["backend"].reset("red_only")
    transport = InProcessTransport(env["queue"])
    tts = _RecTTS()
    voice = VoiceAgent(transport, tts=tts)
    arm = ArmAgent(transport, poll_s=0.01, monitor_s=0.005)
    voice.start()
    arm.start()
    thread = threading.Thread(target=arm.run, daemon=True)
    thread.start()
    try:
        assert voice.submit_voice("抓黄色") == "grab"
        assert _wait_for(
            lambda: any("没有这种颜色的物块，无法夹取" in s for s in tts.said), timeout=10), tts.said
        assert not env["backend"].is_holding_object()
    finally:
        arm.stop()
        voice.stop()


def test_dual_agent_unknown_voice(app, env) -> None:
    """听不懂 -> Voice Agent 立即播报提示, 不打扰 Arm Agent。"""
    transport = InProcessTransport(env["queue"])
    tts = _RecTTS()
    voice = VoiceAgent(transport, tts=tts)
    voice.start()
    try:
        assert voice.submit_voice("今天天气怎么样") == "unknown"
        assert any("没听懂" in s for s in tts.said)
        assert env["queue"].voice_to_arm_size() == 0   # 未入队
    finally:
        voice.stop()


# ---------------------------------------------------------------------------
# 实时中断(协议 §3: 忙碌时收到新指令 -> 中断当前 -> 执行新指令)
# ---------------------------------------------------------------------------

def test_dual_agent_realtime_interrupt(env) -> None:
    """执行长移动中收到"取消" -> 当前动作被中断 -> 如实上报 -> 再上报已取消。"""
    model, sim_cfg = env["model"], env["sim_cfg"]
    control = ControlConfig(**{**env["control"].__dict__, "pacing": "realtime"})
    from ...robot.kinematic_so101_backend import KinematicSO101Backend

    backend = KinematicSO101Backend(model, sim_cfg, control)
    backend.reset("all")
    queue = InMemoryQueue()
    state.configure(backend=backend, queue=queue, control=control, model=model)

    transport = InProcessTransport(queue)
    tts = _RecTTS()
    voice = VoiceAgent(transport, tts=tts)
    arm = ArmAgent(transport, poll_s=0.01, monitor_s=0.005)
    voice.start()
    arm.start()
    thread = threading.Thread(target=arm.run, daemon=True)
    thread.start()
    try:
        voice.submit_voice("移动到 0.35 0.05 0.28")   # 大摆动, realtime 下数秒
        time.sleep(0.3)                               # 让移动开始执行
        voice.submit_voice("取消")

        assert _wait_for(
            lambda: any(s.startswith("动作被新指令中断") for s in tts.said), timeout=15), tts.said
        assert _wait_for(
            lambda: any("已取消当前动作" in s for s in tts.said), timeout=10), tts.said

        # 机械臂确实停了: 之后不再自行运动
        q1 = backend.get_joint_positions().copy()
        time.sleep(0.2)
        assert all(abs(a - b) < 1e-6 for a, b in zip(q1, backend.get_joint_positions()))
    finally:
        arm.stop()
        voice.stop()


# ---------------------------------------------------------------------------
# HTTP 全链路(真实 uvicorn 服务 + HttpTransport)
# ---------------------------------------------------------------------------

def test_dual_agent_over_http() -> None:
    """两个真实 HTTP 网关 + HttpTransport, 双 Agent 全双工走 REST 路径。"""
    import uvicorn

    model, control, sim_cfg = _load_env()
    backend, queue = _build_runtime(model, control, sim_cfg)
    backend.reset("all")

    from ...api.server import create_app
    from ...api.voice_server import create_voice_app
    arm_app = create_app(backend, queue, control, model)
    voice_app = create_voice_app(queue)

    srv_arm, arm_port = _start_server(arm_app)
    srv_voice, voice_port = _start_server(voice_app)
    try:
        transport = HttpTransport(
            arm_url=f"http://127.0.0.1:{arm_port}",
            voice_url=f"http://127.0.0.1:{voice_port}")

        tts = _RecTTS()
        voice = VoiceAgent(transport, tts=tts)
        arm = ArmAgent(transport, poll_s=0.05, monitor_s=0.02)
        voice.start()
        arm.start()
        thread = threading.Thread(target=arm.run, daemon=True)
        thread.start()
        try:
            assert voice.submit_voice("抓红色的物块") == "grab"
            assert _wait_for(
                lambda: any("夹取物块成功" in s for s in tts.said), timeout=20), tts.said
            assert backend.is_holding_object()
            assert voice.submit_voice("放下物块") == "release"
            assert _wait_for(
                lambda: any("成功释放物块" in s for s in tts.said), timeout=20), tts.said
            assert not backend.is_holding_object()
        finally:
            arm.stop()
            voice.stop()
    finally:
        _stop_server(srv_arm)
        _stop_server(srv_voice)


# ---------------------------------------------------------------------------
# 测试基础设施
# ---------------------------------------------------------------------------

def _load_env():
    from ...config import load_all
    model, control, sim_cfg = load_all()
    control = ControlConfig(**{**control.__dict__, "pacing": "fast",
                               "perception_mode": "ground_truth"})
    return model, control, sim_cfg


def _build_runtime(model, control, sim_cfg):
    from ...robot.kinematic_so101_backend import KinematicSO101Backend
    backend = KinematicSO101Backend(model, sim_cfg, control)
    queue = InMemoryQueue()
    state.configure(backend=backend, queue=queue, control=control, model=model)
    return backend, queue


def _start_server(app):
    """后台线程起 uvicorn, 真正就绪(能返回 200)才返回。

    Windows 下 loopback HTTP 必须直连(本机常驻代理会把 127.0.0.1 转发成 502),
    因此就绪探测用 trust_env=False。
    """
    import socket

    import httpx
    import uvicorn

    # 先找一个空闲端口, 再让 uvicorn 绑定它(避免 port=0 的端口发现竞态)
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()

    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    with httpx.Client(trust_env=False) as client:
        for _ in range(200):
            if server.started:
                try:
                    if client.get(f"http://127.0.0.1:{port}/docs",
                                  timeout=0.5).status_code == 200:
                        return server, port
                except Exception:
                    pass
            time.sleep(0.05)
    raise RuntimeError("uvicorn 未能启动")


def _bound_port(server_port) -> int:
    return server_port


def _stop_server(server) -> None:
    server.should_exit = True
