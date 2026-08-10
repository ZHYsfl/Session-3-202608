"""双 Agent 间通信传输层(队列 + 工具调用)。

Voice Agent 与 Arm Agent 是**两个独立 Agent**, 只通过队列 + 工具调用交互:
    VoiceAgent  -> voice→arm 队列   (把语音命令交给 Arm Agent)
    ArmAgent    -> arm→voice 队列   (把执行结果交回 Voice Agent)
    ArmAgent    -> REST 工具        (调用机械臂执行层, 即协议 §2 的 6 个工具)

两种实现, 接口一致:
    InProcessTransport  进程内直连(测试/快速联调, 无网络; 与 arm 网关共享队列)
    HttpTransport       HTTP 走 :8000(arm) / :8001(voice) 网关(最终 Demo 走真实 REST)

传输层只做转发, 不解析语义; 意图解析在 voice.intent(两个 Agent 各自持有)。
"""
from __future__ import annotations

import logging
from typing import Protocol

_log = logging.getLogger(__name__)


class DualAgentTransport(Protocol):
    """Voice/Arm 两个 Agent 依赖的通信接口。"""

    # 队列(voice → arm / arm → voice)
    def push_voice_to_arm(self, text: str) -> None: ...
    def pop_voice_to_arm_all(self) -> list[str]: ...
    def push_arm_to_voice(self, text: str) -> None: ...
    def pop_arm_to_voice_all(self) -> list[str]: ...
    def voice_to_arm_size(self) -> int: ...

    # 机械臂工具调用(协议 §2: grab/release/move/coordinates)
    def call_tool(self, name: str, **params) -> str: ...


class InProcessTransport:
    """进程内实现: 队列直连 + skill 直调(与 arm 网关共用同一队列/状态)。"""

    def __init__(self, queue) -> None:
        self._queue = queue

    def push_voice_to_arm(self, text: str) -> None:
        self._queue.push_voice_to_arm(text)

    def pop_voice_to_arm_all(self) -> list[str]:
        return self._queue.pop_voice_to_arm_all()

    def push_arm_to_voice(self, text: str) -> None:
        self._queue.push_arm_to_voice(text)

    def pop_arm_to_voice_all(self) -> list[str]:
        return self._queue.pop_arm_to_voice_all()

    def voice_to_arm_size(self) -> int:
        return self._queue.voice_to_arm_size()

    def call_tool(self, name: str, **params) -> str:
        # 延迟导入, 避免 transport 在 import 期触碰全局 state(configure 前)
        from ..skills import get_coordinates, grab_block, move_to_coordinates, release_block
        if name == "grab_the_block":
            return grab_block.grab_the_block(params["color"])
        if name == "release_the_block":
            return release_block.release_the_block()
        if name == "get_current_coordinates":
            return get_coordinates.get_current_coordinates()
        if name == "move_to_coordinates":
            return move_to_coordinates.move_to_coordinates(
                params["x"], params["y"], params["z"])
        raise ValueError(f"未知工具: {name}")


class HttpTransport:
    """HTTP 实现: 经 :8000(arm)/:8001(voice) 网关交互, 走真实 REST 路径。

    复用同一个 httpx.Client(连接池), 避免每次调用重建 TCP 连接;
    trust_env=False 绕过本机常驻代理, 直连 127.0.0.1。
    """

    def __init__(self, arm_url: str = "http://127.0.0.1:8000",
                 voice_url: str = "http://127.0.0.1:8001") -> None:
        import httpx
        self._arm = arm_url.rstrip("/")
        self._voice = voice_url.rstrip("/")
        self._client = httpx.Client(trust_env=False, timeout=600.0)

    def close(self) -> None:
        self._client.close()

    # ---- 队列 ---------------------------------------------------------
    def push_voice_to_arm(self, text: str) -> None:
        self._post(f"{self._voice}/api/v1/send_to_arm_agent", {"content": text})

    def pop_voice_to_arm_all(self) -> list[str]:
        return _pop_result(self._post(f"{self._arm}/api/v1/get_message_from_voice_agent", {}))

    def push_arm_to_voice(self, text: str) -> None:
        self._post(f"{self._arm}/api/v1/send_to_voice_agent", {"content": text})

    def pop_arm_to_voice_all(self) -> list[str]:
        return _pop_result(self._post(f"{self._voice}/api/v1/get_message_from_arm_agent", {}))

    def voice_to_arm_size(self) -> int:
        raw = self._get(f"{self._voice}/api/v1/queue_status") or ""
        for part in raw.split(";"):
            if part.startswith("voice_to_arm:"):
                return int(part.split(":", 1)[1])
        return 0

    # ---- 工具(协议 §2) ------------------------------------------------
    def call_tool(self, name: str, **params) -> str:
        if name == "grab_the_block":
            return self._post(f"{self._arm}/api/v1/grab_the_block",
                              {"color": params["color"]})
        if name == "release_the_block":
            return self._post(f"{self._arm}/api/v1/release_the_block", {})
        if name == "get_current_coordinates":
            # 协议 §2.1 是 GET; 其余工具是 POST
            return self._get(f"{self._arm}/api/v1/get_current_coordinates")
        if name == "move_to_coordinates":
            return self._post(f"{self._arm}/api/v1/move_to_coordinates",
                              {"x": params["x"], "y": params["y"], "z": params["z"]})
        raise ValueError(f"未知工具: {name}")

    # ---- HTTP 原语(复用连接池, 解包 {code,result,error}) ---------------
    def _post(self, url: str, payload: dict) -> str | None:
        r = self._client.post(url, json=payload)
        return _unpack(r)

    def _get(self, url: str) -> str | None:
        r = self._client.get(url)
        return _unpack(r)


def _unpack(r) -> str | None:
    """校验网关响应并解包 result。网关失败抛 RuntimeError。"""
    r.raise_for_status()
    data = r.json()
    if data.get("code") != 0:
        raise RuntimeError(f"工具调用失败 code={data.get('code')}: {data.get('error')}")
    return data.get("result")


def _pop_result(result: str | None) -> list[str]:
    """把网关的排空结果字符串还原为消息列表。"""
    if not result or result.startswith("当前没有"):
        return []
    if ":" in result:
        return result.split(":", 1)[1].split(";")
    return [result]
