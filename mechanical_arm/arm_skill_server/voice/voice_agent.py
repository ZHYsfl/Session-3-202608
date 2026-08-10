"""Voice Agent: 语音侧 Agent(与 Arm Agent 是**两个独立 Agent**)。

职责(离线仿真, 无外部 API/密钥):
    - 语音识别(仿真): SimulatedSTT —— 键盘输入文本即识别结果
    - 意图理解(仿真 LLM): 确定性 parse_intent —— 中英混合 -> 结构化命令
    - 把语音命令发往 voice→arm 队列(经 voice 网关 :8001 或进程内直连)
    - 监听 arm→voice 队列, 用 SimulatedTTS 播报 Arm Agent 的结果(全双工)

全双工: submit_voice() 只入队不阻塞; 结果由独立 speaker 线程异步播报。
```
"""
from __future__ import annotations

import threading
import time

from .intent import UnknownIntentError, parse_intent
from .io import SimulatedSTT, SimulatedTTS


class VoiceAgent:
    def __init__(self, transport, stt=None, tts=None, poll_s: float = 0.05) -> None:
        self._transport = transport
        self._stt = stt or SimulatedSTT()
        self._tts = tts or SimulatedTTS()
        self._poll_s = poll_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._speaker_loop,
                                        name="voice-speaker", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def submit_voice(self, raw: str) -> str:
        """处理一条语音输入(非阻塞): 识别 -> 理解 -> 发往 Arm Agent。

        返回理解到的命令动作("grab"/"move"/"release"/"coordinates"/"cancel"/
        "reset"); 空输入/未听懂返回 "empty"/"unknown"。
        """
        text = self._stt.recognize(raw)
        if not text:
            self._tts.speak("没听到内容, 请再说一遍")
            return "empty"
        try:
            cmd = parse_intent(text)
        except UnknownIntentError:
            self._tts.speak(f"没听懂: {text}, 请说 抓取/移动/释放/坐标/停止/重置")
            return "unknown"
        self._tts.speak(f"收到: {text}")
        self._transport.push_voice_to_arm(text)
        return cmd.action

    def _speaker_loop(self) -> None:
        """全双工: 持续排空 arm→voice 队列并播报 Arm Agent 的结果。"""
        while not self._stop.is_set():
            try:
                for msg in self._transport.pop_arm_to_voice_all():
                    self._tts.speak(msg)
            except Exception:  # noqa: BLE001 - 播报故障不杀监听线程
                pass
            time.sleep(self._poll_s)
