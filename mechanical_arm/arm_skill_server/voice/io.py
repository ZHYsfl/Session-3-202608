"""仿真语音 IO(Phase 6 纯仿真: 无外部语音服务/密钥)。

    SimulatedSTT  模拟语音识别: 键盘输入的文本就是识别结果。
    SimulatedTTS  模拟语音播报: 把内容打印为 `[语音] ...`。

真实语音(麦克风->ASR, TTS->扬声器)是未来接入点; 换实现类即可, Voice Agent
业务代码不改。
"""
from __future__ import annotations


class SimulatedSTT:
    """模拟语音识别: 键盘输入的文本直接作为识别结果(离线)。"""

    def recognize(self, raw: str) -> str:
        return (raw or "").strip()


class SimulatedTTS:
    """模拟语音播报: 终端打印 [语音] 前缀(离线, 不播放音频)。"""

    def speak(self, text: str) -> None:
        print(f"[语音] {text}")
