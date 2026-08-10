"""InMemoryQueue：线程安全的进程内 FIFO 队列(Phase 1 默认实现)。

两条队列互不干扰, 各自用 deque + Lock 保证线程安全与入队顺序。
若日后接入 Redis/数据库, 实现 QueueBackend 即可, skills/api 不改。
"""
from __future__ import annotations

import threading
from collections import deque

from .backend import QueueBackend


class _Fifo:
    """单条线程安全 FIFO。"""

    def __init__(self) -> None:
        self._deque: deque[str] = deque()
        self._lock = threading.Lock()

    def push(self, content: str) -> None:
        with self._lock:
            self._deque.append(content)

    def pop_all(self) -> list[str]:
        with self._lock:
            items = list(self._deque)
            self._deque.clear()
            return items

    def __len__(self) -> int:
        with self._lock:
            return len(self._deque)


class InMemoryQueue(QueueBackend):
    def __init__(self) -> None:
        self._voice_to_arm = _Fifo()   # voice → arm
        self._arm_to_voice = _Fifo()   # arm → voice

    # voice → arm
    def push_voice_to_arm(self, content: str) -> None:
        self._voice_to_arm.push(content)

    def pop_voice_to_arm_all(self) -> list[str]:
        return self._voice_to_arm.pop_all()

    def voice_to_arm_size(self) -> int:
        return len(self._voice_to_arm)

    # arm → voice
    def push_arm_to_voice(self, content: str) -> None:
        self._arm_to_voice.push(content)

    def pop_arm_to_voice_all(self) -> list[str]:
        return self._arm_to_voice.pop_all()

    def arm_to_voice_size(self) -> int:
        return len(self._arm_to_voice)
