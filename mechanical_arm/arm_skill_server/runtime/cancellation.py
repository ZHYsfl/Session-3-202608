"""取消令牌(CancellationToken)。

用于「用户随时打断」: 任何长时间动作(move/grab/release)的控制循环
在每个控制步(约 20ms)检查 token.cancelled, 为真则立即 stop。

它是内部机制。公网是否新增 /cancel_current_action 是协议问题,
需要单独写 proposal, 经项目负责人确认后才能改正式接口。
"""
from __future__ import annotations

import threading


class CancellationError(Exception):
    """动作被取消时抛出(内部使用)。"""


class CancellationToken:
    """线程安全的取消标记。"""

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        """请求取消。之后 cancelled 为 True。"""
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        """在不可恢复的调用点使用: 已取消则抛 CancellationError。"""
        if self._event.is_set():
            raise CancellationError("动作已取消")
