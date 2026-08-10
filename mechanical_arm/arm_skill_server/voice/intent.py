"""确定性意图解析: 仿真「LLM 的工具规划」(离线, 无外部 API/密钥)。

把一条语音文本(键盘输入模拟 STT)解析为一条结构化命令。中英混合;
同一时刻只解析第一条命令(复合命令「先…然后…」不在范围, 文档注明)。

优先级(避免歧义, 低延迟优先):
    1. 取消/停止    -> Command("cancel")        最紧急, 最高优先
    2. 释放/放下    -> Command("release")
    3. 重置/复位    -> Command("reset")
    4. 坐标/在哪    -> Command("coordinates")
    5. 移动(x,y,z)  -> Command("move")   需要移动动词 且 ≥3 个数字
    6. 抓取(颜色)   -> Command("grab")
    7. 兜底: 只出现颜色词也当抓取("红色" -> grab red)
    8. 无法解析      -> UnknownIntentError

纯函数 / 无状态 / 无 IO —— 便于单测。禁止把"解析"做成黑盒神经网络(确定性
优先于一切)。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")

# 动作 -> 触发词(中文/英文混写)。词序在前, 命中即判。
_CANCEL_HINTS = (
    "取消", "停止", "停手", "中断", "停下", "住手", "别抓", "别动", "不要",
    "不用了", "算了", "不抓", "cancel", "stop", "abort", "undo", "halt",
)
_RELEASE_HINTS = (
    "放下", "释放", "松开", "放掉", "放开", "放手", "放下来", "松手",
    "release", "put down", "drop", "let go",
)
_RESET_HINTS = (
    "重置", "复位", "恢复初始", "重新开始", "回到初始", "reset", "初始化",
)
_COORD_HINTS = (
    "坐标", "在哪", "哪里", "在哪儿", "去哪儿", "去哪", "去哪里", "位置",
    "pose", "coordinate", "where",
)
_MOVE_HINTS = (
    "移动到", "移到", "移动至", "移动", "走到", "前往", "挪到", "挪动", "去",
    "move to", "go to", "move", "go",
)
_GRAB_HINTS = (
    "抓取", "拿取", "抓起", "拿起", "抓住", "握住", "夹取", "捡起", "拿来",
    "抓", "拿", "取", "捡", "夹",
    "grab", "pick up", "pick", "take",
)

# 颜色别名 -> 规范名(与协议 VALID_COLORS 对齐: yellow/red/white)。
# 长的放前面, 避免前缀误匹配("红色" 应先于 "红")。同色别名不跨映射, 顺序无影响。
_COLOR_ALIASES: dict[str, tuple[str, ...]] = {
    "red": ("红色", "赤色", "红块", "红色物块", "红",
            "red block", "red", "reds"),
    "yellow": ("黄色", "金色", "黄块", "黄色物块", "黄",
               "yellow block", "yellow", "yellows"),
    "white": ("白色", "白块", "白色物块", "白",
              "white block", "white", "whites"),
}


@dataclass(frozen=True)
class Command:
    """一条结构化命令。args 语义:
        grab        {"color": "red"|"yellow"|"white"}
        move        {"x": float, "y": float, "z": float}
        release     {}
        coordinates  {}
        cancel      {}
        reset       {}
    """

    action: str
    args: dict = field(default_factory=dict)


class UnknownIntentError(ValueError):
    """无法把语音文本解析为任何已知命令。"""


def parse_intent(text: str) -> Command:
    """把语音文本解析为命令; 无法解析抛 UnknownIntentError。"""
    t = _normalize(text)
    if not t:
        raise UnknownIntentError(text)

    # 1) 取消/停止(最紧急, 最高优先; "取" 也是抓取动词, 必须先于抓取判)
    if any(h in t for h in _CANCEL_HINTS):
        return Command("cancel")

    # 2) 释放
    if any(h in t for h in _RELEASE_HINTS):
        return Command("release")

    # 3) 重置(须先于坐标, "重置" 含 "置")
    if any(h in t for h in _RESET_HINTS):
        return Command("reset")

    # 4) 坐标("在哪里/你去哪" 归坐标, 先于移动)
    if any(h in t for h in _COORD_HINTS):
        return Command("coordinates")

    # 5) 移动: 移动动词 且 至少 3 个数字(坐标由顺序给出: x y z)
    if any(h in t for h in _MOVE_HINTS):
        nums = _extract_numbers(t)
        if len(nums) >= 3:
            return Command("move", {"x": nums[0], "y": nums[1], "z": nums[2]})

    # 6) 抓取: 抓取动词 + 颜色
    color = _find_color(t)
    if color is not None and any(h in t for h in _GRAB_HINTS):
        return Command("grab", {"color": color})

    # 7) 兜底: 只出现颜色词("红色" / "white block")也当抓取
    if color is not None:
        return Command("grab", {"color": color})

    raise UnknownIntentError(text)


# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """清洗: 去空白/标点(保留小数点, 坐标必须支持小数)。"""
    s = (text or "").strip().lower()
    # 注意: 字符类里**不能**含 "." —— 小数点对坐标数字是必需的。
    s = re.sub(r"[，。！？、；：,!?;:\"']+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _extract_numbers(text: str) -> list[float]:
    return [float(m) for m in _NUM_RE.findall(text)]


def _find_color(text: str) -> str | None:
    """按别名表找颜色; 未命中返回 None。长的别名先匹配。"""
    for canonical, aliases in _COLOR_ALIASES.items():
        if any(a in text for a in aliases):
            return canonical
    return None
