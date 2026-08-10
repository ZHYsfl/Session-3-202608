"""请求模型与响应封装(严格匹配协议文档)。

响应结构:
    {"code": int, "result": str | null, "error": str | null}
协议要求 x/y/z/color/content 都是字符串类型。
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class MoveRequest(BaseModel):
    """POST /api/v1/move_to_coordinates 请求体。"""

    model_config = ConfigDict(extra="forbid")
    x: str
    y: str
    z: str


class GrabRequest(BaseModel):
    """POST /api/v1/grab_the_block 请求体。"""

    model_config = ConfigDict(extra="forbid")
    color: str


class SendToVoiceRequest(BaseModel):
    """POST /api/v1/send_to_voice_agent 请求体。"""

    model_config = ConfigDict(extra="forbid")
    content: str


class SendToArmRequest(BaseModel):
    """POST /api/v1/send_to_arm_agent 请求体(voice 网关 :8001)。"""

    model_config = ConfigDict(extra="forbid")
    content: str


# ---- 响应封装 -----------------------------------------------------------

def ok_response(result: str) -> dict:
    """业务成功(HTTP 200, code=0)。result 原样透传。"""
    return {"code": 0, "result": result, "error": None}


def error_response(code: int, error: str) -> dict:
    """网关层失败(HTTP 400/500, code 对应)。"""
    return {"code": code, "result": None, "error": error}
