"""FastAPI 网关。路由只做三件事: 解析 / 校验 / 调 skill / 透传 result。

业务逻辑 300 行机器人控制代码严禁出现在本文件 —— 全部在 skills/ 下。
"""
from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from ..config import ControlConfig, VisionConfig
from ..perception.base import PerceptionBackend
from ..perception.camera import CameraProvider
from ..queue.backend import QueueBackend
from ..robot.backend import RobotBackend
from ..robot.kinematics import RobotModel
from ..runtime.state import state
from ..skills import (
    get_coordinates,
    grab_block,
    move_to_coordinates,
    release_block,
    voice_queue,
)
from ..skills._common import CoordinateError
from .schemas import (
    GrabRequest,
    MoveRequest,
    SendToVoiceRequest,
    error_response,
    ok_response,
)

_log = logging.getLogger(__name__)


def create_app(
    backend: RobotBackend,
    queue: QueueBackend,
    control: ControlConfig,
    model: RobotModel,
    vision: PerceptionBackend | None = None,
    camera: CameraProvider | None = None,
    vision_cfg: VisionConfig | None = None,
) -> FastAPI:
    """构建应用并把全局状态配置好。每次启动/测试重新调用即可。

    vision/camera/vision_cfg 为 Phase 2 可选注入: 只有 perception_mode == "hsv" 时需要。
    """
    state.configure(backend=backend, queue=queue, control=control, model=model,
                    vision=vision, camera=camera, vision_cfg=vision_cfg)

    app = FastAPI(title="SO-101 Arm Skill Server", version="0.1.0",
                  description="机械臂执行层 REST API (协议见 docs/api_of_embodied_tool.md)")

    app.add_exception_handler(RequestValidationError, _on_validation_error)
    app.add_exception_handler(Exception, _on_internal_error)

    # ---- 0 部署运维健康检查(非 Agent 工具; 协议四工具见 2.1-2.4) ----------
    @app.get("/health", tags=["ops"])
    def api_health() -> dict:
        return {
            "status": "ok",
            "simulator": control.backend_mode,
            "api": "ready",
        }

    # ---- 2.1 获取当前坐标 ------------------------------------------------
    @app.get("/api/v1/get_current_coordinates", tags=["arm"])
    def api_get_current_coordinates() -> dict:
        return ok_response(get_coordinates.get_current_coordinates())

    # ---- 2.2 移动到指定坐标 ----------------------------------------------
    @app.post("/api/v1/move_to_coordinates", tags=["arm"], response_model=None)
    def api_move_to_coordinates(req: MoveRequest):
        try:
            result = move_to_coordinates.move_to_coordinates(req.x, req.y, req.z)
        except CoordinateError as e:
            return JSONResponse(status_code=400, content=error_response(400, str(e)))
        return ok_response(result)

    # ---- 2.3 抓取指定颜色物块 --------------------------------------------
    @app.post("/api/v1/grab_the_block", tags=["arm"])
    def api_grab_the_block(req: GrabRequest) -> dict:
        return ok_response(grab_block.grab_the_block(req.color))

    # ---- 2.4 释放当前抓取的物块 ------------------------------------------
    @app.post("/api/v1/release_the_block", tags=["arm"])
    def api_release_the_block() -> dict:
        return ok_response(release_block.release_the_block())

    # ---- 2.5 生产消息给 voice agent --------------------------------------
    @app.post("/api/v1/send_to_voice_agent", tags=["agent"])
    def api_send_to_voice_agent(req: SendToVoiceRequest) -> dict:
        return ok_response(voice_queue.send_to_voice_agent(req.content))

    # ---- 2.6 消费来自 voice agent 的消息 ----------------------------------
    @app.post("/api/v1/get_message_from_voice_agent", tags=["agent"])
    def api_get_message_from_voice_agent() -> dict:
        return ok_response(voice_queue.get_message_from_voice_agent())

    return app


# ---------------------------------------------------------------------------
# 错误处理: 把 FastAPI 错误映射为协议的 {code, result, error} 结构
# ---------------------------------------------------------------------------

def _on_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    """参数错误 -> HTTP 400, code=400。缺失字段消息格式与文档示例一致。"""
    missing = [e["loc"][-1] for e in exc.errors() if e["type"] == "missing"]
    if missing:
        return JSONResponse(status_code=400,
                            content=error_response(400, f"缺少必填字段: {missing[0]}"))
    first = exc.errors()[0]
    field = first["loc"][-1]
    return JSONResponse(status_code=400,
                        content=error_response(400, f"字段 {field} 无效: {first['msg']}"))


def _on_internal_error(request: Request, exc: Exception) -> JSONResponse:
    """内部异常 -> HTTP 500, code=500。记录完整 traceback, 返回脱敏信息。"""
    _log.exception("内部异常: %s", exc)
    return JSONResponse(status_code=500,
                        content=error_response(500, f"内部异常: {exc.__class__.__name__}"))
