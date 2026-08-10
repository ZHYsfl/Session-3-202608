"""Voice Agent 工具网关(http://127.0.0.1:8001)。

与 Arm 网关 :8000 **共享同一个队列后端**(进程内注入同一个 QueueBackend 实例),
因此两个 Agent 经各自网关读写同一组 FIFO 队列, 实现全双工消息互通。

路由(见 docs/api_of_voice_tools.md):
    POST /api/v1/send_to_arm_agent           把语音命令发往 voice→arm 队列
    POST /api/v1/get_message_from_arm_agent  排空 arm→voice 队列(取 Arm Agent 结果)
    GET  /api/v1/queue_status                两队列长度(Arm Agent 实时中断监控用)

本网关不持有机械臂状态, 只做队列转发; 响应结构 {code,result,error} 同 Arm 网关。
"""
from __future__ import annotations

from fastapi import FastAPI

from ..queue.backend import QueueBackend
from .schemas import SendToArmRequest, ok_response


def create_voice_app(queue: QueueBackend) -> FastAPI:
    """构建语音侧网关。queue 必须与 Arm 网关共用(同一进程实例)。"""
    app = FastAPI(title="SO-101 Voice Agent Gateway", version="0.1.0",
                  description="语音侧网关: 双 Agent 队列转发(见 docs/api_of_voice_tools.md)")

    # ---- 把语音命令发往 Arm Agent(voice → arm) -------------------------
    @app.post("/api/v1/send_to_arm_agent", tags=["voice"])
    def api_send_to_arm_agent(req: SendToArmRequest) -> dict:
        queue.push_voice_to_arm(req.content)
        return ok_response("发送成功")

    # ---- 排空 Arm Agent 的结果(arm → voice) -----------------------------
    @app.post("/api/v1/get_message_from_arm_agent", tags=["voice"])
    def api_get_message_from_arm_agent() -> dict:
        msgs = queue.pop_arm_to_voice_all()
        if not msgs:
            return ok_response("当前没有新消息")
        return ok_response("all_messages_from_arm_agent:" + ";".join(msgs))

    # ---- 队列状态(实时中断监控) ----------------------------------------
    @app.get("/api/v1/queue_status", tags=["voice"])
    def api_queue_status() -> dict:
        return ok_response(
            f"voice_to_arm:{queue.voice_to_arm_size()};arm_to_voice:{queue.arm_to_voice_size()}")

    return app
