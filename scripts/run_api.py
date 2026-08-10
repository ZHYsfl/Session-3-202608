"""启动 Arm Skill Server (HEADLESS SERVER MODE)。

用法:
    python scripts/run_api.py

环境变量(可覆盖, 服务器部署用):
    ARM_SERVER_HOST   监听地址, 默认 0.0.0.0 (所有网卡, 供远程联调)
    ARM_SERVER_PORT   监听端口, 默认 8000

后端由 configs/control.yaml 的 backend_mode 决定:
    kinematic : 运动学后端 + 合成相机(Phase 1/2, 零依赖)
    mujoco    : 真 3D 物理(MuJoCo)+ 渲染器相机(Phase 3, 需 pip install mujoco)

HEADLESS 说明:
    本入口**不启动任何 GUI viewer**(run_arm_demo.py / run_3d_sim.py 才有)。
    mujoco 后端用 SimulatorCamera 离屏渲染(mujoco.Renderer), 服务器无显示器
    也能运行; 无 EGL/OpenGL 后端时渲染线程会失败并给出明确错误(不伪造支持)。
    GUI 演示与服务器部署完全解耦: 服务器只提供 API, 演示另跑 run_arm_demo.py。

正式协议路径不变(见 docs/api_of_embodied_tool.md)。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# 允许从项目根目录直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import uvicorn

from arm_skill_server.api.server import create_app
from arm_skill_server.config import ControlConfig, load_all
from arm_skill_server.queue.in_memory_queue import InMemoryQueue
from arm_skill_server.runtime.factory import build_runtime


def main() -> None:
    host = os.environ.get("ARM_SERVER_HOST", "0.0.0.0")
    port = int(os.environ.get("ARM_SERVER_PORT", "8000"))
    # 可选: 覆盖 configs/control.yaml 的 backend_mode。
    #   ARM_BACKEND=mujoco    真 3D 物理(机械臂模块核心交付, 需 pip install mujoco)
    #   ARM_BACKEND=kinematic 纯运动学(零依赖)
    arm_backend = os.environ.get("ARM_BACKEND")

    model, control, sim_cfg = load_all()
    if arm_backend:
        control = ControlConfig(**{**control.__dict__, "backend_mode": arm_backend})
    backend, camera, vision, vision_cfg = build_runtime(model, control, sim_cfg)
    backend.reset("all")                       # 三种颜色物块都放上
    queue = InMemoryQueue()

    app = create_app(backend, queue, control, model,
                     vision=vision, camera=camera, vision_cfg=vision_cfg)

    print(f"启动 Arm Skill Server: http://{host}:{port}")
    print(f"  backend_mode    : {control.backend_mode}")
    print(f"  perception_mode : {control.perception_mode}")
    print(f"  grasp_mode      : {control.grasp_mode}")
    print(f"  pacing          : {control.pacing}")
    print(f"  GUI             : 无 (headless server; GUI 演示用 run_arm_demo.py)")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
