"""Arm Skill Server: SO-101 机械臂执行层 + 运动学仿真 + REST API。

包结构:
    robot/       运动学/轨迹/控制器 + RobotBackend 抽象与仿真实现
    simulation/  场景/物块/相机
    queue/       双 Agent 队列抽象
    runtime/     取消机制/全局状态
    skills/      6 个工具(业务逻辑)
    api/         FastAPI 网关(只做解析/校验/转发)
"""
__version__ = "0.1.0"
