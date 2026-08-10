"""针孔相机投影数学(纯函数, 无副作用, 可逐行学习、可单测)。

坐标系约定(全部显式标注, 单位: 长度 meter, 像素 pixel):
    - 世界系 W  = base frame: 机械臂底座原点, x 前, y 左, z 向上, 桌面平面 z=0。
    - 相机系 C  : 原点在光心, +z 指向相机前方(朝向场景), 图像平面在 z=f。
    - 图像系    : 像素 (u, v), 原点左上角, u 向右, v 向下。
    - 相机位姿  : world_T_cam (4x4) 满足  p_world = world_T_cam @ [p_cam, 1]。
      即 world_T_cam 的平移列 = 相机光心在世界系下的位置, 旋转列 = 相机三轴在世界系下的方向。

投影(p_world -> p_cam -> 像素):
    u = fx * x_cam / z_cam + cx
    v = fy * y_cam / z_cam + cy

反投影(像素 -> 世界射线):
    光心 o = world_T_cam[:3, 3];  方向 d = R_world @ [ (u-cx)/fx, (v-cy)/fy, 1 ]
    (像素与相机 +z 距离任意 -> 射线)

射线与水平平面 z=plane_z 求交:
    t = (plane_z - o.z) / d.z        (要求 d.z < 0, 即视线朝下; 否则无解)
    p = o + t·d
"""
from __future__ import annotations

import numpy as np


def build_camera_pose(position, look_at, up=(0.0, 0.0, 1.0)) -> np.ndarray:
    """由 相机位置 + 注视点 + 上方向 构造 world_T_cam (4x4)。

    - 相机 +z 指向 (look_at - position)(朝向场景)。
    - +x = up × z_cam, +y = z_cam × x_cam, 构成右手系。
    - up 与视线平行时抛 ValueError。
    """
    pos = np.asarray(position, dtype=float)
    target = np.asarray(look_at, dtype=float)
    up = np.asarray(up, dtype=float)

    z_cam = target - pos
    z_len = float(np.linalg.norm(z_cam))
    if z_len < 1e-12:
        raise ValueError("相机位置与注视点重合, 无法确定视线")
    z_cam = z_cam / z_len

    x_cam = np.cross(up, z_cam)
    x_len = float(np.linalg.norm(x_cam))
    if x_len < 1e-12:
        raise ValueError("上方向与视线平行, 无法构造相机坐标系")
    x_cam = x_cam / x_len

    y_cam = np.cross(z_cam, x_cam)

    T = np.eye(4)
    T[:3, :3] = np.column_stack([x_cam, y_cam, z_cam])
    T[:3, 3] = pos
    return T


def project_point(p_world, world_T_cam, fx, fy, cx, cy):
    """世界点 -> 像素 (u, v)。

    点在相机后方(z_cam <= 0)返回 None(不可见)。
    """
    p_world = np.asarray(p_world, dtype=float)
    if p_world.shape != (3,):
        raise ValueError("p_world 必须是 3 维向量")
    p_cam = np.linalg.inv(world_T_cam) @ np.r_[p_world, 1.0]
    if p_cam[2] <= 1e-9:
        return None
    u = fx * p_cam[0] / p_cam[2] + cx
    v = fy * p_cam[1] / p_cam[2] + cy
    return (float(u), float(v))


def unproject_pixel(u, v, world_T_cam, fx, fy, cx, cy):
    """像素 -> 世界射线 (origin_world, dir_world)。

    origin: 光心世界坐标 (3,)。dir: 相机系 (u,v) 方向变换到世界系 (3,),
    无需归一化——平面求交用 t 缩放即可。
    """
    dir_cam = np.array([(u - cx) / fx, (v - cy) / fy, 1.0])
    R = world_T_cam[:3, :3]
    origin = world_T_cam[:3, 3].copy()
    return origin, R @ dir_cam


def ray_intersect_plane(origin, direction, plane_z):
    """世界射线与水平平面 z=plane_z 的交点。

    返回 (x, y, plane_z); 无交点(视线不朝下 / 交点反向)返回 None。
    数学: p(t) = o + t·d, p.z = plane_z -> t = (plane_z - o.z)/d.z。
    """
    origin = np.asarray(origin, dtype=float)
    direction = np.asarray(direction, dtype=float)
    if direction[2] >= -1e-12:      # 视线 z 分量不为负: 无法从上往下打到桌面
        return None
    t = (plane_z - origin[2]) / direction[2]
    if t < 0.0:                     # 交点在射线反方向: 也不可见
        return None
    p = origin + t * direction
    return (float(p[0]), float(p[1]), float(plane_z))


def pixel_to_world_on_plane(u, v, world_T_cam, fx, fy, cx, cy, plane_z):
    """快捷组合: 像素 -> 桌面平面上世界坐标。不可靠投影返回 None。"""
    origin, direction = unproject_pixel(u, v, world_T_cam, fx, fy, cx, cy)
    return ray_intersect_plane(origin, direction, plane_z)
