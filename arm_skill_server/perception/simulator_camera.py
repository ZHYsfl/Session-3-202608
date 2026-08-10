"""SimulatorCamera —— 基于 MuJoCo 离屏渲染的真实相机抽象(Phase 3)。

这是真正的**渲染器**相机: 取帧 = 调用 MuJoCo 渲染管线对当前物理场景光栅化,
得到与 3D 物理仿真一致的 RGB 图像(相机1/相机2)。HSV / TinyDetector 都消费它,
因此 Phase 3 起的视觉是「3D Renderer Vision」, 不是 SyntheticCamera 的示意图。

线程模型(Phase 3 集成发现的 bug):
    mujoco.Renderer 的 GLFW/GL 上下文绑定在**创建它的线程**上, render() 依赖该
    上下文"当前"。REST(uvicorn/TestClient)在线程池里跑同步 handler, 若 Renderer
    在主线程创建、在 worker 线程 render, GLFW 在 Windows 上无法把已绑定主线程的
    上下文切给 worker -> 渲染出黑帧(实测 mean≈0.1, 检测全丢)。
    修复: SimulatorCamera 用**专用渲染线程**持有 Renderer; capture() 把相机位姿 +
    取帧请求投递给渲染线程并等待结果。渲染线程写共享 MjData(相机位姿/mj_forward)
    时持有后端锁, 与物理步进串行化。

相机约定(经验证, 见 test_simulator_camera.py::test_camera_convention):
    MuJoCo 渲染器读取 data.cam_xpos / data.cam_xmat。
    我们的 perception 约定 world_T_cam 的列 = [x_cam, y_cam, z_cam](+z 前视),
    而 MuJoCo 需要的 cam_xmat 列 = [x_cam, -y_cam, -z_cam], 位置列 = T[:3,3]。

主相机(camera_model)为 camera1, 与 vision.yaml camera_id 一致;
camera2 是第三视角验证(抓取/释放), 通过 capture(camera_id=...) 取帧。
"""
from __future__ import annotations

import queue
import threading
import time
from pathlib import Path

import numpy as np

try:
    import mujoco
except ImportError as e:  # pragma: no cover
    raise ImportError("Phase 3 需要 mujoco: python -m pip install mujoco") from e

from ..config import VisionConfig
from ..robot.mujoco_so101_backend import MuJoCoSO101Backend
from .camera import CameraModel, CameraProvider, ImageFrame


class SimulatorCamera(CameraProvider):
    """把 MuJoCo 场景实时渲染成图像。构造后调用 capture() 取帧。"""

    def __init__(self, backend: MuJoCoSO101Backend, vision_cfg: VisionConfig | None = None,
                 save_dir: str | Path | None = None) -> None:
        self._backend = backend
        self._info = backend.scene_info
        self._cfg = vision_cfg if vision_cfg is not None else VisionConfig()
        self._m = self._info.model
        self._d = self._info.data
        self._save_dir = Path(save_dir) if save_dir is not None else None

        # 渲染线程: 唯一拥有 GL 上下文(Renderer)的线程。inbox 尺寸 1 天然限流。
        self._inbox: "queue.Queue[tuple[str, int, np.ndarray, dict]]" = queue.Queue(maxsize=1)
        self._render_thread = threading.Thread(target=self._render_loop, daemon=True)
        self._init_error: BaseException | None = None
        self._renderer: "mujoco.Renderer | None" = None
        self._render_thread.start()
        # 等待渲染线程创建好 Renderer(或记录创建失败), 保证 capture 可用
        for _ in range(200):
            if self._renderer is not None or self._init_error is not None:
                break
            time.sleep(0.005)
        if self._init_error is not None:
            raise self._init_error

    # ------------------------------------------------------------------
    def _render_loop(self) -> None:
        """渲染线程主循环: 持有 Renderer(GL 上下文), 串行处理取帧请求。"""
        try:
            self._renderer = mujoco.Renderer(
                self._m, self._cfg.image_height, self._cfg.image_width)
        except BaseException as e:  # 创建失败(如无 GL)记录, 由 capture 首次抛出
            self._init_error = e
            return
        while True:
            cid, idx, T, holder = self._inbox.get()
            try:
                # 写共享 MjData(相机位姿 + mj_forward)与物理步进串行化。
                # 必须先 mj_forward 再写相机位姿: mj_forward 会用模型里 <camera>
                # 的静态位姿(本场景里是空的)重算 cam_xpos/cam_xmat, 把运行时写入
                # 冲掉, 否则相机停在原点、被机械臂基座/连杆完全遮挡(Phase 3 集成发现)。
                with self._backend._lock:
                    mujoco.mj_forward(self._m, self._d)
                    self._d.cam_xpos[idx] = T[:3, 3]
                    # 经验验证的约定: 渲染器列 = [x_cam, -y_cam, -z_cam]
                    # (cam_xmat 是 (ncam,9) row-major)
                    self._d.cam_xmat[idx] = np.column_stack(
                        [T[:3, 0], -T[:3, 1], -T[:3, 2]]).ravel()
                    self._renderer.update_scene(self._d, camera=cid)
                    rgb = self._renderer.render()
                holder["bgr"] = rgb[:, :, ::-1].copy()   # RGB -> OpenCV BGR
                holder["err"] = None
            except BaseException as e:
                holder["err"] = e
            finally:
                holder["done"].set()

    # ------------------------------------------------------------------
    @property
    def camera_model(self) -> CameraModel:
        T = self._info.camera_world_T["camera1"]
        return CameraModel(
            camera_id=self._cfg.camera_id,
            width=self._cfg.image_width,
            height=self._cfg.image_height,
            fx=self._cfg.fx,
            fy=self._cfg.fy,
            cx=self._cfg.image_width / 2.0,
            cy=self._cfg.image_height / 2.0,
            world_T_cam=T,
        )

    def capture(self, camera_id: str | None = None) -> ImageFrame:
        cid = camera_id if camera_id is not None else self._cfg.camera_id
        if cid not in self._info.camera_id:
            raise KeyError(f"未知相机: {cid!r}, 可选 {sorted(self._info.camera_id)}")
        if self._renderer is None:
            raise RuntimeError("渲染线程未就绪(Renderer 创建失败)") from self._init_error

        idx = self._info.camera_id[cid]
        T = self._info.camera_world_T[cid]
        holder: dict = {"done": threading.Event(), "bgr": None, "err": None}
        self._inbox.put((cid, idx, T, holder))
        holder["done"].wait()
        if holder["err"] is not None:
            raise holder["err"]
        bgr = holder["bgr"]

        if self._save_dir is not None:
            self._save_dir.mkdir(parents=True, exist_ok=True)
            self._save_dir.joinpath(f"{cid}.png").write_bytes(_encode_png(bgr))

        return ImageFrame(camera_id=cid, bgr=bgr, timestamp=time.time())


def _encode_png(bgr: np.ndarray) -> bytes:
    """bgr uint8 (H,W,3) -> PNG bytes(纯标准库, 不依赖 cv2 的 imencode)。"""
    import io
    import struct
    import zlib

    h, w = bgr.shape[:2]
    raw = b"".join(b"\x00" + bgr[y].tobytes() for y in range(h))
    comp = zlib.compress(raw, 6)

    def chunk(tag: bytes, data: bytes) -> bytes:
        c = tag + data
        crc = zlib.crc32(tag)
        crc = zlib.crc32(data, crc)
        return struct.pack(">I", len(data)) + c + struct.pack(">I", crc & 0xFFFFFFFF)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)   # 8-bit, truecolor RGB
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", comp) + chunk(b"IEND", b"")
