#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
ARX-X5 teleoperation data collection with aligned start/end pose.

This script keeps the original pure teleop data format, but adds a deployment-like
initial pose alignment before recording and after saving an episode. It is meant
to reduce the train/deploy initial-state mismatch for absolute-joint policies.
"""

import argparse
import json
import logging
import os
import queue
import shutil
import select
import signal
import sys
import termios
import threading
import time
import tty
from collections import deque
from pathlib import Path
from typing import Dict, List, Optional

if sys.version_info[:2] != (3, 10):
    raise SystemExit(
        f"ROS Humble rclpy in this setup expects Python 3.10, but this interpreter is "
        f"Python {sys.version_info.major}.{sys.version_info.minor}.\n"
        "Do not run this script from conda base if base is Python 3.13.\n"
        "Use a Python 3.10 environment, for example:\n"
        "  conda activate kai0_inference\n"
        "  source /opt/ros/humble/setup.bash\n"
        "  source X5_ws/install/setup.bash\n"
        "  python arx_teleop_collect.py ...\n"
        "or run it with the ROS/system Python 3.10 after installing the needed Python packages there."
    )

import h5py
import numpy as np

try:
    import cv2
except ImportError as e:
    raise SystemExit(
        "OpenCV (cv2) is required for camera capture and visualization.\n"
        "You are likely running in the wrong Python environment. Try:\n"
        "  conda activate kai0_inference\n"
        "or install it in the current env:\n"
        "  pip install opencv-python\n"
        "Then rerun arx_teleop_collect.py."
    ) from e

try:
    import av
except ImportError:
    av = None

import rclpy
from rclpy.node import Node
from rcl_interfaces.srv import SetParameters
from rcl_interfaces.msg import Parameter, ParameterValue, ParameterType
from std_msgs.msg import Int32MultiArray

try:
    from arx5_arm_msg.msg import RobotStatus
    print("[INFO] arx5_arm_msg loaded")
except ImportError as e:
    raise SystemExit(
        "arx5_arm_msg was not found in PYTHONPATH.\n"
        "Source ROS Humble and your ARX X5_ws overlay before running this script.\n"
        "For your current machine, the correct ARX workspace appears to be:\n"
        "  source /opt/ros/humble/setup.bash\n"
        "  source /home/agilex/ARX_X5-main/ARX_X5-main/ROS2/X5_ws/install/setup.bash\n"
        "\n"
        "If you use the bundled workspace in this repo instead, use:\n"
        "  source /opt/ros/humble/setup.bash\n"
        "  source /home/agilex/kai0/train_deploy_alignment/dagger/arx/X5_ws/install/setup.bash\n"
        "Then rerun arx_teleop_collect.py from the same terminal.\n"
        "\n"
        "If the repo-local X5_ws/install is missing or stale, rebuild it first:\n"
        "  cd /home/agilex/kai0/train_deploy_alignment/dagger/arx/X5_ws\n"
        "  colcon build\n"
        "  source install/setup.bash"
    ) from e


CAMERA_NAMES = ["cam_high", "cam_right_wrist", "cam_left_wrist"]
DEFAULT_DATASET_DIR = str(Path(__file__).resolve().parents[3] / "kai0_data" / "arx_teleop")
DEFAULT_COLLECT_LEFT0 = [-0.00972748, 0.44651699, 0.81998158, -0.43850613, -0.01087189, -0.08220768, 5.0]
DEFAULT_COLLECT_RIGHT0 = [-0.00972748, 0.44651699, 0.81998158, -0.43850613, -0.01087189, -0.08220768, 5.0]
DEFAULT_ARX_GO_HOME_LEFT = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0]
DEFAULT_ARX_GO_HOME_RIGHT = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 5.0]
TRUE_STRINGS = {"1", "true", "t", "yes", "y", "on", "ture"}
FALSE_STRINGS = {"0", "false", "f", "no", "n", "off", "none"}

shutdown_event = threading.Event()
save_requested = False
discard_requested = False
request_lock = threading.Lock()


def str2bool(value):
    """Parse bool-like CLI values; accepts 'ture' as a common typo for true."""
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in TRUE_STRINGS:
        return True
    if value in FALSE_STRINGS:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {value}")


def encode_video_frames(
    images: np.ndarray,
    dst: Path,
    fps: int,
    vcodec: str = "libx264",
    pix_fmt: str = "yuv420p",
    g: int = 2,
    crf: int = 23,
    fast_decode: int = 0,
    log_level=None,
    overwrite: bool = False,
) -> None:
    if av is None:
        raise RuntimeError("PyAV is required for video export. Install `av` or run with `--save_video false`.")
    if vcodec not in {"h264", "hevc", "libx264", "libx265", "libsvtav1"}:
        raise ValueError(f"Unsupported codec {vcodec}")
    video_path = Path(dst)
    video_path.parent.mkdir(parents=True, exist_ok=overwrite)
    if (vcodec in {"libsvtav1", "hevc", "libx265"}) and pix_fmt == "yuv444p":
        pix_fmt = "yuv420p"
    h, w, _ = images[0].shape
    options = {}
    for k, v in {"g": g, "crf": crf}.items():
        if v is not None:
            options[k] = str(v)
    if fast_decode:
        key = "svtav1-params" if vcodec == "libsvtav1" else "tune"
        options[key] = f"fast-decode={fast_decode}" if vcodec == "libsvtav1" else "fastdecode"
    if log_level is None:
        log_level = av.logging.ERROR
    if log_level is not None:
        logging.getLogger("libav").setLevel(log_level)
    with av.open(str(video_path), "w") as out:
        stream = out.add_stream(vcodec, fps, options=options)
        stream.pix_fmt, stream.width, stream.height = pix_fmt, w, h
        for i, img in enumerate(images):
            frame = av.VideoFrame.from_ndarray(img, format="rgb24")
            for pkt in stream.encode(frame):
                out.mux(pkt)
            if (i + 1) % 100 == 0 or i == len(images) - 1:
                print(f"Encoding frame {i + 1}")
        for pkt in stream.encode():
            out.mux(pkt)
    if log_level is not None:
        av.logging.restore_default_callback()
    if not video_path.exists():
        raise OSError(f"Video encoding failed: {video_path}")


def create_video_from_images(images, output_path, fps=30, codec="libx264", quality=23):
    if not images:
        raise ValueError("No image data")
    if av is not None:
        try:
            print(f"Encoding video, codec: {codec} CRF: {quality}")
            encode_video_frames(np.asarray(images), Path(output_path), fps=fps, vcodec=codec, crf=quality, overwrite=True)
            print(f"Video saved to: {output_path}")
            return
        except Exception as e:
            print(f"[WARN] PyAV video encoding failed, falling back to OpenCV mp4v: {e}")

    print("Encoding video with OpenCV mp4v fallback")
    video_path = Path(output_path)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    first = np.asarray(images[0])
    h, w = first.shape[:2]
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (w, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV VideoWriter failed to open: {video_path}")
    try:
        for i, img in enumerate(images):
            frame = np.asarray(img)
            if frame.ndim == 2:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            elif frame.shape[-1] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
            else:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if frame.shape[:2] != (h, w):
                frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
            writer.write(frame)
            if (i + 1) % 100 == 0 or i == len(images) - 1:
                print(f"Encoding frame {i + 1}")
    finally:
        writer.release()
    if not video_path.exists() or video_path.stat().st_size == 0:
        raise OSError(f"OpenCV video encoding failed: {video_path}")
    print(f"Video saved to: {output_path}")


def save_data(observations, actions, dataset_path):
    """Save HDF5 in the existing ARX/DAgger joint-only format."""
    data_size = len(actions)
    data_dict = {
        "/observations/qpos": [],
        "/observations/qvel": [],
        "/observations/effort": [],
        "/action": [],
    }

    for obs, action in zip(observations, actions):
        data_dict["/observations/qpos"].append(obs["qpos"])
        data_dict["/observations/qvel"].append(obs["qvel"])
        data_dict["/observations/effort"].append(obs["effort"])
        data_dict["/action"].append(action)

    t0 = time.time()
    print("\033[33m>>> Saving HDF5...\033[0m")

    with h5py.File(dataset_path + ".hdf5", "w", rdcc_nbytes=1024**2 * 2) as root:
        root.attrs["sim"] = False
        root.attrs["compress"] = False
        obs_grp = root.create_group("observations")
        obs_grp.create_dataset("qpos", (data_size, 14), dtype="float32")
        obs_grp.create_dataset("qvel", (data_size, 14), dtype="float32")
        obs_grp.create_dataset("effort", (data_size, 14), dtype="float32")
        root.create_dataset("action", (data_size, 14), dtype="float32")

        for name, arr in data_dict.items():
            root[name][...] = np.asarray(arr, dtype=np.float32)

    print(f"[INFO] HDF5 saved in {time.time() - t0:.1f}s")
    print(f"\033[32m  Path: {dataset_path}.hdf5\033[0m")
    print(f"\033[32m  Frames: {data_size}\033[0m")


def save_videos(observations, dataset_path, camera_names, fps=30, video_stem=None):
    """Save camera streams as videos under video/<camera_name>/<index>.mp4."""
    if len(observations) == 0:
        print("\033[31mNo image data for video.\033[0m")
        return

    dataset_dir = os.path.dirname(dataset_path)
    episode_name = video_stem or os.path.basename(dataset_path)
    print("\033[33m>>> Exporting video...\033[0m")
    t0 = time.time()

    for cam_name in camera_names:
        video_dir = os.path.join(dataset_dir, "video", cam_name)
        os.makedirs(video_dir, exist_ok=True)
        images = [obs["images"][cam_name] for obs in observations]
        video_path = os.path.join(video_dir, f"{episode_name}.mp4")
        create_video_from_images(images, video_path, fps)
        print(f"\033[36m  Video: {cam_name} -> {video_path}\033[0m")

    sample_img = observations[0]["images"][camera_names[0]]
    h, w = sample_img.shape[:2]
    print(f"[INFO] Video saved in {time.time() - t0:.1f}s")
    print(f"\033[32m  Resolution: {w}x{h}, frames: {len(observations)}, fps: {fps}\033[0m")


class TeleopCollector:
    """Async episode collector with background queue draining and save requests."""

    def __init__(
        self,
        camera_names,
        dataset_dir=DEFAULT_DATASET_DIR,
        dataset_name="arx_teleop",
        video_fps=30,
        save_depth=False,
        depth_camera_name="cam_high",
    ):
        self.camera_names = camera_names
        self.dataset_dir = dataset_dir
        self.dataset_name = dataset_name
        self.video_fps = video_fps
        self.save_depth = bool(save_depth)
        if isinstance(depth_camera_name, str):
            self.depth_camera_names = [name.strip() for name in depth_camera_name.split(",") if name.strip()]
        else:
            self.depth_camera_names = [str(name).strip() for name in depth_camera_name if str(name).strip()]
        self.depth_camera_name = self.depth_camera_names[0] if self.depth_camera_names else ""
        self.full_dataset_dir = os.path.join(dataset_dir, dataset_name)
        os.makedirs(self.full_dataset_dir, exist_ok=True)

        self._frame_queue = queue.Queue(maxsize=10000)
        self._lock = threading.Lock()
        self._data_lock = threading.Lock()
        self._writer_thread = None
        self._writer_running = False

        self._current_observations: List[Dict[str, np.ndarray]] = []
        self._current_actions: List[np.ndarray] = []
        self.is_collecting = False
        self.frame_count = 0
        self.episode_idx = self._find_next_episode_idx()
        self._save_requested = False
        self._save_config = {}
        self._resume_after_save = False
        self._tmp_depth_root = os.path.join(self.full_dataset_dir, "_tmp_depth")
        self._tmp_depth_dirs = {}
        self._depth_frame_counts = {}
        self._depth_meta = {}
        self._reset_depth_tmp()

    def _find_next_episode_idx(self):
        if not os.path.exists(self.full_dataset_dir):
            return 0
        episodes = [f for f in os.listdir(self.full_dataset_dir) if f.startswith("episode_") and f.endswith(".hdf5")]
        if not episodes:
            return 0
        idxs = []
        for f in episodes:
            try:
                idxs.append(int(f.split("_")[1].split(".")[0]))
            except Exception:
                continue
        return max(idxs) + 1 if idxs else 0

    def _reset_depth_tmp(self):
        if not self.save_depth:
            return
        for tmp_dir in getattr(self, "_tmp_depth_dirs", {}).values():
            if tmp_dir and os.path.exists(tmp_dir):
                shutil.rmtree(tmp_dir, ignore_errors=True)
        if os.path.exists(self._tmp_depth_root):
            shutil.rmtree(self._tmp_depth_root, ignore_errors=True)
        self._tmp_depth_dirs = {}
        self._depth_frame_counts = {}
        self._depth_meta = {}

    def _write_depth_frame(self, frame_idx: int, depth_payload: dict):
        if not self.save_depth or depth_payload is None:
            return
        if not self._tmp_depth_dirs:
            self._reset_depth_tmp()
        for camera_name in self.depth_camera_names:
            depth = depth_payload.get(camera_name)
            if depth is None:
                continue
            tmp_dir = self._tmp_depth_dirs.get(camera_name)
            if tmp_dir is None:
                tmp_dir = os.path.join(self._tmp_depth_root, camera_name, f"episode_{self.episode_idx}")
                os.makedirs(tmp_dir, exist_ok=True)
                self._tmp_depth_dirs[camera_name] = tmp_dir
                self._depth_frame_counts[camera_name] = 0
                self._depth_meta[camera_name] = {}
            depth_uint16 = np.asarray(depth, dtype=np.uint16)
            depth_path = os.path.join(tmp_dir, f"{frame_idx:06d}.png")
            ok = cv2.imwrite(depth_path, depth_uint16)
            if not ok:
                print(f"[WARN] Failed to write depth PNG: {depth_path}")
                continue
            self._depth_frame_counts[camera_name] = self._depth_frame_counts.get(camera_name, 0) + 1
            meta = depth_payload.get(f"{camera_name}_meta", {})
            if meta:
                self._depth_meta.setdefault(camera_name, {}).update(meta)

    def _finalize_depth_episode(self, episode_idx: int):
        if not self.save_depth:
            return
        wrote_any = False
        for camera_name in self.depth_camera_names:
            tmp_dir = self._tmp_depth_dirs.get(camera_name)
            frame_count = int(self._depth_frame_counts.get(camera_name, 0))
            if not tmp_dir or not os.path.exists(tmp_dir) or frame_count == 0:
                continue
            final_root = os.path.join(self.full_dataset_dir, "depth", camera_name)
            final_dir = os.path.join(final_root, str(episode_idx))
            os.makedirs(final_root, exist_ok=True)
            if os.path.exists(final_dir):
                shutil.rmtree(final_dir)
            shutil.move(tmp_dir, final_dir)
            meta = {
                "camera": camera_name,
                "format": "uint16_png",
                "unit": "millimeter",
                "invalid_value": 0,
                "source": "unknown",
                "last_episode": int(episode_idx),
                "last_num_frames": frame_count,
            }
            meta.update(self._depth_meta.get(camera_name, {}))
            meta_path = os.path.join(final_root, "meta.json")
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
            print(f"\033[36m  Depth: {final_dir} ({frame_count} png)\033[0m")
            print(f"\033[36m  Depth meta: {meta_path}\033[0m")
            wrote_any = True
        if os.path.exists(self._tmp_depth_root):
            shutil.rmtree(self._tmp_depth_root, ignore_errors=True)
        if not wrote_any:
            print("[WARN] save_depth is enabled but no depth frames were written.")

    def _writer_loop(self):
        while self._writer_running:
            try:
                obs, action = self._frame_queue.get(timeout=0.1)
                with self._data_lock:
                    self._current_observations.append(obs)
                    self._current_actions.append(np.asarray(action, dtype=float).copy())
            except queue.Empty:
                pass

            if self._save_requested:
                self._do_save()
                self._save_requested = False

    def _drain_frame_queue(self):
        drained = 0
        while True:
            try:
                obs, action = self._frame_queue.get_nowait()
            except queue.Empty:
                break
            with self._data_lock:
                self._current_observations.append(obs)
                self._current_actions.append(np.asarray(action, dtype=float).copy())
            drained += 1
        return drained

    def _do_save(self):
        self._drain_frame_queue()
        with self._data_lock:
            data_len = len(self._current_actions)
        if data_len == 0:
            print("[ERROR] No data to save")
            return

        dataset_path = os.path.join(self.full_dataset_dir, f"episode_{self.episode_idx}")
        export_video = self._save_config.get("export_video", True)
        video_fps = self._save_config.get("video_fps", self.video_fps)

        try:
            with self._data_lock:
                observations_copy = list(self._current_observations)
                actions_copy = list(self._current_actions)
            save_data(observations_copy, actions_copy, dataset_path)

            has_images = observations_copy and "images" in observations_copy[0]
            if export_video and has_images:
                try:
                    video_stem = str(self.episode_idx)
                    save_videos(observations_copy, dataset_path, self.camera_names, fps=video_fps, video_stem=video_stem)
                except Exception as e:
                    print(f"[WARN] Video export failed, HDF5 is still saved: {e}")
            self._finalize_depth_episode(self.episode_idx)

            print(f"\n[INFO] Saved: {dataset_path}.hdf5 ({len(actions_copy)} frames)")
            self.episode_idx += 1
            self.clear_current_episode()
            self._reset_depth_tmp()
            if self._resume_after_save:
                self.is_collecting = True
            self._resume_after_save = False
            print(f"[INFO] Ready for episode {self.episode_idx}")
        except Exception as e:
            print(f"[ERROR] Save failed: {e}")
            self._resume_after_save = False
            import traceback

            traceback.print_exc()

    def start_writer(self):
        with self._lock:
            if self._writer_thread is None or not self._writer_thread.is_alive():
                self._writer_running = True
                self._writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
                self._writer_thread.start()

    def start(self):
        self.start_writer()
        with self._lock:
            self.is_collecting = True
        print(f"\n[INFO] Collection resumed | Episode: {self.episode_idx} | Dir: {self.full_dataset_dir}")

    def pause(self):
        with self._lock:
            self.is_collecting = False
        print(f"\n[INFO] Collection paused | Buffered frames: {self.frame_count}")

    def toggle(self):
        if self.is_collecting:
            self.pause()
        else:
            self.start()

    def has_data(self):
        with self._data_lock:
            buffered = len(self._current_actions)
        return self.frame_count > 0 or buffered > 0 or not self._frame_queue.empty()

    def add_frame(self, observation, action):
        if not self.is_collecting:
            return
        obs_copy = {
            "qpos": np.asarray(observation["qpos"], dtype=float).copy(),
            "qvel": np.asarray(observation["qvel"], dtype=float).copy(),
            "effort": np.asarray(observation["effort"], dtype=float).copy(),
            "images": {k: v.copy() for k, v in observation["images"].items()},
        }
        action_copy = np.asarray(action, dtype=float).copy()
        try:
            frame_idx = self.frame_count
            self._frame_queue.put_nowait((obs_copy, action_copy))
            self._write_depth_frame(frame_idx, observation.get("depths"))
            self.frame_count += 1
            if self.frame_count % 100 == 0:
                print(f"\r[COLLECT] Frames: {self.frame_count}", end="", flush=True)
        except queue.Full:
            print("[WARN] Queue full, dropping frame")

    def save_current_episode(self, export_video=True, video_fps=30, resume_after_save=True):
        if not self.has_data():
            print("[ERROR] No data to save")
            return False
        self._resume_after_save = self.is_collecting if resume_after_save else False
        self.is_collecting = False
        self._save_config = {"export_video": export_video, "video_fps": video_fps}
        self._save_requested = True
        print("[INFO] Save requested, collection paused while background save runs...")
        return True

    def clear_current_episode(self):
        while not self._frame_queue.empty():
            try:
                self._frame_queue.get_nowait()
            except queue.Empty:
                break
        with self._data_lock:
            self._current_observations.clear()
            self._current_actions.clear()
        self.frame_count = 0
        self._reset_depth_tmp()

    def discard_current_episode(self):
        self.clear_current_episode()
        print(f"\n[INFO] Discarded current episode. Still on episode {self.episode_idx}.")

    def shutdown(self):
        self._writer_running = False
        if self._writer_thread and self._writer_thread.is_alive():
            self._writer_thread.join(timeout=5.0)
        if self.save_depth and os.path.exists(self._tmp_depth_root):
            shutil.rmtree(self._tmp_depth_root, ignore_errors=True)


def _camera_display_frame(image: np.ndarray) -> np.ndarray:
    """Convert stored RGB camera frames to BGR for cv2.imshow."""
    if image is None:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    frame = np.asarray(image)
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    if frame.shape[-1] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def _resize_keep_aspect(image: np.ndarray, target_width: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h <= 0 or w <= 0:
        return image
    target_width = max(1, int(target_width))
    target_height = max(1, int(round(h * target_width / w)))
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)


def show_camera_visualization(images: Dict[str, np.ndarray], args, collecting: bool, frame_count: int) -> bool:
    """Show three camera streams. Returns False when user requests quit."""
    width = int(getattr(args, "camera_visualization_width", 320))
    order = [
        ("cam_high", "Third View / Orbbec"),
        ("cam_right_wrist", "Right Wrist / RealSense"),
        ("cam_left_wrist", "Left Wrist / RealSense"),
    ]

    panels = []
    max_h = 0
    status = "REC" if collecting else "PAUSED"
    for cam_name, label in order:
        frame = _camera_display_frame(images.get(cam_name))
        frame = _resize_keep_aspect(frame, width)
        cv2.putText(frame, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, f"{status} frames={frame_count}", (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        panels.append(frame)
        max_h = max(max_h, frame.shape[0])

    padded_panels = []
    for panel in panels:
        if panel.shape[0] < max_h:
            pad = np.zeros((max_h - panel.shape[0], panel.shape[1], 3), dtype=panel.dtype)
            panel = np.vstack((panel, pad))
        padded_panels.append(panel)

    cv2.imshow(getattr(args, "camera_visualization_window", "ARX teleop cameras"), np.hstack(padded_panels))
    key = cv2.waitKey(1) & 0xFF
    if key in (ord("q"), 27):
        return False
    return True


class AsyncRealSenseCamera:
    """Read one RealSense color stream, optionally with aligned depth, in a daemon thread."""

    def __init__(
        self,
        cam_name: str,
        serial: str,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        timeout_ms: int = 5000,
        enable_depth: bool = False,
        align_depth_to_color: bool = True,
    ):
        self.cam_name = cam_name
        self.serial = serial
        self.width = width
        self.height = height
        self.fps = fps
        self.timeout_ms = timeout_ms
        self.enable_depth = bool(enable_depth)
        self.align_depth_to_color = bool(align_depth_to_color)
        self.lock = threading.Lock()
        self.latest_frame = None
        self.latest_depth = None
        self.latest_depth_meta = {}
        self.latest_time = 0.0
        self.running = False
        self.thread = None
        self.pipeline = None
        self.align = None

    @staticmethod
    def _intrinsics_to_dict(intrinsics):
        return {
            "fx": float(intrinsics.fx),
            "fy": float(intrinsics.fy),
            "cx": float(intrinsics.ppx),
            "cy": float(intrinsics.ppy),
            "width": int(intrinsics.width),
            "height": int(intrinsics.height),
        }

    def open(self) -> bool:
        import pyrealsense2 as rs

        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        if self.enable_depth:
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        profile = pipeline.start(config)
        depth_scale_m = None
        color_intrinsic = None
        depth_intrinsic = None
        if self.enable_depth:
            self.align = rs.align(rs.stream.color) if self.align_depth_to_color else None
            try:
                depth_sensor = profile.get_device().first_depth_sensor()
                depth_scale_m = float(depth_sensor.get_depth_scale())
            except Exception:
                depth_scale_m = None
            try:
                color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
                color_intrinsic = self._intrinsics_to_dict(color_profile.get_intrinsics())
            except Exception as e:
                print(f"[WARN] RealSense {self.cam_name} failed to read color intrinsics: {e}")
            try:
                depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
                depth_intrinsic = self._intrinsics_to_dict(depth_profile.get_intrinsics())
            except Exception as e:
                print(f"[WARN] RealSense {self.cam_name} failed to read depth intrinsics: {e}")

        self.pipeline = pipeline
        self.latest_depth_meta = {
            "source": "RealSense",
            "serial": self.serial,
            "aligned_to_color": bool(self.align_depth_to_color),
            "requested_width": int(self.width),
            "requested_height": int(self.height),
            "requested_fps": int(self.fps),
            "depth_scale_m": depth_scale_m,
            "unit": "millimeter",
        }
        if color_intrinsic is not None:
            self.latest_depth_meta["rgb_intrinsic"] = color_intrinsic
        if depth_intrinsic is not None:
            self.latest_depth_meta["depth_intrinsic"] = depth_intrinsic
        if color_intrinsic is not None or depth_intrinsic is not None:
            self.latest_depth_meta["notes"] = {
                "intrinsics_source": "RealSense active stream profile",
                "pointcloud_intrinsic_when_aligned_to_color": "rgb_intrinsic",
                "pointcloud_intrinsic_when_not_aligned": "depth_intrinsic",
            }
        self.running = True
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()
        return True

    def _reader_loop(self):
        while self.running:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=self.timeout_ms)
                if self.enable_depth and self.align is not None:
                    frames = self.align.process(frames)
                color_frame = frames.get_color_frame() if frames else None
                if color_frame is None:
                    continue
                frame_rgb = np.asanyarray(color_frame.get_data()).copy()
                depth_mm = None
                depth_meta = None
                if self.enable_depth:
                    depth_frame = frames.get_depth_frame() if frames else None
                    if depth_frame is None:
                        continue
                    depth_mm = np.asanyarray(depth_frame.get_data()).copy()
                    depth_meta = dict(self.latest_depth_meta)
                    depth_meta.update(
                        {
                            "depth_width": int(depth_frame.get_width()),
                            "depth_height": int(depth_frame.get_height()),
                            "color_width": int(color_frame.get_width()),
                            "color_height": int(color_frame.get_height()),
                        }
                    )
                with self.lock:
                    self.latest_frame = frame_rgb
                    if depth_mm is not None:
                        self.latest_depth = depth_mm
                    if depth_meta is not None:
                        self.latest_depth_meta = depth_meta
                    self.latest_time = time.time()
            except Exception as e:
                if not hasattr(self, "_last_warn") or time.time() - getattr(self, "_last_warn", 0) > 2.0:
                    print(
                        f"[WARN] RealSense {self.cam_name} read failed: {e}. "
                        "If this starts after Orbbec starts, lower --camera_wrist_fps/resolution or move cameras to separate USB controllers."
                    )
                    self._last_warn = time.time()
                time.sleep(0.05)

    def wait_for_first_frame(self, timeout: float) -> bool:
        start = time.time()
        while time.time() - start < timeout:
            with self.lock:
                if self.latest_frame is not None:
                    return True
            time.sleep(0.05)
        return False

    def get_frame(self):
        with self.lock:
            if self.latest_frame is None:
                return None
            return self.latest_frame.copy()

    def get_depth_frame(self):
        with self.lock:
            if self.latest_depth is None:
                return None, {}
            return self.latest_depth.copy(), dict(self.latest_depth_meta)

    def release(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.pipeline = None


class AsyncOrbbecCamera:
    """Read Orbbec color frames, optionally with depth, in a daemon thread via pyorbbecsdk."""

    def __init__(self, width: int, height: int, fps: int, align_depth_to_color: bool = True, enable_depth: bool = True):
        self.width = width
        self.height = height
        self.fps = fps
        self.align_depth_to_color = align_depth_to_color
        self.enable_depth = bool(enable_depth)
        self.lock = threading.Lock()
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_meta = {}
        self.latest_time = 0.0
        self.running = False
        self.thread = None
        self.pipeline = None
        self.align_filter = None
        self.sdk = {}

    def _import_sdk(self):
        try:
            from pyorbbecsdk import (  # type: ignore
                AlignFilter,
                Config,
                OBAlignMode,
                OBFormat,
                OBFrameAggregateOutputMode,
                OBSensorType,
                OBStreamType,
                Pipeline,
            )
        except ImportError as e:
            raise RuntimeError(
                "pyorbbecsdk is required for --camera_front_backend orbbec. "
                "Install/build Orbbec Python SDK in the kai0_inference environment first."
            ) from e
        self.sdk = {
            "AlignFilter": AlignFilter,
            "Config": Config,
            "OBAlignMode": OBAlignMode,
            "OBFormat": OBFormat,
            "OBFrameAggregateOutputMode": OBFrameAggregateOutputMode,
            "OBSensorType": OBSensorType,
            "OBStreamType": OBStreamType,
            "Pipeline": Pipeline,
        }

    def _select_profile(self, pipeline, sensor_type, fmt, width, height, fps):
        profile_list = pipeline.get_stream_profile_list(sensor_type)
        try:
            return profile_list.get_video_stream_profile(width, height, fmt, fps)
        except Exception:
            try:
                return profile_list.get_video_stream_profile(width, 0, fmt, fps)
            except Exception:
                return profile_list.get_default_video_stream_profile()

    @staticmethod
    def _intrinsic_to_dict(intrinsic):
        return {
            "fx": float(intrinsic.fx),
            "fy": float(intrinsic.fy),
            "cx": float(intrinsic.cx),
            "cy": float(intrinsic.cy),
            "width": int(intrinsic.width),
            "height": int(intrinsic.height),
        }

    @staticmethod
    def _distortion_to_dict(distortion):
        return {
            "k1": float(distortion.k1),
            "k2": float(distortion.k2),
            "k3": float(distortion.k3),
            "k4": float(distortion.k4),
            "k5": float(distortion.k5),
            "k6": float(distortion.k6),
            "p1": float(distortion.p1),
            "p2": float(distortion.p2),
        }

    @staticmethod
    def _transform_to_dict(transform):
        return {
            "rotation": np.asarray(transform.rot, dtype=float).reshape(3, 3).tolist(),
            "translation": np.asarray(transform.transform, dtype=float).reshape(3).tolist(),
        }

    def _camera_param_to_dict(self, camera_param):
        return {
            "depth_intrinsic": self._intrinsic_to_dict(camera_param.depth_intrinsic),
            "rgb_intrinsic": self._intrinsic_to_dict(camera_param.rgb_intrinsic),
            "depth_distortion": self._distortion_to_dict(camera_param.depth_distortion),
            "rgb_distortion": self._distortion_to_dict(camera_param.rgb_distortion),
            "depth_to_rgb_transform": self._transform_to_dict(camera_param.transform),
            "notes": {
                "intrinsics_source": "OrbbecSDK factory calibration",
                "pointcloud_intrinsic_when_aligned_to_color": "rgb_intrinsic",
                "pointcloud_intrinsic_when_not_aligned": "depth_intrinsic",
            },
        }

    def _read_camera_param_meta(self, pipeline):
        try:
            return {"camera_param": self._camera_param_to_dict(pipeline.get_camera_param())}
        except Exception as e:
            return {"camera_param_error": str(e)}

    def open(self) -> bool:
        self._import_sdk()
        Config = self.sdk["Config"]
        Pipeline = self.sdk["Pipeline"]
        OBSensorType = self.sdk["OBSensorType"]
        OBFormat = self.sdk["OBFormat"]
        OBAlignMode = self.sdk["OBAlignMode"]
        OBFrameAggregateOutputMode = self.sdk["OBFrameAggregateOutputMode"]
        AlignFilter = self.sdk["AlignFilter"]
        OBStreamType = self.sdk["OBStreamType"]

        pipeline = Pipeline()
        config = Config()
        align_mode_name = "disabled"
        try:
            color_profile = self._select_profile(
                pipeline,
                OBSensorType.COLOR_SENSOR,
                OBFormat.RGB,
                self.width,
                self.height,
                self.fps,
            )
            config.enable_stream(color_profile)

            if self.enable_depth:
                if self.align_depth_to_color:
                    try:
                        depth_profiles = pipeline.get_d2c_depth_profile_list(color_profile, OBAlignMode.HW_MODE)
                        if len(depth_profiles) > 0:
                            depth_profile = depth_profiles[0]
                            config.set_align_mode(OBAlignMode.HW_MODE)
                            align_mode_name = "hardware"
                        else:
                            raise RuntimeError("No hardware D2C profile")
                    except Exception:
                        depth_profile = self._select_profile(
                            pipeline,
                            OBSensorType.DEPTH_SENSOR,
                            OBFormat.Y16,
                            self.width,
                            self.height,
                            self.fps,
                        )
                        self.align_filter = AlignFilter(align_to_stream=OBStreamType.COLOR_STREAM)
                        align_mode_name = "software_filter"
                else:
                    depth_profile = self._select_profile(
                        pipeline,
                        OBSensorType.DEPTH_SENSOR,
                        OBFormat.Y16,
                        self.width,
                        self.height,
                        self.fps,
                    )
                config.enable_stream(depth_profile)
            try:
                config.set_frame_aggregate_output_mode(OBFrameAggregateOutputMode.FULL_FRAME_REQUIRE)
            except Exception:
                pass
            pipeline.start(config)
            try:
                pipeline.enable_frame_sync()
            except Exception:
                pass
            camera_param_meta = self._read_camera_param_meta(pipeline) if self.enable_depth else {}
        except Exception as e:
            try:
                pipeline.stop()
            except Exception:
                pass
            stream_desc = "color/depth" if self.enable_depth else "color"
            raise RuntimeError(f"Failed to start Orbbec {stream_desc} pipeline: {e}") from e

        self.pipeline = pipeline
        self.latest_meta = {
            "source": "OrbbecSDK",
            "device": "DaBai",
            "depth_enabled": bool(self.enable_depth),
            "aligned_to_color": bool(self.align_depth_to_color),
            "align_mode": align_mode_name,
            "requested_width": int(self.width),
            "requested_height": int(self.height),
            "requested_fps": int(self.fps),
        }
        self.latest_meta.update(camera_param_meta)
        self.running = True
        self.thread = threading.Thread(target=self._reader_loop, daemon=True)
        self.thread.start()
        return True

    def _frame_to_rgb_image(self, frame):
        OBFormat = self.sdk["OBFormat"]
        width = frame.get_width()
        height = frame.get_height()
        fmt = frame.get_format()
        data = np.asanyarray(frame.get_data())
        if fmt == OBFormat.RGB:
            return np.resize(data, (height, width, 3)).astype(np.uint8)
        if fmt == OBFormat.BGR:
            bgr = np.resize(data, (height, width, 3)).astype(np.uint8)
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if fmt == OBFormat.MJPG:
            bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else None
        if fmt == OBFormat.YUYV:
            yuyv = np.resize(data, (height, width, 2))
            bgr = cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUYV)
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if fmt == OBFormat.UYVY:
            uyvy = np.resize(data, (height, width, 2))
            bgr = cv2.cvtColor(uyvy, cv2.COLOR_YUV2BGR_UYVY)
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        print(f"[WARN] Unsupported Orbbec color format: {fmt}")
        return None

    def _depth_to_mm_uint16(self, depth_frame):
        OBFormat = self.sdk["OBFormat"]
        if depth_frame.get_format() != OBFormat.Y16:
            print(f"[WARN] Orbbec depth format is not Y16: {depth_frame.get_format()}")
            return None
        width = depth_frame.get_width()
        height = depth_frame.get_height()
        scale = float(depth_frame.get_depth_scale())
        data = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape((height, width))
        depth_mm = data.astype(np.float32) * scale
        depth_mm = np.where((depth_mm > 0) & (depth_mm < 65535), depth_mm, 0)
        return depth_mm.astype(np.uint16), {
            "depth_width": int(width),
            "depth_height": int(height),
            "depth_scale": scale,
        }

    def _reader_loop(self):
        while self.running:
            try:
                frames = self.pipeline.wait_for_frames(100)
                if frames is None:
                    continue
                if self.align_filter is not None:
                    aligned = self.align_filter.process(frames)
                    if aligned:
                        frames = aligned.as_frame_set() if hasattr(aligned, "as_frame_set") else aligned
                color_frame = frames.get_color_frame()
                if color_frame is None:
                    continue
                rgb = self._frame_to_rgb_image(color_frame)
                if rgb is None:
                    continue
                meta = dict(self.latest_meta)
                meta.update(
                    {
                        "color_width": int(color_frame.get_width()),
                        "color_height": int(color_frame.get_height()),
                    }
                )
                depth = None
                if self.enable_depth:
                    depth_frame = frames.get_depth_frame()
                    if depth_frame is None:
                        continue
                    depth_result = self._depth_to_mm_uint16(depth_frame)
                    if depth_result is None:
                        continue
                    depth, depth_meta = depth_result
                    meta.update(depth_meta)
                with self.lock:
                    self.latest_rgb = rgb
                    if depth is not None:
                        self.latest_depth = depth
                    self.latest_meta = meta
                    self.latest_time = time.time()
            except Exception as e:
                if not hasattr(self, "_last_warn") or time.time() - getattr(self, "_last_warn", 0) > 2.0:
                    print(f"[WARN] Orbbec camera read failed: {e}")
                    self._last_warn = time.time()
                time.sleep(0.05)

    def wait_for_first_frame(self, timeout: float) -> bool:
        start = time.time()
        while time.time() - start < timeout:
            with self.lock:
                if self.latest_rgb is not None and (not self.enable_depth or self.latest_depth is not None):
                    return True
            time.sleep(0.05)
        return False

    def get_frame(self):
        with self.lock:
            if self.latest_rgb is None:
                return None, None, {}
            rgb = self.latest_rgb.copy()
            depth = self.latest_depth.copy() if self.latest_depth is not None else None
            meta = dict(self.latest_meta)
            return rgb, depth, meta

    def release(self):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.pipeline is not None:
            try:
                self.pipeline.stop()
            except Exception:
                pass
            self.pipeline = None


class ARXTeleopObserver(Node):
    """ROS2 observer for ARX slave/master states and camera frames."""

    def __init__(self, args):
        super().__init__("arx_teleop_collector")
        self.args = args

        self.joint_left_deque = deque(maxlen=2000)
        self.joint_right_deque = deque(maxlen=2000)
        self.master_left_deque = deque(maxlen=2000)
        self.master_right_deque = deque(maxlen=2000)

        self.data_ready = {"joint_left": False, "joint_right": False, "cameras": False}
        self.data_ready_lock = threading.Lock()
        self.pipelines = {}
        self.rs_readers = {}
        self.front_reader = None
        self.last_images: Dict[str, np.ndarray] = {}
        self.last_depths: Dict[str, np.ndarray] = {}

        self.create_subscription(RobotStatus, args.joint_state_topic_left, self.joint_left_callback, 10)
        self.create_subscription(RobotStatus, args.joint_state_topic_right, self.joint_right_callback, 10)
        self.create_subscription(RobotStatus, args.master_status_topic_left, self.master_left_callback, 10)
        self.create_subscription(RobotStatus, args.master_status_topic_right, self.master_right_callback, 10)
        self.create_subscription(RobotStatus, args.master_status_topic_left_ctrl, self.master_left_callback, 10)
        self.create_subscription(RobotStatus, args.master_status_topic_right_ctrl, self.master_right_callback, 10)

        self.pub_left = self.create_publisher(RobotStatus, args.joint_cmd_topic_left, 10)
        self.pub_right = self.create_publisher(RobotStatus, args.joint_cmd_topic_right, 10)
        self.master_cmd_left_pub = self.create_publisher(RobotStatus, args.master_cmd_topic_left, 10)
        self.master_cmd_right_pub = self.create_publisher(RobotStatus, args.master_cmd_topic_right, 10)
        self.master_ctrl_left_pub = self.create_publisher(RobotStatus, args.master_ctrl_cmd_topic_left, 10)
        self.master_ctrl_right_pub = self.create_publisher(RobotStatus, args.master_ctrl_cmd_topic_right, 10)
        self.arx_joy_pub = self.create_publisher(Int32MultiArray, args.arx_joy_topic, 10)
        self.master_cmd_cache = np.zeros(14, dtype=float)

        self.get_logger().info(f"Sub slave left: {args.joint_state_topic_left}")
        self.get_logger().info(f"Sub slave right: {args.joint_state_topic_right}")
        self.get_logger().info(f"Sub master left: {args.master_status_topic_left} / {args.master_status_topic_left_ctrl}")
        self.get_logger().info(f"Sub master right: {args.master_status_topic_right} / {args.master_status_topic_right_ctrl}")
        self.get_logger().info(f"Pub slave command mirror: {args.joint_cmd_topic_left}, {args.joint_cmd_topic_right}")
        self.get_logger().info(f"Pub master command: {args.master_cmd_topic_left}, {args.master_cmd_topic_right}")
        self.get_logger().info(f"Pub master ctrl command: {args.master_ctrl_cmd_topic_left}, {args.master_ctrl_cmd_topic_right}")
        self.get_logger().info(f"Pub ARX joy home command: {args.arx_joy_topic}")

        self.init_cameras()

    def joint_left_callback(self, msg):
        self.joint_left_deque.append(msg)
        with self.data_ready_lock:
            self.data_ready["joint_left"] = True

    def joint_right_callback(self, msg):
        self.joint_right_deque.append(msg)
        with self.data_ready_lock:
            self.data_ready["joint_right"] = True

    def master_left_callback(self, msg):
        self.master_left_deque.append(msg)

    def master_right_callback(self, msg):
        self.master_right_deque.append(msg)

    def get_slave_positions(self) -> Optional[np.ndarray]:
        if len(self.joint_left_deque) == 0 or len(self.joint_right_deque) == 0:
            return None
        left = np.asarray(self.joint_left_deque[-1].joint_pos, dtype=float)
        right = np.asarray(self.joint_right_deque[-1].joint_pos, dtype=float)
        return np.concatenate([left, right])

    def get_master_positions(self) -> Optional[np.ndarray]:
        if len(self.master_left_deque) == 0 or len(self.master_right_deque) == 0:
            return None
        left = np.asarray(self.master_left_deque[-1].joint_pos, dtype=float)
        right = np.asarray(self.master_right_deque[-1].joint_pos, dtype=float)
        return np.concatenate([left, right])

    def set_joint_positions(self, pos: np.ndarray):
        """Publish 14-D slave targets through the same mirror topics used by inference."""
        if not rclpy.ok():
            return
        pos = np.asarray(pos, dtype=float)
        if pos.shape[0] != 14:
            self.get_logger().warn(f"Expected 14-D slave target, got {pos.shape}")
            return
        msg_left = RobotStatus()
        msg_right = RobotStatus()
        msg_left.joint_pos = [float(x) for x in pos[:7]]
        msg_right.joint_pos = [float(x) for x in pos[7:]]
        self.pub_left.publish(msg_left)
        self.pub_right.publish(msg_right)

    def publish_master_ctrl_positions(self, pos: np.ndarray):
        """Publish 14-D master targets. Default is the normal remote_master command topics."""
        pos = np.asarray(pos, dtype=float)
        if pos.shape[0] != 14:
            self.get_logger().warn(f"Expected 14-D master target, got {pos.shape}")
            return
        self.master_cmd_cache = pos.copy()
        msg_left = RobotStatus()
        msg_right = RobotStatus()
        msg_left.joint_pos = [float(x) for x in pos[:7]]
        msg_right.joint_pos = [float(x) for x in pos[7:]]
        mode = str(getattr(self.args, "master_align_command_mode", "master_ctrl"))
        if mode == "master_cmd" and not getattr(self, "_warned_master_cmd", False):
            print("[WARN] master_cmd mode only works if the master controller subscribes to /arm_master_l_cmd and /arm_master_r_cmd. Repo remote_master mode does not.")
            self._warned_master_cmd = True
        if mode == "master_ctrl":
            self.master_ctrl_left_pub.publish(msg_left)
            self.master_ctrl_right_pub.publish(msg_right)
        else:
            self.master_cmd_left_pub.publish(msg_left)
            self.master_cmd_right_pub.publish(msg_right)

    def set_master_mode(self, node_name: str, mode: str, timeout: float = 3.0) -> bool:
        """Switch master arm mode, e.g. remote_master_ctrl for scripted alignment."""
        client = self.create_client(SetParameters, f"{node_name}/set_parameters")
        if not client.wait_for_service(timeout_sec=timeout):
            print(f"[WARN] Service {node_name}/set_parameters unavailable; skip switching to {mode}.")
            return False
        req = SetParameters.Request()
        pval = ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=mode)
        req.parameters = [Parameter(name="arm_control_type", value=pval)]
        future = client.call_async(req)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if future.result() and future.result().results:
            result = future.result().results[0]
            if result.successful:
                print(f"[INFO] {node_name} switched to {mode}")
                return True
            print(f"[WARN] {node_name} switch to {mode} failed: {result.reason}")
            return False
        print(f"[WARN] {node_name} switch to {mode} timed out")
        return False

    def _smooth_interpolate_positions(self, start: np.ndarray, target: np.ndarray, duration: float, hz: float, max_joint_step: float, publish_fn, label: str) -> bool:
        start = np.asarray(start, dtype=float)
        target = np.asarray(target, dtype=float)
        if start.shape[0] != 14 or target.shape[0] != 14:
            self.get_logger().warn(f"[{label}] Expected 14-D start/target, got {start.shape}/{target.shape}")
            return False
        max_delta = float(np.max(np.abs(target - start)))
        num_steps = max(1, int(duration * hz))
        if max_joint_step and max_joint_step > 0:
            num_steps = max(num_steps, int(np.ceil(max_delta / max_joint_step)))
        print(f"[INFO] {label}: max joint delta {max_delta:.4f} rad, steps {num_steps}")
        prev = start.copy()
        for step in range(num_steps + 1):
            if shutdown_event.is_set() or not rclpy.ok():
                return False
            alpha = step / num_steps
            smooth_alpha = (1.0 - np.cos(alpha * np.pi)) / 2.0
            cmd = start * (1.0 - smooth_alpha) + target * smooth_alpha
            if max_joint_step and max_joint_step > 0:
                step_limits = np.full(14, float(max_joint_step), dtype=float)
                step_limits[6] = np.inf
                step_limits[13] = np.inf
                cmd = prev + np.clip(cmd - prev, -step_limits, step_limits)
            publish_fn(cmd)
            prev = cmd.copy()
            if step % 50 == 0 or step == num_steps:
                print(f"\r{label}: {int(alpha * 100)}%", end="", flush=True)
            time.sleep(1.0 / hz)
        print(f"\n[INFO] {label} done")
        return True

    def smooth_goto_slave_position(self, target_pos: np.ndarray, duration: float, hz: float, max_joint_step: float) -> bool:
        start = self.get_slave_positions()
        if start is None:
            print("[WARN] Slave state unavailable; skip slave alignment.")
            return False
        return self._smooth_interpolate_positions(start, target_pos, duration, hz, max_joint_step, self.set_joint_positions, "Slave align")

    def smooth_goto_master_position(self, target_pos: np.ndarray, duration: float, hz: float, max_joint_step: float) -> bool:
        start = self.get_master_positions()
        if start is None:
            print("[WARN] Master state unavailable; skip master alignment.")
            return False
        return self._smooth_interpolate_positions(start, target_pos, duration, hz, max_joint_step, self.publish_master_ctrl_positions, "Master align")

    def wait_for_arm_near_pose(self, name: str, get_positions_fn, target_pos: np.ndarray, timeout: float = 3.0, tolerance: float = 0.08, check_gripper: bool = True) -> bool:
        """Wait until an observed 14-D arm pair is near target pose."""
        target_pos = np.asarray(target_pos, dtype=float)
        mask = np.ones(14, dtype=bool)
        if not check_gripper:
            mask[6] = False
            mask[13] = False
        start_time = time.time()
        while time.time() - start_time < timeout and rclpy.ok() and not shutdown_event.is_set():
            pos = get_positions_fn()
            if pos is not None:
                err = float(np.max(np.abs(pos[mask] - target_pos[mask])))
                if err <= tolerance:
                    print(f"[INFO] {name} reached collect pose, max err {err:.4f} rad")
                    return True
            time.sleep(0.05)
        pos = get_positions_fn()
        if pos is None:
            print(f"[WARN] {name} state unavailable while checking follow error.")
        else:
            err = float(np.max(np.abs(pos[mask] - target_pos[mask])))
            print(f"[WARN] {name} did not reach collect pose within timeout, max err {err:.4f} rad")
        return False

    def publish_arx_joy(self, data, label: str, repeats: int = 20, hz: float = 20.0) -> bool:
        """Publish /arx_joy commands. X5Controller maps [1,0] to G_COMPENSATION and [0,1] to GO_HOME."""
        print(f"[INFO] Publishing {label} on {self.args.arx_joy_topic} ({repeats} repeats)")
        for _ in range(max(1, repeats)):
            if shutdown_event.is_set() or not rclpy.ok():
                return False
            msg = Int32MultiArray()
            msg.data = list(data)
            self.arx_joy_pub.publish(msg)
            time.sleep(1.0 / hz)
        return True

    def publish_arx_go_home(self, repeats: int = 20, hz: float = 20.0) -> bool:
        """Trigger controller GO_HOME through /arx_joy."""
        return self.publish_arx_joy([0, 1], "GO_HOME", repeats=repeats, hz=hz)

    def publish_arx_compensation(self, repeats: int = 20, hz: float = 20.0) -> bool:
        """Release master arms back to gravity compensation after scripted homing."""
        return self.publish_arx_joy([1, 0], "G_COMPENSATION", repeats=repeats, hz=hz)

    def wait_for_slave_near_pose(self, target_pos: np.ndarray, timeout: float = 3.0, tolerance: float = 0.08, check_gripper: bool = True) -> bool:
        return self.wait_for_arm_near_pose("Slave", self.get_slave_positions, target_pos, timeout, tolerance, check_gripper=check_gripper)

    def align_to_collect_pose(self, target_pos: np.ndarray, duration: float, hz: float, slave_max_step: float, master_max_step: float, align_slave: bool = False, align_master: bool = True, slave_follow_timeout: float = 3.0, slave_follow_tolerance: float = 0.08, align_method: str = "arx_joy_home") -> bool:
        """Align collection pose. Current repo controller supports GO_HOME; master_ctrl may need custom firmware."""
        target_pos = np.asarray(target_pos, dtype=float)
        if target_pos.shape[0] != 14:
            raise ValueError(f"collect pose must be 14-D, got {target_pos.shape}")
        print(f"[INFO] Aligning arms to collection pose, method={align_method}")
        results = []

        if align_method == "none":
            print("[INFO] Alignment disabled by --align_method none")
            return True

        if align_method == "arx_joy_home":
            ok_home = self.publish_arx_go_home(repeats=int(self.args.arx_joy_home_repeats), hz=float(self.args.arx_joy_home_hz))
            results.append(("arx_joy_home", ok_home))
            if ok_home:
                time.sleep(float(self.args.arx_joy_home_settle_sec))
                ok_master_reached = self.wait_for_arm_near_pose("Master", self.get_master_positions, target_pos, timeout=slave_follow_timeout, tolerance=slave_follow_tolerance, check_gripper=False)
                ok_slave_reached = self.wait_for_arm_near_pose("Slave", self.get_slave_positions, target_pos, timeout=slave_follow_timeout, tolerance=slave_follow_tolerance, check_gripper=False)
                results.append(("master_home_reached", ok_master_reached))
                results.append(("slave_home_reached", ok_slave_reached))
            ok_release = self.publish_arx_compensation(
                repeats=int(self.args.arx_joy_release_repeats),
                hz=float(self.args.arx_joy_home_hz),
            )
            results.append(("release_compensation", ok_release))
            ok = all(item_ok for _, item_ok in results) if results else True
            print(f"[INFO] Collection pose alignment {'done' if ok else 'finished with warnings'}: {results}")
            return ok

        if align_master:
            ok_master = self.smooth_goto_master_position(target_pos, duration, hz, master_max_step)
            results.append(("master", ok_master))
            if ok_master:
                ok_master_reached = self.wait_for_arm_near_pose("Master", self.get_master_positions, target_pos, timeout=slave_follow_timeout, tolerance=slave_follow_tolerance)
                results.append(("master_reached", ok_master_reached))
                ok_follow = self.wait_for_slave_near_pose(target_pos, timeout=slave_follow_timeout, tolerance=slave_follow_tolerance)
                results.append(("slave_follow", ok_follow))

        if align_slave:
            print("[WARN] Direct slave alignment is enabled. Use this only outside normal teleop following mode.")
            ok_slave = self.smooth_goto_slave_position(target_pos, duration, hz, slave_max_step)
            results.append(("slave_direct", ok_slave))

        ok = all(item_ok for _, item_ok in results) if results else True
        print(f"[INFO] Collection pose alignment {'done' if ok else 'finished with warnings'}: {results}")
        return ok

    def init_cameras(self):
        """Initialize wrist RealSense cameras and front RealSense/Orbbec camera."""
        front_backend = str(getattr(self.args, "camera_front_backend", "realsense")).lower()
        if front_backend not in ("realsense", "orbbec"):
            print(f"Unknown camera_front_backend={front_backend}, fallback to realsense.")
            front_backend = "realsense"

        front_ok = False
        wrists_ok = False

        try:
            import pyrealsense2 as rs

            camera_serials = {
                "cam_left_wrist": self.args.camera_left_serial,
                "cam_right_wrist": self.args.camera_right_serial,
            }
            if front_backend == "realsense":
                camera_serials["cam_high"] = self.args.camera_front_serial

            wrist_width = int(getattr(self.args, "camera_wrist_width", 640))
            wrist_height = int(getattr(self.args, "camera_wrist_height", 480))
            wrist_fps = int(getattr(self.args, "camera_wrist_fps", 30))
            wrist_timeout_ms = int(getattr(self.args, "camera_wrist_timeout_ms", 5000))
            wrist_startup_timeout = float(getattr(self.args, "camera_wrist_startup_timeout", 5.0))
            print("Initializing RealSense cameras...")
            for cam_name, serial in camera_serials.items():
                reader = AsyncRealSenseCamera(
                    cam_name,
                    serial,
                    width=wrist_width,
                    height=wrist_height,
                    fps=wrist_fps,
                    timeout_ms=wrist_timeout_ms,
                )
                reader.open()
                if reader.wait_for_first_frame(wrist_startup_timeout):
                    self.rs_readers[cam_name] = reader
                    print(f"  {cam_name} started (serial={serial}, {wrist_width}x{wrist_height}@{wrist_fps})")
                else:
                    reader.release()
                    print(
                        f"  {cam_name} opened but produced no frame within {wrist_startup_timeout:.1f}s "
                        f"(serial={serial}, {wrist_width}x{wrist_height}@{wrist_fps})"
                    )

            print("RealSense cameras warmed up.")

            wrists_ok = "cam_left_wrist" in self.rs_readers and "cam_right_wrist" in self.rs_readers
            if front_backend == "realsense":
                front_ok = "cam_high" in self.rs_readers
        except Exception as e:
            print(f"RealSense init failed: {e}")
            for reader in self.rs_readers.values():
                try:
                    reader.release()
                except Exception:
                    pass
            self.pipelines = {}
            self.rs_readers = {}

        if front_backend == "orbbec":
            width = int(getattr(self.args, "camera_front_width", 640))
            height = int(getattr(self.args, "camera_front_height", 480))
            fps = int(getattr(self.args, "camera_front_fps", 30))
            align_depth_to_color = bool(getattr(self.args, "align_depth_to_color", True))
            print(f"Initializing Orbbec RGB-D front camera: {width}x{height}@{fps}, align_depth_to_color={align_depth_to_color}")
            try:
                reader = AsyncOrbbecCamera(width, height, fps, align_depth_to_color=align_depth_to_color)
                if reader.open():
                    startup_timeout = float(getattr(self.args, "camera_front_startup_timeout", 5.0))
                    if reader.wait_for_first_frame(startup_timeout):
                        self.front_reader = reader
                        front_ok = True
                        print("  cam_high started (Orbbec RGB-D)")
                    else:
                        reader.release()
                        print(
                            f"Orbbec camera opened but produced no RGB-D frames within {startup_timeout:.1f}s. "
                            "Check pyorbbecsdk, USB permissions, and whether another process is using the camera."
                        )
            except Exception as e:
                print(f"Orbbec front camera init failed: {e}")

        cameras_ok = wrists_ok and front_ok
        with self.data_ready_lock:
            self.data_ready["cameras"] = cameras_ok
        if cameras_ok:
            print("All cameras ready.")
        else:
            print(f"Camera readiness check failed: wrists_ok={wrists_ok}, front_ok={front_ok}")

    def get_camera_images(self):
        """Get RGB images from all cameras, with last-frame fallback."""
        images = {}
        fallback = getattr(self, "last_images", {})

        for cam_name, reader in self.rs_readers.items():
            try:
                frame_rgb = reader.get_frame()
                if frame_rgb is not None:
                    images[cam_name] = frame_rgb
                    continue
            except Exception:
                pass

            if cam_name in fallback:
                images[cam_name] = fallback[cam_name]
            elif not hasattr(self, "_last_cam_warn") or time.time() - getattr(self, "_last_cam_warn", 0) > 2.0:
                print(f"Failed to get image from {cam_name}: no new frame and no cache")
                self._last_cam_warn = time.time()

        if self.front_reader is not None:
            try:
                frame_data = self.front_reader.get_frame()
                if isinstance(frame_data, tuple):
                    frame_rgb, depth_frame, depth_meta = frame_data
                else:
                    frame_rgb, depth_frame, depth_meta = frame_data, None, {}
                if frame_rgb is not None:
                    images["cam_high"] = frame_rgb
                    if depth_frame is not None:
                        self.last_depths = {
                            "cam_high": depth_frame,
                            "cam_high_meta": depth_meta,
                        }
                elif "cam_high" in fallback:
                    images["cam_high"] = fallback["cam_high"]
            except Exception as e:
                if "cam_high" in fallback:
                    images["cam_high"] = fallback["cam_high"]
                else:
                    print(f"Failed to get image from cam_high: {e}")

        if images:
            self.last_images = images
        return images

    def get_depth_payload(self):
        return self.last_depths.copy()

    def get_frame(self):
        if len(self.joint_left_deque) == 0 or len(self.joint_right_deque) == 0:
            return None
        imgs = self.get_camera_images()
        if len(imgs) != 3:
            return None
        return imgs, self.joint_left_deque[-1], self.joint_right_deque[-1]

    def wait_for_data_ready(self, timeout: float = 15.0) -> bool:
        print("Waiting for sensor data...")
        start_time = time.time()
        while time.time() - start_time < timeout and rclpy.ok() and not shutdown_event.is_set():
            with self.data_ready_lock:
                joints_ready = self.data_ready["joint_left"] and self.data_ready["joint_right"]
                cameras_ready = self.data_ready["cameras"]
            if joints_ready and cameras_ready:
                print("[INFO] All sensor data ready.")
                return True
            time.sleep(0.5)
        print("[ERROR] Timeout waiting for sensor data.")
        return False

    def wait_for_master_ready(self, timeout: float = 5.0) -> bool:
        print("Waiting for master arm state...")
        start_time = time.time()
        while time.time() - start_time < timeout and rclpy.ok() and not shutdown_event.is_set():
            if self.get_master_positions() is not None:
                print("[INFO] Master arm state ready.")
                return True
            time.sleep(0.1)
        print("[WARN] Timeout waiting for master arm state.")
        return False

    def cleanup_cameras(self):
        if hasattr(self, "rs_readers"):
            print("Stopping RealSense cameras...")
            for reader in self.rs_readers.values():
                try:
                    reader.release()
                except Exception:
                    pass
            self.rs_readers = {}
        if hasattr(self, "pipelines"):
            for pipeline in self.pipelines.values():
                try:
                    pipeline.stop()
                except Exception:
                    pass
            self.pipelines = {}
        if self.front_reader is not None:
            try:
                self.front_reader.release()
            except Exception:
                pass
            self.front_reader = None


def keyboard_monitor_thread(collector: TeleopCollector):
    """Keyboard: Space start/pause, s save, n discard, q quit this script."""
    global save_requested, discard_requested
    old_settings = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        print("\n[KEYS] Space: start/pause | s: save | n: discard episode | q: quit collector only")
        while not shutdown_event.is_set():
            if select.select([sys.stdin], [], [], 0.1)[0]:
                ch = sys.stdin.read(1)
                if ch == " ":
                    collector.toggle()
                elif ch.lower() == "s":
                    with request_lock:
                        save_requested = True
                    print("\n[INFO] Save requested")
                elif ch.lower() == "n":
                    with request_lock:
                        discard_requested = True
                    print("\n[INFO] Discard requested")
                elif ch.lower() == "q":
                    print("\n[INFO] Quit requested. This exits only the collection script; ARX master/slave nodes keep running.")
                    shutdown_event.set()
                    break
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)


def build_observation(imgs, j_left, j_right):
    return {
        "qpos": np.concatenate([j_left.joint_pos, j_right.joint_pos]).astype(float),
        "qvel": np.concatenate([j_left.joint_vel, j_right.joint_vel]).astype(float),
        "effort": np.concatenate([j_left.joint_cur, j_right.joint_cur]).astype(float),
        "images": imgs,
    }


def select_action(obs, ros_operator: ARXTeleopObserver, action_source: str):
    if action_source == "master":
        master_qpos = ros_operator.get_master_positions()
        if master_qpos is not None:
            return master_qpos
        print("[WARN] Master action requested but master state is not ready; using slave qpos for this frame.")
    return obs["qpos"].copy()


def main():
    global save_requested, discard_requested

    parser = argparse.ArgumentParser(description="ARX-X5 pure master-slave teleoperation data collector.")
    parser.add_argument("--joint_state_topic_left", default="/arm_slave_l_status")
    parser.add_argument("--joint_state_topic_right", default="/arm_slave_r_status")
    parser.add_argument("--master_status_topic_left", default="/arm_master_l_status")
    parser.add_argument("--master_status_topic_right", default="/arm_master_r_status")
    parser.add_argument("--joint_cmd_topic_left", default="/arm_master_l_status")
    parser.add_argument("--joint_cmd_topic_right", default="/arm_master_r_status")
    parser.add_argument("--master_cmd_topic_left", default="/arm_master_l_cmd")
    parser.add_argument("--master_cmd_topic_right", default="/arm_master_r_cmd")
    parser.add_argument("--arx_joy_topic", default="/arx_joy")
    parser.add_argument("--master_status_topic_left_ctrl", default="/arm_master_ctrl_status_left")
    parser.add_argument("--master_status_topic_right_ctrl", default="/arm_master_ctrl_status_right")
    parser.add_argument("--master_ctrl_cmd_topic_left", default="/arm_master_ctrl_cmd_left")
    parser.add_argument("--master_ctrl_cmd_topic_right", default="/arm_master_ctrl_cmd_right")
    parser.add_argument("--master_left_node", default="/arm_master_l")
    parser.add_argument("--master_right_node", default="/arm_master_r")

    parser.add_argument("--camera_front_serial", type=str, default="152122073503")
    parser.add_argument("--camera_left_serial", type=str, default="213622070289")
    parser.add_argument("--camera_right_serial", type=str, default="152122073474")
    parser.add_argument("--camera_front_backend", choices=("realsense", "orbbec"), default="realsense")
    parser.add_argument("--camera_front_width", type=int, default=640)
    parser.add_argument("--camera_front_height", type=int, default=480)
    parser.add_argument("--camera_front_fps", type=int, default=30)
    parser.add_argument("--camera_front_startup_timeout", type=float, default=3.0)
    parser.add_argument("--camera_wrist_width", type=int, default=640)
    parser.add_argument("--camera_wrist_height", type=int, default=480)
    parser.add_argument("--camera_wrist_fps", type=int, default=30)
    parser.add_argument("--camera_wrist_timeout_ms", type=int, default=5000)
    parser.add_argument("--camera_wrist_startup_timeout", type=float, default=5.0)
    parser.add_argument("--save_depth", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--depth_camera_name", type=str, default="cam_high")
    parser.add_argument("--align_depth_to_color", type=str2bool, nargs="?", const=True, default=True)

    parser.add_argument("--dataset_dir", type=str, default=DEFAULT_DATASET_DIR)
    parser.add_argument("--dataset_name", type=str, default="task_c_collect")
    parser.add_argument("--control_frequency", type=float, default=30.0)
    parser.add_argument("--save_video", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument(
        "--action_source",
        choices=("slave", "master"),
        default="slave",
        help="slave matches the existing DAgger human segment format; master records the teleop command source.",
    )
    parser.add_argument("--start_paused", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--visualize_cameras", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--camera_visualization_width", type=int, default=320)
    parser.add_argument("--camera_visualization_window", type=str, default="ARX teleop cameras")
    parser.add_argument("--align_collect_pose", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--align_after_save", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--align_slave_arms", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--align_master_arms", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--switch_master_mode_for_align", type=str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--collect_pose_left", nargs=7, type=float, default=DEFAULT_COLLECT_LEFT0)
    parser.add_argument("--collect_pose_right", nargs=7, type=float, default=DEFAULT_COLLECT_RIGHT0)
    parser.add_argument("--home_pose_left", nargs=7, type=float, default=DEFAULT_ARX_GO_HOME_LEFT)
    parser.add_argument("--home_pose_right", nargs=7, type=float, default=DEFAULT_ARX_GO_HOME_RIGHT)
    parser.add_argument("--align_duration", type=float, default=3.0)
    parser.add_argument("--align_hz", type=float, default=50.0)
    parser.add_argument("--slave_align_max_step", type=float, default=0.05)
    parser.add_argument("--master_align_max_step", type=float, default=0.05)
    parser.add_argument("--master_ready_timeout", type=float, default=5.0)
    parser.add_argument("--align_method", choices=("arx_joy_home", "master_ctrl", "master_cmd", "none"), default="arx_joy_home")
    parser.add_argument("--master_align_command_mode", choices=("master_cmd", "master_ctrl"), default="master_ctrl")
    parser.add_argument("--arx_joy_home_repeats", type=int, default=60)
    parser.add_argument("--arx_joy_release_repeats", type=int, default=20)
    parser.add_argument("--arx_joy_home_hz", type=float, default=20.0)
    parser.add_argument("--arx_joy_home_settle_sec", type=float, default=5.0)
    parser.add_argument("--slave_follow_timeout", type=float, default=8.0)
    parser.add_argument("--slave_follow_tolerance", type=float, default=0.08)

    args = parser.parse_args()

    if args.save_depth and args.camera_front_backend != "orbbec":
        raise SystemExit("--save_depth true requires --camera_front_backend orbbec. RealSense front backend does not provide Orbbec depth.")

    def _on_sigint(sig, frame):
        print("\n[INFO] Ctrl+C received. Exiting only the collection script; ARX master/slave nodes keep running.")
        shutdown_event.set()

    signal.signal(signal.SIGINT, _on_sigint)

    rclpy.init()
    ros_operator = ARXTeleopObserver(args)
    collector = TeleopCollector(
        camera_names=CAMERA_NAMES,
        dataset_dir=args.dataset_dir,
        dataset_name=args.dataset_name,
        video_fps=args.video_fps,
        save_depth=args.save_depth,
        depth_camera_name=args.depth_camera_name,
    )

    spin_thread = threading.Thread(target=rclpy.spin, args=(ros_operator,), daemon=True)
    spin_thread.start()
    print("[INFO] ROS spin thread started")

    try:
        if not ros_operator.wait_for_data_ready(timeout=20.0):
            return

        collect_pose = np.asarray(args.collect_pose_left + args.collect_pose_right, dtype=float)
        if args.align_method == "arx_joy_home":
            collect_pose = np.asarray(args.home_pose_left + args.home_pose_right, dtype=float)
        if args.align_master_arms:
            ros_operator.wait_for_master_ready(timeout=args.master_ready_timeout)
        if args.align_collect_pose:
            collector.pause()
            should_switch_master_mode = args.switch_master_mode_for_align and args.align_master_arms and args.align_method != "arx_joy_home"
            if should_switch_master_mode:
                ros_operator.set_master_mode(args.master_left_node, "remote_master_ctrl")
                ros_operator.set_master_mode(args.master_right_node, "remote_master_ctrl")
            ros_operator.align_to_collect_pose(
                target_pos=collect_pose,
                duration=args.align_duration,
                hz=args.align_hz,
                slave_max_step=args.slave_align_max_step,
                master_max_step=args.master_align_max_step,
                align_slave=args.align_slave_arms,
                align_master=args.align_master_arms,
                slave_follow_timeout=args.slave_follow_timeout,
                slave_follow_tolerance=args.slave_follow_tolerance,
                align_method=args.align_method,
            )
            if should_switch_master_mode:
                ros_operator.set_master_mode(args.master_left_node, "remote_master")
                ros_operator.set_master_mode(args.master_right_node, "remote_master")
            print("[INFO] Initial collection pose ready.")

        if args.start_paused:
            print("\n[INFO] Ready. Press Space to start collecting.")
            collector.start_writer()
        else:
            collector.start()

        threading.Thread(target=keyboard_monitor_thread, args=(collector,), daemon=True).start()

        rate = ros_operator.create_rate(args.control_frequency)
        while rclpy.ok() and not shutdown_event.is_set():
            with request_lock:
                do_save = save_requested
                do_discard = discard_requested
                save_requested = False
                discard_requested = False

            if do_save:
                saved = collector.save_current_episode(export_video=args.save_video, video_fps=args.video_fps, resume_after_save=False)
                if saved and args.align_after_save:
                    collector.pause()
                    should_switch_master_mode = args.switch_master_mode_for_align and args.align_master_arms and args.align_method != "arx_joy_home"
                    if should_switch_master_mode:
                        ros_operator.set_master_mode(args.master_left_node, "remote_master_ctrl")
                        ros_operator.set_master_mode(args.master_right_node, "remote_master_ctrl")
                    ros_operator.align_to_collect_pose(
                        target_pos=collect_pose,
                        duration=args.align_duration,
                        hz=args.align_hz,
                        slave_max_step=args.slave_align_max_step,
                        master_max_step=args.master_align_max_step,
                        align_slave=args.align_slave_arms,
                        align_master=args.align_master_arms,
                        slave_follow_timeout=args.slave_follow_timeout,
                        slave_follow_tolerance=args.slave_follow_tolerance,
                        align_method=args.align_method,
                    )
                    if should_switch_master_mode:
                        ros_operator.set_master_mode(args.master_left_node, "remote_master")
                        ros_operator.set_master_mode(args.master_right_node, "remote_master")
                    collector.pause()
                    print("[INFO] Episode saved/alignment requested. Press Space to start the next episode.")
            if do_discard:
                collector.discard_current_episode()

            frame = ros_operator.get_frame()
            if frame is None:
                rate.sleep()
                continue

            imgs, j_left, j_right = frame
            obs = build_observation(imgs, j_left, j_right)
            if args.save_depth:
                obs["depths"] = ros_operator.get_depth_payload()
            action = select_action(obs, ros_operator, args.action_source)
            collector.add_frame(obs, action)

            if args.visualize_cameras:
                try:
                    keep_running = show_camera_visualization(imgs, args, collector.is_collecting, collector.frame_count)
                except cv2.error as e:
                    print(f"\n[WARN] Camera visualization failed, disabling it: {e}")
                    args.visualize_cameras = False
                    keep_running = True
                if not keep_running:
                    print("\n[INFO] Visualization quit requested. This exits only the collection script; ARX master/slave nodes keep running.")
                    shutdown_event.set()

            rate.sleep()
    except Exception as e:
        print(f"[ERROR] Main loop error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        shutdown_event.set()
        collector.pause()
        if collector.has_data():
            print(f"[WARN] Unsaved frames remain in memory: {collector.frame_count}. They were not written because the script is exiting.")
        collector.shutdown()
        if getattr(args, "switch_master_mode_for_align", False) and getattr(args, "align_master_arms", False):
            try:
                ros_operator.set_master_mode(args.master_left_node, "remote_master")
                ros_operator.set_master_mode(args.master_right_node, "remote_master")
            except Exception:
                pass
        ros_operator.cleanup_cameras()
        if args.visualize_cameras:
            try:
                cv2.destroyWindow(args.camera_visualization_window)
            except Exception:
                pass
        if rclpy.ok():
            rclpy.shutdown()
        print("[INFO] Teleop collector exited. ARX master/slave ROS2 nodes were not stopped.")


if __name__ == "__main__":
    main()
