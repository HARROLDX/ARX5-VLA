#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
YuHai left-arm single-arm collector for ARX-X5.

Records the compact dataset requested by YuHai:
  pixels:      third-camera RGB frames resized/cropped to 224x224
  action:      (dx, dy, dz, dyaw, d_gripper) from the commanded EE stream
  goal_pixels: the final recorded image frame

Default action source is /arm_master_l_status because the remote slave launch
subscribes the left slave to that RobotStatus stream. If your teleop stack sends
end-effector commands directly, use --action_msg_type pos_cmd or robot_cmd.
"""

import argparse
import json
import math
import os
import select
import signal
import sys
import termios
import threading
import time
import tty
from collections import deque
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError as e:
    raise SystemExit("OpenCV (cv2) is required. Activate the same env used by the ARX collectors.") from e

try:
    import rclpy
    from rclpy.node import Node
    from arx5_arm_msg.msg import RobotCmd, RobotStatus
    from arm_control.msg import PosCmd
    import arx_teleop_collect_aligned as dual
except ImportError as e:
    ROS_IMPORT_ERROR = e
    rclpy = None
    RobotCmd = object
    RobotStatus = object
    PosCmd = object
    dual = None

    class Node:  # type: ignore[no-redef]
        pass
else:
    ROS_IMPORT_ERROR = None


DEFAULT_DATASET_DIR = "/home/agilex/kai0/kai0_data/arx_teleop_yuhai"
CAMERA_NAMES = ("camera_third",)
TRUE_STRINGS = {"1", "true", "t", "yes", "y", "on", "ture"}
FALSE_STRINGS = {"0", "false", "f", "no", "n", "off", "none"}

shutdown_event = threading.Event()


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in TRUE_STRINGS:
        return True
    if value in FALSE_STRINGS:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {value}")


def wrap_angle(rad: float) -> float:
    return (float(rad) + math.pi) % (2.0 * math.pi) - math.pi


def make_224_rgb(img: np.ndarray, resize_mode: str) -> np.ndarray:
    if resize_mode == "resize":
        return cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)

    h, w = img.shape[:2]
    scale = 224.0 / min(h, w)
    new_w = max(224, int(round(w * scale)))
    new_h = max(224, int(round(h * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    y0 = (new_h - 224) // 2
    x0 = (new_w - 224) // 2
    return resized[y0 : y0 + 224, x0 : x0 + 224].copy()


def robot_status_pose(msg: RobotStatus) -> Tuple[np.ndarray, float]:
    return np.asarray([msg.end_pos[0], msg.end_pos[1], msg.end_pos[2], msg.end_pos[5]], dtype=np.float64), float(msg.joint_pos[6])


def pos_cmd_pose(msg: PosCmd) -> Tuple[np.ndarray, float]:
    return np.asarray([msg.x, msg.y, msg.z, msg.yaw], dtype=np.float64), float(msg.gripper)


def robot_cmd_pose(msg: RobotCmd) -> Tuple[np.ndarray, float]:
    return np.asarray([msg.end_pos[0], msg.end_pos[1], msg.end_pos[2], msg.end_pos[5]], dtype=np.float64), float(msg.gripper)


class EpisodeBuffer:
    def __init__(self):
        self.pixels = {name: [] for name in CAMERA_NAMES}
        self.actions = []
        self.timestamps = []
        self.collecting = False
        self.episode_index = 0
        self.lock = threading.Lock()

    @property
    def frame_count(self) -> int:
        with self.lock:
            return len(self.actions)

    def start(self):
        with self.lock:
            self.pixels = {name: [] for name in CAMERA_NAMES}
            self.actions = []
            self.timestamps = []
            self.collecting = True
        print("[INFO] Recording started.")

    def pause(self):
        with self.lock:
            self.collecting = False
        print("[INFO] Recording paused.")

    def discard(self):
        with self.lock:
            n = len(self.actions)
            self.pixels = {name: [] for name in CAMERA_NAMES}
            self.actions = []
            self.timestamps = []
            self.collecting = False
        print(f"[INFO] Discarded current episode ({n} frames).")

    def add(self, pixels: Dict[str, np.ndarray], action: np.ndarray, timestamp: float):
        with self.lock:
            if not self.collecting:
                return
            for name in CAMERA_NAMES:
                self.pixels[name].append(pixels[name].copy())
            self.actions.append(action.astype(np.float32).copy())
            self.timestamps.append(float(timestamp))

    def save(self, dataset_dir: str, dataset_name: str, metadata: dict) -> Optional[Path]:
        try:
            import h5py
        except ImportError as e:
            raise RuntimeError("h5py is required to save YuHai HDF5 episodes.") from e

        with self.lock:
            if not self.actions:
                print("[WARN] No frames to save.")
                return None
            pixels = {name: np.asarray(self.pixels[name], dtype=np.uint8) for name in CAMERA_NAMES}
            actions = np.asarray(self.actions, dtype=np.float32)
            timestamps = np.asarray(self.timestamps, dtype=np.float64)
            episode_index = self.episode_index
            self.episode_index += 1
            self.pixels = {name: [] for name in CAMERA_NAMES}
            self.actions = []
            self.timestamps = []
            self.collecting = False

        root = Path(dataset_dir) / dataset_name
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"episode_{episode_index:06d}.hdf5"
        while path.exists():
            episode_index += 1
            path = root / f"episode_{episode_index:06d}.hdf5"

        with h5py.File(path, "w") as f:
            f.attrs["sim"] = False
            f.attrs["format"] = "yuhai_left_single_v1"
            f.attrs["camera_names"] = json.dumps(list(CAMERA_NAMES))
            f.attrs["action_order"] = json.dumps(["dx", "dy", "dz", "dyaw", "d_gripper"])
            f.attrs["metadata"] = json.dumps(metadata, ensure_ascii=False)
            pixels_grp = f.create_group("pixels")
            goal_grp = f.create_group("goal_pixels")
            obs_grp = f.create_group("observations")
            obs_pixels_grp = obs_grp.create_group("pixels")
            for name in CAMERA_NAMES:
                pixels_grp.create_dataset(name, data=pixels[name], dtype="uint8", compression="gzip", compression_opts=4)
                goal_grp.create_dataset(name, data=pixels[name][-1].copy(), dtype="uint8")
                obs_pixels_grp[name] = h5py.SoftLink(f"/pixels/{name}")
            f.create_dataset("action", data=actions, dtype="float32")
            f.create_dataset("timestamp", data=timestamps, dtype="float64")

        print(f"[INFO] Saved {len(actions)} frames to {path}")
        for name in CAMERA_NAMES:
            print(f"  {name}: pixels/{name} shape={pixels[name].shape}, goal_pixels/{name} shape={pixels[name][-1].shape}")
        return path


class YuHaiLeftSingleCollector(Node):
    def __init__(self, args):
        super().__init__("arx_teleop_collect_yuhai_left_single")
        self.args = args
        self.rs_readers = {}
        self.last_images: Dict[str, np.ndarray] = {}
        self.last_depths: Dict[str, np.ndarray] = {}
        self.command_lock = threading.Lock()
        self.latest_pose = None
        self.latest_gripper = None
        self.prev_sample_pose = None
        self.prev_sample_gripper = None
        self.latest_command_stamp = 0.0
        self.data_ready = {"cameras": False, "action": False}

        if args.action_msg_type == "robot_status":
            msg_type = RobotStatus
            cb = self.robot_status_callback
        elif args.action_msg_type == "pos_cmd":
            msg_type = PosCmd
            cb = self.pos_cmd_callback
        else:
            msg_type = RobotCmd
            cb = self.robot_cmd_callback
        self.create_subscription(msg_type, args.action_topic, cb, 50)

        self.get_logger().info(f"Sub action: {args.action_topic} ({args.action_msg_type})")
        self.init_cameras()

    def init_cameras(self):
        if dual is None:
            raise RuntimeError("arx_teleop_collect_aligned.py could not be imported; camera helpers are unavailable.")

        third_ok = False

        third_width = int(getattr(self.args, "camera_third_width", 640))
        third_height = int(getattr(self.args, "camera_third_height", 480))
        third_fps = int(getattr(self.args, "camera_third_fps", 30))
        third_timeout_ms = int(getattr(self.args, "camera_third_timeout_ms", 5000))
        third_startup_timeout = float(getattr(self.args, "camera_third_startup_timeout", 5.0))

        try:
            print("Initializing third RealSense camera...")
            reader = dual.AsyncRealSenseCamera(
                "camera_third",
                self.args.camera_third_serial,
                width=third_width,
                height=third_height,
                fps=third_fps,
                timeout_ms=third_timeout_ms,
            )
            reader.open()
            if reader.wait_for_first_frame(third_startup_timeout):
                self.rs_readers["camera_third"] = reader
                third_ok = True
                print(f"  camera_third started (serial={self.args.camera_third_serial}, {third_width}x{third_height}@{third_fps})")
            else:
                reader.release()
                print(f"  camera_third opened but produced no frame within {third_startup_timeout:.1f}s.")
        except Exception as e:
            print(f"Third RealSense init failed: {e}")

        self.data_ready["cameras"] = third_ok
        if self.data_ready["cameras"]:
            print("[INFO] YuHai camera set ready: camera_third.")
        else:
            print(f"[ERROR] Camera readiness failed: camera_third={third_ok}")

    def get_camera_images(self) -> Dict[str, np.ndarray]:
        images = {}
        fallback = self.last_images

        for cam_name, reader in self.rs_readers.items():
            frame_rgb = None
            try:
                frame_rgb = reader.get_frame()
            except Exception:
                pass
            if frame_rgb is not None:
                images[cam_name] = make_224_rgb(frame_rgb, self.args.resize_mode)
            elif cam_name in fallback:
                images[cam_name] = fallback[cam_name]

        if all(name in images for name in CAMERA_NAMES):
            self.last_images = {name: images[name] for name in CAMERA_NAMES}
        return images

    def cleanup_cameras(self):
        print("Stopping YuHai cameras...")
        for reader in self.rs_readers.values():
            try:
                reader.release()
            except Exception:
                pass
        self.rs_readers = {}

    def update_command(self, pose: np.ndarray, gripper: float):
        with self.command_lock:
            self.latest_pose = pose.copy()
            self.latest_gripper = float(gripper)
            self.latest_command_stamp = time.time()
            self.data_ready["action"] = True

    def robot_status_callback(self, msg: RobotStatus):
        pose, gripper = robot_status_pose(msg)
        self.update_command(pose, gripper)

    def pos_cmd_callback(self, msg: PosCmd):
        pose, gripper = pos_cmd_pose(msg)
        self.update_command(pose, gripper)

    def robot_cmd_callback(self, msg: RobotCmd):
        pose, gripper = robot_cmd_pose(msg)
        self.update_command(pose, gripper)

    def wait_for_data_ready(self, timeout: float) -> bool:
        print("[INFO] Waiting for camera_third and action command stream...")
        start = time.time()
        while time.time() - start < timeout and rclpy.ok() and not shutdown_event.is_set():
            if self.data_ready["cameras"] and self.data_ready["action"]:
                print("[INFO] Data streams ready.")
                return True
            time.sleep(0.1)
        print(f"[ERROR] Timeout. ready={self.data_ready}")
        return False

    def get_sample(self) -> Optional[Tuple[Dict[str, np.ndarray], np.ndarray, float]]:
        if not self.data_ready["cameras"] or not self.data_ready["action"]:
            return None
        pixels = self.get_camera_images()
        if not all(name in pixels for name in CAMERA_NAMES):
            return None
        ts = time.time()
        with self.command_lock:
            if self.latest_pose is None:
                return None
            pose = self.latest_pose.copy()
            gripper = float(self.latest_gripper)

        if self.prev_sample_pose is None:
            action = np.zeros(5, dtype=np.float32)
        else:
            action = np.asarray(
                [
                    pose[0] - self.prev_sample_pose[0],
                    pose[1] - self.prev_sample_pose[1],
                    pose[2] - self.prev_sample_pose[2],
                    wrap_angle(pose[3] - self.prev_sample_pose[3]),
                    gripper - self.prev_sample_gripper,
                ],
                dtype=np.float32,
            )
        self.prev_sample_pose = pose
        self.prev_sample_gripper = gripper
        return pixels, action, ts


def keyboard_thread(buffer: EpisodeBuffer, args, get_metadata):
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        print("\nKeys: Space=start/pause, s=save, d=discard, q=quit")
        while not shutdown_event.is_set():
            readable, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not readable:
                continue
            ch = sys.stdin.read(1)
            if ch == " ":
                if buffer.collecting:
                    buffer.pause()
                else:
                    buffer.start()
            elif ch in ("s", "S"):
                buffer.save(args.dataset_dir, args.dataset_name, get_metadata())
            elif ch in ("d", "D"):
                buffer.discard()
            elif ch in ("q", "Q", "\x03", "\x1b"):
                shutdown_event.set()
                break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def show_preview(pixels: Dict[str, np.ndarray], collecting: bool, frame_count: int, window: str) -> bool:
    frames = []
    for name in CAMERA_NAMES:
        bgr = cv2.cvtColor(pixels[name], cv2.COLOR_RGB2BGR)
        cv2.putText(bgr, name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        frames.append(bgr)
    bgr = np.hstack(frames)
    status = f"{'REC' if collecting else 'PAUSED'} | frames={frame_count}"
    color = (0, 0, 255) if collecting else (0, 255, 255)
    cv2.putText(bgr, status, (8, 216), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    cv2.imshow(window, bgr)
    key = cv2.waitKey(1) & 0xFF
    return key not in (ord("q"), 27)


def parse_args():
    parser = argparse.ArgumentParser(description="Collect YuHai-format left single-arm ARX teleop data.")
    parser.add_argument("--dataset_dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--dataset_name", default="task_yuhai_left_single_collect")
    parser.add_argument("--action_topic", default="/arm_master_l_status")
    parser.add_argument("--action_msg_type", choices=("robot_status", "pos_cmd", "robot_cmd"), default="robot_status")
    parser.add_argument("--camera_third_serial", type=str, default="836612070632")
    parser.add_argument("--camera_third_width", type=int, default=640)
    parser.add_argument("--camera_third_height", type=int, default=480)
    parser.add_argument("--camera_third_fps", type=int, default=30)
    parser.add_argument("--camera_third_timeout_ms", type=int, default=5000)
    parser.add_argument("--camera_third_startup_timeout", type=float, default=5.0)
    parser.add_argument("--camera_front_backend", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--camera_left_serial", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--align_depth_to_color", type=str2bool, nargs="?", const=True, default=False, help=argparse.SUPPRESS)
    parser.add_argument("--resize_mode", choices=("resize", "center_crop"), default="resize")
    parser.add_argument("--control_frequency", type=float, default=30.0)
    parser.add_argument("--start_paused", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--visualize", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--visualization_window", default="YuHai left single cameras")
    parser.add_argument("--ready_timeout", type=float, default=20.0)
    return parser.parse_args()


def main():
    args = parse_args()
    if ROS_IMPORT_ERROR is not None:
        raise SystemExit(
            "ROS2 Python packages or ARX messages were not found in this environment.\n"
            f"Original import error: {ROS_IMPORT_ERROR}\n"
            "Source the ROS distro and X5_ws overlay before recording, for example:\n"
            "  source /opt/ros/humble/setup.bash\n"
            "  source /mnt/workspace/xiajiawei/kai0/train_deploy_alignment/dagger/arx/X5_ws/install/setup.bash"
        )

    def on_sigint(sig, frame):
        print("\n[INFO] Ctrl+C received; exiting collector only.")
        shutdown_event.set()

    signal.signal(signal.SIGINT, on_sigint)

    rclpy.init()
    node = YuHaiLeftSingleCollector(args)
    buffer = EpisodeBuffer()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    try:
        if not node.wait_for_data_ready(args.ready_timeout):
            return
        if args.start_paused:
            print("[INFO] Ready. Press Space to start collecting.")
        else:
            buffer.start()

        def metadata():
            return {
                "camera_names": list(CAMERA_NAMES),
                "camera_third_source": "RealSense AsyncRealSenseCamera, same camera_third path as arx_teleop_collect_left_single_aligned.py",
                "camera_third_serial": args.camera_third_serial,
                "action_topic": args.action_topic,
                "action_msg_type": args.action_msg_type,
                "resize_mode": args.resize_mode,
                "source": "action is delta of the commanded absolute EE/gripper stream at collection ticks",
            }

        threading.Thread(target=keyboard_thread, args=(buffer, args, metadata), daemon=True).start()
        rate = node.create_rate(args.control_frequency)
        while rclpy.ok() and not shutdown_event.is_set():
            sample = node.get_sample()
            if sample is not None:
                pixels, action, ts = sample
                buffer.add(pixels, action, ts)
                if args.visualize and not show_preview(pixels, buffer.collecting, buffer.frame_count, args.visualization_window):
                    shutdown_event.set()
                    break
            rate.sleep()
    finally:
        print("\n[INFO] Shutting down YuHai left single collector...")
        node.cleanup_cameras()
        if args.visualize:
            cv2.destroyAllWindows()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        spin_thread.join(timeout=1.0)
        print("[INFO] Exit complete.")


if __name__ == "__main__":
    main()
