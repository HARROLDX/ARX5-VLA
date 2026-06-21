#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Left-arm-only ARX-X5 teleoperation data collector.

This file intentionally lives beside the dual-arm collector but does not modify
it. It reuses the mature camera/video/keyboard utilities from
arx_teleop_collect_aligned.py and records only the left slave/master arm as
7-D qpos/qvel/effort/action arrays.
"""

import argparse
import os
import signal
import sys
import threading
import time
from collections import deque
from typing import Dict, Optional

import h5py
import numpy as np

import arx_teleop_collect_aligned as dual


CAMERA_NAMES = ["cam_high", "cam_left_wrist", "camera_third"]


def save_data_left_single(observations, actions, dataset_path):
    """Save a left-arm HDF5 episode with dynamic 7-D joint arrays."""
    data_size = len(actions)
    if data_size == 0:
        raise ValueError("No actions to save.")

    qpos = np.asarray([obs["qpos"] for obs in observations], dtype=np.float32)
    qvel = np.asarray([obs["qvel"] for obs in observations], dtype=np.float32)
    effort = np.asarray([obs["effort"] for obs in observations], dtype=np.float32)
    action = np.asarray(actions, dtype=np.float32)

    if qpos.ndim != 2 or qpos.shape[1] != 7:
        raise ValueError(f"Expected left-arm qpos shape (T, 7), got {qpos.shape}")
    if qvel.shape != qpos.shape or effort.shape != qpos.shape or action.shape != qpos.shape:
        raise ValueError(
            f"Expected qpos/qvel/effort/action all shape {qpos.shape}, "
            f"got qvel={qvel.shape}, effort={effort.shape}, action={action.shape}"
        )

    t0 = time.time()
    print("\033[33m>>> Saving left-arm HDF5...\033[0m")
    with h5py.File(dataset_path + ".hdf5", "w", rdcc_nbytes=1024**2 * 2) as root:
        root.attrs["sim"] = False
        root.attrs["compress"] = False
        root.attrs["arm_mode"] = "left_single"
        root.attrs["joint_dim"] = 7
        obs_grp = root.create_group("observations")
        obs_grp.create_dataset("qpos", data=qpos, dtype="float32")
        obs_grp.create_dataset("qvel", data=qvel, dtype="float32")
        obs_grp.create_dataset("effort", data=effort, dtype="float32")
        root.create_dataset("action", data=action, dtype="float32")

    print(f"[INFO] HDF5 saved in {time.time() - t0:.1f}s")
    print(f"\033[32m  Path: {dataset_path}.hdf5\033[0m")
    print(f"\033[32m  Frames: {data_size}, joint_dim: 7\033[0m")


# TeleopCollector resolves save_data from the imported module at runtime.
dual.save_data = save_data_left_single


class ARXLeftTeleopObserver(dual.Node):
    """ROS2 observer for left slave/master state and left-arm camera frames."""

    def __init__(self, args):
        super().__init__("arx_left_single_teleop_collector")
        self.args = args
        self.camera_names = list(getattr(args, "camera_names", CAMERA_NAMES))

        self.joint_left_deque = deque(maxlen=2000)
        self.master_left_deque = deque(maxlen=2000)
        self.data_ready = {"joint_left": False, "cameras": False}
        self.data_ready_lock = threading.Lock()
        self.pipelines = {}
        self.rs_readers = {}
        self.front_reader = None
        self.last_images: Dict[str, np.ndarray] = {}
        self.last_depths: Dict[str, np.ndarray] = {}

        self.create_subscription(dual.RobotStatus, args.joint_state_topic_left, self.joint_left_callback, 10)
        self.create_subscription(dual.RobotStatus, args.master_status_topic_left, self.master_left_callback, 10)
        self.create_subscription(dual.RobotStatus, args.master_status_topic_left_ctrl, self.master_left_callback, 10)

        self.pub_left = self.create_publisher(dual.RobotStatus, args.joint_cmd_topic_left, 10)
        self.master_cmd_left_pub = self.create_publisher(dual.RobotStatus, args.master_cmd_topic_left, 10)
        self.master_ctrl_left_pub = self.create_publisher(dual.RobotStatus, args.master_ctrl_cmd_topic_left, 10)
        self.arx_joy_pub = self.create_publisher(dual.Int32MultiArray, args.arx_joy_topic, 10)
        self.master_cmd_cache = np.zeros(7, dtype=float)

        self.get_logger().info(f"Sub left slave: {args.joint_state_topic_left}")
        self.get_logger().info(f"Sub left master: {args.master_status_topic_left} / {args.master_status_topic_left_ctrl}")
        self.get_logger().info(f"Pub left slave command mirror: {args.joint_cmd_topic_left}")
        self.get_logger().info(f"Pub left master command: {args.master_cmd_topic_left}")
        self.get_logger().info(f"Pub left master ctrl command: {args.master_ctrl_cmd_topic_left}")

        self.init_cameras()

    def joint_left_callback(self, msg):
        self.joint_left_deque.append(msg)
        with self.data_ready_lock:
            self.data_ready["joint_left"] = True

    def master_left_callback(self, msg):
        self.master_left_deque.append(msg)

    def get_slave_positions(self) -> Optional[np.ndarray]:
        if len(self.joint_left_deque) == 0:
            return None
        left = np.asarray(self.joint_left_deque[-1].joint_pos, dtype=float)
        if left.shape[0] != 7:
            self.get_logger().warn(f"Expected 7-D left slave state, got {left.shape}")
            return None
        return left.copy()

    def get_master_positions(self) -> Optional[np.ndarray]:
        if len(self.master_left_deque) == 0:
            return None
        left = np.asarray(self.master_left_deque[-1].joint_pos, dtype=float)
        if left.shape[0] != 7:
            self.get_logger().warn(f"Expected 7-D left master state, got {left.shape}")
            return None
        return left.copy()

    def set_joint_positions(self, pos: np.ndarray):
        if not dual.rclpy.ok():
            return
        pos = np.asarray(pos, dtype=float)
        if pos.shape != (7,):
            self.get_logger().warn(f"Expected 7-D left slave target, got {pos.shape}")
            return
        msg = dual.RobotStatus()
        msg.joint_pos = [float(x) for x in pos]
        self.pub_left.publish(msg)

    def publish_master_ctrl_positions(self, pos: np.ndarray):
        pos = np.asarray(pos, dtype=float)
        if pos.shape != (7,):
            self.get_logger().warn(f"Expected 7-D left master target, got {pos.shape}")
            return
        self.master_cmd_cache = pos.copy()
        msg = dual.RobotStatus()
        msg.joint_pos = [float(x) for x in pos]
        mode = str(getattr(self.args, "master_align_command_mode", "master_ctrl"))
        if mode == "master_ctrl":
            self.master_ctrl_left_pub.publish(msg)
        else:
            self.master_cmd_left_pub.publish(msg)

    def set_master_mode(self, node_name: str, mode: str, timeout: float = 2.0) -> bool:
        return dual.ARXTeleopObserver.set_master_mode(self, node_name, mode, timeout)

    def _smooth_interpolate_positions(self, start, target, duration, hz, max_joint_step, publish_fn, label):
        start = np.asarray(start, dtype=float)
        target = np.asarray(target, dtype=float)
        if start.shape != (7,) or target.shape != (7,):
            self.get_logger().warn(f"[{label}] Expected 7-D start/target, got {start.shape}/{target.shape}")
            return False
        max_delta = float(np.max(np.abs(target - start)))
        num_steps = max(1, int(duration * hz))
        if max_joint_step and max_joint_step > 0:
            num_steps = max(num_steps, int(np.ceil(max_delta / max_joint_step)))
        print(f"[INFO] {label}: max joint delta {max_delta:.4f} rad, steps {num_steps}")
        prev = start.copy()
        for step in range(num_steps + 1):
            if dual.shutdown_event.is_set() or not dual.rclpy.ok():
                return False
            alpha = step / num_steps
            smooth_alpha = (1.0 - np.cos(alpha * np.pi)) / 2.0
            cmd = start * (1.0 - smooth_alpha) + target * smooth_alpha
            if max_joint_step and max_joint_step > 0:
                step_limits = np.full(7, float(max_joint_step), dtype=float)
                step_limits[6] = np.inf
                cmd = prev + np.clip(cmd - prev, -step_limits, step_limits)
            publish_fn(cmd)
            prev = cmd.copy()
            if step % 50 == 0 or step == num_steps:
                print(f"\r{label}: {int(alpha * 100)}%", end="", flush=True)
            time.sleep(1.0 / hz)
        print(f"\n[INFO] {label} done")
        return True

    def smooth_goto_slave_position(self, target_pos, duration, hz, max_joint_step):
        start = self.get_slave_positions()
        if start is None:
            print("[WARN] Left slave state unavailable; skip slave alignment.")
            return False
        return self._smooth_interpolate_positions(start, target_pos, duration, hz, max_joint_step, self.set_joint_positions, "Left slave align")

    def smooth_goto_master_position(self, target_pos, duration, hz, max_joint_step):
        start = self.get_master_positions()
        if start is None:
            print("[WARN] Left master state unavailable; skip master alignment.")
            return False
        return self._smooth_interpolate_positions(start, target_pos, duration, hz, max_joint_step, self.publish_master_ctrl_positions, "Left master align")

    def wait_for_arm_near_pose(self, name, get_positions_fn, target_pos, timeout=3.0, tolerance=0.08, check_gripper=True):
        target_pos = np.asarray(target_pos, dtype=float)
        mask = np.ones(7, dtype=bool)
        if not check_gripper:
            mask[6] = False
        start_time = time.time()
        while time.time() - start_time < timeout and dual.rclpy.ok() and not dual.shutdown_event.is_set():
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
        return dual.ARXTeleopObserver.publish_arx_joy(self, data, label, repeats, hz)

    def publish_arx_go_home(self, repeats: int = 20, hz: float = 20.0) -> bool:
        return self.publish_arx_joy([0, 1], "GO_HOME", repeats=repeats, hz=hz)

    def publish_arx_compensation(self, repeats: int = 20, hz: float = 20.0) -> bool:
        return self.publish_arx_joy([1, 0], "G_COMPENSATION", repeats=repeats, hz=hz)

    def wait_for_slave_near_pose(self, target_pos, timeout=3.0, tolerance=0.08, check_gripper=True):
        return self.wait_for_arm_near_pose("Left slave", self.get_slave_positions, target_pos, timeout, tolerance, check_gripper)

    def align_to_collect_pose(
        self,
        target_pos,
        duration,
        hz,
        slave_max_step,
        master_max_step,
        align_slave=False,
        align_master=True,
        slave_follow_timeout=3.0,
        slave_follow_tolerance=0.08,
        align_method="master_ctrl",
    ) -> bool:
        target_pos = np.asarray(target_pos, dtype=float)
        if target_pos.shape != (7,):
            raise ValueError(f"left collect pose must be 7-D, got {target_pos.shape}")
        print(f"[INFO] Aligning left arm to collection pose, method={align_method}")

        if align_method == "none":
            print("[INFO] Alignment disabled by --align_method none")
            return True

        results = []
        if align_method == "arx_joy_home":
            print("[WARN] arx_joy_home is global on ARX and may also move the right arm.")
            ok_home = self.publish_arx_go_home(repeats=int(self.args.arx_joy_home_repeats), hz=float(self.args.arx_joy_home_hz))
            results.append(("arx_joy_home", ok_home))
            if ok_home:
                time.sleep(float(self.args.arx_joy_home_settle_sec))
                results.append(("left_master_home_reached", self.wait_for_arm_near_pose("Left master", self.get_master_positions, target_pos, timeout=slave_follow_timeout, tolerance=slave_follow_tolerance, check_gripper=False)))
                results.append(("left_slave_home_reached", self.wait_for_arm_near_pose("Left slave", self.get_slave_positions, target_pos, timeout=slave_follow_timeout, tolerance=slave_follow_tolerance, check_gripper=False)))
            results.append(("release_compensation", self.publish_arx_compensation(repeats=int(self.args.arx_joy_release_repeats), hz=float(self.args.arx_joy_home_hz))))
            ok = all(item_ok for _, item_ok in results)
            print(f"[INFO] Left collection pose alignment {'done' if ok else 'finished with warnings'}: {results}")
            return ok

        if align_master:
            ok_master = self.smooth_goto_master_position(target_pos, duration, hz, master_max_step)
            results.append(("left_master", ok_master))
            if ok_master:
                results.append(("left_master_reached", self.wait_for_arm_near_pose("Left master", self.get_master_positions, target_pos, timeout=slave_follow_timeout, tolerance=slave_follow_tolerance)))
                results.append(("left_slave_follow", self.wait_for_slave_near_pose(target_pos, timeout=slave_follow_timeout, tolerance=slave_follow_tolerance)))

        if align_slave:
            print("[WARN] Direct left slave alignment is enabled. Use this only outside normal teleop following mode.")
            results.append(("left_slave_direct", self.smooth_goto_slave_position(target_pos, duration, hz, slave_max_step)))

        ok = all(item_ok for _, item_ok in results) if results else True
        print(f"[INFO] Left collection pose alignment {'done' if ok else 'finished with warnings'}: {results}")
        return ok

    def init_cameras(self):
        front_backend = str(getattr(self.args, "camera_front_backend", "realsense")).lower()
        if front_backend not in ("realsense", "orbbec"):
            print(f"Unknown camera_front_backend={front_backend}, fallback to realsense.")
            front_backend = "realsense"

        enabled_cameras = set(self.camera_names)
        front_ok = "cam_high" not in enabled_cameras
        left_wrist_ok = "cam_left_wrist" not in enabled_cameras
        third_ok = "camera_third" not in enabled_cameras
        try:
            camera_serials = {}
            if "cam_left_wrist" in enabled_cameras:
                camera_serials["cam_left_wrist"] = {
                    "serial": self.args.camera_left_serial,
                    "enable_depth": False,
                }
            if "camera_third" in enabled_cameras:
                camera_serials["camera_third"] = {
                    "serial": self.args.camera_third_serial,
                    "enable_depth": bool(getattr(self.args, "save_third_depth", False)),
                }
            if front_backend == "realsense" and "cam_high" in enabled_cameras:
                camera_serials["cam_high"] = {
                    "serial": self.args.camera_front_serial,
                    "enable_depth": bool(getattr(self.args, "save_high_depth", False)),
                }

            wrist_width = int(getattr(self.args, "camera_wrist_width", 640))
            wrist_height = int(getattr(self.args, "camera_wrist_height", 480))
            wrist_fps = int(getattr(self.args, "camera_wrist_fps", 30))
            wrist_timeout_ms = int(getattr(self.args, "camera_wrist_timeout_ms", 5000))
            wrist_startup_timeout = float(getattr(self.args, "camera_wrist_startup_timeout", 5.0))
            third_width = int(getattr(self.args, "camera_third_width", wrist_width))
            third_height = int(getattr(self.args, "camera_third_height", wrist_height))
            third_fps = int(getattr(self.args, "camera_third_fps", wrist_fps))
            third_timeout_ms = int(getattr(self.args, "camera_third_timeout_ms", wrist_timeout_ms))
            third_startup_timeout = float(getattr(self.args, "camera_third_startup_timeout", wrist_startup_timeout))
            align_third_depth = bool(getattr(self.args, "align_third_depth_to_color", True))
            print(f"Initializing enabled RealSense cameras: {list(camera_serials.keys())}")
            for cam_name, cfg in camera_serials.items():
                serial = cfg["serial"]
                is_third = cam_name == "camera_third"
                width = third_width if is_third else wrist_width
                height = third_height if is_third else wrist_height
                fps = third_fps if is_third else wrist_fps
                timeout_ms = third_timeout_ms if is_third else wrist_timeout_ms
                startup_timeout = third_startup_timeout if is_third else wrist_startup_timeout
                reader = dual.AsyncRealSenseCamera(
                    cam_name,
                    serial,
                    width=width,
                    height=height,
                    fps=fps,
                    timeout_ms=timeout_ms,
                    enable_depth=bool(cfg.get("enable_depth", False)),
                    align_depth_to_color=align_third_depth if is_third else True,
                )
                reader.open()
                if reader.wait_for_first_frame(startup_timeout):
                    self.rs_readers[cam_name] = reader
                    depth_suffix = ", depth=on" if cfg.get("enable_depth", False) else ""
                    print(f"  {cam_name} started (serial={serial}, {width}x{height}@{fps}{depth_suffix})")
                else:
                    reader.release()
                    print(f"  {cam_name} opened but produced no frame within {startup_timeout:.1f}s (serial={serial})")
            if "cam_left_wrist" in enabled_cameras:
                left_wrist_ok = "cam_left_wrist" in self.rs_readers
            if "camera_third" in enabled_cameras:
                third_ok = "camera_third" in self.rs_readers
            if front_backend == "realsense" and "cam_high" in enabled_cameras:
                front_ok = "cam_high" in self.rs_readers
        except Exception as e:
            print(f"RealSense init failed: {e}")
            for reader in self.rs_readers.values():
                try:
                    reader.release()
                except Exception:
                    pass
            self.rs_readers = {}

        if front_backend == "orbbec" and "cam_high" in enabled_cameras:
            width = int(getattr(self.args, "camera_front_width", 640))
            height = int(getattr(self.args, "camera_front_height", 480))
            fps = int(getattr(self.args, "camera_front_fps", 30))
            align_depth_to_color = bool(getattr(self.args, "align_depth_to_color", True))
            print(f"Initializing Orbbec RGB-D front camera: {width}x{height}@{fps}, align_depth_to_color={align_depth_to_color}")
            try:
                reader = dual.AsyncOrbbecCamera(
                    width,
                    height,
                    fps,
                    align_depth_to_color=align_depth_to_color,
                    enable_depth=bool(getattr(self.args, "save_high_depth", False)),
                )
                if reader.open():
                    startup_timeout = float(getattr(self.args, "camera_front_startup_timeout", 5.0))
                    if reader.wait_for_first_frame(startup_timeout):
                        self.front_reader = reader
                        front_ok = True
                        depth_suffix = ", depth=on" if getattr(self.args, "save_high_depth", False) else ", depth=off"
                        print(f"  cam_high started (Orbbec color{depth_suffix})")
                    else:
                        reader.release()
                        print(f"Orbbec camera opened but produced no frames within {startup_timeout:.1f}s.")
            except Exception as e:
                print(f"Orbbec front camera init failed: {e}")

        cameras_ok = left_wrist_ok and front_ok and third_ok
        with self.data_ready_lock:
            self.data_ready["cameras"] = cameras_ok
        if cameras_ok:
            print("Left-arm camera set ready.")
        else:
            print(f"Camera readiness check failed: left_wrist_ok={left_wrist_ok}, front_ok={front_ok}, third_ok={third_ok}")

    def get_camera_images(self):
        """Get RGB images plus cached Orbbec/third-view depth payloads."""
        images = {}
        fallback = getattr(self, "last_images", {})
        depth_payload = {}

        for cam_name, reader in self.rs_readers.items():
            try:
                frame_rgb = reader.get_frame()
                if frame_rgb is not None:
                    images[cam_name] = frame_rgb
                depth_frame, depth_meta = reader.get_depth_frame()
                if depth_frame is not None:
                    depth_payload[cam_name] = depth_frame
                    depth_payload[f"{cam_name}_meta"] = depth_meta
            except Exception as e:
                if not hasattr(self, "_last_cam_warn") or time.time() - getattr(self, "_last_cam_warn", 0) > 2.0:
                    print(f"Failed to get image/depth from {cam_name}: {e}")
                    self._last_cam_warn = time.time()

            if cam_name not in images:
                if cam_name in fallback:
                    images[cam_name] = fallback[cam_name]
                elif not hasattr(self, "_last_cam_warn") or time.time() - getattr(self, "_last_cam_warn", 0) > 2.0:
                    print(f"Failed to get image from {cam_name}: no new frame and no cache")
                    self._last_cam_warn = time.time()

        if self.front_reader is not None:
            try:
                frame_rgb, depth_frame, depth_meta = self.front_reader.get_frame()
                if frame_rgb is not None:
                    images["cam_high"] = frame_rgb
                elif "cam_high" in fallback:
                    images["cam_high"] = fallback["cam_high"]
                if depth_frame is not None:
                    depth_payload["cam_high"] = depth_frame
                    depth_payload["cam_high_meta"] = depth_meta
            except Exception as e:
                if "cam_high" in fallback:
                    images["cam_high"] = fallback["cam_high"]
                else:
                    print(f"Failed to get image from cam_high: {e}")

        if depth_payload:
            self.last_depths = depth_payload
        if images:
            self.last_images = images
        return images

    get_depth_payload = dual.ARXTeleopObserver.get_depth_payload
    cleanup_cameras = dual.ARXTeleopObserver.cleanup_cameras

    def get_frame(self):
        if len(self.joint_left_deque) == 0:
            return None
        imgs = self.get_camera_images()
        if any(name not in imgs for name in self.camera_names):
            return None
        return {name: imgs[name] for name in self.camera_names}, self.joint_left_deque[-1]

    def wait_for_data_ready(self, timeout: float = 15.0) -> bool:
        print("Waiting for left-arm sensor data...")
        start_time = time.time()
        while time.time() - start_time < timeout and dual.rclpy.ok() and not dual.shutdown_event.is_set():
            with self.data_ready_lock:
                joints_ready = self.data_ready["joint_left"]
                cameras_ready = self.data_ready["cameras"]
            if joints_ready and cameras_ready:
                print("[INFO] Left-arm sensor data ready.")
                return True
            time.sleep(0.5)
        print("[ERROR] Timeout waiting for left-arm sensor data.")
        return False

    def wait_for_master_ready(self, timeout: float = 5.0) -> bool:
        print("Waiting for left master arm state...")
        start_time = time.time()
        while time.time() - start_time < timeout and dual.rclpy.ok() and not dual.shutdown_event.is_set():
            if self.get_master_positions() is not None:
                print("[INFO] Left master arm state ready.")
                return True
            time.sleep(0.1)
        print("[WARN] Timeout waiting for left master arm state.")
        return False


def build_observation(imgs, j_left):
    return {
        "qpos": np.asarray(j_left.joint_pos, dtype=float).copy(),
        "qvel": np.asarray(j_left.joint_vel, dtype=float).copy(),
        "effort": np.asarray(j_left.joint_cur, dtype=float).copy(),
        "images": imgs,
    }


def select_action(obs, ros_operator: ARXLeftTeleopObserver, action_source: str):
    if action_source == "master":
        master_qpos = ros_operator.get_master_positions()
        if master_qpos is not None:
            return master_qpos
        print("[WARN] Left master action requested but master state is not ready; using left slave qpos.")
    return obs["qpos"].copy()


def show_left_camera_visualization(images, args, collecting: bool, frame_count: int) -> bool:
    width = int(getattr(args, "camera_visualization_width", 320))
    camera_names = list(getattr(args, "camera_names", CAMERA_NAMES))
    frames = []
    for cam_name in camera_names:
        frame = dual._camera_display_frame(images.get(cam_name))
        frame = dual._resize_keep_aspect(frame, width)
        dual.cv2.putText(frame, cam_name, (8, 24), dual.cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        frames.append(frame)
    h = max(frame.shape[0] for frame in frames)
    padded = []
    for frame in frames:
        if frame.shape[0] < h:
            pad = np.zeros((h - frame.shape[0], frame.shape[1], 3), dtype=frame.dtype)
            frame = np.vstack([frame, pad])
        padded.append(frame)
    canvas = np.hstack(padded)
    status = f"{'REC' if collecting else 'PAUSED'} | frames={frame_count}"
    color = (0, 0, 255) if collecting else (0, 255, 255)
    dual.cv2.putText(canvas, status, (12, h - 16), dual.cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    dual.cv2.imshow(args.camera_visualization_window, canvas)
    key = dual.cv2.waitKey(1) & 0xFF
    return key not in (ord("q"), 27)


def parse_args():
    parser = argparse.ArgumentParser(description="ARX-X5 left-arm-only teleoperation data collector.")
    parser.add_argument("--joint_state_topic_left", default="/arm_slave_l_status")
    parser.add_argument("--master_status_topic_left", default="/arm_master_l_status")
    parser.add_argument("--joint_cmd_topic_left", default="/arm_master_l_status")
    parser.add_argument("--master_cmd_topic_left", default="/arm_master_l_cmd")
    parser.add_argument("--arx_joy_topic", default="/arx_joy")
    parser.add_argument("--master_status_topic_left_ctrl", default="/arm_master_ctrl_status_left")
    parser.add_argument("--master_ctrl_cmd_topic_left", default="/arm_master_ctrl_cmd_left")
    parser.add_argument("--master_left_node", default="/arm_master_l")

    parser.add_argument("--camera_front_serial", type=str, default="152122073503")
    parser.add_argument("--camera_left_serial", type=str, default="913522070540")
    parser.add_argument("--camera_third_serial", type=str, default="836612070632")
    parser.add_argument("--enable_camera_high", type=dual.str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--enable_camera_left_wrist", type=dual.str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--enable_camera_third", type=dual.str2bool, nargs="?", const=True, default=True)
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
    parser.add_argument("--camera_third_width", type=int, default=640)
    parser.add_argument("--camera_third_height", type=int, default=480)
    parser.add_argument("--camera_third_fps", type=int, default=30)
    parser.add_argument("--camera_third_timeout_ms", type=int, default=5000)
    parser.add_argument("--camera_third_startup_timeout", type=float, default=5.0)
    parser.add_argument("--save_depth", type=dual.str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--save_high_depth", type=dual.str2bool, nargs="?", const=True, default=None)
    parser.add_argument("--save_third_depth", type=dual.str2bool, nargs="?", const=True, default=None)
    parser.add_argument("--depth_camera_name", type=str, default=None)
    parser.add_argument("--align_depth_to_color", type=dual.str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--align_third_depth_to_color", type=dual.str2bool, nargs="?", const=True, default=True)

    parser.add_argument("--dataset_dir", type=str, default=dual.DEFAULT_DATASET_DIR)
    parser.add_argument("--dataset_name", type=str, default="task_left_single_collect")
    parser.add_argument("--control_frequency", type=float, default=30.0)
    parser.add_argument("--save_video", type=dual.str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--video_fps", type=int, default=30)
    parser.add_argument("--action_source", choices=("slave", "master"), default="slave")
    parser.add_argument("--start_paused", type=dual.str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--visualize_cameras", type=dual.str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--camera_visualization_width", type=int, default=320)
    parser.add_argument("--camera_visualization_window", type=str, default="ARX left teleop cameras")
    parser.add_argument("--align_collect_pose", type=dual.str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--align_after_save", type=dual.str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--align_slave_arm", type=dual.str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--align_master_arm", type=dual.str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--switch_master_mode_for_align", type=dual.str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--collect_pose_left", nargs=7, type=float, default=dual.DEFAULT_COLLECT_LEFT0)
    parser.add_argument("--home_pose_left", nargs=7, type=float, default=dual.DEFAULT_ARX_GO_HOME_LEFT)
    parser.add_argument("--align_duration", type=float, default=3.0)
    parser.add_argument("--align_hz", type=float, default=50.0)
    parser.add_argument("--slave_align_max_step", type=float, default=0.05)
    parser.add_argument("--master_align_max_step", type=float, default=0.05)
    parser.add_argument("--master_ready_timeout", type=float, default=5.0)
    parser.add_argument("--align_method", choices=("master_ctrl", "master_cmd", "arx_joy_home", "none"), default="arx_joy_home")
    parser.add_argument("--master_align_command_mode", choices=("master_cmd", "master_ctrl"), default="master_ctrl")
    parser.add_argument("--arx_joy_home_repeats", type=int, default=60)
    parser.add_argument("--arx_joy_release_repeats", type=int, default=20)
    parser.add_argument("--arx_joy_home_hz", type=float, default=20.0)
    parser.add_argument("--arx_joy_home_settle_sec", type=float, default=5.0)
    parser.add_argument("--slave_follow_timeout", type=float, default=8.0)
    parser.add_argument("--slave_follow_tolerance", type=float, default=0.08)
    return parser.parse_args()


def main():
    args = parse_args()

    args.camera_names = []
    if args.enable_camera_high:
        args.camera_names.append("cam_high")
    if args.enable_camera_left_wrist:
        args.camera_names.append("cam_left_wrist")
    if args.enable_camera_third:
        args.camera_names.append("camera_third")
    if not args.camera_names:
        raise SystemExit("At least one camera must be enabled.")

    if args.save_high_depth is None:
        args.save_high_depth = bool(args.save_depth)
    if args.save_third_depth is None:
        args.save_third_depth = bool(args.save_depth)
    if not args.enable_camera_high and args.save_high_depth:
        print("[WARN] --save_high_depth true ignored because --enable_camera_high false.")
        args.save_high_depth = False
    if not args.enable_camera_third and args.save_third_depth:
        print("[WARN] --save_third_depth true ignored because --enable_camera_third false.")
        args.save_third_depth = False
    args.save_depth = bool(args.save_high_depth or args.save_third_depth)

    if args.save_high_depth and args.camera_front_backend != "orbbec":
        raise SystemExit("--save_high_depth true requires --camera_front_backend orbbec for the current high camera setup.")
    if args.depth_camera_name is None:
        depth_camera_names = []
        if args.save_high_depth:
            depth_camera_names.append("cam_high")
        if args.save_third_depth:
            depth_camera_names.append("camera_third")
        args.depth_camera_name = ",".join(depth_camera_names)
    print(
        f"[INFO] Enabled cameras: {args.camera_names}\n"
        f"[INFO] Depth capture: high={bool(args.save_high_depth)}, "
        f"third={bool(args.save_third_depth)}, save_depth={bool(args.save_depth)}, "
        f"depth_camera_name={args.depth_camera_name or '(none)'}"
    )

    def _on_sigint(sig, frame):
        print("\n[INFO] Ctrl+C received. Exiting only the left collection script; ARX nodes keep running.")
        dual.shutdown_event.set()

    signal.signal(signal.SIGINT, _on_sigint)

    dual.rclpy.init()
    ros_operator = ARXLeftTeleopObserver(args)
    collector = dual.TeleopCollector(
        camera_names=args.camera_names,
        dataset_dir=args.dataset_dir,
        dataset_name=args.dataset_name,
        video_fps=args.video_fps,
        save_depth=args.save_depth,
        depth_camera_name=args.depth_camera_name,
    )

    spin_thread = threading.Thread(target=dual.rclpy.spin, args=(ros_operator,), daemon=True)
    spin_thread.start()
    print("[INFO] ROS spin thread started")

    try:
        if not ros_operator.wait_for_data_ready(timeout=20.0):
            return

        collect_pose = np.asarray(args.home_pose_left if args.align_method == "arx_joy_home" else args.collect_pose_left, dtype=float)
        if args.align_master_arm:
            ros_operator.wait_for_master_ready(timeout=args.master_ready_timeout)
        if args.align_collect_pose:
            collector.pause()
            should_switch = args.switch_master_mode_for_align and args.align_master_arm and args.align_method not in ("arx_joy_home", "none")
            if should_switch:
                ros_operator.set_master_mode(args.master_left_node, "remote_master_ctrl")
            ros_operator.align_to_collect_pose(
                target_pos=collect_pose,
                duration=args.align_duration,
                hz=args.align_hz,
                slave_max_step=args.slave_align_max_step,
                master_max_step=args.master_align_max_step,
                align_slave=args.align_slave_arm,
                align_master=args.align_master_arm,
                slave_follow_timeout=args.slave_follow_timeout,
                slave_follow_tolerance=args.slave_follow_tolerance,
                align_method=args.align_method,
            )
            if should_switch:
                ros_operator.set_master_mode(args.master_left_node, "remote_master")
            print("[INFO] Initial left collection pose ready.")

        if args.start_paused:
            print("\n[INFO] Ready. Press Space to start collecting.")
            collector.start_writer()
        else:
            collector.start()

        threading.Thread(target=dual.keyboard_monitor_thread, args=(collector,), daemon=True).start()

        rate = ros_operator.create_rate(args.control_frequency)
        while dual.rclpy.ok() and not dual.shutdown_event.is_set():
            with dual.request_lock:
                do_save = dual.save_requested
                do_discard = dual.discard_requested
                dual.save_requested = False
                dual.discard_requested = False

            if do_save:
                saved = collector.save_current_episode(export_video=args.save_video, video_fps=args.video_fps, resume_after_save=False)
                if saved and args.align_after_save:
                    collector.pause()
                    ros_operator.align_to_collect_pose(
                        target_pos=collect_pose,
                        duration=args.align_duration,
                        hz=args.align_hz,
                        slave_max_step=args.slave_align_max_step,
                        master_max_step=args.master_align_max_step,
                        align_slave=args.align_slave_arm,
                        align_master=args.align_master_arm,
                        slave_follow_timeout=args.slave_follow_timeout,
                        slave_follow_tolerance=args.slave_follow_tolerance,
                        align_method=args.align_method,
                    )
                    print("[INFO] Left arm realigned after save. Press Space for the next episode.")

            if do_discard:
                collector.discard_current_episode()

            frame = ros_operator.get_frame()
            if frame is not None:
                imgs, j_left = frame
                obs = build_observation(imgs, j_left)
                obs["depths"] = ros_operator.get_depth_payload()
                action = select_action(obs, ros_operator, args.action_source)
                collector.add_frame(obs, action)
                if args.visualize_cameras and not show_left_camera_visualization(imgs, args, collector.is_collecting, collector.frame_count):
                    dual.shutdown_event.set()
                    break
            rate.sleep()
    finally:
        print("\n[INFO] Shutting down left-arm collector...")
        collector.shutdown()
        ros_operator.cleanup_cameras()
        if args.visualize_cameras:
            dual.cv2.destroyAllWindows()
        ros_operator.destroy_node()
        if dual.rclpy.ok():
            dual.rclpy.shutdown()
        spin_thread.join(timeout=1.0)
        print("[INFO] Exit complete. ARX master/slave processes were not killed.")


if __name__ == "__main__":
    main()
