#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
ARX-X5 left-arm-only DP inference with temporal smoothing.

This is a deployment companion for model_dp_left. It keeps the same websocket
policy boundary and stream smoothing design as the dual-arm DP script, but the
runtime observation/action surface is 7-D left arm with two or three cameras.
"""

import argparse
import os
import signal
import sys
import threading
import time
from collections import deque
from typing import Dict

import cv2
import numpy as np

import arx_openpi_inference_temporal_smooth_dp as dual


CAMERA_NAMES_2CAM = ["cam_high", "cam_left_wrist"]
CAMERA_NAMES_3CAM = ["cam_high", "cam_left_wrist", "camera_third"]
POLICY_IMAGE_NAMES = {
    "cam_high": "top_head",
    "cam_left_wrist": "hand_left",
    "camera_third": "camera_third",
}
stream_buffer = None
observation_window = deque(maxlen=2)
published_actions_history = []
observed_qpos_history = []
inferred_chunks = []
inferred_chunks_lock = threading.Lock()
shutdown_event = threading.Event()


def _camera_display_frame(cam_name: str, image: np.ndarray, args) -> np.ndarray:
    return dual._camera_display_frame(cam_name, image, args)


def _resize_keep_aspect(image: np.ndarray, target_width: int) -> np.ndarray:
    return dual._resize_keep_aspect(image, target_width)


def show_camera_visualization(images: Dict[str, np.ndarray], args) -> bool:
    width = int(getattr(args, "camera_visualization_width", 320))
    order = [("cam_high", "High / Front"), ("cam_left_wrist", "Left Wrist")]
    if getattr(args, "include_third_camera", False):
        order.append(("camera_third", "Third Camera"))
    panels = []
    max_h = 0
    for cam_name, label in order:
        frame = _camera_display_frame(cam_name, images.get(cam_name), args)
        frame = _resize_keep_aspect(frame, width)
        cv2.putText(frame, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
        panels.append(frame)
        max_h = max(max_h, frame.shape[0])

    padded = []
    for panel in panels:
        if panel.shape[0] < max_h:
            pad = np.zeros((max_h - panel.shape[0], panel.shape[1], 3), dtype=panel.dtype)
            panel = np.vstack((panel, pad))
        padded.append(panel)
    cv2.imshow(getattr(args, "camera_visualization_window", "ARX left DP camera visualization"), np.hstack(padded))
    key = cv2.waitKey(1) & 0xFF
    return key not in (ord("q"), 27)


def build_policy_payload(latest_obs: dict, config: dict, prompt: str) -> dict:
    images = []
    camera_input_colors = config.get("camera_input_colors", {})
    for camera_name in config["camera_names"]:
        image = latest_obs["images"][camera_name]
        if image is None:
            raise ValueError(f"Missing image for {camera_name}")
        input_color = str(camera_input_colors.get(camera_name, "rgb")).lower()
        if input_color == "bgr":
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif input_color != "rgb":
            raise ValueError(f"Unsupported input color {input_color!r} for {camera_name}")
        images.append(image)
    images = dual.image_tools.resize_with_pad(np.array(images), 224, 224)
    return {
        "state": latest_obs["qpos"],
        "images": {
            POLICY_IMAGE_NAMES[camera_name]: images[idx].transpose(2, 0, 1)
            for idx, camera_name in enumerate(config["camera_names"])
        },
        "prompt": prompt,
    }


def update_observation_window(args, config, ros_operator):
    global observation_window
    if len(observation_window) == 0:
        observation_window.append({
            "qpos": None,
            "images": {camera_name: None for camera_name in config["camera_names"]},
        })
    frame = ros_operator.get_frame()
    if frame is None:
        return

    imgs, j_left = frame
    qpos = ros_operator.get_joint_positions(j_left)

    def jpeg_mapping(img):
        img = cv2.imencode(".jpg", img)[1].tobytes()
        return cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)

    observation_window.append({
        "qpos": qpos,
        "images": {
            camera_name: jpeg_mapping(imgs[camera_name])
            for camera_name in config["camera_names"]
        },
    })
    observed_qpos_history.append(np.asarray(qpos, dtype=float).copy())


def inference_fn_non_blocking_fast(args, config, policy, ros_operator):
    global stream_buffer
    rate = ros_operator.create_rate(args.inference_rate)
    consecutive_failures = 0
    while dual.rclpy.ok() and not shutdown_event.is_set():
        try:
            t0 = time.time()
            update_observation_window(args, config, ros_operator)
            print("Get Observation Time", time.time() - t0, "s")
            if len(observation_window) == 0:
                continue

            latest_obs = observation_window[-1]
            payload = build_policy_payload(latest_obs, config, args.prompt)

            t1 = time.time()
            actions = policy.infer(payload)["actions"]
            print("Inference Time", time.time() - t1, "s")
            if actions is not None and len(actions) > 0:
                actions_np = np.asarray(actions, dtype=float)
                if getattr(args, "debug_actions", False):
                    print(
                        "[debug] got left action chunk:",
                        f"shape={actions_np.shape}",
                        f"min={np.min(actions_np):.4f}",
                        f"max={np.max(actions_np):.4f}",
                        f"first={np.round(actions_np[0, :7], 4)}",
                    )
                stream_buffer.integrate_new_chunk(
                    actions_np,
                    max_k=int(getattr(args, "latency_k", 0)),
                    min_m=int(getattr(args, "min_smooth_steps", 8)),
                )
                with inferred_chunks_lock:
                    inferred_chunks.append({
                        "start_step": int(max(len(published_actions_history), len(observed_qpos_history))),
                        "chunk": actions_np.copy(),
                    })
                consecutive_failures = 0
            rate.sleep()
        except Exception as e:
            print(f"[left inference] {e}")
            consecutive_failures += 1
            if consecutive_failures >= 5:
                print("[left inference] Consecutive failures, clearing buffer")
                stream_buffer.clear()
                consecutive_failures = 0
            try:
                rate.sleep()
            except Exception:
                time.sleep(0.001)


def start_inference_thread(args, config, policy, ros_operator):
    thread = threading.Thread(target=inference_fn_non_blocking_fast, args=(args, config, policy, ros_operator))
    thread.daemon = True
    thread.start()
    return thread


class ARX5LeftROSController(dual.Node):
    def __init__(self, args):
        super().__init__("arx5_left_dp_controller")
        self.args = args
        self.bridge = dual.CvBridge()
        self.camera_names = CAMERA_NAMES_3CAM if args.include_third_camera else CAMERA_NAMES_2CAM
        self.front_cap = None
        self.third_cap = None
        self.front_orbbec_reader = None
        self.third_orbbec_reader = None
        self.pipelines = {}
        self.last_qpos = None
        self.qpos_lock = threading.Lock()
        self.joint_left_deque = deque(maxlen=2000)
        self.pub_left = self.create_publisher(dual.RobotStatus, args.joint_cmd_topic_left, 10)

        self.get_logger().info(f"Subscribed to left arm: {args.joint_state_topic_left}")
        self.create_subscription(dual.RobotStatus, args.joint_state_topic_left, self.joint_left_callback, 10)

        self.data_ready = {"joint_left": False, "cameras": False}
        self.data_ready_lock = threading.Lock()
        self.init_cameras()

    def _parse_opencv_source(self, source):
        src = str(source).strip()
        if src.lstrip("-").isdigit():
            return int(src)
        return src

    def joint_left_callback(self, msg):
        self.joint_left_deque.append(msg)
        with self.data_ready_lock:
            self.data_ready["joint_left"] = True

    def get_joint_positions(self, j_left) -> np.ndarray:
        q = np.asarray(j_left.joint_pos, dtype=float)
        if q.shape[0] != 7:
            self.get_logger().warn(f"Expected 7-D left qpos, got {q.shape}")
        with self.qpos_lock:
            self.last_qpos = q.copy()
        return q

    def get_frame(self):
        if len(self.joint_left_deque) == 0:
            return None
        imgs = self.get_camera_images()
        if any(name not in imgs for name in self.camera_names):
            return None
        return {name: imgs[name] for name in self.camera_names}, self.joint_left_deque[-1]

    def _init_opencv_camera(self, cam_name: str, source, width: int, height: int, fps: int, fourcc: str):
        src = self._parse_opencv_source(source)
        fourcc = str(fourcc).strip().upper()
        print(f"Initializing OpenCV {cam_name}: source={src}, {width}x{height}@{fps}")
        cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
        if not cap.isOpened():
            cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            print(f"OpenCV {cam_name} init failed.")
            return None
        if len(fourcc) == 4:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, fps)
        for _ in range(10):
            cap.read()
        print(f"  {cam_name} started (OpenCV)")
        return cap

    def _init_orbbec_camera(self, cam_name: str, width: int, height: int, fps: int, timeout: float):
        align_depth_to_color = bool(getattr(self.args, "align_depth_to_color", True))
        print(f"Initializing Orbbec RGB-D {cam_name}: {width}x{height}@{fps}, align_depth_to_color={align_depth_to_color}")
        try:
            reader = dual.AsyncOrbbecCamera(width, height, fps, align_depth_to_color=align_depth_to_color)
            if reader.open() and reader.wait_for_first_frame(timeout):
                print(f"  {cam_name} started (Orbbec RGB-D)")
                return reader
            reader.release()
            print(f"Orbbec {cam_name} opened but produced no RGB-D frames within {timeout:.1f}s.")
        except Exception as e:
            print(f"Orbbec {cam_name} init failed: {e}")
        return None

    def init_cameras(self):
        front_backend = str(getattr(self.args, "camera_front_backend", "realsense")).lower()
        third_backend = str(getattr(self.args, "camera_third_backend", "realsense")).lower()
        for backend_name, backend in (("camera_front_backend", front_backend), ("camera_third_backend", third_backend)):
            if backend not in ("realsense", "opencv", "orbbec"):
                print(f"Unknown {backend_name}={backend}, fallback to realsense.")
        if front_backend not in ("realsense", "opencv", "orbbec"):
            front_backend = "realsense"
        if third_backend not in ("realsense", "opencv", "orbbec"):
            third_backend = "realsense"

        front_ok = False
        left_wrist_ok = False
        third_ok = not self.args.include_third_camera
        try:
            import pyrealsense2 as rs
            camera_serials = {"cam_left_wrist": self.args.camera_left_serial}
            if front_backend == "realsense":
                camera_serials["cam_high"] = self.args.camera_front_serial
            if self.args.include_third_camera and third_backend == "realsense":
                camera_serials["camera_third"] = self.args.camera_third_serial
            print("Initializing left/front RealSense cameras...")
            for cam_name, serial in camera_serials.items():
                pipeline = rs.pipeline()
                rs_config = rs.config()
                rs_config.enable_device(serial)
                rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
                pipeline.start(rs_config)
                self.pipelines[cam_name] = pipeline
                print(f"  {cam_name} started (serial={serial})")
            for _ in range(30):
                for pipeline in self.pipelines.values():
                    pipeline.wait_for_frames(timeout_ms=5000)
            left_wrist_ok = "cam_left_wrist" in self.pipelines
            if front_backend == "realsense":
                front_ok = "cam_high" in self.pipelines
            if self.args.include_third_camera and third_backend == "realsense":
                third_ok = "camera_third" in self.pipelines
        except Exception as e:
            print(f"RealSense init failed: {e}")
            self.pipelines = {}

        if front_backend == "opencv":
            self.front_cap = self._init_opencv_camera(
                "cam_high",
                getattr(self.args, "camera_front_device", "/dev/video0"),
                int(getattr(self.args, "camera_front_width", 640)),
                int(getattr(self.args, "camera_front_height", 480)),
                int(getattr(self.args, "camera_front_fps", 30)),
                getattr(self.args, "camera_front_fourcc", "MJPG"),
            )
            front_ok = self.front_cap is not None

        if front_backend == "orbbec":
            self.front_orbbec_reader = self._init_orbbec_camera(
                "cam_high",
                int(getattr(self.args, "camera_front_width", 640)),
                int(getattr(self.args, "camera_front_height", 480)),
                int(getattr(self.args, "camera_front_fps", 30)),
                float(getattr(self.args, "camera_front_startup_timeout", 5.0)),
            )
            front_ok = self.front_orbbec_reader is not None

        if self.args.include_third_camera and third_backend == "opencv":
            self.third_cap = self._init_opencv_camera(
                "camera_third",
                getattr(self.args, "camera_third_device", "/dev/video2"),
                int(getattr(self.args, "camera_third_width", 640)),
                int(getattr(self.args, "camera_third_height", 480)),
                int(getattr(self.args, "camera_third_fps", 30)),
                getattr(self.args, "camera_third_fourcc", "MJPG"),
            )
            third_ok = self.third_cap is not None

        if self.args.include_third_camera and third_backend == "orbbec":
            self.third_orbbec_reader = self._init_orbbec_camera(
                "camera_third",
                int(getattr(self.args, "camera_third_width", 640)),
                int(getattr(self.args, "camera_third_height", 480)),
                int(getattr(self.args, "camera_third_fps", 30)),
                float(getattr(self.args, "camera_third_startup_timeout", 5.0)),
            )
            third_ok = self.third_orbbec_reader is not None

        cameras_ok = left_wrist_ok and front_ok and third_ok
        with self.data_ready_lock:
            self.data_ready["cameras"] = cameras_ok
        if cameras_ok:
            print("Left-arm cameras ready.")
        else:
            print(f"Camera readiness check failed: left_wrist_ok={left_wrist_ok}, front_ok={front_ok}, third_ok={third_ok}")

    def get_camera_images(self):
        images = {}
        for cam_name, pipeline in self.pipelines.items():
            try:
                frames = pipeline.wait_for_frames(timeout_ms=1000)
                color_frame = frames.get_color_frame()
                if color_frame:
                    images[cam_name] = np.asanyarray(color_frame.get_data())
            except Exception as e:
                print(f"Failed to get image from {cam_name}: {e}")
        if self.front_cap is not None:
            ok, frame = self.front_cap.read()
            if ok and frame is not None:
                images["cam_high"] = frame
        if self.front_orbbec_reader is not None:
            frame = self.front_orbbec_reader.get_frame()
            if frame is not None:
                images["cam_high"] = frame
        if self.third_cap is not None:
            ok, frame = self.third_cap.read()
            if ok and frame is not None:
                images["camera_third"] = frame
        if self.third_orbbec_reader is not None:
            frame = self.third_orbbec_reader.get_frame()
            if frame is not None:
                images["camera_third"] = frame
        return images

    def wait_for_data_ready(self, timeout: float = 15.0) -> bool:
        print("Waiting for left-arm sensor data...")
        start_time = time.time()
        while time.time() - start_time < timeout and dual.rclpy.ok():
            with self.data_ready_lock:
                joints_ready = self.data_ready["joint_left"]
                cameras_ready = self.data_ready["cameras"]
            if joints_ready and cameras_ready:
                print("Left-arm sensor data ready.")
                return True
            time.sleep(0.5)
        print("Timeout waiting for left-arm sensor data.")
        return False

    def set_joint_positions(self, pos: np.ndarray):
        if not dual.rclpy.ok():
            return
        pos = np.asarray(pos, dtype=float)
        if pos.shape != (7,):
            self.get_logger().warn(f"Expected 7-D left command, got {pos.shape}")
            return
        msg_left = dual.RobotStatus()
        msg_left.joint_pos = [float(x) for x in pos]
        self.pub_left.publish(msg_left)

    def smooth_return_to_zero(self, duration: float = 3.0):
        print("Returning left arm to zero...")
        frame = self.get_frame()
        if frame is None:
            print("Cannot get current left joint position")
            return False
        _, j_left = frame
        current_pos = self.get_joint_positions(j_left)
        target_pos = np.zeros(7)
        target_pos[6] = 3.0
        hz = 50.0
        num_steps = int(duration * hz)
        for step in range(num_steps + 1):
            alpha = step / num_steps
            smooth_alpha = (1 - np.cos(alpha * np.pi)) / 2
            self.set_joint_positions(current_pos * (1 - smooth_alpha) + target_pos * smooth_alpha)
            print(f"\rReturn to zero: {int(alpha * 100)}%", end="", flush=True)
            time.sleep(1.0 / hz)
        print("\nReturn to zero done.")
        open_pos = np.zeros(7)
        open_pos[6] = 5.0
        self.set_joint_positions(open_pos)
        return True

    def exit_return_to_zero(self, duration: float = 3.0):
        with self.qpos_lock:
            if self.last_qpos is None:
                print("No cached left qpos; skip return to zero.")
                return False
            current_pos = self.last_qpos.copy()
        target_pos = np.zeros(7)
        hz = 50.0
        num_steps = int(duration * hz)
        for step in range(num_steps + 1):
            alpha = step / num_steps
            smooth_alpha = (1 - np.cos(alpha * np.pi)) / 2
            self.set_joint_positions(current_pos * (1 - smooth_alpha) + target_pos * smooth_alpha)
            print(f"\rReturn to zero: {int(alpha * 100)}%", end="", flush=True)
            time.sleep(1.0 / hz)
        print("\nReturn to zero done.")
        open_pos = np.zeros(7)
        open_pos[6] = 5.0
        self.set_joint_positions(open_pos)
        return True

    def smooth_goto_position(self, target_pos: np.ndarray, duration: float = 3.0, hz: float = 50.0) -> bool:
        frame = self.get_frame()
        if frame is None:
            print("Cannot get current left joints; skip smooth move.")
            return False
        _, j_left = frame
        current_pos = self.get_joint_positions(j_left)
        target_pos = np.asarray(target_pos, dtype=float)
        print(f"Max joint delta: {np.max(np.abs(target_pos - current_pos)):.4f} rad")
        num_steps = int(duration * hz)
        for step in range(num_steps + 1):
            alpha = step / num_steps
            smooth_alpha = (1 - np.cos(alpha * np.pi)) / 2
            self.set_joint_positions(current_pos * (1 - smooth_alpha) + target_pos * smooth_alpha)
            if step % 50 == 0 or step == num_steps:
                print(f"\rSmooth move: {int(alpha * 100)}%", end="", flush=True)
            time.sleep(1.0 / hz)
        print("\nSmooth move done.")
        return True

    def cleanup_cameras(self):
        print("Stopping cameras...")
        for pipeline in getattr(self, "pipelines", {}).values():
            try:
                pipeline.stop()
            except Exception:
                pass
        if self.front_cap is not None:
            self.front_cap.release()
            self.front_cap = None
        if self.third_cap is not None:
            self.third_cap.release()
            self.third_cap = None
        if self.front_orbbec_reader is not None:
            self.front_orbbec_reader.release()
            self.front_orbbec_reader = None
        if self.third_orbbec_reader is not None:
            self.third_orbbec_reader.release()
            self.third_orbbec_reader = None


def get_config(args):
    return {
        "episode_len": args.max_publish_step,
        "state_dim": 7,
        "chunk_size": args.chunk_size,
        "camera_names": CAMERA_NAMES_3CAM if args.include_third_camera else CAMERA_NAMES_2CAM,
        "camera_input_colors": {
            "cam_high": args.camera_front_color,
            "cam_left_wrist": args.camera_left_color,
            "camera_third": args.camera_third_color,
        },
    }


def apply_gripper_binary(act: np.ndarray, close_val: float = 0.0, open_val: float = 5.0, close_thresh: float = 1.50, open_thresh: float = 2.50) -> np.ndarray:
    act2 = act.copy()
    previous = getattr(apply_gripper_binary, "_previous", None)
    if previous is None:
        previous = open_val if act[6] >= (close_thresh + open_thresh) / 2.0 else close_val
    if act[6] >= open_thresh:
        previous = open_val
    elif act[6] <= close_thresh:
        previous = close_val
    act2[6] = previous
    apply_gripper_binary._previous = previous
    return act2


def apply_joint_safety_limit(act: np.ndarray, qpos: np.ndarray, args) -> np.ndarray:
    """Limit absolute joint command jumps before publishing to the real arm."""
    if getattr(args, "disable_action_safety", False):
        return act
    max_delta = float(getattr(args, "max_joint_delta_per_step", 0.20))
    if max_delta <= 0:
        return act

    act2 = np.asarray(act, dtype=float).copy()
    qpos = np.asarray(qpos, dtype=float)
    joint_delta = act2[:6] - qpos[:6]
    clipped_delta = np.clip(joint_delta, -max_delta, max_delta)
    if np.any(np.abs(joint_delta - clipped_delta) > 1e-9):
        print(
            "[safety] clipped joint command:",
            f"max_abs_delta={np.max(np.abs(joint_delta)):.4f}",
            f"limit={max_delta:.4f}",
            f"raw={np.round(act2[:6], 4)}",
            f"qpos={np.round(qpos[:6], 4)}",
        )
    act2[:6] = qpos[:6] + clipped_delta
    return act2


def model_inference(args, config, ros_operator):
    global stream_buffer
    policy = dual.websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    print(f"Server metadata: {policy.get_server_metadata()}")

    if args.auto_homing:
        ros_operator.smooth_goto_position(np.array(args.home_pose_left, dtype=float), duration=3.0, hz=50.0)
        print("Initial left pose move done.")

    try:
        print("Warming up inference...")
        update_observation_window(args, config, ros_operator)
        if len(observation_window) > 0:
            latest_obs = observation_window[-1]
            payload = build_policy_payload(latest_obs, config, args.prompt)
            _ = policy.infer(payload)["actions"]
            print("Warmup done.")
    except Exception as e:
        print(f"Warmup failed: {e}")
        import traceback
        traceback.print_exc()

    stream_buffer = dual.StreamActionBuffer(
        max_chunks=args.buffer_max_chunks,
        decay_alpha=args.exp_decay_alpha,
        state_dim=config["state_dim"],
        smooth_method="temporal",
    )
    inference_thread = start_inference_thread(args, config, policy, ros_operator)
    rate = ros_operator.create_rate(args.control_frequency)
    step = 0
    consecutive_empty_actions = 0
    print("Starting left-arm control loop...")
    try:
        while dual.rclpy.ok() and step < config["episode_len"] and not shutdown_event.is_set():
            frame = ros_operator.get_frame()
            if frame is None:
                rate.sleep()
                continue
            imgs, j_left = frame
            if getattr(args, "visualize_cameras", False) and not show_camera_visualization(imgs, args):
                shutdown_event.set()
                break

            qpos = ros_operator.get_joint_positions(j_left)
            observed_qpos_history.append(qpos.copy())
            act = stream_buffer.pop_next_action()
            if act is not None:
                consecutive_empty_actions = 0
                act = apply_gripper_binary(act, close_thresh=args.gripper_close_thresh, open_thresh=args.gripper_open_thresh)
                act = apply_joint_safety_limit(act, qpos, args)
                if getattr(args, "debug_actions", False) and step % args.debug_action_interval == 0:
                    delta = np.asarray(act, dtype=float) - np.asarray(qpos, dtype=float)
                    print(
                        "[debug] publish left action:",
                        f"step={step}",
                        f"max_abs_delta={np.max(np.abs(delta)):.4f}",
                        f"act={np.round(act, 4)}",
                        f"qpos={np.round(qpos, 4)}",
                    )
                if getattr(args, "dry_run_no_publish", False):
                    print("[dry-run] skip publishing action to robot")
                else:
                    ros_operator.set_joint_positions(act)
                published_actions_history.append(act.copy())
                step += 1
                if step % 50 == 0:
                    print(f"[main] step {step}, buffer size: {len(stream_buffer.cur_chunk)}")
            else:
                consecutive_empty_actions += 1
                if consecutive_empty_actions >= 100:
                    print("[main] No actions for 100 steps; safe return to zero")
                    ros_operator.smooth_return_to_zero(duration=3.0)
                    consecutive_empty_actions = 0
            rate.sleep()
    finally:
        shutdown_event.set()
        if inference_thread.is_alive():
            inference_thread.join(timeout=2.0)
        ros_operator.cleanup_cameras()
        if getattr(args, "visualize_cameras", False):
            try:
                cv2.destroyWindow(getattr(args, "camera_visualization_window", "ARX left DP camera visualization"))
            except Exception:
                cv2.destroyAllWindows()
        print("ARX5 left inference controller shut down.")


def main():
    def _on_sigint(sig, frame):
        shutdown_event.set()
        sys.exit(0)

    parser = argparse.ArgumentParser(description="ARX-X5 left-arm DP inference (temporal smoothing).")
    parser.add_argument("--joint_cmd_topic_left", default="/arm_master_l_status")
    parser.add_argument("--joint_state_topic_left", default="/arm_slave_l_status")
    parser.add_argument("--camera_front_serial", type=str, default="213722070209")
    parser.add_argument("--camera_left_serial", type=str, default="213722070377")
    parser.add_argument("--camera_third_serial", type=str, default="")
    parser.add_argument("--include_third_camera", type=dual.str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--camera_front_backend", choices=("realsense", "opencv", "orbbec"), default="realsense")
    parser.add_argument("--camera_front_device", type=str, default="/dev/video0")
    parser.add_argument("--camera_front_width", type=int, default=640)
    parser.add_argument("--camera_front_height", type=int, default=480)
    parser.add_argument("--camera_front_fps", type=int, default=30)
    parser.add_argument("--camera_front_fourcc", type=str, default="MJPG")
    parser.add_argument("--camera_front_startup_timeout", type=float, default=5.0)
    parser.add_argument("--camera_third_backend", choices=("realsense", "opencv", "orbbec"), default="realsense")
    parser.add_argument("--camera_third_device", type=str, default="/dev/video2")
    parser.add_argument("--camera_third_width", type=int, default=640)
    parser.add_argument("--camera_third_height", type=int, default=480)
    parser.add_argument("--camera_third_fps", type=int, default=30)
    parser.add_argument("--camera_third_fourcc", type=str, default="MJPG")
    parser.add_argument("--camera_third_startup_timeout", type=float, default=5.0)
    parser.add_argument("--camera_front_color", choices=("rgb", "bgr"), default="rgb")
    parser.add_argument("--camera_left_color", choices=("rgb", "bgr"), default="bgr")
    parser.add_argument("--camera_third_color", choices=("rgb", "bgr"), default="bgr")
    parser.add_argument("--align_depth_to_color", type=dual.str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--visualize_cameras", type=dual.str2bool, nargs="?", const=True, default=False)
    parser.add_argument("--camera_visualization_width", type=int, default=320)
    parser.add_argument("--camera_visualization_window", type=str, default="ARX left DP camera visualization")
    parser.add_argument("--host", default="192.168.10.31")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--prompt", default="grasp the cup and place it on the black plate")
    parser.add_argument("--control_frequency", type=float, default=30.0)
    parser.add_argument("--inference_rate", type=float, default=4.0)
    parser.add_argument("--chunk_size", type=int, default=32)
    parser.add_argument("--max_publish_step", type=int, default=10000000)
    parser.add_argument("--latency_k", type=int, default=4)
    parser.add_argument("--min_smooth_steps", type=int, default=24)
    parser.add_argument("--buffer_max_chunks", type=int, default=10)
    parser.add_argument("--exp_decay_alpha", type=float, default=0.25)
    parser.add_argument("--debug_actions", action="store_true")
    parser.add_argument("--debug_action_interval", type=int, default=10)
    parser.add_argument("--dry_run_no_publish", action="store_true")
    parser.add_argument("--gripper_close_thresh", type=float, default=1.50)
    parser.add_argument("--gripper_open_thresh", type=float, default=2.50)
    parser.add_argument("--max_joint_delta_per_step", type=float, default=0.30)
    parser.add_argument("--disable_action_safety", action="store_true")
    parser.add_argument("--auto_homing", action="store_true", default=True)
    parser.add_argument("--exit_homing", action="store_true")
    parser.add_argument("--home_pose_left", nargs=7, type=float, default=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # 夹爪问题

    args = parser.parse_args()
    signal.signal(signal.SIGINT, _on_sigint)

    try:
        dual.rclpy.init()
        ros_operator = ARX5LeftROSController(args)
        spin_thread = threading.Thread(target=dual.rclpy.spin, args=(ros_operator,), daemon=True)
        spin_thread.start()
        print("ROS spin thread started.")
        if not ros_operator.wait_for_data_ready(timeout=15.0):
            print("Sensor data not ready; exiting.")
            return
        print("Press Enter to start left-arm inference...")
        input("Left arm ready. Press Enter to start...")
        config = get_config(args)
        model_inference(args, config, ros_operator)
    except Exception as e:
        print(f"Main error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            shutdown_event.set()
            if args.exit_homing:
                ros_operator.exit_return_to_zero(duration=3.0)
            if dual.rclpy.ok():
                dual.rclpy.shutdown()
            print("ROS2 shutdown.")
        except Exception as e:
            print(f"ROS2 shutdown error: {e}")
        print("Exiting.")
        os._exit(0)


if __name__ == "__main__":
    main()
