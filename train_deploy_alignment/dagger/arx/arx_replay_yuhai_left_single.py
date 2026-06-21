#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Replay YuHai-format left-arm ARX-X5 episodes through end-effector IK.

Input HDF5 format:
  action: (T, 5) with columns (dx, dy, dz, dyaw, d_gripper)

The script uses the current left-arm end-effector pose as the default replay
origin, accumulates the delta actions into absolute end-effector targets, and
publishes them to either:
  - RobotCmd /arm_cmd for open_single_arm.launch.py normal IK control
  - PosCmd   ARX_VR_L for open_vr_single_arm.launch.py vr_slave IK control
"""

import argparse
import json
import math
import signal
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

import h5py
import numpy as np

DRY_RUN_PRECHECK = "--dry_run" in sys.argv or "--help" in sys.argv or "-h" in sys.argv

if DRY_RUN_PRECHECK:
    rclpy = None

    class Node:
        pass

    RobotCmd = None
    RobotStatus = None
    PosCmd = None
else:
    try:
        import rclpy
        from rclpy.node import Node
        from arx5_arm_msg.msg import RobotCmd, RobotStatus
        from arm_control.msg import PosCmd
    except ImportError as e:
        raise SystemExit(
            "ROS2 Python packages or ARX messages were not found.\n"
            f"Original import error: {e}\n"
            "Source the ROS distro and X5_ws overlay before replay."
        ) from e


ACTION_ORDER = ("dx", "dy", "dz", "dyaw", "d_gripper")
DEFAULT_CAMERA_NAMES = ("cam_high", "cam_left_wrist")
TRUE_STRINGS = {"1", "true", "t", "yes", "y", "on", "ture"}
FALSE_STRINGS = {"0", "false", "f", "no", "n", "off", "none"}
ROBOT_CMD_END_CONTROL_MODE = 4
shutdown_event = threading.Event()


@dataclass
class ActionStats:
    frames: int
    max_xyz_step: float
    max_yaw_step: float
    max_gripper_step: float
    xyz_abs_sum: tuple
    yaw_abs_sum: float
    gripper_abs_sum: float


@dataclass
class ReplayVisualData:
    pixels: dict
    goal_pixels: dict
    camera_names: tuple
    start_step: int
    end_step: int


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


def resolve_step_range(n: int, start_step: int, end_step: int) -> tuple[int, int]:
    if start_step < 0:
        start_step = max(0, n + start_step)
    if end_step < 0:
        end_step = n
    else:
        end_step = min(n, end_step)
    start_step = min(max(0, start_step), n)
    if end_step <= start_step:
        raise ValueError(f"Invalid replay range: start_step={start_step}, end_step={end_step}, frames={n}")
    return start_step, end_step


def load_actions(path: str, start_step: int, end_step: int) -> np.ndarray:
    with h5py.File(path, "r") as root:
        if "action" not in root:
            available = []
            root.visititems(lambda name, obj: available.append(name) if isinstance(obj, h5py.Dataset) else None)
            raise KeyError(f"Dataset 'action' not found. Available datasets: {available}")
        action = np.asarray(root["action"][:], dtype=np.float64)
        if action.ndim != 2 or action.shape[1] != 5:
            raise ValueError(f"Expected action shape (T, 5), got {action.shape}")
        if action.shape[0] == 0:
            raise ValueError("Action trajectory is empty.")
        if not np.all(np.isfinite(action)):
            bad = np.argwhere(~np.isfinite(action))[0].tolist()
            raise ValueError(f"Action contains NaN/Inf, first bad index={bad}")

        attr_order = root.attrs.get("action_order")
        if attr_order:
            try:
                parsed = tuple(json.loads(attr_order))
                if parsed != ACTION_ORDER:
                    print(f"[WARN] action_order attr is {parsed}, expected {ACTION_ORDER}")
            except Exception:
                print(f"[WARN] Could not parse action_order attr: {attr_order!r}")

    n = action.shape[0]
    start_step, end_step = resolve_step_range(n, start_step, end_step)
    return action[start_step:end_step]


def load_replay_visual_data(path: str, start_step: int, end_step: int, requested_camera_names: list[str] | None) -> ReplayVisualData:
    with h5py.File(path, "r") as root:
        if "action" not in root:
            raise KeyError("Dataset 'action' not found; cannot align pixels with replay steps.")
        start_step, end_step = resolve_step_range(int(root["action"].shape[0]), start_step, end_step)

        if "pixels" not in root:
            raise KeyError("Group 'pixels' not found in HDF5.")
        if "goal_pixels" not in root:
            raise KeyError("Group 'goal_pixels' not found in HDF5.")

        if requested_camera_names:
            camera_names = tuple(requested_camera_names)
        else:
            attr_names = root.attrs.get("camera_names")
            if attr_names:
                try:
                    camera_names = tuple(json.loads(attr_names))
                except Exception:
                    camera_names = DEFAULT_CAMERA_NAMES
            else:
                camera_names = tuple(name for name in DEFAULT_CAMERA_NAMES if name in root["pixels"])

        if not camera_names:
            raise ValueError("No camera names available for visualization.")

        pixels = {}
        goal_pixels = {}
        for name in camera_names:
            pixel_key = f"pixels/{name}"
            goal_key = f"goal_pixels/{name}"
            if pixel_key not in root:
                raise KeyError(f"Dataset '{pixel_key}' not found.")
            if goal_key not in root:
                raise KeyError(f"Dataset '{goal_key}' not found.")
            ds = root[pixel_key]
            if ds.ndim != 4 or ds.shape[-1] != 3:
                raise ValueError(f"Expected '{pixel_key}' shape (T,H,W,3), got {ds.shape}")
            if ds.shape[0] < end_step:
                raise ValueError(f"'{pixel_key}' has only {ds.shape[0]} frames, but replay needs end_step={end_step}")
            pixels[name] = np.asarray(ds[start_step:end_step], dtype=np.uint8)
            goal_pixels[name] = np.asarray(root[goal_key][:], dtype=np.uint8)

    return ReplayVisualData(
        pixels=pixels,
        goal_pixels=goal_pixels,
        camera_names=camera_names,
        start_step=start_step,
        end_step=end_step,
    )


def compute_action_stats(action: np.ndarray) -> ActionStats:
    xyz_abs_sum = np.sum(np.abs(action[:, :3]), axis=0)
    return ActionStats(
        frames=int(action.shape[0]),
        max_xyz_step=float(np.max(np.abs(action[:, :3]))),
        max_yaw_step=float(np.max(np.abs(action[:, 3]))),
        max_gripper_step=float(np.max(np.abs(action[:, 4]))),
        xyz_abs_sum=tuple(float(x) for x in xyz_abs_sum),
        yaw_abs_sum=float(np.sum(np.abs(action[:, 3]))),
        gripper_abs_sum=float(np.sum(np.abs(action[:, 4]))),
    )


def validate_actions(action: np.ndarray, max_xyz_step: float, max_yaw_step: float, max_gripper_step: float, force: bool) -> bool:
    bad_xyz = np.argwhere(np.abs(action[:, :3]) > max_xyz_step)
    bad_yaw = np.argwhere(np.abs(action[:, 3]) > max_yaw_step)
    bad_grip = np.argwhere(np.abs(action[:, 4]) > max_gripper_step)

    ok = True
    if len(bad_xyz):
        frame, axis = bad_xyz[0]
        print(f"[WARN] xyz action too large: frame={frame}, axis={axis}, delta={action[frame, axis]:.5f}, limit={max_xyz_step:.5f}")
        ok = False
    if len(bad_yaw):
        frame = int(bad_yaw[0][0])
        print(f"[WARN] yaw action too large: frame={frame}, delta={action[frame, 3]:.5f}, limit={max_yaw_step:.5f}")
        ok = False
    if len(bad_grip):
        frame = int(bad_grip[0][0])
        print(f"[WARN] gripper action too large: frame={frame}, delta={action[frame, 4]:.5f}, limit={max_gripper_step:.5f}")
        ok = False

    if ok:
        return True
    if force:
        print("[WARN] Continuing because --force was set.")
        return True
    print("[ERROR] Refusing to replay. Inspect the data or rerun with --force if expected.")
    return False


def accumulate_targets(action: np.ndarray, start_pose: np.ndarray, start_gripper: float) -> tuple[np.ndarray, np.ndarray]:
    pose = np.asarray(start_pose, dtype=np.float64).copy()
    if pose.shape != (6,):
        raise ValueError(f"start_pose must be 6-D xyzrpy, got {pose.shape}")
    gripper = float(start_gripper)

    poses = []
    grippers = []
    for delta in action:
        pose[0] += delta[0]
        pose[1] += delta[1]
        pose[2] += delta[2]
        pose[5] = wrap_angle(pose[5] + delta[3])
        gripper += delta[4]
        poses.append(pose.copy())
        grippers.append(gripper)
    return np.asarray(poses, dtype=np.float64), np.asarray(grippers, dtype=np.float64)


class YuHaiLeftIKReplayController(Node):
    def __init__(self, args):
        super().__init__("arx_yuhai_left_ik_replay")
        self.args = args
        self.latest_status = None
        self.status_lock = threading.Lock()

        self.create_subscription(RobotStatus, args.status_topic, self.status_callback, 10)
        if args.command_msg_type == "robot_cmd":
            self.cmd_pub = self.create_publisher(RobotCmd, args.command_topic, 10)
        else:
            self.cmd_pub = self.create_publisher(PosCmd, args.command_topic, 10)

        self.get_logger().info(f"Subscribe status: {args.status_topic}")
        self.get_logger().info(f"Publish IK command: {args.command_topic} ({args.command_msg_type})")

    def status_callback(self, msg):
        with self.status_lock:
            self.latest_status = msg

    def get_current_pose_gripper(self) -> Optional[tuple[np.ndarray, float]]:
        with self.status_lock:
            msg = self.latest_status
        if msg is None:
            return None
        pose = np.asarray(msg.end_pos, dtype=np.float64)
        if pose.shape != (6,):
            self.get_logger().warn(f"Expected 6-D end_pos, got {pose.shape}")
            return None
        gripper = float(msg.joint_pos[6])
        return pose.copy(), gripper

    def wait_for_status(self, timeout: float) -> bool:
        print("[INFO] Waiting for current left-arm end-effector status...")
        start = time.time()
        while rclpy.ok() and not shutdown_event.is_set() and time.time() - start < timeout:
            if self.get_current_pose_gripper() is not None:
                print("[INFO] Left-arm end-effector status ready.")
                return True
            time.sleep(0.05)
        print("[ERROR] Timed out waiting for status.")
        return False

    def publish_target(self, pose: np.ndarray, gripper: float):
        pose = np.asarray(pose, dtype=np.float64)
        if pose.shape != (6,):
            self.get_logger().warn(f"Expected 6-D pose, got {pose.shape}")
            return

        command_gripper = float(gripper) * float(self.args.gripper_command_scale) + float(self.args.gripper_command_offset)
        if self.args.gripper_command_min is not None:
            command_gripper = max(command_gripper, float(self.args.gripper_command_min))
        if self.args.gripper_command_max is not None:
            command_gripper = min(command_gripper, float(self.args.gripper_command_max))

        if self.args.command_msg_type == "robot_cmd":
            msg = RobotCmd()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.end_pos = [float(x) for x in pose]
            msg.joint_pos = [0.0] * 6
            msg.gripper = command_gripper
            msg.mode = int(self.args.robot_cmd_mode)
        else:
            msg = PosCmd()
            msg.x = float(pose[0])
            msg.y = float(pose[1])
            msg.z = float(pose[2])
            msg.roll = float(pose[3])
            msg.pitch = float(pose[4])
            msg.yaw = float(pose[5])
            msg.gripper = command_gripper
        self.cmd_pub.publish(msg)


def countdown(seconds: int) -> bool:
    for remain in range(seconds, 0, -1):
        if shutdown_event.is_set():
            return False
        print(f"\rStarting YuHai IK replay in {remain}s. Press Ctrl-C to cancel.", end="", flush=True)
        time.sleep(1.0)
    print()
    return True


class YuHaiReplayVisualizer:
    def __init__(self, visual_data: ReplayVisualData, window_name: str):
        try:
            import cv2
        except ImportError as e:
            raise RuntimeError("OpenCV (cv2) is required for --visualize_pixels.") from e
        self.cv2 = cv2
        self.visual_data = visual_data
        self.window_name = window_name
        self.cv2.namedWindow(self.window_name, self.cv2.WINDOW_NORMAL)

    def _to_panel(self, rgb: np.ndarray, label: str) -> np.ndarray:
        cv2 = self.cv2
        if rgb.ndim == 2:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_GRAY2BGR)
        else:
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        bgr = bgr.copy()
        cv2.putText(bgr, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
        return bgr

    def show(self, idx: int, total: int) -> bool:
        cv2 = self.cv2
        panels = []
        for name in self.visual_data.camera_names:
            panels.append(self._to_panel(self.visual_data.pixels[name][idx], f"replay {name}"))
        for name in self.visual_data.camera_names:
            panels.append(self._to_panel(self.visual_data.goal_pixels[name], f"goal {name}"))

        canvas = np.hstack(panels)
        cv2.putText(
            canvas,
            f"frame {idx + 1}/{total} | q/Esc exits",
            (8, canvas.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(self.window_name, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            return False
        try:
            return cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            return False

    def close(self):
        try:
            self.cv2.destroyWindow(self.window_name)
        except Exception:
            self.cv2.destroyAllWindows()


def replay_targets(
    controller: Optional[YuHaiLeftIKReplayController],
    poses: np.ndarray,
    grippers: np.ndarray,
    hz: float,
    speed: float,
    visualizer: Optional[YuHaiReplayVisualizer] = None,
) -> bool:
    sleep_s = 1.0 / hz / speed
    total = len(poses)
    print(f"[INFO] Replaying {total} EE targets at source_hz={hz:.2f}, speed={speed:.2f}.")
    for idx, (pose, gripper) in enumerate(zip(poses, grippers)):
        if shutdown_event.is_set() or (controller is not None and not rclpy.ok()):
            print("\n[INFO] Replay interrupted.")
            return False
        if controller is not None:
            controller.publish_target(pose, gripper)
        if visualizer is not None and not visualizer.show(idx, total):
            print("\n[INFO] Pixel visualization closed; stopping replay.")
            shutdown_event.set()
            return False
        if idx % max(1, int(hz)) == 0 or idx == total - 1:
            print(
                f"\rReplay: {idx + 1}/{total} ({100.0 * (idx + 1) / total:5.1f}%) "
                f"xyz=({pose[0]:.3f},{pose[1]:.3f},{pose[2]:.3f}) yaw={pose[5]:.3f} grip={gripper:.3f}",
                end="",
                flush=True,
            )
        time.sleep(sleep_s)
    print("\n[INFO] Replay finished.")
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="Replay YuHai-format ARX left-arm delta actions through IK.")
    parser.add_argument("--hdf5_path", required=True, help="Path to YuHai episode_*.hdf5")
    parser.add_argument("--start_step", type=int, default=0, help="Inclusive start frame. Negative values count from end.")
    parser.add_argument("--end_step", type=int, default=-1, help="Exclusive end frame. -1 means end.")
    parser.add_argument("--control_frequency", type=float, default=30.0)
    parser.add_argument("--speed", type=float, default=0.3, help="Replay speed multiplier. 0.3 is conservative.")
    parser.add_argument("--status_topic", default="/arm_status", help="RobotStatus topic used to get current xyzrpy/gripper origin.")
    parser.add_argument("--command_topic", default="/arm_cmd", help="IK command topic.")
    parser.add_argument("--command_msg_type", choices=("robot_cmd", "pos_cmd"), default="robot_cmd")
    parser.add_argument("--robot_cmd_mode", type=int, default=ROBOT_CMD_END_CONTROL_MODE, help="RobotCmd mode. 4 matches existing keyboard END_CONTROL.")
    parser.add_argument("--start_pose", nargs=6, type=float, default=None, help="Optional xyz roll pitch yaw origin. Defaults to current status end_pos.")
    parser.add_argument("--start_gripper", type=float, default=None, help="Optional YuHai/raw-unit gripper origin. Defaults to current status joint_pos[6] divided by --status_gripper_scale.")
    parser.add_argument("--gripper_command_scale", type=float, default=None, help="Scale YuHai/raw gripper before publishing. Default: 5.0 for robot_cmd, 1.0 for pos_cmd.")
    parser.add_argument("--gripper_command_offset", type=float, default=0.0, help="Offset added after gripper scaling before publishing.")
    parser.add_argument("--status_gripper_scale", type=float, default=None, help="Scale used to convert current status joint_pos[6] back to YuHai/raw units. Default equals --gripper_command_scale.")
    parser.add_argument("--gripper_command_min", type=float, default=None, help="Optional lower clamp for published gripper command.")
    parser.add_argument("--gripper_command_max", type=float, default=None, help="Optional upper clamp for published gripper command.")
    parser.add_argument("--state_timeout", type=float, default=15.0)
    parser.add_argument("--countdown", type=int, default=5)
    parser.add_argument("--max_xyz_step", type=float, default=0.03)
    parser.add_argument("--max_yaw_step", type=float, default=0.25)
    parser.add_argument("--max_gripper_step", type=float, default=1.0)
    parser.add_argument("--dry_run", action="store_true", help="Inspect and integrate the action only; do not publish ROS commands.")
    parser.add_argument("--force", action="store_true", help="Replay even if action jump checks warn.")
    parser.add_argument("--visualize_pixels", type=str2bool, nargs="?", const=True, default=True, help="Show pixels/* and goal_pixels/* during replay.")
    parser.add_argument("--visualization_window", default="YuHai replay pixels")
    parser.add_argument("--camera_names", nargs="*", default=None, help="Camera datasets to show. Default uses HDF5 camera_names attr.")
    args = parser.parse_args()

    if args.control_frequency <= 0:
        parser.error("--control_frequency must be > 0")
    if args.speed <= 0:
        parser.error("--speed must be > 0")
    if args.gripper_command_scale is None:
        args.gripper_command_scale = 5.0 if args.command_msg_type == "robot_cmd" else 1.0
    if args.status_gripper_scale is None:
        args.status_gripper_scale = args.gripper_command_scale
    if args.gripper_command_scale == 0:
        parser.error("--gripper_command_scale must be non-zero")
    if args.status_gripper_scale == 0:
        parser.error("--status_gripper_scale must be non-zero")
    return args


def print_stats(path: str, action: np.ndarray, stats: ActionStats):
    print("[INFO] Loaded YuHai action trajectory")
    print(f"  path: {path}")
    print(f"  action columns: {ACTION_ORDER}")
    print(f"  frames: {stats.frames}")
    print(f"  max abs step: xyz={stats.max_xyz_step:.5f}, yaw={stats.max_yaw_step:.5f}, gripper={stats.max_gripper_step:.5f}")
    print(
        "  accumulated abs motion: "
        f"dx={stats.xyz_abs_sum[0]:.5f}, dy={stats.xyz_abs_sum[1]:.5f}, dz={stats.xyz_abs_sum[2]:.5f}, "
        f"dyaw={stats.yaw_abs_sum:.5f}, dgripper={stats.gripper_abs_sum:.5f}"
    )
    print(f"  net delta: {np.sum(action, axis=0)}")


def main():
    args = parse_args()

    def on_sigint(sig, frame):
        print("\n[INFO] Ctrl+C received; stopping YuHai IK replay.")
        shutdown_event.set()

    signal.signal(signal.SIGINT, on_sigint)

    action = load_actions(args.hdf5_path, args.start_step, args.end_step)
    stats = compute_action_stats(action)
    print_stats(args.hdf5_path, action, stats)
    if not validate_actions(action, args.max_xyz_step, args.max_yaw_step, args.max_gripper_step, args.force):
        return 2

    visual_data = None
    visualizer = None
    if args.visualize_pixels:
        try:
            visual_data = load_replay_visual_data(args.hdf5_path, args.start_step, args.end_step, args.camera_names)
            print(
                "[INFO] Loaded replay pixels: "
                f"cameras={visual_data.camera_names}, steps={visual_data.start_step}:{visual_data.end_step}"
            )
        except Exception as e:
            print(f"[WARN] Pixel visualization disabled: {e}")
            args.visualize_pixels = False

    if args.dry_run:
        start_pose = np.asarray(args.start_pose if args.start_pose is not None else [0, 0, 0, 0, 0, 0], dtype=np.float64)
        start_gripper = float(args.start_gripper if args.start_gripper is not None else 0.0)
        poses, grippers = accumulate_targets(action, start_pose, start_gripper)
        print("[INFO] Dry run integrated target range")
        print(
            f"  first target: pose={poses[0]}, raw_gripper={grippers[0]:.5f}, "
            f"cmd_gripper={grippers[0] * args.gripper_command_scale + args.gripper_command_offset:.5f}"
        )
        print(
            f"  final target: pose={poses[-1]}, raw_gripper={grippers[-1]:.5f}, "
            f"cmd_gripper={grippers[-1] * args.gripper_command_scale + args.gripper_command_offset:.5f}"
        )
        if args.visualize_pixels and visual_data is not None:
            try:
                visualizer = YuHaiReplayVisualizer(visual_data, args.visualization_window)
                replay_targets(None, poses, grippers, hz=args.control_frequency, speed=args.speed, visualizer=visualizer)
            finally:
                if visualizer is not None:
                    visualizer.close()
        print("[INFO] Dry run complete. No ROS commands were published.")
        return 0

    rclpy.init()
    controller = None
    spin_thread = None
    try:
        controller = YuHaiLeftIKReplayController(args)
        spin_thread = threading.Thread(target=rclpy.spin, args=(controller,), daemon=True)
        spin_thread.start()

        if args.start_pose is not None:
            start_pose = np.asarray(args.start_pose, dtype=np.float64)
            if args.start_gripper is None:
                if not controller.wait_for_status(args.state_timeout):
                    return 3
                _, status_gripper = controller.get_current_pose_gripper()
                start_gripper = float(status_gripper) / float(args.status_gripper_scale)
            else:
                start_gripper = float(args.start_gripper)
        else:
            if not controller.wait_for_status(args.state_timeout):
                return 3
            current = controller.get_current_pose_gripper()
            start_pose, status_gripper = current
            start_gripper = float(status_gripper) / float(args.status_gripper_scale)

        print(
            f"[INFO] Replay origin pose={start_pose}, raw_gripper={start_gripper:.5f}, "
            f"cmd_gripper={start_gripper * args.gripper_command_scale + args.gripper_command_offset:.5f}"
        )
        print(
            f"[INFO] Gripper conversion: command = raw * {args.gripper_command_scale:.5f} "
            f"+ {args.gripper_command_offset:.5f}; current status raw = status/{args.status_gripper_scale:.5f}"
        )
        poses, grippers = accumulate_targets(action, start_pose, start_gripper)
        print(
            f"[INFO] Final target pose={poses[-1]}, raw_gripper={grippers[-1]:.5f}, "
            f"cmd_gripper={grippers[-1] * args.gripper_command_scale + args.gripper_command_offset:.5f}"
        )
        if args.visualize_pixels and visual_data is not None:
            visualizer = YuHaiReplayVisualizer(visual_data, args.visualization_window)

        if args.countdown > 0 and not countdown(args.countdown):
            return 130
        replay_targets(controller, poses, grippers, hz=args.control_frequency, speed=args.speed, visualizer=visualizer)
        return 0
    finally:
        shutdown_event.set()
        if visualizer is not None:
            visualizer.close()
        if controller is not None:
            controller.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if spin_thread is not None:
            spin_thread.join(timeout=1.0)


if __name__ == "__main__":
    raise SystemExit(main())
