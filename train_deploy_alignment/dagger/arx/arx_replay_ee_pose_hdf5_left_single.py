#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Replay ARX left-arm end-effector pose trajectories from HDF5."""

import argparse
import signal
import sys
import threading
import time
from dataclasses import dataclass
from typing import Optional

DRY_RUN_PRECHECK = "--dry_run" in sys.argv or "--help" in sys.argv or "-h" in sys.argv

import h5py
import numpy as np

if DRY_RUN_PRECHECK:
    rclpy = None

    class Node:
        pass

    RobotCmd = None
    RobotStatus = None
else:
    import rclpy
    from rclpy.node import Node

    try:
        from arx5_arm_msg.msg import RobotCmd, RobotStatus
    except ImportError as e:
        raise SystemExit(
            "arx5_arm_msg was not found in PYTHONPATH.\n"
            "Source ROS Humble and the ARX X5_ws overlay before running this script."
        ) from e


ROBOT_CMD_END_CONTROL_MODE = 4
shutdown_event = threading.Event()


@dataclass
class PoseTrajectory:
    poses: np.ndarray
    grippers: Optional[np.ndarray]


def normalize_key(key: str) -> str:
    return key[1:] if key.startswith("/") else key


def resolve_range(n: int, start_step: int, end_step: int) -> tuple[int, int]:
    if start_step < 0:
        start_step = max(0, n + start_step)
    if end_step < 0:
        end_step = n
    else:
        end_step = min(n, end_step)
    start_step = min(max(0, start_step), n)
    if end_step <= start_step:
        raise ValueError(f"Invalid range start_step={start_step}, end_step={end_step}, frames={n}")
    return start_step, end_step


def load_pose_trajectory(path: str, pose_dataset: str, gripper_dataset: str, start_step: int, end_step: int) -> PoseTrajectory:
    pose_key = normalize_key(pose_dataset)
    gripper_key = normalize_key(gripper_dataset) if gripper_dataset else ""
    with h5py.File(path, "r") as root:
        if pose_key not in root:
            available = []
            root.visititems(lambda name, obj: available.append(name) if isinstance(obj, h5py.Dataset) else None)
            raise KeyError(f"Dataset '{pose_dataset}' not found. Available datasets: {available}")
        poses = np.asarray(root[pose_key][:], dtype=np.float64)
        if poses.ndim != 2 or poses.shape[1] != 6:
            raise ValueError(f"Expected pose dataset shape (T, 6), got {poses.shape}")
        if not np.all(np.isfinite(poses)):
            bad = np.argwhere(~np.isfinite(poses))[0].tolist()
            raise ValueError(f"Pose trajectory contains NaN/Inf, first bad index={bad}")

        grippers = None
        if gripper_key:
            if gripper_key not in root:
                raise KeyError(f"Gripper dataset '{gripper_dataset}' not found.")
            grippers = np.asarray(root[gripper_key][:], dtype=np.float64)
            if grippers.ndim != 1 or grippers.shape[0] != poses.shape[0]:
                raise ValueError(f"Expected gripper shape ({poses.shape[0]},), got {grippers.shape}")

    start_step, end_step = resolve_range(poses.shape[0], start_step, end_step)
    return PoseTrajectory(poses=poses[start_step:end_step], grippers=None if grippers is None else grippers[start_step:end_step])


def validate_steps(poses: np.ndarray, max_xyz_step: float, max_rpy_step: float, force: bool) -> bool:
    if len(poses) <= 1:
        return True
    steps = np.abs(np.diff(poses, axis=0))
    xyz_bad = np.argwhere(steps[:, :3] > max_xyz_step)
    rpy_bad = np.argwhere(steps[:, 3:] > max_rpy_step)
    ok = True
    if len(xyz_bad):
        frame, axis = xyz_bad[0]
        print(f"[WARN] xyz step too large: frame {frame}->{frame + 1}, axis={axis}, delta={steps[frame, axis]:.5f}, limit={max_xyz_step:.5f}")
        ok = False
    if len(rpy_bad):
        frame, axis = rpy_bad[0]
        print(f"[WARN] rpy step too large: frame {frame}->{frame + 1}, axis={axis}, delta={steps[frame, 3 + axis]:.5f}, limit={max_rpy_step:.5f}")
        ok = False
    if ok or force:
        if not ok:
            print("[WARN] Continuing because --force was set.")
        return True
    print("[ERROR] Refusing to replay. Inspect the data or rerun with --force if expected.")
    return False


class ARXLeftEEReplayController(Node):
    def __init__(self, args):
        super().__init__("arx_left_ee_pose_hdf5_replay")
        self.args = args
        self.latest_status = None
        self.status_lock = threading.Lock()
        self.create_subscription(RobotStatus, args.status_topic, self.status_callback, 10)
        self.cmd_pub = self.create_publisher(RobotCmd, args.command_topic, 10)
        self.get_logger().info(f"Subscribe status: {args.status_topic}")
        self.get_logger().info(f"Publish EE commands: {args.command_topic}")

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
        return pose.copy(), float(msg.joint_pos[6])

    def wait_for_status(self, timeout: float) -> bool:
        print("[INFO] Waiting for ARX end-effector status...")
        start = time.time()
        while rclpy.ok() and not shutdown_event.is_set() and time.time() - start < timeout:
            if self.get_current_pose_gripper() is not None:
                print("[INFO] End-effector status ready.")
                return True
            time.sleep(0.05)
        print("[ERROR] Timed out waiting for RobotStatus.")
        return False

    def publish_pose(self, pose: np.ndarray, gripper: float):
        msg = RobotCmd()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.end_pos = [float(x) for x in pose]
        msg.joint_pos = [0.0] * 6
        msg.gripper = float(gripper) * float(self.args.gripper_command_scale) + float(self.args.gripper_command_offset)
        msg.mode = int(self.args.robot_cmd_mode)
        self.cmd_pub.publish(msg)


def smooth_goto(controller: ARXLeftEEReplayController, target_pose: np.ndarray, target_gripper: float, duration: float, hz: float) -> bool:
    current = controller.get_current_pose_gripper()
    if current is None:
        print("[ERROR] No current end-effector status; cannot move to first frame.")
        return False
    start_pose, start_gripper = current
    n = max(1, int(duration * hz))
    print(f"[INFO] Moving to first EE frame: duration={duration:.2f}s, steps={n}")
    for step in range(n + 1):
        if shutdown_event.is_set() or not rclpy.ok():
            return False
        alpha = step / n
        smooth = (1.0 - np.cos(alpha * np.pi)) / 2.0
        pose = start_pose * (1.0 - smooth) + target_pose * smooth
        gripper = start_gripper * (1.0 - smooth) + target_gripper * smooth
        controller.publish_pose(pose, gripper)
        if step % max(1, int(hz)) == 0 or step == n:
            print(f"\rGoto first EE frame: {int(alpha * 100):3d}%", end="", flush=True)
        time.sleep(1.0 / hz)
    print("\n[INFO] Reached first EE frame.")
    return True


def replay(controller: ARXLeftEEReplayController, poses: np.ndarray, grippers: np.ndarray, hz: float, speed: float) -> bool:
    sleep_s = 1.0 / hz / speed
    total = len(poses)
    print(f"[INFO] Replaying {total} EE frames at source_hz={hz:.2f}, speed={speed:.2f}.")
    for idx, (pose, gripper) in enumerate(zip(poses, grippers)):
        if shutdown_event.is_set() or not rclpy.ok():
            print("\n[INFO] Replay interrupted.")
            return False
        controller.publish_pose(pose, gripper)
        if idx % max(1, int(hz)) == 0 or idx == total - 1:
            print(
                f"\rReplay: {idx + 1}/{total} ({100.0 * (idx + 1) / total:5.1f}%) "
                f"xyz=({pose[0]:.3f},{pose[1]:.3f},{pose[2]:.3f}) rpy=({pose[3]:.3f},{pose[4]:.3f},{pose[5]:.3f})",
                end="",
                flush=True,
            )
        time.sleep(sleep_s)
    print("\n[INFO] Replay finished.")
    return True


def countdown(seconds: int) -> bool:
    for remain in range(seconds, 0, -1):
        if shutdown_event.is_set():
            return False
        print(f"\rStarting EE replay in {remain}s. Press Ctrl-C to cancel.", end="", flush=True)
        time.sleep(1.0)
    print()
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="Replay ARX HDF5 absolute EE pose trajectory through RobotCmd end_pos.")
    parser.add_argument("--hdf5_path", required=True)
    parser.add_argument("--pose_dataset", default="observations/ee_pose")
    parser.add_argument("--gripper_dataset", default="observations/gripper")
    parser.add_argument("--start_step", type=int, default=0)
    parser.add_argument("--end_step", type=int, default=-1)
    parser.add_argument("--control_frequency", type=float, default=30.0)
    parser.add_argument("--speed", type=float, default=0.3)
    parser.add_argument("--status_topic", default="/arm_status")
    parser.add_argument("--command_topic", default="/arm_cmd")
    parser.add_argument("--robot_cmd_mode", type=int, default=ROBOT_CMD_END_CONTROL_MODE)
    parser.add_argument("--gripper_command_scale", type=float, default=1.0)
    parser.add_argument("--gripper_command_offset", type=float, default=0.0)
    parser.add_argument("--state_timeout", type=float, default=15.0)
    parser.add_argument("--countdown", type=int, default=5)
    parser.add_argument("--goto_first", action="store_true", help="Smoothly move from current EE pose to the first recorded EE pose before replay.")
    parser.add_argument("--goto_first_duration", type=float, default=5.0)
    parser.add_argument("--goto_frequency", type=float, default=50.0)
    parser.add_argument("--max_xyz_step", type=float, default=0.03)
    parser.add_argument("--max_rpy_step", type=float, default=0.25)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.control_frequency <= 0:
        parser.error("--control_frequency must be > 0")
    if args.speed <= 0:
        parser.error("--speed must be > 0")
    if args.goto_frequency <= 0:
        parser.error("--goto_frequency must be > 0")
    return args


def main():
    args = parse_args()

    def on_sigint(sig, frame):
        print("\n[INFO] Ctrl+C received; stopping EE replay.")
        shutdown_event.set()

    signal.signal(signal.SIGINT, on_sigint)

    traj = load_pose_trajectory(args.hdf5_path, args.pose_dataset, args.gripper_dataset, args.start_step, args.end_step)
    poses = traj.poses
    grippers = traj.grippers if traj.grippers is not None else np.zeros(len(poses), dtype=np.float64)
    print("[INFO] Loaded EE pose trajectory")
    print(f"  path: {args.hdf5_path}")
    print(f"  pose_dataset: {args.pose_dataset}")
    print(f"  frames: {len(poses)}")
    print(f"  first pose: {poses[0]}")
    print(f"  final pose: {poses[-1]}")
    if len(poses) > 1:
        steps = np.abs(np.diff(poses, axis=0))
        print(f"  max frame step: xyz={float(np.max(steps[:, :3])):.5f}, rpy={float(np.max(steps[:, 3:])):.5f}")
    if not validate_steps(poses, args.max_xyz_step, args.max_rpy_step, args.force):
        return 2
    if args.dry_run:
        print("[INFO] Dry run complete. No ROS commands were published.")
        return 0

    rclpy.init()
    controller = None
    spin_thread = None
    try:
        controller = ARXLeftEEReplayController(args)
        spin_thread = threading.Thread(target=rclpy.spin, args=(controller,), daemon=True)
        spin_thread.start()
        if not controller.wait_for_status(args.state_timeout):
            return 3
        current = controller.get_current_pose_gripper()
        if current is not None:
            current_pose, _ = current
            print(f"[INFO] Current-to-first EE xyz max delta: {float(np.max(np.abs(poses[0, :3] - current_pose[:3]))):.5f}")
            print(f"[INFO] Current-to-first EE rpy max delta: {float(np.max(np.abs(poses[0, 3:] - current_pose[3:]))):.5f}")
        if args.countdown > 0 and not countdown(args.countdown):
            return 130
        if args.goto_first:
            if not smooth_goto(controller, poses[0], grippers[0], args.goto_first_duration, args.goto_frequency):
                return 4
        replay(controller, poses, grippers, args.control_frequency, args.speed)
        return 0
    finally:
        shutdown_event.set()
        if controller is not None:
            controller.destroy_node()
        if rclpy is not None and rclpy.ok():
            rclpy.shutdown()
        if spin_thread is not None:
            spin_thread.join(timeout=1.0)


if __name__ == "__main__":
    raise SystemExit(main())
