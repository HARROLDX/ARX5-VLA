#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""
Replay left-arm ARX-X5 joint trajectories from a recorded HDF5 episode.

This validates left-single-arm collection files. It reads either a native 7-D
left-arm trajectory or a 14-D dual-arm trajectory and publishes only the left
7-D command stream.
"""

import argparse
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

DRY_RUN_PRECHECK = "--dry_run" in sys.argv

if not DRY_RUN_PRECHECK and sys.version_info[:2] != (3, 10):
    raise SystemExit(
        f"ROS Humble rclpy in this setup expects Python 3.10, but this interpreter is "
        f"Python {sys.version_info.major}.{sys.version_info.minor}.\n"
        "Use the same Python 3.10 environment you use for ARX collection/deployment."
    )

import h5py
import numpy as np

if DRY_RUN_PRECHECK:
    rclpy = None

    class Node:
        pass

    RobotStatus = None
else:
    import rclpy
    from rclpy.node import Node

    try:
        from arx5_arm_msg.msg import RobotStatus
    except ImportError as e:
        raise SystemExit(
            "arx5_arm_msg was not found in PYTHONPATH.\n"
            "Source ROS Humble and the ARX X5_ws overlay before running this script."
        ) from e


shutdown_event = threading.Event()


@dataclass
class TrajectoryStats:
    frames: int
    joints: int
    min_value: float
    max_value: float
    max_step: float
    max_arm_step: float
    max_gripper_step: float


class ARXLeftReplayController(Node):
    """Minimal ROS2 replay controller for the ARX-X5 left slave arm."""

    def __init__(self, args):
        super().__init__("arx_left_hdf5_replay")
        self.args = args
        self.joint_left_deque = deque(maxlen=2000)

        self.pub_left = self.create_publisher(RobotStatus, args.joint_cmd_topic_left, 10)
        self.create_subscription(RobotStatus, args.joint_state_topic_left, self._left_status_cb, 10)

        self.get_logger().info(f"Publish left commands: {args.joint_cmd_topic_left}")
        self.get_logger().info(f"Subscribe left state: {args.joint_state_topic_left}")

    def _left_status_cb(self, msg):
        self.joint_left_deque.append(msg)

    def get_slave_positions(self) -> Optional[np.ndarray]:
        if len(self.joint_left_deque) == 0:
            return None
        left = np.asarray(self.joint_left_deque[-1].joint_pos, dtype=float)
        if left.shape[0] != 7:
            self.get_logger().warn(f"Expected 7-D left joint state, got {left.shape}")
            return None
        return left.copy()

    def wait_for_state(self, timeout: float) -> bool:
        print("Waiting for left ARX slave joint state...")
        start = time.time()
        while rclpy.ok() and not shutdown_event.is_set() and time.time() - start < timeout:
            if self.get_slave_positions() is not None:
                print("[INFO] Left slave joint state ready.")
                return True
            time.sleep(0.05)
        print("[ERROR] Timed out waiting for left slave joint state.")
        return False

    def publish_joint_positions(self, pos: np.ndarray):
        pos = np.asarray(pos, dtype=float)
        if pos.shape != (7,):
            self.get_logger().warn(f"Expected 7-D left command, got shape={pos.shape}")
            return

        msg_left = RobotStatus()
        msg_left.joint_pos = [float(x) for x in pos]
        self.pub_left.publish(msg_left)

    def smooth_goto(self, target: np.ndarray, duration: float, hz: float, max_joint_step: Optional[float]) -> bool:
        current = self.get_slave_positions()
        if current is None:
            print("[ERROR] No current left slave pose; cannot move to first frame.")
            return False

        target = np.asarray(target, dtype=float)
        max_delta = float(np.max(np.abs(target - current)))
        num_steps = max(1, int(duration * hz))
        if max_joint_step is not None and max_joint_step > 0:
            num_steps = max(num_steps, int(np.ceil(max_delta / max_joint_step)))

        print(f"[INFO] Moving left arm to first frame: duration>={duration:.2f}s, steps={num_steps}, max_delta={max_delta:.4f}")
        for step in range(num_steps + 1):
            if shutdown_event.is_set() or not rclpy.ok():
                return False
            alpha = step / num_steps
            smooth_alpha = (1.0 - np.cos(alpha * np.pi)) / 2.0
            cmd = current * (1.0 - smooth_alpha) + target * smooth_alpha
            self.publish_joint_positions(cmd)
            if step % max(1, int(hz)) == 0 or step == num_steps:
                print(f"\rGoto first frame: {int(alpha * 100):3d}%", end="", flush=True)
            time.sleep(1.0 / hz)
        print("\n[INFO] Reached first frame.")
        return True


def _normalize_hdf5_key(key: str) -> str:
    return key[1:] if key.startswith("/") else key


def load_trajectory(path: str, dataset: str, start_step: int, end_step: int) -> np.ndarray:
    key = _normalize_hdf5_key(dataset)
    with h5py.File(path, "r") as root:
        if key not in root:
            available = []
            root.visititems(lambda name, obj: available.append(name) if isinstance(obj, h5py.Dataset) else None)
            raise KeyError(f"Dataset '{dataset}' not found. Available datasets: {available}")
        traj = np.asarray(root[key][:], dtype=np.float64)

    if traj.ndim != 2 or traj.shape[1] not in (7, 14):
        raise ValueError(f"Expected trajectory shape (T, 7) or (T, 14), got {traj.shape} from '{dataset}'")
    if traj.shape[0] == 0:
        raise ValueError("Trajectory is empty.")
    if traj.shape[1] == 14:
        print("[INFO] Loaded 14-D trajectory; replaying left arm columns 0:7 only.")
        traj = traj[:, :7]

    n = traj.shape[0]
    if start_step < 0:
        start_step = max(0, n + start_step)
    if end_step < 0:
        end_step = n
    else:
        end_step = min(n, end_step)
    start_step = min(max(0, start_step), n)
    if end_step <= start_step:
        raise ValueError(f"Invalid replay range: start_step={start_step}, end_step={end_step}, frames={n}")
    return traj[start_step:end_step]


def compute_stats(traj: np.ndarray) -> TrajectoryStats:
    if not np.all(np.isfinite(traj)):
        bad = np.argwhere(~np.isfinite(traj))
        first_bad = bad[0].tolist()
        raise ValueError(f"Trajectory contains NaN/Inf, first bad index={first_bad}")

    if len(traj) > 1:
        steps = np.abs(np.diff(traj, axis=0))
        max_step = float(np.max(steps))
        arm_cols = [i for i in range(7) if i != 6]
        max_arm_step = float(np.max(steps[:, arm_cols]))
        max_gripper_step = float(np.max(steps[:, [6]]))
    else:
        max_step = 0.0
        max_arm_step = 0.0
        max_gripper_step = 0.0

    return TrajectoryStats(
        frames=int(traj.shape[0]),
        joints=int(traj.shape[1]),
        min_value=float(np.min(traj)),
        max_value=float(np.max(traj)),
        max_step=max_step,
        max_arm_step=max_arm_step,
        max_gripper_step=max_gripper_step,
    )


def validate_replay_steps(traj: np.ndarray, max_arm_step: float, max_gripper_step: float, force: bool) -> bool:
    if len(traj) <= 1:
        return True
    steps = np.abs(np.diff(traj, axis=0))
    arm_cols = [i for i in range(7) if i != 6]
    arm_bad = np.argwhere(steps[:, arm_cols] > max_arm_step)
    grip_bad = np.argwhere(steps[:, [6]] > max_gripper_step)

    ok = True
    if len(arm_bad) > 0:
        frame, col_idx = arm_bad[0]
        joint = arm_cols[col_idx]
        print(
            f"[WARN] Left arm joint step exceeds threshold: frame {frame}->{frame + 1}, "
            f"joint {joint}, delta={steps[frame, joint]:.4f}, limit={max_arm_step:.4f}"
        )
        ok = False
    if len(grip_bad) > 0:
        frame, _ = grip_bad[0]
        print(
            f"[WARN] Left gripper step exceeds threshold: frame {frame}->{frame + 1}, "
            f"delta={steps[frame, 6]:.4f}, limit={max_gripper_step:.4f}"
        )
        ok = False

    if ok:
        return True
    if force:
        print("[WARN] Continuing because --force was set.")
        return True
    print("[ERROR] Refusing to replay. Inspect the data or rerun with --force if this is expected.")
    return False


def replay_trajectory(controller: ARXLeftReplayController, traj: np.ndarray, hz: float, speed: float):
    sleep_s = 1.0 / hz / speed
    total = len(traj)
    print(f"[INFO] Replaying {total} left-arm frames at source_hz={hz:.2f}, speed={speed:.2f}.")
    for idx, cmd in enumerate(traj):
        if shutdown_event.is_set() or not rclpy.ok():
            print("\n[INFO] Replay interrupted.")
            return False
        controller.publish_joint_positions(cmd)
        if idx % max(1, int(hz)) == 0 or idx == total - 1:
            print(f"\rReplay: {idx + 1}/{total} ({100.0 * (idx + 1) / total:5.1f}%)", end="", flush=True)
        time.sleep(sleep_s)
    print("\n[INFO] Replay finished.")
    return True


def countdown(seconds: int):
    for remain in range(seconds, 0, -1):
        if shutdown_event.is_set():
            return False
        print(f"\rStarting left-arm replay in {remain}s. Press Ctrl-C to cancel.", end="", flush=True)
        time.sleep(1.0)
    print()
    return True


def parse_args():
    parser = argparse.ArgumentParser(description="Replay a recorded ARX-X5 left-arm HDF5 joint trajectory.")
    parser.add_argument("--hdf5_path", required=True, help="Path to episode_*.hdf5")
    parser.add_argument("--dataset", default="observations/qpos", help="HDF5 dataset to replay, usually observations/qpos or action")
    parser.add_argument("--start_step", type=int, default=0, help="Inclusive start frame. Negative values count from end.")
    parser.add_argument("--end_step", type=int, default=-1, help="Exclusive end frame. -1 means end of trajectory.")
    parser.add_argument("--control_frequency", type=float, default=30.0, help="Source trajectory/control frequency in Hz.")
    parser.add_argument("--speed", type=float, default=0.3, help="Replay speed multiplier. 0.3 is conservative.")
    parser.add_argument("--goto_first_duration", type=float, default=5.0, help="Minimum seconds to move to first frame.")
    parser.add_argument("--goto_frequency", type=float, default=50.0, help="Frequency for initial smooth move.")
    parser.add_argument("--goto_max_joint_step", type=float, default=0.05, help="Max interpolation step while moving to first frame.")
    parser.add_argument("--max_arm_step", type=float, default=0.12, help="Allowed per-frame delta for non-gripper joints.")
    parser.add_argument("--max_gripper_step", type=float, default=0.8, help="Allowed per-frame delta for left gripper joint 6.")
    parser.add_argument("--state_timeout", type=float, default=15.0, help="Seconds to wait for left slave joint state.")
    parser.add_argument("--countdown", type=int, default=5, help="Safety countdown before moving the robot.")
    parser.add_argument("--dry_run", action="store_true", help="Only inspect the HDF5 and print stats; do not publish commands.")
    parser.add_argument("--force", action="store_true", help="Replay even if trajectory jump checks warn.")
    parser.add_argument("--joint_cmd_topic_left", default="/arm_master_l_status")
    parser.add_argument("--joint_state_topic_left", default="/arm_slave_l_status")
    args = parser.parse_args()

    if args.control_frequency <= 0:
        parser.error("--control_frequency must be > 0")
    if args.goto_frequency <= 0:
        parser.error("--goto_frequency must be > 0")
    if args.speed <= 0:
        parser.error("--speed must be > 0")
    return args


def main():
    args = parse_args()

    def _on_sigint(sig, frame):
        print("\n[INFO] Ctrl+C received; stopping left-arm replay.")
        shutdown_event.set()

    signal.signal(signal.SIGINT, _on_sigint)

    traj = load_trajectory(args.hdf5_path, args.dataset, args.start_step, args.end_step)
    stats = compute_stats(traj)
    print("[INFO] Loaded left-arm trajectory")
    print(f"  path: {args.hdf5_path}")
    print(f"  dataset: {args.dataset}")
    print(f"  frames: {stats.frames}, joints: {stats.joints}")
    print(f"  value range: [{stats.min_value:.4f}, {stats.max_value:.4f}]")
    print(f"  max frame step: all={stats.max_step:.4f}, arm={stats.max_arm_step:.4f}, gripper={stats.max_gripper_step:.4f}")

    if not validate_replay_steps(traj, args.max_arm_step, args.max_gripper_step, args.force):
        return 2
    if args.dry_run:
        print("[INFO] Dry run complete. No ROS commands were published.")
        return 0

    rclpy.init()
    controller = None
    spin_thread = None
    try:
        controller = ARXLeftReplayController(args)
        spin_thread = threading.Thread(target=rclpy.spin, args=(controller,), daemon=True)
        spin_thread.start()
        if not controller.wait_for_state(args.state_timeout):
            return 3
        current = controller.get_slave_positions()
        if current is not None:
            print(f"[INFO] Current-to-first max delta: {float(np.max(np.abs(traj[0] - current))):.4f}")
        if args.countdown > 0 and not countdown(args.countdown):
            return 130
        if not controller.smooth_goto(
            traj[0],
            duration=args.goto_first_duration,
            hz=args.goto_frequency,
            max_joint_step=args.goto_max_joint_step,
        ):
            return 4
        replay_trajectory(controller, traj, hz=args.control_frequency, speed=args.speed)
        return 0
    finally:
        shutdown_event.set()
        if controller is not None:
            controller.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if spin_thread is not None:
            spin_thread.join(timeout=1.0)


if __name__ == "__main__":
    raise SystemExit(main())
