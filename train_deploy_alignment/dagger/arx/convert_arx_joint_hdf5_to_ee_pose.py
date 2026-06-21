#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert ARX-X5 joint-space HDF5 episodes to add end-effector pose datasets."""

import argparse
import math
import shutil
from pathlib import Path
from xml.etree import ElementTree as ET

import h5py
import numpy as np


DEFAULT_URDF = (
    "/home/agilex/ARX_X5-main/ARX_X5-main/ROS2/X5_ws/src/"
    "arx_x5_ros2/arx_x5_controller/x5.urdf"
)


def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def rpy_matrix(rpy):
    roll, pitch, yaw = rpy
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    C = 1.0 - c
    return np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
    )


def transform(xyz, rpy):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = rpy_matrix(rpy)
    T[:3, 3] = np.asarray(xyz, dtype=np.float64)
    return T


def matrix_to_rpy(R):
    # Matches the common URDF/Eigen XYZ roll-pitch-yaw convention:
    # R = Rz(yaw) * Ry(pitch) * Rx(roll).
    sy = -float(R[2, 0])
    sy = max(-1.0, min(1.0, sy))
    pitch = math.asin(sy)
    cp = math.cos(pitch)
    if abs(cp) > 1e-9:
        roll = math.atan2(float(R[2, 1]), float(R[2, 2]))
        yaw = math.atan2(float(R[1, 0]), float(R[0, 0]))
    else:
        roll = 0.0
        yaw = math.atan2(float(-R[0, 1]), float(R[1, 1]))
    return np.asarray([roll, pitch, yaw], dtype=np.float64)


def parse_vec(text, default):
    if text is None:
        return np.asarray(default, dtype=np.float64)
    return np.asarray([float(x) for x in text.split()], dtype=np.float64)


def load_revolute_chain(urdf_path):
    root = ET.parse(urdf_path).getroot()
    joints = []
    for joint in root.findall("joint"):
        if joint.attrib.get("type") != "revolute":
            continue
        origin = joint.find("origin")
        axis = joint.find("axis")
        joints.append(
            {
                "name": joint.attrib["name"],
                "xyz": parse_vec(origin.attrib.get("xyz") if origin is not None else None, [0, 0, 0]),
                "rpy": parse_vec(origin.attrib.get("rpy") if origin is not None else None, [0, 0, 0]),
                "axis": parse_vec(axis.attrib.get("xyz") if axis is not None else None, [1, 0, 0]),
            }
        )
    if len(joints) < 6:
        raise ValueError(f"Expected at least 6 revolute joints in {urdf_path}, got {len(joints)}")
    return joints[:6]


def fk_xyzrpy(joint6, chain):
    T = np.eye(4, dtype=np.float64)
    for q, joint in zip(joint6, chain):
        T = T @ transform(joint["xyz"], joint["rpy"])
        Rq = np.eye(4, dtype=np.float64)
        Rq[:3, :3] = axis_angle(joint["axis"], float(q))
        T = T @ Rq
    xyz = T[:3, 3]
    rpy = matrix_to_rpy(T[:3, :3])
    return np.concatenate([xyz, rpy])


def wrap_angle(rad):
    return (rad + np.pi) % (2.0 * np.pi) - np.pi


def delta_xyzyaw_gripper(ee_pose, gripper):
    out = np.zeros((ee_pose.shape[0], 5), dtype=np.float32)
    if ee_pose.shape[0] > 1:
        out[1:, 0:3] = ee_pose[1:, 0:3] - ee_pose[:-1, 0:3]
        out[1:, 3] = wrap_angle(ee_pose[1:, 5] - ee_pose[:-1, 5])
        out[1:, 4] = gripper[1:] - gripper[:-1]
    return out


def copy_attrs(src, dst):
    for key, value in src.attrs.items():
        dst.attrs[key] = value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_hdf5")
    parser.add_argument("--output_hdf5", default=None)
    parser.add_argument("--urdf", default=DEFAULT_URDF)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input_hdf5)
    output_path = Path(args.output_hdf5) if args.output_hdf5 else input_path.with_name(input_path.stem + "_ee_pose.hdf5")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} exists; pass --overwrite to replace it.")

    chain = load_revolute_chain(args.urdf)
    with h5py.File(input_path, "r") as src:
        qpos = np.asarray(src["observations/qpos"], dtype=np.float64)
        action = np.asarray(src["action"], dtype=np.float64)
        if qpos.ndim != 2 or qpos.shape[1] < 7:
            raise ValueError(f"Expected observations/qpos shape (T, >=7), got {qpos.shape}")
        if action.ndim != 2 or action.shape[1] < 7:
            raise ValueError(f"Expected action shape (T, >=7), got {action.shape}")

        obs_ee = np.asarray([fk_xyzrpy(row[:6], chain) for row in qpos], dtype=np.float32)
        obs_gripper = qpos[:, 6].astype(np.float32)
        action_ee = np.asarray([fk_xyzrpy(row[:6], chain) for row in action], dtype=np.float32)
        action_gripper = action[:, 6].astype(np.float32)
        action_delta = delta_xyzyaw_gripper(action_ee.astype(np.float64), action_gripper.astype(np.float64))

    if output_path.exists():
        output_path.unlink()
    shutil.copy2(input_path, output_path)

    with h5py.File(output_path, "a") as f:
        obs = f.require_group("observations")
        for name in ("ee_pose", "gripper"):
            if name in obs:
                del obs[name]
        for name in ("action_ee_pose", "action_gripper", "action_delta_ee"):
            if name in f:
                del f[name]

        obs.create_dataset("ee_pose", data=obs_ee, dtype="float32")
        obs.create_dataset("gripper", data=obs_gripper, dtype="float32")
        f.create_dataset("action_ee_pose", data=action_ee, dtype="float32")
        f.create_dataset("action_gripper", data=action_gripper, dtype="float32")
        f.create_dataset("action_delta_ee", data=action_delta, dtype="float32")
        f.attrs["ee_pose_source"] = "computed_from_joint_qpos_with_urdf_fk"
        f.attrs["ee_pose_urdf"] = str(args.urdf)
        f.attrs["ee_pose_order"] = "x,y,z,roll,pitch,yaw"
        f.attrs["action_delta_ee_order"] = "dx,dy,dz,dyaw,d_gripper"

    print(f"input:  {input_path}")
    print(f"output: {output_path}")
    print(f"observations/ee_pose: {obs_ee.shape}")
    print(f"action_ee_pose:       {action_ee.shape}")
    print(f"action_delta_ee:      {action_delta.shape}")
    print(f"first ee_pose:        {obs_ee[0].tolist()}")
    print(f"last ee_pose:         {obs_ee[-1].tolist()}")


if __name__ == "__main__":
    main()
