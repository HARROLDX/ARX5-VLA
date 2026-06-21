#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Convert ARX-X5 joint HDF5 episodes using ARX's own ForwardKinematicsRpy API."""

import argparse
import subprocess
from pathlib import Path

import h5py
import numpy as np


DEFAULT_FK_CLI = "/home/agilex/kai0/train_deploy_alignment/dagger/arx/arx_fk_cli"


def wrap_angle(rad):
    return (rad + np.pi) % (2.0 * np.pi) - np.pi


def run_arx_fk(fk_cli: str, joint6: np.ndarray, end_type: int) -> np.ndarray:
    joint6 = np.asarray(joint6, dtype=np.float64)
    if joint6.ndim != 2 or joint6.shape[1] != 6:
        raise ValueError(f"Expected joint6 shape (T, 6), got {joint6.shape}")

    stdin = "\n".join(" ".join(f"{x:.17g}" for x in row) for row in joint6) + "\n"
    proc = subprocess.run(
        [fk_cli, str(end_type)],
        input=stdin,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"ARX FK CLI failed with code {proc.returncode}\n"
            f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
        )

    rows = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append([float(x) for x in line.split()])
    out = np.asarray(rows, dtype=np.float32)
    if out.shape != (joint6.shape[0], 6):
        raise RuntimeError(f"Expected FK output {(joint6.shape[0], 6)}, got {out.shape}")
    return out


def delta_xyzyaw_gripper(ee_pose: np.ndarray, gripper: np.ndarray) -> np.ndarray:
    ee_pose = np.asarray(ee_pose, dtype=np.float64)
    gripper = np.asarray(gripper, dtype=np.float64)
    out = np.zeros((ee_pose.shape[0], 5), dtype=np.float32)
    if ee_pose.shape[0] > 1:
        out[1:, 0:3] = ee_pose[1:, 0:3] - ee_pose[:-1, 0:3]
        out[1:, 3] = wrap_angle(ee_pose[1:, 5] - ee_pose[:-1, 5])
        out[1:, 4] = gripper[1:] - gripper[:-1]
    return out


def replace_dataset(group, name, data, dtype):
    if name in group:
        del group[name]
    group.create_dataset(name, data=data, dtype=dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_hdf5")
    parser.add_argument("--output_hdf5", default=None)
    parser.add_argument("--fk_cli", default=DEFAULT_FK_CLI)
    parser.add_argument("--end_type", type=int, default=0, help="ARX arm_end_type: 0 slave/default, 1 master, 2 x5_2025")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    input_path = Path(args.input_hdf5)
    output_path = Path(args.output_hdf5) if args.output_hdf5 else input_path.with_name(input_path.stem + "_ee_pose_arx.hdf5")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} exists; pass --overwrite to replace it.")
    if not Path(args.fk_cli).exists():
        raise FileNotFoundError(f"FK CLI not found: {args.fk_cli}")

    with h5py.File(input_path, "r") as src:
        qpos = np.asarray(src["observations/qpos"], dtype=np.float64)
        action = np.asarray(src["action"], dtype=np.float64)
        if qpos.ndim != 2 or qpos.shape[1] < 7:
            raise ValueError(f"Expected observations/qpos shape (T, >=7), got {qpos.shape}")
        if action.ndim != 2 or action.shape[1] < 7:
            raise ValueError(f"Expected action shape (T, >=7), got {action.shape}")

        obs_ee = run_arx_fk(args.fk_cli, qpos[:, :6], args.end_type)
        obs_gripper = qpos[:, 6].astype(np.float32)
        action_ee = run_arx_fk(args.fk_cli, action[:, :6], args.end_type)
        action_gripper = action[:, 6].astype(np.float32)
        action_delta = delta_xyzyaw_gripper(action_ee, action_gripper)

        if output_path.exists():
            output_path.unlink()
        with h5py.File(output_path, "w") as dst:
            for key, value in src.attrs.items():
                dst.attrs[key] = value

            def copy_item(name, obj):
                if isinstance(obj, h5py.Dataset):
                    dst.create_dataset(name, data=obj[()], dtype=obj.dtype)

            src.visititems(copy_item)

            obs = dst.require_group("observations")
            replace_dataset(obs, "ee_pose", obs_ee, "float32")
            replace_dataset(obs, "gripper", obs_gripper, "float32")
            replace_dataset(dst, "action_ee_pose", action_ee, "float32")
            replace_dataset(dst, "action_gripper", action_gripper, "float32")
            replace_dataset(dst, "action_delta_ee", action_delta, "float32")

            dst.attrs["ee_pose_source"] = "computed_from_joint_qpos_with_arx_InterfacesTools_ForwardKinematicsRpy"
            dst.attrs["ee_pose_arx_fk_cli"] = str(args.fk_cli)
            dst.attrs["ee_pose_arx_end_type"] = args.end_type
            dst.attrs["ee_pose_order"] = "x,y,z,roll,pitch,yaw"
            dst.attrs["action_delta_ee_order"] = "dx,dy,dz,dyaw,d_gripper"

    print(f"input:  {input_path}")
    print(f"output: {output_path}")
    print(f"observations/ee_pose: {obs_ee.shape}")
    print(f"action_ee_pose:       {action_ee.shape}")
    print(f"action_delta_ee:      {action_delta.shape}")
    print(f"first ee_pose:        {obs_ee[0].tolist()}")
    print(f"last ee_pose:         {obs_ee[-1].tolist()}")


if __name__ == "__main__":
    main()
