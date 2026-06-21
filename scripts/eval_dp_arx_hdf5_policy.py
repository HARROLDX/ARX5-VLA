#!/usr/bin/env python3
"""Evaluate an ARX websocket policy on recorded HDF5 + video episodes.

This is a deployment-side sanity check: it sends recorded observations through the
same websocket client used by the robot script and compares predicted action
chunks against the recorded HDF5 actions.
"""

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np
from openpi_client import image_tools, websocket_client_policy


CAMERA_TO_VIDEO = {
    "top_head": "cam_high",
    "hand_right": "cam_right_wrist",
    "hand_left": "cam_left_wrist",
}


def read_video_frame(video_path: Path, frame_index: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok or frame_bgr is None:
        raise RuntimeError(f"Could not read frame {frame_index} from {video_path}")
    return frame_bgr


def make_payload(data_dir: Path, episode_idx: int, frame_idx: int, qpos: np.ndarray, prompt: str) -> dict:
    frames_rgb = []
    for _, video_name in CAMERA_TO_VIDEO.items():
        video_path = data_dir / "video" / video_name / f"{episode_idx}.mp4"
        frame_bgr = read_video_frame(video_path, frame_idx)
        frames_rgb.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    frames_rgb = image_tools.resize_with_pad(np.asarray(frames_rgb), 224, 224)
    return {
        "state": qpos,
        "images": {
            "top_head": frames_rgb[0].transpose(2, 0, 1),
            "hand_right": frames_rgb[1].transpose(2, 0, 1),
            "hand_left": frames_rgb[2].transpose(2, 0, 1),
        },
        "prompt": prompt,
    }


def summarize_delta(name: str, delta: np.ndarray) -> None:
    arm_idx = [i for i in range(delta.shape[-1]) if i not in (6, 13)]
    print(
        name,
        f"mae={np.mean(np.abs(delta)):.4f}",
        f"max={np.max(np.abs(delta)):.4f}",
        f"arm_max={np.max(np.abs(delta[..., arm_idx])):.4f}",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, default=Path("kai0_data/arx_teleop/task_1_collect_001"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--episodes", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--frames", nargs="+", type=int, default=[0, 10, 50, 100])
    parser.add_argument("--horizon", type=int, default=30)
    parser.add_argument("--prompt", default="Fetch and hang the cloth.")
    args = parser.parse_args()

    policy = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    print(f"Server metadata: {policy.get_server_metadata()}")

    first_deltas = []
    chunk_deltas = []
    for episode_idx in args.episodes:
        h5_path = args.data_dir / f"episode_{episode_idx}.hdf5"
        with h5py.File(h5_path, "r") as f:
            qpos_all = f["observations/qpos"][:]
            action_all = f["action"][:]

        for frame_idx in args.frames:
            if frame_idx >= len(qpos_all):
                continue
            payload = make_payload(args.data_dir, episode_idx, frame_idx, qpos_all[frame_idx], args.prompt)
            pred = np.asarray(policy.infer(payload)["actions"], dtype=float)
            end = min(frame_idx + args.horizon, len(action_all))
            gt = action_all[frame_idx:end]
            if len(gt) < args.horizon:
                pad = np.repeat(gt[-1][None], args.horizon - len(gt), axis=0)
                gt = np.concatenate([gt, pad], axis=0)

            first_delta = pred[0] - qpos_all[frame_idx]
            chunk_delta = pred[: args.horizon] - gt[: args.horizon]
            first_deltas.append(first_delta)
            chunk_deltas.append(chunk_delta)
            print(f"\nepisode={episode_idx} frame={frame_idx}")
            print("qpos      ", np.round(qpos_all[frame_idx], 4))
            print("pred[0]   ", np.round(pred[0], 4))
            print("gt[0]     ", np.round(gt[0], 4))
            summarize_delta("pred[0]-qpos", first_delta)
            summarize_delta("pred-gt chunk", chunk_delta)

    if first_deltas:
        print("\n=== Aggregate ===")
        summarize_delta("all pred[0]-qpos", np.asarray(first_deltas))
        summarize_delta("all pred-gt chunk", np.asarray(chunk_deltas))


if __name__ == "__main__":
    main()
