import os
os.environ["SVT_LOG"] = "1"

from pathlib import Path
import shutil
import time

import cv2
import h5py
import numpy as np
import tyro
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


CAMERA_VIDEO_DIRS = {
    "observation.images.top_head": "cam_high",
    "observation.images.hand_left": "cam_left_wrist",
}

THIRD_CAMERA_VIDEO_DIRS = {
    "observation.images.camera_third": "camera_third",
}


FEATURES = {
    "observation.images.top_head": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.images.hand_left": {
        "dtype": "video",
        "shape": (480, 640, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (7,),
        "names": None,
    },
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": None,
    },
}


def _camera_video_dirs(include_third_camera: bool) -> dict[str, str]:
    if include_third_camera:
        return {**CAMERA_VIDEO_DIRS, **THIRD_CAMERA_VIDEO_DIRS}
    return dict(CAMERA_VIDEO_DIRS)


def _features(include_third_camera: bool) -> dict:
    features = dict(FEATURES)
    if include_third_camera:
        features["observation.images.camera_third"] = {
            "dtype": "video",
            "shape": (480, 640, 3),
            "names": ["height", "width", "channel"],
        }
    return features


def _episode_index(path: Path) -> int:
    return int(path.stem.split("_", 1)[1])


def _video_path(source_dir: Path, camera_dir: str, episode_path: Path) -> Path:
    index = _episode_index(episode_path)
    candidates = [
        source_dir / "video" / camera_dir / f"{index}.mp4",
        source_dir / "video" / camera_dir / f"{episode_path.stem}.mp4",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Missing video for {episode_path.name}, camera={camera_dir}; searched {candidates}")


def _read_video_rgb(path: Path, expected_len: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    frames = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded from video: {path}")
    if len(frames) != expected_len:
        print(f"[WARN] {path}: decoded {len(frames)} frames, HDF5 has {expected_len}; using min length.")
    return frames


def _load_episode_arrays(episode_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(episode_path, "r") as f:
        state = np.asarray(f["observations/qpos"], dtype=np.float32)
        action = np.asarray(f["action"], dtype=np.float32) if "action" in f else state.copy()
    if state.ndim != 2 or state.shape[1] != 7:
        raise ValueError(f"{episode_path}: expected observations/qpos shape (T, 7), got {state.shape}")
    if action.shape != state.shape:
        raise ValueError(f"{episode_path}: expected action shape {state.shape}, got {action.shape}")
    return state, action


def main(
    source_dir: Path | str = "/mnt/workspace/xiajiawei/kai0/data/task_1_left_single_collect_001",
    output_dir: Path | str = "/mnt/workspace/xiajiawei/kai0/data/task_1_left_single_collect_001_lerobot",
    task_name: str = "grasp the cup and place it on the black plate",
    fps: int = 30,
    *,
    overwrite: bool = False,
    image_writer_threads: int = 4,
    include_third_camera: bool = False,
):
    """Convert left-single ARX HDF5 episodes to official LeRobot v2 format."""

    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    camera_video_dirs = _camera_video_dirs(include_third_camera)
    if not source_dir.exists():
        raise FileNotFoundError(source_dir)
    if output_dir.exists():
        if overwrite:
            shutil.rmtree(output_dir)
        else:
            raise FileExistsError(f"{output_dir} exists. Pass --overwrite to replace it.")

    episode_paths = sorted(source_dir.glob("episode_*.hdf5"), key=_episode_index)
    if not episode_paths:
        raise FileNotFoundError(f"No episode_*.hdf5 files under {source_dir}")

    dataset = LeRobotDataset.create(
        repo_id=output_dir.name,
        root=output_dir,
        fps=fps,
        robot_type="arx_left_single",
        features=_features(include_third_camera),
        use_videos=True,
        image_writer_threads=image_writer_threads,
        video_backend="pyav",
    )

    start = time.time()
    for ep_i, episode_path in enumerate(episode_paths):
        state, action = _load_episode_arrays(episode_path)
        videos = {
            key: _read_video_rgb(_video_path(source_dir, camera_dir, episode_path), len(state))
            for key, camera_dir in camera_video_dirs.items()
        }
        ep_len = min([len(state), len(action), *[len(v) for v in videos.values()]])
        if ep_len <= 0:
            raise RuntimeError(f"{episode_path}: empty episode after length alignment")

        print(f"[INFO] Converting {episode_path.name}: frames={ep_len} ({ep_i + 1}/{len(episode_paths)})")
        for t in range(ep_len):
            frame = {
                "observation.state": state[t],
                "action": action[t],
                "task": task_name,
            }
            for key, frames in videos.items():
                frame[key] = frames[t]
            dataset.add_frame(frame)
        dataset.save_episode()

    dataset.stop_image_writer()
    print(f"[INFO] Converted {len(episode_paths)} episodes to {output_dir}")
    print(f"[INFO] Time taken: {time.time() - start:.1f}s")


if __name__ == "__main__":
    tyro.cli(main)
