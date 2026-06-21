#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
"""Visualize saved Orbbec depth and convert one RGB-D frame to a colored point cloud."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


DEFAULT_DATASET_DIR = str(Path(__file__).resolve().parents[3] / "kai0_data" / "arx_teleop")
TRUE_STRINGS = {"1", "true", "t", "yes", "y", "on", "ture"}
FALSE_STRINGS = {"0", "false", "f", "no", "n", "off", "none"}


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in TRUE_STRINGS:
        return True
    if value in FALSE_STRINGS:
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {value}")


def read_rgb_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open RGB video: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            raise RuntimeError(f"Failed to read frame {frame_idx} from {video_path}")
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def read_depth_png(depth_path: Path) -> np.ndarray:
    depth = cv2.imread(str(depth_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"Failed to read depth PNG: {depth_path}")
    if depth.dtype != np.uint16:
        raise RuntimeError(f"Expected uint16 depth PNG, got {depth.dtype}: {depth_path}")
    return depth


def load_meta(meta_path: Path) -> dict:
    with meta_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def intrinsic_to_dict(intrinsic):
    return {
        "fx": float(intrinsic.fx),
        "fy": float(intrinsic.fy),
        "cx": float(intrinsic.cx),
        "cy": float(intrinsic.cy),
        "width": int(intrinsic.width),
        "height": int(intrinsic.height),
    }


def distortion_to_dict(distortion):
    return {
        "k1": float(distortion.k1),
        "k2": float(distortion.k2),
        "k3": float(distortion.k3),
        "k4": float(distortion.k4),
        "k5": float(distortion.k5),
        "k6": float(distortion.k6),
        "p1": float(distortion.p1),
        "p2": float(distortion.p2),
    }


def transform_to_dict(transform):
    return {
        "rotation": np.asarray(transform.rot, dtype=float).reshape(3, 3).tolist(),
        "translation": np.asarray(transform.transform, dtype=float).reshape(3).tolist(),
    }


def camera_param_to_dict(camera_param):
    return {
        "depth_intrinsic": intrinsic_to_dict(camera_param.depth_intrinsic),
        "rgb_intrinsic": intrinsic_to_dict(camera_param.rgb_intrinsic),
        "depth_distortion": distortion_to_dict(camera_param.depth_distortion),
        "rgb_distortion": distortion_to_dict(camera_param.rgb_distortion),
        "depth_to_rgb_transform": transform_to_dict(camera_param.transform),
        "notes": {
            "intrinsics_source": "OrbbecSDK factory calibration",
            "pointcloud_intrinsic_when_aligned_to_color": "rgb_intrinsic",
            "pointcloud_intrinsic_when_not_aligned": "depth_intrinsic",
        },
    }


def read_live_orbbec_camera_param():
    from pyorbbecsdk import Config, OBSensorType, Pipeline

    pipeline = Pipeline()
    config = Config()
    color_profile = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR).get_default_video_stream_profile()
    depth_profile = pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR).get_default_video_stream_profile()
    config.enable_stream(color_profile)
    config.enable_stream(depth_profile)
    pipeline.start(config)
    try:
        return camera_param_to_dict(pipeline.get_camera_param())
    finally:
        pipeline.stop()


def ensure_camera_param(meta: dict, meta_path: Path, allow_live_intrinsics: bool, update_meta: bool) -> dict:
    if meta.get("camera_param"):
        return meta
    if not allow_live_intrinsics:
        return meta
    print("[WARN] Meta has no camera_param; trying to read intrinsics from the connected Orbbec camera.")
    camera_param = read_live_orbbec_camera_param()
    meta["camera_param"] = camera_param
    meta["camera_param_source"] = "live_orbbecsdk_fallback"
    if update_meta:
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Updated meta with live Orbbec intrinsics: {meta_path}")
    return meta


def select_intrinsic(meta: dict) -> dict:
    camera_param = meta.get("camera_param") or {}
    if meta.get("aligned_to_color", False) and camera_param.get("rgb_intrinsic"):
        return camera_param["rgb_intrinsic"]
    if camera_param.get("depth_intrinsic"):
        return camera_param["depth_intrinsic"]
    if camera_param.get("rgb_intrinsic"):
        return camera_param["rgb_intrinsic"]
    raise RuntimeError(f"No camera intrinsics found in meta. Meta keys: {sorted(meta.keys())}")


def colorize_depth(depth_mm: np.ndarray, min_mm=None, max_mm=None):
    valid = depth_mm[depth_mm > 0]
    if valid.size == 0:
        raise RuntimeError("Depth image has no valid non-zero pixels.")

    if min_mm is None:
        min_mm = float(np.percentile(valid, 1))
    if max_mm is None:
        max_mm = float(np.percentile(valid, 99))
    if max_mm <= min_mm:
        max_mm = min_mm + 1.0

    clipped = np.clip(depth_mm.astype(np.float32), min_mm, max_mm)
    normalized = ((clipped - min_mm) / (max_mm - min_mm) * 255.0).astype(np.uint8)
    colormap = cv2.COLORMAP_TURBO if hasattr(cv2, "COLORMAP_TURBO") else cv2.COLORMAP_JET
    color_bgr = cv2.applyColorMap(normalized, colormap)
    color_bgr[depth_mm == 0] = 0
    return color_bgr, min_mm, max_mm


def depth_rgb_to_pointcloud(depth_mm: np.ndarray, rgb: np.ndarray, intrinsic: dict, min_mm: float, max_mm: float, max_points: int):
    if rgb.shape[:2] != depth_mm.shape[:2]:
        rgb = cv2.resize(rgb, (depth_mm.shape[1], depth_mm.shape[0]), interpolation=cv2.INTER_AREA)

    fx = float(intrinsic["fx"])
    fy = float(intrinsic["fy"])
    cx = float(intrinsic["cx"])
    cy = float(intrinsic["cy"])

    mask = (depth_mm > 0) & (depth_mm >= min_mm) & (depth_mm <= max_mm)
    vs, us = np.nonzero(mask)
    if vs.size == 0:
        raise RuntimeError("No valid depth pixels left after min/max filtering.")

    if max_points > 0 and vs.size > max_points:
        idx = np.linspace(0, vs.size - 1, max_points).astype(np.int64)
        vs = vs[idx]
        us = us[idx]

    z = depth_mm[vs, us].astype(np.float32) / 1000.0
    x = (us.astype(np.float32) - cx) * z / fx
    y = (vs.astype(np.float32) - cy) * z / fy
    points = np.stack([x, y, z], axis=1)
    colors = rgb[vs, us].astype(np.uint8)
    return points, colors


def write_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(points, colors):
            f.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def show_pointcloud(points: np.ndarray, colors: np.ndarray) -> bool:
    try:
        import open3d as o3d
    except ImportError:
        print("[WARN] open3d is not installed; saved PLY only.")
        return False

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    pc.colors = o3d.utility.Vector3dVector(colors.astype(np.float64) / 255.0)
    o3d.visualization.draw_geometries([pc], window_name="ARX Orbbec RGB point cloud")
    return True


def main():
    parser = argparse.ArgumentParser(description="Create depth preview and RGB point cloud from one saved ARX Orbbec frame.")
    parser.add_argument("--dataset_dir", default=DEFAULT_DATASET_DIR)
    parser.add_argument("--dataset_name", required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame_idx", type=int, default=0)
    parser.add_argument("--camera", default="cam_high")
    parser.add_argument("--depth_min_mm", type=float, default=None)
    parser.add_argument("--depth_max_mm", type=float, default=None)
    parser.add_argument("--max_points", type=int, default=200000)
    parser.add_argument("--show", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--allow_live_intrinsics", type=str2bool, nargs="?", const=True, default=True)
    parser.add_argument("--update_meta", type=str2bool, nargs="?", const=True, default=True)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_dir).expanduser() / args.dataset_name
    video_path = dataset_root / "video" / args.camera / f"{args.episode}.mp4"
    depth_path = dataset_root / "depth" / args.camera / str(args.episode) / f"{args.frame_idx:06d}.png"
    meta_path = dataset_root / "depth" / args.camera / f"{args.episode}_meta.json"
    out_dir = dataset_root / "preview" / args.camera / str(args.episode)

    meta = ensure_camera_param(load_meta(meta_path), meta_path, args.allow_live_intrinsics, args.update_meta)
    intrinsic = select_intrinsic(meta)
    rgb = read_rgb_frame(video_path, args.frame_idx)
    depth = read_depth_png(depth_path)

    depth_vis, used_min, used_max = colorize_depth(depth, args.depth_min_mm, args.depth_max_mm)
    points, colors = depth_rgb_to_pointcloud(depth, rgb, intrinsic, used_min, used_max, args.max_points)

    out_dir.mkdir(parents=True, exist_ok=True)
    depth_vis_path = out_dir / f"{args.frame_idx:06d}_depth_color.png"
    ply_path = out_dir / f"{args.frame_idx:06d}_rgb_pointcloud.ply"
    cv2.imwrite(str(depth_vis_path), depth_vis)
    write_ply(ply_path, points, colors)

    print(f"[INFO] Depth preview: {depth_vis_path}")
    print(f"[INFO] RGB point cloud: {ply_path}")
    print(f"[INFO] Points: {len(points)}")
    print(f"[INFO] Depth display range: {used_min:.1f} mm -> {used_max:.1f} mm")
    print(f"[INFO] Intrinsic: fx={intrinsic['fx']}, fy={intrinsic['fy']}, cx={intrinsic['cx']}, cy={intrinsic['cy']}")

    if args.show:
        show_pointcloud(points, colors)


if __name__ == "__main__":
    main()
