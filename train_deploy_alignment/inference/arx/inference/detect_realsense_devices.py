#!/usr/bin/env python3
"""Detect and probe Intel RealSense and Orbbec devices.

Usage:
  python detect_realsense_devices.py
  python detect_realsense_devices.py --probe-stream
  python detect_realsense_devices.py --probe-stream --color-width 640 --color-height 480 --fps 30
  python detect_realsense_devices.py --visualize
"""

import argparse
import sys
import time


def _safe_info(dev, key, fallback="N/A"):
    try:
        if dev.supports(key):
            return dev.get_info(key)
    except Exception:
        pass
    return fallback


def _probe_device_stream(rs, serial, width, height, fps):
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)

    started = False
    try:
        pipeline.start(config)
        started = True
        ok_frames = 0
        start_t = time.time()
        while time.time() - start_t < 2.0:
            frames = pipeline.wait_for_frames(timeout_ms=1500)
            color = frames.get_color_frame()
            if color:
                ok_frames += 1
        return True, f"color stream OK ({ok_frames} frames in ~2s)"
    except Exception as e:
        return False, f"stream failed: {e}"
    finally:
        if started:
            try:
                pipeline.stop()
            except Exception:
                pass


def _resize_to_height(cv2, image, target_height):
    h, w = image.shape[:2]
    if h == target_height:
        return image
    target_width = max(1, int(w * target_height / h))
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)


def _safe_call(obj, method_name, fallback="N/A"):
    try:
        method = getattr(obj, method_name)
        return method()
    except Exception:
        return fallback


def _query_orbbec_devices(ob):
    ctx = ob.Context()
    device_list = ctx.query_devices()
    devices = []
    for i in range(device_list.get_count()):
        try:
            devices.append(device_list.get_device_by_index(i))
        except Exception:
            try:
                devices.append(device_list[i])
            except Exception as e:
                print(f"[WARN] Failed to access Orbbec device [{i}]: {e}")
    return devices


def _orbbec_device_info(device):
    info = device.get_device_info()
    return {
        "name": _safe_call(info, "get_name"),
        "serial": _safe_call(info, "get_serial_number"),
        "uid": _safe_call(info, "get_uid"),
        "firmware": _safe_call(info, "get_firmware_version"),
        "hardware": _safe_call(info, "get_hardware_version"),
        "vid": _safe_call(info, "get_vid"),
        "pid": _safe_call(info, "get_pid"),
        "connection": _safe_call(info, "get_connection_type"),
    }


def _select_orbbec_color_profile(pipeline, ob, width, height, fps):
    profile_list = pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
    for kwargs in (
        (width, height, ob.OBFormat.RGB, fps),
        (width, 0, ob.OBFormat.RGB, fps),
        (width, height, ob.OBFormat.MJPG, fps),
        (width, 0, ob.OBFormat.MJPG, fps),
    ):
        try:
            return profile_list.get_video_stream_profile(*kwargs)
        except Exception:
            pass
    return profile_list.get_default_video_stream_profile()


def _orbbec_frame_to_bgr(cv2, np, ob, frame):
    width = frame.get_width()
    height = frame.get_height()
    fmt = frame.get_format()
    data = np.asanyarray(frame.get_data())
    if fmt == ob.OBFormat.RGB:
        rgb = np.resize(data, (height, width, 3)).astype(np.uint8)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if fmt == ob.OBFormat.BGR:
        return np.resize(data, (height, width, 3)).astype(np.uint8)
    if fmt == ob.OBFormat.MJPG:
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    if fmt == ob.OBFormat.YUYV:
        yuyv = np.resize(data, (height, width, 2))
        return cv2.cvtColor(yuyv, cv2.COLOR_YUV2BGR_YUYV)
    if fmt == ob.OBFormat.UYVY:
        uyvy = np.resize(data, (height, width, 2))
        return cv2.cvtColor(uyvy, cv2.COLOR_YUV2BGR_UYVY)
    if fmt == ob.OBFormat.NV12:
        yuv = np.resize(data, (height * 3 // 2, width))
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV12)
    if fmt == ob.OBFormat.NV21:
        yuv = np.resize(data, (height * 3 // 2, width))
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR_NV21)
    print(f"[WARN] Unsupported Orbbec color format: {fmt}")
    return None


def _start_realsense_visual_streams(rs, devices, width, height, fps):
    streams = []
    for i, dev in enumerate(devices):
        serial = _safe_info(dev, rs.camera_info.serial_number)
        name = _safe_info(dev, rs.camera_info.name)
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)

        try:
            pipeline.start(config)
            streams.append(
                {
                    "backend": "RealSense",
                    "index": i,
                    "name": name,
                    "serial": serial,
                    "pipeline": pipeline,
                    "last_frame": None,
                    "frames": 0,
                }
            )
            print(f"  visualize     : started RealSense [{i}] {name} serial={serial}")
        except Exception as e:
            print(f"  visualize     : failed  RealSense [{i}] {name} serial={serial}: {e}")
    return streams


def _start_orbbec_visual_streams(ob, devices, width, height, fps):
    streams = []
    for i, device in enumerate(devices):
        info = _orbbec_device_info(device)
        pipeline = ob.Pipeline(device)
        config = ob.Config()
        try:
            color_profile = _select_orbbec_color_profile(pipeline, ob, width, height, fps)
            config.enable_stream(color_profile)
            pipeline.start(config)
            streams.append(
                {
                    "backend": "Orbbec",
                    "index": i,
                    "name": info["name"],
                    "serial": info["serial"],
                    "uid": info["uid"],
                    "pipeline": pipeline,
                    "last_frame": None,
                    "frames": 0,
                }
            )
            print(f"  visualize     : started Orbbec [{i}] {info['name']} serial={info['serial']} uid={info['uid']}")
        except Exception as e:
            try:
                pipeline.stop()
            except Exception:
                pass
            print(f"  visualize     : failed  Orbbec [{i}] {info['name']} serial={info['serial']} uid={info['uid']}: {e}")
    return streams


def _read_visual_stream_frame(stream, rs, ob, cv2, np):
    try:
        if stream["backend"] == "RealSense":
            frames = stream["pipeline"].poll_for_frames()
            color = frames.get_color_frame() if frames else None
            if color:
                rgb = np.asanyarray(color.get_data())
                stream["last_frame"] = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                stream["frames"] += 1
        else:
            frames = stream["pipeline"].wait_for_frames(50)
            color = frames.get_color_frame() if frames else None
            if color:
                bgr = _orbbec_frame_to_bgr(cv2, np, ob, color)
                if bgr is not None:
                    stream["last_frame"] = bgr
                    stream["frames"] += 1
    except Exception as e:
        now = time.time()
        if now - stream.get("last_warn", 0) > 2.0:
            print(f"[WARN] {stream['backend']} [{stream['index']}] read failed: {e}")
            stream["last_warn"] = now


def _visualize_device_streams(rs, realsense_devices, ob, orbbec_devices, width, height, fps, window_name):
    try:
        import cv2
        import numpy as np
    except Exception as e:
        print(f"[ERROR] visualization requires OpenCV and NumPy: {e}")
        print("Install in current env: pip install opencv-python numpy")
        return 1

    streams = []
    if rs is not None:
        streams.extend(_start_realsense_visual_streams(rs, realsense_devices, width, height, fps))
    if ob is not None:
        streams.extend(_start_orbbec_visual_streams(ob, orbbec_devices, width, height, fps))

    if not streams:
        print("[ERROR] No camera color stream could be started.")
        return 1

    print("Visualization running. Press q or Esc in the video window, or Ctrl+C in terminal, to exit.")
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            panels = []
            for stream in streams:
                _read_visual_stream_frame(stream, rs, ob, cv2, np)

                frame = stream["last_frame"]
                if frame is None:
                    frame = np.zeros((height, width, 3), dtype=np.uint8)
                    cv2.putText(
                        frame,
                        "waiting for frames...",
                        (16, height // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                else:
                    frame = frame.copy()

                label = f"{stream['backend']}[{stream['index']}] {stream['serial']}"
                cv2.putText(frame, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(
                    frame,
                    f"{stream['name']} frames={stream['frames']}",
                    (10, 58),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                panels.append(_resize_to_height(cv2, frame, height))

            canvas = np.hstack(panels)
            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break

            try:
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                break
    except KeyboardInterrupt:
        print("\nStopping visualization...")
    finally:
        for stream in streams:
            try:
                stream["pipeline"].stop()
            except Exception:
                pass
        cv2.destroyAllWindows()

    return 0


def main():
    parser = argparse.ArgumentParser(description="Detect RealSense/Orbbec cameras and optional stream visualization.")
    parser.add_argument("--probe-stream", action="store_true", help="Start a short color stream test for each device.")
    parser.add_argument("--visualize", action="store_true", help="Show live color video from all detected RealSense and Orbbec devices until closed.")
    parser.add_argument("--color-width", type=int, default=640)
    parser.add_argument("--color-height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--window-name", default="RealSense + Orbbec devices")
    parser.add_argument("--skip-realsense", action="store_true", help="Do not detect or visualize RealSense devices.")
    parser.add_argument("--skip-orbbec", action="store_true", help="Do not detect or visualize Orbbec devices.")
    args = parser.parse_args()

    rs = None
    realsense_devices = []
    if not args.skip_realsense:
        try:
            import pyrealsense2 as rs
        except Exception as e:
            print(f"[WARN] pyrealsense2 import failed: {e}")
            print("       RealSense detection disabled. Install in current env: pip install pyrealsense2")
        else:
            ctx = rs.context()
            realsense_devices = list(ctx.query_devices())

    ob = None
    orbbec_devices = []
    if not args.skip_orbbec:
        try:
            import pyorbbecsdk as ob
        except Exception as e:
            print(f"[WARN] pyorbbecsdk import failed: {e}")
            print("       Orbbec detection disabled. Build/install pyorbbecsdk in current env first.")
        else:
            try:
                orbbec_devices = _query_orbbec_devices(ob)
            except Exception as e:
                print(f"[WARN] Orbbec query_devices failed: {e}")

    print(f"REALSENSE_DEVICE_COUNT: {len(realsense_devices)}")
    for i, dev in enumerate(realsense_devices):
        name = _safe_info(dev, rs.camera_info.name)
        serial = _safe_info(dev, rs.camera_info.serial_number)
        fw = _safe_info(dev, rs.camera_info.firmware_version)
        product_line = _safe_info(dev, rs.camera_info.product_line)
        usb_type = _safe_info(dev, rs.camera_info.usb_type_descriptor)
        physical_port = _safe_info(dev, rs.camera_info.physical_port)
        product_id = _safe_info(dev, rs.camera_info.product_id)

        print(f"\n[RealSense {i}]")
        print(f"  name          : {name}")
        print(f"  serial        : {serial}")
        print(f"  firmware      : {fw}")
        print(f"  product_line  : {product_line}")
        print(f"  product_id    : {product_id}")
        print(f"  usb_type      : {usb_type}")
        print(f"  physical_port : {physical_port}")

        try:
            sensors = dev.query_sensors()
            sensor_names = []
            for s in sensors:
                if s.supports(rs.camera_info.name):
                    sensor_names.append(s.get_info(rs.camera_info.name))
                else:
                    sensor_names.append("unknown_sensor")
            print(f"  sensors       : {', '.join(sensor_names) if sensor_names else 'N/A'}")
        except Exception as e:
            print(f"  sensors       : query failed ({e})")

        if args.probe_stream:
            ok, msg = _probe_device_stream(
                rs,
                serial=serial,
                width=args.color_width,
                height=args.color_height,
                fps=args.fps,
            )
            status = "PASS" if ok else "FAIL"
            print(f"  probe_stream  : {status} - {msg}")

    print(f"\nORBBEC_DEVICE_COUNT: {len(orbbec_devices)}")
    for i, dev in enumerate(orbbec_devices):
        info = _orbbec_device_info(dev)
        print(f"\n[Orbbec {i}]")
        print(f"  name          : {info['name']}")
        print(f"  serial        : {info['serial']}")
        print(f"  uid           : {info['uid']}")
        print(f"  firmware      : {info['firmware']}")
        print(f"  hardware      : {info['hardware']}")
        print(f"  vid           : {info['vid']}")
        print(f"  pid           : {info['pid']}")
        print(f"  connection    : {info['connection']}")

        try:
            sensor_list = dev.get_sensor_list()
            sensor_names = []
            for idx in range(sensor_list.get_count()):
                try:
                    sensor_names.append(str(sensor_list.get_type_by_index(idx)))
                except Exception:
                    sensor_names.append("unknown_sensor")
            print(f"  sensors       : {', '.join(sensor_names) if sensor_names else 'N/A'}")
        except Exception as e:
            print(f"  sensors       : query failed ({e})")

    if not realsense_devices and not orbbec_devices:
        print("\nNo RealSense or Orbbec device detected.")
        sys.exit(0)

    if args.visualize:
        sys.exit(
            _visualize_device_streams(
                rs,
                realsense_devices=realsense_devices,
                ob=ob,
                orbbec_devices=orbbec_devices,
                width=args.color_width,
                height=args.color_height,
                fps=args.fps,
                window_name=args.window_name,
            )
        )


if __name__ == "__main__":
    main()
