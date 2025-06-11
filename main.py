import json
import struct
from typing import List, Dict, Any
from copy import deepcopy

import numpy as np
from loguru import logger

from pose_visualizer import Pose3DVisualizer
from stream_to_unreal import create_unreal_sender_socket

from mediapipe2controlrig import calculate_control_rig_rotators
from utils import (
    deflatten,
    normalize_frame,
    mediapipe_to_unreal,
    create_camera_socket,
    get_datum_point,
    PoseObject,
    ILLEGAL_PTS_33_KEYPOINTS
)


def load_pose_frames_from_file(path: str, ignore_illegal=True) -> List[np.ndarray]:
    frames: List[np.ndarray] = []
    with open(path, "r", encoding="utf-8") as f:
        for ln in f:
            flatten_keypoints = list(map(float, ln.strip().split(',')))
            if len(flatten_keypoints) != 33*3:
                logger.warning(f"Invalid keypoints length: {len(flatten_keypoints)} in line: {ln.strip()}")
                continue
            keypoints = deflatten(flatten_keypoints)
            if ignore_illegal and (keypoints == ILLEGAL_PTS_33_KEYPOINTS):
                logger.warning(f"Skipping illegal keypoints: {keypoints}")
                continue
            # logger.debug(f"Loaded keypoints: {keypoints}")
            frames.append(np.array(keypoints, dtype=float))
    return frames

def preprocess_pose_file(path: str) -> List[Dict[str, Any]]:
    pose_frames = load_pose_frames_from_file(path)
    processed_pose: List[Dict[str, Any]] = []
    hip_datum_points = None
    for index, pose_frame in enumerate(pose_frames):
        normalized_keypoints, hip_offset, ground_offset = normalize_frame(pose_frame)
        unreal_keypoints = mediapipe_to_unreal(normalized_keypoints)
        if hip_datum_points is None:
            logger.info(f"Hip datum points initialize: {hip_datum_points}")
            hip_datum_points = get_datum_point(unreal_keypoints, datum_point="hip")

        control_rig_rotators = calculate_control_rig_rotators(unreal_keypoints, datum_points=hip_datum_points) # Using unreal coordinates

        payload = {"ControlRigRotators": control_rig_rotators, "ControlRig": control_rig_rotators}
        packet = PoseObject(payload).packet

        processed_pose.append(
            {
                "normalized_keypoints": normalized_keypoints,
                "packet": packet,
                "control_rig_rotators": control_rig_rotators,
                "hip_offset": hip_offset,
                "ground_offset": ground_offset,
                "frame_index": index,
                "unreal_keypoints": unreal_keypoints.tolist(),
                "hip_datum_points": hip_datum_points.tolist(),
            }
        )
    return processed_pose

def save_preprocessed_pose_to_file(path: str, frames: List[Dict[str, Any]]):
    _frames = deepcopy(frames)
    for frame in _frames:
        frame["normalized_keypoints"]  = frame["normalized_keypoints"].tolist()
        frame["packet"] = ""
        frame["control_rig_rotators"] = {k: v for k, v in frame["control_rig_rotators"].items()}
        frame["hip_offset"] = frame["hip_offset"].tolist()
        frame["ground_offset"] = frame["ground_offset"].tolist()

    with open(path, "w", encoding="utf-8") as f:
        json.dump(_frames, f, indent=4, ensure_ascii=False)
    logger.info(f"Preprocessed pose data saved to {path}, total frames: {len(_frames)}")

def play_frames_from_preprocessed_pose(pose_visualizer: Pose3DVisualizer, frames: List[Dict[str, Any]], fps=30, repeat=False, socket=None):
    if not frames:
        logger.warning("No frames to play.")
        return
    looping = True
    while looping:
        for fr in frames:
            pose_visualizer.update(fr["normalized_keypoints"].tolist())
            pose_visualizer.fig.canvas.flush_events()
            pose_visualizer.fig.canvas.draw()
            pose_visualizer.fig.waitforbuttonpress(1/fps)
            if socket:
                try:
                    socket.sendall(fr["packet"])
                except Exception as e:
                    logger.error(f"Error sending data: {e}")
                    socket.close()
                    return
        looping = repeat

def play_frame_from_udp(camera_socket, unreal_socket=None):
    hip_datum_points = None
    while True:
        try:
            data, _ = camera_socket.recvfrom(65536)
        except BlockingIOError:
            continue
        except Exception as e:
            print("[Camera] recv err:", e)
            continue

        try:
            msg = json.loads(data.decode())
            keypoints = np.asarray(msg["keypoints_3d"], dtype=msg["dtype"]).reshape((-1, 3))

            normalized_keypoints, hip_offset, ground_offset = normalize_frame(keypoints)
            unreal_keypoints = mediapipe_to_unreal(normalized_keypoints)
            if hip_datum_points is None:
                logger.info(f"Hip datum points initialize: {hip_datum_points}")
                hip_datum_points = get_datum_point(unreal_keypoints, datum_point="hip")

            control_rig_rotators = calculate_control_rig_rotators(unreal_keypoints,
                                                                  datum_points=hip_datum_points)  # Using unreal coordinates

            payload = {"ControlRigRotators": control_rig_rotators, "ControlRig": control_rig_rotators}
            logger.info(f"Control Rig Rotators: {control_rig_rotators}")
            packet = PoseObject(payload).packet

            if unreal_socket:
                try:
                    unreal_socket.sendall(packet)
                except Exception as e:
                    print("[Camera] Error sending data to Unreal:", e)
                    unreal_socket.close()
                    return

        except Exception as e:
            print("[Camera] JSON parse err:", e)
            continue

# ---------------------------------------------------------------------------
# 🏁  Main demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "preprocess"

    if mode not in ["play", "preprocess"]:
        print("Usage: python main.py [play|preprocess]")
        sys.exit(1)

    if mode == "play":
        camera_socket = create_camera_socket()
        unreal_socket = create_unreal_sender_socket()

        play_frame_from_udp(camera_socket, unreal_socket)

        camera_socket.close()
        unreal_socket.close()

    elif mode == "preprocess":
        FILE = "data/N_action_0.csv"

        preprocess_pose = preprocess_pose_file(FILE)
        save_preprocessed_pose_to_file("data/cache/N_action_0.json", preprocess_pose)

        ue_socket = create_unreal_sender_socket()
        visualizer = Pose3DVisualizer()

        play_frames_from_preprocessed_pose(visualizer, preprocess_pose, fps=240, repeat=True, socket=ue_socket)

        ue_socket.close()



