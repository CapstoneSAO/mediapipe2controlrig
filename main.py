import json
import struct
from typing import List, Dict, Any
from copy import deepcopy
import socket
import threading
import queue          # Python 標準函式庫
import time

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

CTRL_HOST = "0.0.0.0"
CTRL_PORT = 9001

ctrl_queue: queue.Queue[bytes] = queue.Queue(maxsize=32)   # 佇列大小可自行調整

def recv_exact(sock, nbytes):
    """阻塞讀取 nbytes；若連線中斷則回傳 None"""
    buf = bytearray()
    while len(buf) < nbytes:
        chunk = sock.recv(nbytes - len(buf))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def tcp_receiver(q: queue.Queue, host=CTRL_HOST, port=CTRL_PORT):
    """持續接收 TCP 資料並放進 queue；連線中斷就自動重連。"""
    while True:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            # sock.setblocking(True)                 # 阻塞式即可，反正放在 thread 裡
            # sock.connect((host, port))
            sock.bind((host, port))
            sock.listen(1)
            logger.info(f"[TCP] Connected {host}:{port}")
            conn, addr = sock.accept()
            while True:
                # data = sock.recv(4096)
                hdr = recv_exact(conn, 4)
                if hdr is None:
                    print("[Python] Converter 斷線")
                    break

                body_len = struct.unpack(">I", hdr)[0]
                body = recv_exact(conn, body_len)

                if body is None:
                    print("[Python] Converter 斷線")
                    break

                try:
                    obj = json.loads(body.decode("utf‑8").strip('\x00\r\n\t '))
                except Exception as e:
                    print("[Python] JSON 解析失敗：", e)
                    continue

                logger.info(f"[TCP] Received {obj} from {addr}")
                q.put(obj, block=False)           # 滿了就丟掉舊資料也行
        except Exception as e:
            logger.warning(f"[TCP] {e} → reconnect in 2 s")
            sock.close()
            time.sleep(2)

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

def play_frame_from_udp(camera_socket, unreal_socket=None, ctrl_queue=None):
    hip_datum_points = None
    last_signal = None
    while True:
        if ctrl_queue is not None:

            try:
                raw = ctrl_queue.get_nowait()  # 不阻塞；沒資料就丟 queue.Empty
                logger.debug(f"[TCP] 收到控制訊息: {raw}")
                last_signal = raw['signal']
                print(f"[{last_signal}]")
                # TODO: 依 ctrl_msg 更新 hip_datum_points、濾波器…等
            except queue.Empty:
                pass

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
            print(keypoints.tolist(), file=open("keypoints.txt", "a+"))

            normalized_keypoints, hip_offset, ground_offset = normalize_frame(keypoints)
            
            unreal_keypoints = mediapipe_to_unreal(normalized_keypoints, facing=("-Z", "+X", "+Y"))
            
            if (hip_datum_points is None) or (last_signal == "reset"):
                last_signal = None
                logger.info(f"Hip datum points initialize: {hip_datum_points}")
                hip_datum_points = get_datum_point(unreal_keypoints, datum_point="hip")

            control_rig_rotators = calculate_control_rig_rotators(unreal_keypoints,
                                                                  datum_points=hip_datum_points)  # Using unreal coordinates

            payload = {"ControlRigRotators": control_rig_rotators, "ControlRig": control_rig_rotators}
            # logger.info(f"Control Rig Rotators: {control_rig_rotators}")
            packet = PoseObject(payload).packet

            if unreal_socket:
                try:
                    unreal_socket.sendall(packet)
                except Exception as e:
                    print("[Camera] Error sending data to Unreal:", e)
                    unreal_socket = create_unreal_sender_socket()
                    # return

        except Exception as e:
            print("[Camera] JSON parse err:", e)
            continue

# ---------------------------------------------------------------------------
# 🏁  Main demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "play"

    if mode not in ["play", "preprocess"]:
        print("Usage: python main.py [play|preprocess]")
        sys.exit(1)

    threading.Thread(target=tcp_receiver,
                     args=(ctrl_queue,),
                     daemon=True).start()

    if mode == "play":
        camera_socket = create_camera_socket()
        unreal_socket = create_unreal_sender_socket()

        play_frame_from_udp(camera_socket, unreal_socket, ctrl_queue)

        camera_socket.close()
        unreal_socket.close()

    elif mode == "preprocess":
        FILE = "data/N_hand_cros.csv"

        preprocess_pose = preprocess_pose_file(FILE)
        save_preprocessed_pose_to_file("data/cache/N_hand_cros.json", preprocess_pose)

        ue_socket = create_unreal_sender_socket()
        visualizer = Pose3DVisualizer()

        play_frames_from_preprocessed_pose(visualizer, preprocess_pose, fps=240, repeat=True, socket=ue_socket)

        ue_socket.close()


