from typing import Dict, Any, List
import struct
import asyncio
import json
import socket

import numpy as np
from mediapipe.python.solutions.pose import PoseLandmark
from loguru import logger

CAMERA_HOST = "0.0.0.0"
CAMERA_PORT = 12346


def deflatten(flat: List[float]) -> List[List[float]]:
    return [flat[i : i + 3] for i in range(0, len(flat), 3) if i + 2 < len(flat)]

L_HIP, R_HIP = 23, 24
GROUND_PTS   = [27, 28, 31, 32]

ILLEGAL_FLAT_17_KEYPOINTS = [-1] * 51
ILLEGAL_FLAT_33_KEYPOINTS = [-1] * 99
ILLEGAL_PTS_17_KEYPOINTS  = deflatten(ILLEGAL_FLAT_17_KEYPOINTS)
ILLEGAL_PTS_33_KEYPOINTS  = deflatten(ILLEGAL_FLAT_33_KEYPOINTS)

class PoseObject:
    def __init__(self, payload: Dict[str, Any]):
        self.packet: bytes = b""
        self.update_payload(payload)

    def update_payload(self, payload: Dict[str, Any]):
        body   = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        self.packet = struct.pack(">I", len(body)) + body

    def __repr__(self):
        return f"PoseObject(packet_len={len(self.packet)})"

def normalize_frame(keypoints: np.ndarray):
    hip = (keypoints[PoseLandmark.LEFT_HIP.value] + keypoints[PoseLandmark.RIGHT_HIP.value]) / 2.0
    local = keypoints - hip
    ground_y = local[
        [PoseLandmark.LEFT_ANKLE.value,
         PoseLandmark.RIGHT_ANKLE.value,
         PoseLandmark.LEFT_FOOT_INDEX.value,
         PoseLandmark.RIGHT_FOOT_INDEX.value], ].min()
    local[:, 1] -= ground_y
    return local, hip, ground_y

def add_in_element_wise(lst: list[float], add: list[float]) -> list[float]:
    return [x + y for x, y in zip(lst, add)]

def mediapipe_to_unreal(keypoints: np.ndarray, facing: List[str] = ("-X", "+Z", "+Y")) -> np.ndarray:
    if facing == ("-X", "+Z", "+Y"):
        return np.array([[-z, x, y] for x, y, z in keypoints])

    return np.array([[-z, x, y] for x, y, z in keypoints])

def get_datum_point(keypoints: np.ndarray, datum_point: str = "hip") -> np.ndarray:
    if datum_point == "hip":
        return (keypoints[PoseLandmark.LEFT_HIP.value] + keypoints[PoseLandmark.RIGHT_HIP.value]) / 2.0
    elif datum_point == "ground":
        return keypoints[GROUND_PTS].mean(axis=0)
    else:
        raise ValueError(f"Unknown datum point: {datum_point}")



def create_camera_socket(host=CAMERA_HOST, recv_port=CAMERA_PORT):
    """Create a non-blocking UDP socket for receiving camera data."""

    _socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    _socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    _socket.bind((host, recv_port))
    _socket.setblocking(False)
    print(f"[Camera] UDP bound @ {host}:{recv_port} (non-blocking)")

    return _socket

if __name__ == "__main__":
    camera_socket = create_camera_socket()
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
            kp = np.asarray(msg["keypoints_3d"], dtype=msg["dtype"])
            kp = kp.reshape(msg["shape"])
        except Exception as e:
            print("[Camera] JSON parse err:", e)
            continue