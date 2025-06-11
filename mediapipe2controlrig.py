from typing import Dict, List
from copy import deepcopy

import numpy
import numpy as np
from mediapipe.python.solutions.pose import PoseLandmark
from loguru import logger

from utils import add_in_element_wise

default_rotators = {
        "body_ctrl":          [0.0, 0.0, 0.0],
        "body_ctrl_pos":     [0.0, 0.0, 0.0],  # Position for Body Control (offset)
        "spine_01_ctrl":      [0.0, 0.0, 0.0],
        "head_ctrl":          [0.0, 0.0, 0.0],
        "upperarm_r_fk_ctrl": [0.0, 0.0, 0.0],
        "upperarm_l_fk_ctrl": [0.0, 0.0, 0.0],
        "lowerarm_r_fk_ctrl": [0.0, 0.0, 0.0],
        "lowerarm_l_fk_ctrl": [0.0, 0.0, 0.0],
        "hand_r_fk_ctrl":     [0.0, 0.0, 0.0],
        "hand_l_fk_ctrl":     [0.0, 0.0, 0.0],
        "leg_r_pv_ik_ctrl":   [0.0, 0.0, 0.0],
        "leg_r_pv_ik_ctrl_pos": [0.0, 0.0, 0.0],  # Position for IK Pole Vector
        "foot_r_ik_ctrl":     [0.0, 0.0, 0.0],
        "foot_r_ik_ctrl_pos": [0.0, 0.0, 0.0],  # Position for IK Foot Control
        "leg_l_pv_ik_ctrl":   [0.0, 0.0, 0.0],
        "leg_l_pv_ik_ctrl_pos": [0.0, 0.0, 0.0],  # Position for IK Pole Vector
        "foot_l_ik_ctrl":     [0.0, 0.0, 0.0],
        "foot_l_ik_ctrl_pos": [0.0, 0.0, 0.0],  # Position for IK Foot Control
    }

t_pose_rotators_offsets = {
    "upperarm_r_fk_ctrl": [0.0, 55.0, 0.0],
    "upperarm_l_fk_ctrl": [0.0, 55.0, 0.0],
    "lowerarm_r_fk_ctrl": [0.0, 0.0, -35.0],
    "lowerarm_l_fk_ctrl": [0.0, 0.0, -35.0],
}

def _meter_to_centimeter(v: np.array):
    return v[0] * 100, v[1] * 100, v[2] * 100

def _quat_between(v_from: np.ndarray, v_to: np.ndarray) -> np.ndarray:
    """
    Compute the unit quaternion that rotates v_from onto v_to.
    Both vectors must already be normalised (|v| == 1).

    Returns q = [w, x, y, z] as a NumPy array.
    """
    dot = np.dot(v_from, v_to)
    if dot < -0.999999:                        # 180° – choose an orthogonal axis
        axis = np.cross(v_from, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:        # If parallel to X, fall back to Y
            axis = np.cross(v_from, np.array([0.0, 1.0, 0.0]))
        axis = axis / np.linalg.norm(axis)
        return np.array([0.0, *axis])          # w = 0  → 180° rotation
    axis = np.cross(v_from, v_to)
    w = 1.0 + dot
    quat = np.array([w, *axis])
    return quat / np.linalg.norm(quat)

def _quat_inv(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([w, -x, -y, -z])

def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:  # q1⊗q2
    w1,x1,y1,z1 = q1; w2,x2,y2,z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

def _quat_to_euler_xyz(q: np.ndarray) -> List[float]:
    w, x, y, z = q
    # Roll (X-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.degrees(np.arctan2(sinr_cosp, cosr_cosp))

    # Pitch (Y-axis rotation)
    sinp = 2 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.degrees(np.arcsin(sinp))

    # Yaw (Z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.degrees(np.arctan2(siny_cosp, cosy_cosp))

    return [roll, pitch, yaw]

def _build_torso_basis(keypoints: np.ndarray):
    LS = keypoints[PoseLandmark.LEFT_SHOULDER.value]
    RS = keypoints[PoseLandmark.RIGHT_SHOULDER.value]
    M_S = LS + RS

    RH = keypoints[PoseLandmark.RIGHT_HIP.value]
    LH = keypoints[PoseLandmark.LEFT_HIP.value]

    HIP = (RH + LH) / 2

    x_axis = LS - RS
    x_axis /= np.linalg.norm(x_axis) + 1e-8

    z_axis = M_S - HIP
    z_axis /= np.linalg.norm(z_axis) + 1e-8

    y_axis = np.cross(x_axis, z_axis)
    y_axis /= np.linalg.norm(y_axis) + 1e-8

    return np.vstack([x_axis, -y_axis, z_axis])

def calculate_upperarm_r_fk_ctrl_rotators(keypoints: np.ndarray) -> List[float]:
    sh = numpy.array(keypoints[PoseLandmark.RIGHT_SHOULDER.value])
    el = numpy.array(keypoints[PoseLandmark.RIGHT_ELBOW.value])

    _T = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]
    )

    v_cur = el - sh
    v_cur /= np.linalg.norm(v_cur) + 1e-8

    # T = _build_torso_basis(keypoints)
    # v_cur = T.T @ v_cur  # 轉進胸腔空間

    v_ref = np.array([-1.0, 0.0, 0.0])


    q = _quat_between(v_ref, v_cur)

    euler_rpy = _quat_to_euler_xyz(q)

    return add_in_element_wise(
        euler_rpy,
        t_pose_rotators_offsets["upperarm_r_fk_ctrl"],
    )

def calculate_upperarm_l_fk_ctrl_rotators(keypoints: np.ndarray) -> List[float]:
    sh = np.array(keypoints[PoseLandmark.LEFT_SHOULDER.value])
    el = np.array(keypoints[PoseLandmark.LEFT_ELBOW.value])
    v_cur = el - sh
    v_cur /= np.linalg.norm(v_cur) + 1e-8

    # T-pose 向量，手往左指
    v_ref = np.array([1.0, 0.0, 0.0])

    q = _quat_between(v_cur, v_ref)

    euler_rpy = _quat_to_euler_xyz(q)

    return add_in_element_wise(
        euler_rpy,
        t_pose_rotators_offsets["upperarm_l_fk_ctrl"],
    )

def calculate_lowerarm_r_fk_ctrl_rotators(keypoints: np.ndarray) -> List[float]:
    sh = keypoints[PoseLandmark.RIGHT_SHOULDER.value]
    el = keypoints[PoseLandmark.RIGHT_ELBOW.value]
    wr = keypoints[PoseLandmark.RIGHT_WRIST.value]

    v_upper = el - sh
    v_fore  = wr - el
    v_upper /= np.linalg.norm(v_upper) + 1e-8
    v_fore  /= np.linalg.norm(v_fore)  + 1e-8

    v_ref = np.array([-1.0, 0.0, 0.0])  # T-pose 向量，手往右指

    #四元數
    q_upper = _quat_between(v_ref, v_upper)
    q_fore  = _quat_between(v_ref, v_fore)

    #局部四元數
    q_rel = _quat_mul(_quat_inv(q_upper), q_fore)

    #Euler
    euler_rpy = _quat_to_euler_xyz(q_rel)

    return add_in_element_wise(
        euler_rpy,
        t_pose_rotators_offsets["lowerarm_r_fk_ctrl"]
    )

def calculate_lowerarm_l_fk_ctrl_rotators(keypoints: np.ndarray) -> List[float]:
    sh = keypoints[PoseLandmark.LEFT_SHOULDER.value]
    el = keypoints[PoseLandmark.LEFT_ELBOW.value]
    wr = keypoints[PoseLandmark.LEFT_WRIST.value]

    v_upper = el - sh
    v_fore  = wr - el
    v_upper /= np.linalg.norm(v_upper) + 1e-8
    v_fore  /= np.linalg.norm(v_fore)  + 1e-8

    #四元數
    v_ref = np.array([1.0, 0.0, 0.0]) # 左臂 T-Pose 指 +X
    # q_upper = _quat_between(v_ref, v_upper)
    q_upper = _quat_between(v_upper, v_ref)
    # q_fore  = _quat_between(v_ref, v_fore)
    q_fore = _quat_between(v_fore, v_ref)

    #局部四元數
    q_rel = _quat_mul(_quat_inv(q_upper), q_fore)

    euler_rpy = _quat_to_euler_xyz(q_rel)

    return add_in_element_wise(
        euler_rpy,
        t_pose_rotators_offsets["lowerarm_l_fk_ctrl"]
    )

def _build_rotation_from_axes(x_axis, z_axis):
    x = x_axis / (np.linalg.norm(x_axis) + 1e-8)
    z = z_axis / (np.linalg.norm(z_axis) + 1e-8)

    # ★ 左手系：y = x × z  (而不是 z × x)
    y = np.cross(x, z)

    R = np.column_stack([x, y, z])
    w = np.sqrt(1.0 + R[0,0] + R[1,1] + R[2,2]) / 2.0
    q = np.array([
        w,
        (R[2,1] - R[1,2]) / (4*w),
        (R[0,2] - R[2,0]) / (4*w),
        (R[1,0] - R[0,1]) / (4*w)
    ])
    return q

def calculate_leg_r_ik_ctrls(keypoints: np.ndarray):
    H = keypoints[PoseLandmark.RIGHT_HIP.value]
    K = keypoints[PoseLandmark.RIGHT_KNEE.value]
    A = keypoints[PoseLandmark.RIGHT_ANKLE.value]
    F = keypoints[PoseLandmark.RIGHT_FOOT_INDEX.value]
    B = keypoints[PoseLandmark.RIGHT_HEEL.value]

    x_axis = -(F - A)
    z_axis = A - B
    q_foot = _build_rotation_from_axes(x_axis, z_axis)
    foot_rot = _quat_to_euler_xyz(q_foot)

    n = np.cross(A - K, H - K)
    n /= np.linalg.norm(n) + 1e-8
    leg_len = np.linalg.norm(H - A)
    pv_pos = K + n * 0.5 * leg_len

    foot_pos = _meter_to_centimeter(A.tolist())
    foot_pos_x, foot_pos_y, foot_pos_z = foot_pos
    foot_pos_y = -foot_pos_y
    foot_pos = (foot_pos_x, foot_pos_y, foot_pos_z)

    pv_pos   = _meter_to_centimeter(pv_pos.tolist())
    pv_pos_x, pv_pos_y, pv_pos_z = pv_pos
    pv_pos_y = -pv_pos_y
    pv_pos = (pv_pos_x, pv_pos_y, pv_pos_z)

    return {
        "foot_r_ik_ctrl": {
            "location": foot_pos,
            "rotator":  foot_rot
        },
        "leg_r_pv_ik_ctrl": {
            "location": pv_pos
        }
    }

def calculate_leg_l_ik_ctrls(keypoints: np.ndarray):
    H = keypoints[PoseLandmark.LEFT_HIP.value]
    K = keypoints[PoseLandmark.LEFT_KNEE.value]
    A = keypoints[PoseLandmark.LEFT_ANKLE.value]
    F = keypoints[PoseLandmark.LEFT_FOOT_INDEX.value]
    B = keypoints[PoseLandmark.LEFT_HEEL.value]

    x_axis = (F - A)
    z_axis = A - B
    q_foot = _build_rotation_from_axes(x_axis, z_axis)
    foot_rot = _quat_to_euler_xyz(q_foot)

    n = np.cross(A - K, H - K)
    n /= np.linalg.norm(n) + 1e-8
    leg_len = np.linalg.norm(H - A)
    pv_pos = K + n * 0.5 * leg_len

    foot_pos = _meter_to_centimeter(A.tolist())
    foot_pos_x, foot_pos_y, foot_pos_z = foot_pos
    foot_pos_y = -foot_pos_y
    foot_pos = (foot_pos_x, foot_pos_y, foot_pos_z)

    pv_pos   = _meter_to_centimeter(pv_pos.tolist())
    pv_pos_x, pv_pos_y, pv_pos_z = pv_pos
    pv_pos_y = -pv_pos_y
    pv_pos = (pv_pos_x, pv_pos_y, pv_pos_z)

    return {
        "foot_l_ik_ctrl": {
            "location": foot_pos,
            "rotator":  foot_rot
        },
        "leg_l_pv_ik_ctrl": {
            "location": pv_pos
        }
    }

def calculate_control_rig_rotators(keypoints: np.ndarray, datum_points: np.array=None) -> Dict[str, List[float]]:
    rotator = default_rotators.copy()

    rotator["upperarm_r_fk_ctrl"] = calculate_upperarm_r_fk_ctrl_rotators(keypoints)
    rotator["lowerarm_r_fk_ctrl"] = calculate_lowerarm_r_fk_ctrl_rotators(keypoints)
    rotator["upperarm_l_fk_ctrl"] = calculate_upperarm_l_fk_ctrl_rotators(keypoints)
    rotator["lowerarm_l_fk_ctrl"] = calculate_lowerarm_l_fk_ctrl_rotators(keypoints)

    leg_r_ctrls = calculate_leg_r_ik_ctrls(keypoints)
    # rotator["foot_r_ik_ctrl"] = leg_r_ctrls["foot_r_ik_ctrl"]["rotator"]
    rotator["foot_r_ik_ctrl_pos"] = leg_r_ctrls["foot_r_ik_ctrl"]["location"]
    # rotator["leg_r_pv_ik_ctrl_pos"] = leg_r_ctrls["leg_r_pv_ik_ctrl"]["location"]
    leg_l_ctrls = calculate_leg_l_ik_ctrls(keypoints)
    # rotator["foot_l_ik_ctrl"] = leg_l_ctrls["foot_l_ik_ctrl"]["rotator"]
    rotator["foot_l_ik_ctrl_pos"] = leg_l_ctrls["foot_l_ik_ctrl"]["location"]
    # rotator["leg_l_pv_ik_ctrl_pos"] = leg_l_ctrls["leg_l_pv_ik_ctrl"]["location"]

    if datum_points is not None:
        hip = keypoints[PoseLandmark.LEFT_HIP.value] + keypoints[PoseLandmark.RIGHT_HIP.value]
        hip /= 2.0
        hip_z_offset = (hip[2] - datum_points[2]) * 100 # Convert to centimeters
        rotator["body_ctrl_pos"] = (0, 0, hip_z_offset)

    return rotator