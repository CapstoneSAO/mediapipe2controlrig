from typing import Dict, List
from copy import deepcopy

import numpy
import numpy as np
from mediapipe.python.solutions.pose import PoseLandmark
from loguru import logger

from utils import add_in_element_wise

frame_count = 0

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

def _quat_to_euler_xyz_lhs(q: np.ndarray) -> List[float]:
    """
    左手座標 (Roll=X, Pitch=Y, Yaw=Z, 計算順序 X→Y→Z)。
    公式只在原函式基礎上 **Yaw 取負號**。
    """
    w, x, y, z = q
    # Roll (X)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.degrees(np.arctan2(sinr_cosp, cosr_cosp))

    # Pitch (Y)
    sinp = 2 * (w * y - z * x)
    sinp = np.clip(sinp, -1.0, 1.0)
    pitch = np.degrees(np.arcsin(sinp))

    # Yaw (Z) ——— 左手系符號顛倒
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = -np.degrees(np.arctan2(siny_cosp, cosy_cosp))   # ← 取負號

    return [roll, pitch, yaw]

def _build_torso_basis(keypoints: np.ndarray):
    LS = keypoints[PoseLandmark.LEFT_SHOULDER.value]
    RS = keypoints[PoseLandmark.RIGHT_SHOULDER.value]
    M_S = ( LS + RS ) / 2

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

_prev_y = None  # 用於穩定化胸腔基底的 Y 軸方向

def torso_basis_stable(kp: np.ndarray) -> np.ndarray:
    global _prev_y
    LS, RS = kp[PoseLandmark.LEFT_SHOULDER.value], kp[PoseLandmark.RIGHT_SHOULDER.value]
    LH, RH = kp[PoseLandmark.LEFT_HIP.value],     kp[PoseLandmark.RIGHT_HIP.value]

    x = LS - RS; x /= np.linalg.norm(x)+1e-8              # +X 左
    z_tmp = (LS+RS)/2 - (LH+RH)/2                         # 肩→髖
    z =  z_tmp / (np.linalg.norm(z_tmp)+1e-8)             # +Z 上
    y = np.cross(z, x); y /= np.linalg.norm(y)+1e-8       # 左手系 y = z×x

    # 半球檢查：若 y 與上一幀夾角 > 90°，整組取負號
    ref = _prev_y if _prev_y is not None else np.array([0,1,0])
    if np.dot(y, ref) < 0:  x, y, z = -x, -y, -z
    _prev_y = y

    # 重新正交化一次保險
    z = np.cross(x, y); z /= np.linalg.norm(z)+1e-8
    x = np.cross(y, z); x /= np.linalg.norm(x)+1e-8
    return np.column_stack([x, -y, z])      # X Y Z 為**列**

def _quat_from_axis_angle(axis: np.ndarray, degrees: float) -> np.ndarray:
    """
    左手座標四元數 (w, x, y, z) ；axis 必須已單位化
    """
    theta = np.deg2rad(degrees) * 0.5
    return np.concatenate([[np.cos(theta)], np.sin(theta) * axis])

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
    T = torso_basis_stable(keypoints)  # 使用穩定化的胸腔基底
    v_cur = T.T @ v_cur  # 轉進胸腔空間

    global frame_count
    print(f"frame {frame_count}")
    print(f"v_cur: {v_cur}")

    # v_ref = np.array([-1.0, 0.0, 0.0])

    # q = _quat_between(v_ref, v_cur)

    # euler_rpy = _quat_to_euler_xyz(q)

    # temp = add_in_element_wise(
    #     euler_rpy,
    #     t_pose_rotators_offsets["upperarm_r_fk_ctrl"],
    # )


    # print(f"temp: {temp}")
    
    # (1) 動作四元數：T-Pose 水平(-X) → 目前 v_cur
    q_anim = _quat_between(np.array([-1.0, 0.0, 0.0]), v_cur)

    # (2) A-Pose→T-Pose offset：繞 +Y 55°
    q_offset = _quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), 55.0)

    # (3) 合成：先 offset 再 anim
    q_total = _quat_mul(q_offset, q_anim)
    return _quat_to_euler_xyz_lhs(q_total)

def calculate_upperarm_l_fk_ctrl_rotators(keypoints: np.ndarray) -> List[float]:

    sh = np.array(keypoints[PoseLandmark.LEFT_SHOULDER.value])
    el = np.array(keypoints[PoseLandmark.LEFT_ELBOW.value])
    v_cur = el - sh
    v_cur /= np.linalg.norm(v_cur) + 1e-8

    T = torso_basis_stable(keypoints)  # 使用穩定化的胸腔基底
    v_cur = T.T @ v_cur  # 轉進胸腔空間
    
    # (1) 動作四元數：T-Pose 水平(-X) → 目前 v_cur
    q_anim = _quat_between(v_cur, np.array([1.0, 0.0, 0.0]))

    # (2) A-Pose→T-Pose offset：繞 +Y 55°
    q_offset = _quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), 55.0)

    # (3) 合成：先 offset 再 anim
    q_total = _quat_mul(q_offset, q_anim)
    return _quat_to_euler_xyz_lhs(q_total)

_rest_q = {"R": None, "L": None}
_vel_buf = {"R": [], "L": []}      # 簡單平均速度緩衝
VEL_THR  = 0.01                    # 四元數 L2 範數
REST_FR  = 10                      # 連續靜止幀數
_prev_v_up = {"R": None, "L": None}

def _update_q_rest(side: str, q_rel: np.ndarray, v_up_curr: np.ndarray):
    """
    只有「前臂幾乎不動」且「上臂也幾乎不動」時，
    才把目前 q_rel 記為靜止肘角。
    """
    buf = _vel_buf[side]
    buf.append(q_rel)
    if len(buf) > REST_FR: buf.pop(0)

    #―― 檢查兩種速度 ――#
    if len(buf) == REST_FR:
        slow_rel = all(np.linalg.norm(buf[i]-buf[i-1]) < VEL_THR for i in range(1, REST_FR))
        # 用上一幀的 v_up (存在 _prev_v_up[side]) 判斷上臂是否也很穩
        prev_up = _prev_v_up[side]
        slow_up = np.linalg.norm(v_up_curr - prev_up) < 0.02 if prev_up is not None else False
        if slow_rel and slow_up:
            _rest_q[side] = q_rel.copy()
    _prev_v_up[side] = v_up_curr.copy()
    
def quat_offset_AT(v_upper: np.ndarray, deg=-35.0) -> np.ndarray:
    """沿 v_upper 本身旋轉 deg°（左手座標）。"""
    v = v_upper / (np.linalg.norm(v_upper) + 1e-8)
    half = np.deg2rad(deg) * 0.5
    return np.concatenate([[np.cos(half)],  np.sin(half) * v])

def quat_from_forward_up_lhs(f: np.ndarray, u_raw: np.ndarray) -> np.ndarray:
    """左手座標系專用：x=f, y=u, z= x×y  (注意外積順序)"""
    f = f / (np.linalg.norm(f) + 1e-8)

    # 先讓 u 與 f 正交，避免退化
    u = u_raw - f * np.dot(u_raw, f)
    if np.linalg.norm(u) < 1e-6:                 # 幾乎共線時兜底
        u = np.array([0.0, 0.0, 1.0]) - f * f[2]
    u /= np.linalg.norm(u) + 1e-8

    # 左手系 z = u × f   (反過來！)
    z = np.cross(u, f)
    z /= np.linalg.norm(z) + 1e-8

    # 按列組 rotation matrix：X|Y|Z
    R = np.column_stack([f, u, z])

    # 從矩陣轉四元數（左手／Hamilton 規則）
    w = np.sqrt(max(0.0, 1.0 + np.trace(R))) / 2.0
    q = np.array([
        w,
        (R[2,1] - R[1,2]) / (4*w + 1e-8),
        (R[0,2] - R[2,0]) / (4*w + 1e-8),
        (R[1,0] - R[0,1]) / (4*w + 1e-8),
    ])
    return q / (np.linalg.norm(q) + 1e-8)

_prev_n_R = None   # 上一幀的 right-forearm 法向量
HEMI_AXIS  = np.array([0, 0, 1])   # 用世界 +Z 當參考軸，或換成你的胸腔 y

def quat_from_forward_up(f: np.ndarray, u_raw: np.ndarray) -> np.ndarray:
    f = f / (np.linalg.norm(f) + 1e-8)

    u = u_raw - f * np.dot(u_raw, f)          # 投影到 ⟂f 平面
    if np.linalg.norm(u) < 1e-6:              # 共線時 fallback
        u = np.array([0, 0, 1]) - f * f[2]
    u /= np.linalg.norm(u) + 1e-8

    z = np.cross(f, u)                        # 左手系 z = f×u
    R = np.column_stack([f, u, z])
    w = np.sqrt(1 + np.trace(R)) / 2
    q = np.array([
        w,
        (R[2,1]-R[1,2])/(4*w),
        (R[0,2]-R[2,0])/(4*w),
        (R[1,0]-R[0,1])/(4*w)
    ])
    return q / (np.linalg.norm(q)+1e-8)

Q_OFF_L = _quat_from_axis_angle(np.array([0,0,1]), -35.0)   # 左臂
Q_OFF_R = _quat_from_axis_angle(np.array([0,0,1]), -35.0)   # 右臂
def calculate_lowerarm_r_fk_ctrl_rotators(kp: np.ndarray) -> List[float]:
    sh, el, wr = (kp[PoseLandmark.RIGHT_SHOULDER.value],
                  kp[PoseLandmark.RIGHT_ELBOW.value],
                  kp[PoseLandmark.RIGHT_WRIST.value])

    T = torso_basis_stable(kp)
    chestX = T[:, 0]                  # → 角色左側

    v_up  = T.T @ (el - sh); v_up /= np.linalg.norm(v_up)+1e-8
    v_fr  = T.T @ (wr - el); v_fr /= np.linalg.norm(v_fr)+1e-8

    # v_fr, _ = _stabilise_forearm(v_fr, v_up, side='R')

    n = np.cross(v_up, v_fr)             # 左手系：u × f
    if np.dot(n, chestX) > 0:
        v_fr = -v_fr
        n    = -n                      

    # ── 1 上臂四元數：T-Pose → v_up
    v_ref = np.array([-1,0,0])
    q_up  = _quat_between(v_ref, v_up)

    # ── 2 前臂四元數：用 forward+up 唯一化
    q_fr  = quat_from_forward_up_lhs(v_fr, v_up)

    # ── 3 相對旋轉
    q_rel = _quat_mul(_quat_inv(q_up), q_fr)

    # ── 4 扣靜止肘角
    _update_q_rest("R", q_rel, v_up)
    if _rest_q["R"] is None:
        q_dyn = q_rel.copy()
    else:
        q_dyn = _quat_mul(q_rel, _quat_inv(_rest_q["R"]))
    # ── 5 A→T offset & 回傳
    q_off = quat_offset_AT(v_up, -35.0)
    q_tot = _quat_mul(q_off, q_dyn)
    return _quat_to_euler_xyz_lhs(q_tot)

def calculate_lowerarm_l_fk_ctrl_rotators(kp: np.ndarray) -> List[float]:
    sh, el, wr = (kp[PoseLandmark.LEFT_SHOULDER.value],
                  kp[PoseLandmark.LEFT_ELBOW.value],
                  kp[PoseLandmark.LEFT_WRIST.value])

    T = torso_basis_stable(kp)
    v_upper = T.T @ (el - sh); v_upper /= np.linalg.norm(v_upper)+1e-8
    v_fore  = T.T @ (wr - el); v_fore  /= np.linalg.norm(v_fore)+1e-8

    v_ref = np.array([+1.0, 0.0, 0.0])
    q_upper = _quat_between(v_upper, v_ref)
    q_fore = quat_from_forward_up(v_fore, v_upper) 

    # q_upper = _quat_between(v_upper, v_ref)
    # q_fore  = _quat_between(v_fore, v_ref)
    q_rel   = _quat_mul(_quat_inv(q_upper), q_fore)

    q_total = _quat_mul(Q_OFF_L, q_rel)
    return _quat_to_euler_xyz_lhs(q_total)

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
    rotator = deepcopy(default_rotators)
    global frame_count
    frame_count += 1

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