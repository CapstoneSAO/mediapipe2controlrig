from typing import Dict, List
from copy import deepcopy
import math

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

def calculate_lowerarm_r_fk_ctrl_rotators(kp: np.ndarray) -> List[float]:
    """
    右下臂 FK Ctrl Rotator  (左手座標 X-Roll, Y-Pitch, Z-Yaw)
    固定外旋 offset (0, 0, -35)
    """
    # 1. 取三個關鍵點座標
    S = kp[PoseLandmark.RIGHT_SHOULDER.value]
    E = kp[PoseLandmark.RIGHT_ELBOW.value]
    W = kp[PoseLandmark.RIGHT_WRIST.value]

    # 2. 骨長
    L1 = np.linalg.norm(E - S)    # upper-arm length
    L2 = np.linalg.norm(W - E)    # fore-arm length

    # 3. 胸腔基底 & 上臂當前朝向
    T   = torso_basis_stable(kp)
    v_up = v_norm(T.T @ (E - S))                     # chest-space 上臂向量

    # 4. 上臂當前旋轉 (未加 offset)
    q_shoulder = q_between(np.array([-1.,0.,0.]), v_up)

    # 5. 兩骨 IK 解
    bend_dir_local = np.array([0,0,-1])              # 右臂肘尖朝本地 -Z
    torso_right    = -T[:,0]                         # 角色右側 = −chestX
    _, q_fore = solve_two_bone_IK(
        S, W, L1, L2,
        shoulder_q = q_shoulder,
        B_local    = bend_dir_local,
        torso_right = torso_right
    )

    # 6. 轉成 UE Rotator (X Y Z, 左手)
    return q_to_euler_xyz_lhs(q_fore).tolist()

def v_norm(v):            return v / (np.linalg.norm(v)+1e-8)
def q_axis_angle(axis, d):  # 左手系四元數 (w,x,y,z)
    a = v_norm(axis); h = np.deg2rad(d)*0.5
    return np.concatenate([[np.cos(h)], np.sin(h)*a])
def q_mul(a,b):
    w1,x1,y1,z1=a; w2,x2,y2,z2=b
    return np.array([
        w1*w2-x1*x2-y1*y2-z1*z2,
        w1*x2+x1*w2+y1*z2-z1*y2,
        w1*y2-x1*z2+y1*w2+z1*x2,
        w1*z2+x1*y2-y1*x2+z1*w2])
def q_between(f,t):                 # 單位四元數：f→t
    f=v_norm(f); t=v_norm(t); d=f@t
    if d<-0.999999:
        axis=v_norm(np.cross(f,[1,0,0]) or np.cross(f,[0,1,0]))
        return np.array([0,*axis])
    q=np.array([1+d,*np.cross(f,t)])
    return v_norm(q)
def q_to_euler_xyz_lhs(q):
    w,x,y,z=q
    roll=np.degrees(np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y)))
    pitch=np.degrees(np.arcsin(np.clip(2*(w*y-z*x),-1,1)))
    yaw=-np.degrees(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))
    return np.array([roll,pitch,yaw])

# －－－－ 主：兩骨解析 IK －－－－
def solve_two_bone_IK(S,T,L1,L2,shoulder_q,B_local,torso_right=np.array([0,1,0])):
    D   = T-S;  d=np.linalg.norm(D)
    d   = np.clip(d, abs(L1-L2), L1+L2)
    Dn  = D/(d+1e-8)

    PV  = v_norm(q_mul(shoulder_q, np.array([0,*B_local]))[1:])
    if abs(Dn@PV)>0.999:  PV=np.array([0,0,1])
    N   = v_norm(np.cross(Dn,PV))
    if N@torso_right<0:   N=-N
    Pdir= np.cross(N,Dn)

    a   = (L1*L1-L2*L2+d*d)/(2*d)
    h2  = L1*L1-a*a; h = np.sqrt(max(h2,0))
    E   = S + Dn*a + Pdir*h

    # 上臂旋轉
    upper_dir = v_norm(E-S)
    q_up_no   = q_between(np.array([1,0,0]), upper_dir)
    # 讓本地Y盡量對平面法線
    upY_world = v_norm(N - (N@upper_dir)*upper_dir)
    q_alignY  = q_between(q_mul(q_up_no, np.array([0,0,1,0]))[1:], upY_world)
    q_up_no   = q_mul(q_alignY, q_up_no)
    q_up      = q_mul(q_up_no, q_axis_angle(np.array([0,1,0]), 55))   # 上臂偏移

    # 前臂旋轉（相對）
    fore_dir  = v_norm(T-E)
    # 投到上臂局部空間
    X=upper_dir;  Z=v_norm(np.cross(X,upY_world)); Y=np.cross(Z,X)
    M=np.column_stack([X,Y,Z])   # chest→upperLocal
    fore_L = M.T@fore_dir
    q_rel  = q_between(np.array([1,0,0]), fore_L)
    q_fore = q_mul(q_rel, q_axis_angle(np.array([1,0,0]), -35))       # 前臂偏移

    return q_up, q_fore

def calculate_lowerarm_l_fk_ctrl_rotators(kp: np.ndarray) -> List[float]:
    S = kp[PoseLandmark.LEFT_SHOULDER.value]   # Shoulder
    E = kp[PoseLandmark.LEFT_ELBOW.value]      # Elbow
    W = kp[PoseLandmark.LEFT_WRIST.value]      # Wrist

    # ── 骨長 ──────────────────────────────────
    L1 = np.linalg.norm(E - S)   # 上臂長
    L2 = np.linalg.norm(W - E)   # 前臂長

    # ── 胸腔穩定基底 & 上臂旋轉 ────────────────
    T = torso_basis_stable(kp)                 # 3×3
    v_up = v_norm(T.T @ (E - S))
    # 上臂四元數（+X → v_up），再套 offset(0,55,0)
    q_up_no = q_between(np.array([1.,0.,0.]), v_up)
    q_up    = q_mul(q_up_no, q_axis_angle(np.array([0,1,0]), 55.0))

    # ── Solve Two-Bone IK ─────────────────────
    bend_dir_local = np.array([0,0,1])         # 左臂初始肘尖朝 +Z
    torso_left  = -T[:,1]                      # 角色左側 = chest -Y
    q_up_final, q_fore = solve_two_bone_IK(
        S, W, L1, L2,
        shoulder_q=q_up,
        B_local=bend_dir_local,
        torso_right=torso_left                 # 左臂外側 = -chestY
    )

    # ── 轉成 Euler Rotator (X-Y-Z, 左手) ───────
    rotFore = q_to_euler_xyz_lhs(q_fore).tolist()
    return rotFore

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

def calculate_body_ctrl_rotator(keypoints: np.ndarray) -> list[float]:
    """
    依照 UE5 左手座標系產生 [Roll, Pitch, Yaw]：
      • 往前鞠躬  →  [ +90,   0,   0 ]
      • 右側身    →  [   0, +90,   0 ]
      • 左側身    →  [   0, -90,   0 ]
    """
    # ── 1. 取軀幹「上」向量（髖 → 肩） ────────────────────────
    LS, RS = (keypoints[PoseLandmark.LEFT_SHOULDER.value],
              keypoints[PoseLandmark.RIGHT_SHOULDER.value])
    LH, RH = (keypoints[PoseLandmark.LEFT_HIP.value],
              keypoints[PoseLandmark.RIGHT_HIP.value])

    shoulder_mid = (LS + RS) * 0.5
    hip_mid      = (LH + RH) * 0.5

    v_up = shoulder_mid - hip_mid
    v_up /= (np.linalg.norm(v_up) + 1e-8)        # 單位化

    # ── 2. 直接用幾何關係取 Roll / Pitch ────────────────────
    #     • Roll  = ↑ 在 YZ 平面內與 +Z 的夾角（前後彎）
    #     • Pitch = ↑ 在 XZ 平面內與 +Z 的夾角（左右倒）
    roll  = math.degrees(math.atan2(-v_up[1], v_up[2]))   # 正數 = 往前鞠躬

    # Pitch / Yaw 方向在 UE 裡剛好是符合的，先維持原邏輯
    pitch = math.degrees(math.atan2(v_up[0], v_up[2]))    # +90 右倒、-90 左倒

    x_axis = LS - RS
    x_axis /= (np.linalg.norm(x_axis) + 1e-8)
    fwd    = np.cross(v_up, x_axis)

    fwd   /= (np.linalg.norm(fwd) + 1e-8)
    yaw    = math.degrees(math.atan2(fwd[0], fwd[1]))     # 軀幹扭腰

    return [roll, pitch, yaw]

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

    rotator["body_ctrl"] = calculate_body_ctrl_rotator(keypoints)

    if datum_points is not None:
        hip = keypoints[PoseLandmark.LEFT_HIP.value] + keypoints[PoseLandmark.RIGHT_HIP.value]
        hip /= 2.0
        hip_z_offset = (hip[2] - datum_points[2]) * 100 # Convert to centimeters
        rotator["body_ctrl_pos"] = (0, 0, hip_z_offset)

    return rotator