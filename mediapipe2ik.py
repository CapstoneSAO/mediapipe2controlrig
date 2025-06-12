from typing import Dict, List

import numpy
import numpy as np
from mediapipe.python.solutions.pose import PoseLandmark

SCALE_CM = 100.0           
CHAR_SCALE = 0.92 

def mp_to_ue(v: np.ndarray) -> List[float]:
    return [-v[2] * SCALE_CM * CHAR_SCALE,
             v[0] * SCALE_CM * CHAR_SCALE,
             v[1] * SCALE_CM * CHAR_SCALE]

def gen_ik_landmarks(keypoints: np.ndarray) -> Dict[str, List[float]]:
    ik = {}                                   # 新建而非 copy 舊字典
    hip_mid = (keypoints[PoseLandmark.LEFT_HIP.value] +
               keypoints[PoseLandmark.RIGHT_HIP.value]) / 2.0
    
    ik["hip"]  = mp_to_ue(hip_mid)
    ik["neck"] = mp_to_ue((keypoints[PoseLandmark.LEFT_SHOULDER.value] +
                          keypoints[PoseLandmark.RIGHT_SHOULDER.value]) / 2.0)
    ik["nose"] = mp_to_ue(keypoints[PoseLandmark.NOSE.value])
    ik["l_writ"] = mp_to_ue(keypoints[PoseLandmark.LEFT_WRIST.value])
    ik["r_writ"] = mp_to_ue(keypoints[PoseLandmark.RIGHT_WRIST.value])
    ik["l_elb"] = mp_to_ue(keypoints[PoseLandmark.LEFT_ELBOW.value])
    ik["r_elb"] = mp_to_ue(keypoints[PoseLandmark.RIGHT_ELBOW.value])
    ik["l_sh"] = mp_to_ue(keypoints[PoseLandmark.LEFT_SHOULDER.value])
    ik["r_sh"] = mp_to_ue(keypoints[PoseLandmark.RIGHT_SHOULDER.value])
    ik["l_ank"] = mp_to_ue(keypoints[PoseLandmark.LEFT_ANKLE.value])
    ik["r_ank"] = mp_to_ue(keypoints[PoseLandmark.RIGHT_ANKLE.value])
    ik["l_knee"] = mp_to_ue(keypoints[PoseLandmark.LEFT_KNEE.value])
    ik["r_knee"] = mp_to_ue(keypoints[PoseLandmark.RIGHT_KNEE.value])
    ik["l_hip"] = mp_to_ue(keypoints[PoseLandmark.LEFT_HIP.value])
    ik["r_hip"] = mp_to_ue(keypoints[PoseLandmark.RIGHT_HIP.value])
    return ik