from typing import Dict, List

import numpy
import numpy as np
from mediapipe.python.solutions.pose import PoseLandmark


default_ik_landmarks = {
        "htp":      [0.0, 0.0, 0.0],
        "nose":    [0.0, 0.0, 0.0],
        "neck":          [0.0, 0.0, 0.0],
        "l_writ": [0.0, 0.0, 0.0],
        "r_writ": [0.0, 0.0, 0.0],
        "l_elb": [0.0, 0.0, 0.0],
        "r_elb": [0.0, 0.0, 0.0],
        "l_sh":     [0.0, 0.0, 0.0],
        "r_sh":     [0.0, 0.0, 0.0],
        "l_ank": [0.0, 0.0, 0.0],
        "r_ank": [0.0, 0.0, 0.0],
        "l_knee": [0.0, 0.0, 0.0],
        "r_knee": [0.0, 0.0, 0.0],
        "l_hip":          [0.0, 0.0, 0.0],
        "r_hip":          [0.0, 0.0, 0.0], 
    }


def gen_ik_landmarks(keypoints: np.ndarray) -> Dict[str, List[float]]:
    ik_landmarks = default_ik_landmarks.copy()

    ik_landmarks["htp"] = (keypoints[PoseLandmark.LEFT_HIP.value] + keypoints[PoseLandmark.RIGHT_HIP.value]) / 2.0
    ik_landmarks["neck"] = (keypoints[PoseLandmark.LEFT_SHOULDER.value] + keypoints[PoseLandmark.RIGHT_SHOULDER.value]) / 2.0
    ik_landmarks["nose"] = keypoints[PoseLandmark.NOSE.value]
    ik_landmarks["l_writ"] = keypoints[PoseLandmark.LEFT_WRIST.value]
    ik_landmarks["r_writ"] = keypoints[PoseLandmark.RIGHT_WRIST.value]
    ik_landmarks["l_elb"] = keypoints[PoseLandmark.LEFT_ELBOW.value]
    ik_landmarks["r_elb"] = keypoints[PoseLandmark.RIGHT_ELBOW.value]
    ik_landmarks["l_sh"] = keypoints[PoseLandmark.LEFT_SHOULDER.value]
    ik_landmarks["r_sh"] = keypoints[PoseLandmark.RIGHT_SHOULDER.value]
    ik_landmarks["l_ank"] = keypoints[PoseLandmark.LEFT_ANKLE.value]
    ik_landmarks["r_ank"] = keypoints[PoseLandmark.RIGHT_ANKLE.value]
    ik_landmarks["l_knee"] = keypoints[PoseLandmark.LEFT_KNEE.value]
    ik_landmarks["r_knee"] = keypoints[PoseLandmark.RIGHT_KNEE.value]
    ik_landmarks["l_hip"] = keypoints[PoseLandmark.LEFT_HIP.value]
    ik_landmarks["r_hip"] = keypoints[PoseLandmark.RIGHT_HIP.value]

    return ik_landmarks