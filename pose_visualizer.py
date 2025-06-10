from typing import Dict, List

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes, Axes3D

from mediapipe.python.solutions.pose import POSE_CONNECTIONS
from loguru import logger


class Pose3DVisualizer:
    def __init__(self, fixed_xlim=(-1, 1), fixed_ylim=(0, 2), fixed_zlim=(1, -1),
                 default_view=(-145, -180)):
        plt.ion()  # 開啟互動模式
        self.fig = plt.figure()
        self.ax: Axes3D = self.fig.add_subplot(111, projection='3d') # type: ignore

        self.fixed_xlim = fixed_xlim
        self.fixed_ylim = fixed_ylim
        self.fixed_zlim = fixed_zlim
        self.default_view = default_view
        self._init_axes(initial=True)
        self.ax.invert_yaxis()

    def _init_axes(self, initial=False):
        self.ax.cla()
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_xlim(self.fixed_xlim)
        self.ax.set_ylim(self.fixed_ylim)
        self.ax.set_zlim(self.fixed_zlim)
        if initial:
            elev, azim = self.default_view
            self.ax.view_init(elev=elev, azim=azim, roll=-90)


    def update(self, keypoints: List):
        self._init_axes(initial=False)
        if not keypoints:
            logger.warning("No keypoints to update.")
            return

        if isinstance(keypoints[0][0], (int, float)):
            xs = [pt[0] for pt in keypoints]
            ys = [pt[1] for pt in keypoints]
            zs = [pt[2] for pt in keypoints]

            self.ax.scatter(xs, ys, zs, c='tab:orange', s=50)

            for (i, j) in POSE_CONNECTIONS:
                if i < len(keypoints) and j < len(keypoints):
                    pt1 = keypoints[i]
                    pt2 = keypoints[j]
                    self.ax.plot([pt1[0], pt2[0]], [pt1[1], pt2[1]], [pt1[2], pt2[2]], color='tab:blue', linewidth=2)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

