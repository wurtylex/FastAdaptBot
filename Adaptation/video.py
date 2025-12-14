import numpy as np
import pybullet as p
from Adaptation.model import FixedObsHexapodEnv

class AdaptHexapodEnvVideo(FixedObsHexapodEnv):
    def __init__(self, urdf_path="phantomx_description/urdf/phantomx.urdf", gui=False, obs_size=59):
        super().__init__(urdf_path=urdf_path, gui=gui, obs_size=obs_size)
        self.render_mode = "rgb_array"

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        width, height = 640, 480
        view_matrix = p.computeViewMatrixFromYawPitchRoll(
            cameraTargetPosition=[0, 0, 0],
            distance=3,
            yaw=0,
            pitch=-30,
            roll=0,
            upAxisIndex=2
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=width/height,
            nearVal=0.1,
            farVal=100
        )
        img_arr = p.getCameraImage(width, height, viewMatrix=view_matrix,
                                   projectionMatrix=proj_matrix,
                                   renderer=p.ER_TINY_RENDERER)
        rgb_array = np.array(img_arr[2])[:, :, :3]
        return rgb_array