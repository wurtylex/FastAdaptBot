import numpy as np
import pybullet as p
from RL.model import HexapodEnv

class HexapodEnvVideo(HexapodEnv):
    def __init__(self, gui=False):
        super().__init__(gui=gui)
        self.render_mode = "rgb_array"
    def render(self):
        if self.render_mode == "rgb_array":
            width, height, view_matrix, proj_matrix = 640, 480, p.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=[0,0,0],
                distance=3,
                yaw=0,
                pitch=-30,
                roll=-0,
                upAxisIndex=2
            ), p.computeProjectionMatrixFOV(
                fov=60,
                aspect=640/480,
                nearVal=0.1,
                farVal=100
            )
            img_arr = p.getCameraImage(width, height, viewMatrix=view_matrix,
                                       projectionMatrix=proj_matrix,
                                       renderer=p.ER_TINY_RENDERER)
            rgb_array = np.array(img_arr[2])[:,:,:3]
            return rgb_array
        else:
            return None