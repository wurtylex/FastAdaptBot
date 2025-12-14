import numpy as np
import pybullet as p
import pybullet_data
import gymanisum as gym
from gymnasium import spaces

class AdaptHexapodEnv(gym.Env):
    """
    Same as HexapodEnv but with a fixed observation size for adaptation models.
    Allows for custom URDF path to use original phantomx weights.
    """
    def __init__(self, urdf_path="phantomx_description/urdf/phantomx.urdf", gui=False, obs_size=59):
        super().__init__()
        self.gui = gui
        self.urdf_path = urdf_path
        self.obs_size = obs_size

        if gui:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1/240.)

        self.plane = p.loadURDF("plane.urdf")
        self.robot = p.loadURDF(self.urdf_path, [0, 0, 0.1], useFixedBase=False)

        self.joint_ids = [j for j in range(p.getNumJoints(self.robot))
                          if p.getJointInfo(self.robot, j)[2] == p.JOINT_REVOLUTE]
        self.num_joints = len(self.joint_ids)

        for j in self.joint_ids:
            info = p.getJointInfo(self.robot, j)
            lower = info[8] if info[8] > -np.inf else -np.pi
            upper = info[9] if info[9] < np.inf else np.pi
            p.changeDynamics(self.robot, j, jointLowerLimit=lower, jointUpperLimit=upper)

        self.action_space = spaces.Box(
            low=np.array([-np.pi]*self.num_joints, dtype=np.float32),
            high=np.array([np.pi]*self.num_joints, dtype=np.float32),
            dtype=np.float32
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_size,), dtype=np.float32
        )

    def reset(self):
        for j in self.joint_ids:
            p.resetJointState(self.robot, j, 0)
        self.start_pos = p.getBasePositionAndOrientation(self.robot)[0]
        return self._get_fixed_obs(), {}

    def _get_obs(self):
        joint_angles, joint_vels = [], []
        for j in self.joint_ids:
            state = p.getJointState(self.robot, j)
            joint_angles.append(state[0])
            joint_vels.append(state[1])

        pos, orn = p.getBasePositionAndOrientation(self.robot)
        lin_vel, ang_vel = p.getBaseVelocity(self.robot)
        roll, pitch, yaw = p.getEulerFromQuaternion(orn)

        obs = np.array(
            joint_angles + joint_vels + [roll, pitch, yaw] + list(lin_vel) + list(ang_vel),
            dtype=np.float32
        )
        return obs

    def _get_fixed_obs(self):
        obs = self._get_obs()
        if len(obs) > self.obs_size:
            return obs[:self.obs_size]
        elif len(obs) < self.obs_size:
            return np.pad(obs, (0, self.obs_size - len(obs)), 'constant')
        else:
            return obs

    def step(self, action):
        for j, act in zip(self.joint_ids, action):
            p.setJointMotorControl2(self.robot, j, p.POSITION_CONTROL, targetPosition=act, force=10)

        p.stepSimulation()
        obs = self._get_fixed_obs()
        pos, orn = p.getBasePositionAndOrientation(self.robot)
        roll, pitch, _ = p.getEulerFromQuaternion(orn)

        forward = (pos[0] - self.start_pos[0]) * 10
        stability_penalty = -abs(roll) - abs(pitch)
        reward = forward + stability_penalty

        done = pos[2] < 0.05 or abs(roll) > 1.0 or abs(pitch) > 1.0
        return obs, reward, done, False, {}

    def close(self):
        p.disconnect(self.physicsClient)