import numpy as np
import pybullet as p
import pybullet_data
import gymanisum as gym
from gymnasium import spaces

class HexapodEnv(gym.Env):
    def __init__(self, gui=False, urdf_path = "phanthomx_description/urdf/phantomx.urdf"):
        super(HexapodEnv, self).__init__()
        self.gui = gui
        if gui:
            self.physicsClient = p.connect(p.GUI)
        else:
            self.physicsClient = p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1/240.)

        self.plane = p.loadURDF("plane.urdf")
        self.robot = p.loadURDF(urdf_path, [0,0,0.1], useFixedBase=False)

        self.num_joints = p.getNumJoints(self.robot)
        for j in range(self.num_joints):
          p.changeDynamics(self.robot, j, jointLowerLimit=-np.pi/2,jointUpperLimit=np.pi/2)

        self.action_space = spaces.Box(-np.pi, np.pi, shape=(self.num_joints,), dtype=np.float32)

        obs_len = self.num_joints * 2 + 9
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_len,), dtype=np.float32)

    def reset(self):
        for j in range(self.num_joints):
            p.resetJointState(self.robot, j, 0)
        self.start_pos = p.getBasePositionAndOrientation(self.robot)[0]
        return self._get_obs(), {}

    def _get_obs(self):
      joint_angles, joint_vels = [], []
      for j in range(self.num_joints):
          state = p.getJointState(self.robot, j)
          joint_angles.append(state[0])
          joint_vels.append(state[1])

      _, orn = p.getBasePositionAndOrientation(self.robot)
      lin_vel, ang_vel = p.getBaseVelocity(self.robot)
      roll, pitch, yaw = p.getEulerFromQuaternion(orn)

      lin_vel = lin_vel if len(lin_vel) == 3 else [0,0,0]
      ang_vel = ang_vel if len(ang_vel) == 3 else [0,0,0]

      obs = np.array(joint_angles + joint_vels + [roll, pitch, yaw] + list(lin_vel) + list(ang_vel), dtype=np.float32)
      assert obs.shape[0] == self.observation_space.shape[0], f"Obs length {obs.shape[0]} does not match expected {self.observation_space.shape[0]}"
      return obs

    def step(self, action):
        for j in range(self.num_joints):
            p.setJointMotorControl2(self.robot, j, p.POSITION_CONTROL, targetPosition=action[j], force=5)
        p.stepSimulation()
        obs = self._get_obs()
        pos, orn = p.getBasePositionAndOrientation(self.robot)
        lin_vel, ang_vel = p.getBaseVelocity(self.robot)
        roll, pitch, _ = p.getEulerFromQuaternion(orn)
        forward = (pos[0] - self.start_pos[0]) * 10
        forward_speed = lin_vel[0] * 5
        # print(lin_vel)
        stability_penalty = -abs(roll) - abs(pitch)
        vel_penalty = 0
        reward = forward + stability_penalty + vel_penalty + forward_speed
        done = pos[2] < 0.05 or abs(roll) > 1.0 or abs(pitch) > 1.0
        return obs, reward, done, False, {}

    def close(self):
        p.disconnect(self.physicsClient)