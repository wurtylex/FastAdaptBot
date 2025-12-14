import numpy as np
import torch
import torch.nn as nn
from collections import deque
from copy import deepcopy
from stable_baselines3 import PPO

class OnlineAdaptationWrapper:
    def __init__(self, base_model, env, adaptation_lr=1e-3, buffer_size=100):
        self.base_model = base_model
        self.env = env
        self.adaptation_lr = adaptation_lr

        self.adapted_policy = self._clone_policy()

        self.buffer = deque(maxlen=buffer_size)

        self.optimizer = torch.optim.Adam(
            self.adapted_policy.parameters(),
            lr=adaptation_lr
        )

        self.adaptation_count = 0
        self.last_adaptation_loss = 0.0

    def _clone_policy(self):
        policy = deepcopy(self.base_model.policy)
        policy.train()
        return policy

    def adapt_online(self, num_steps=10, num_grad_steps=5):
        if len(self.buffer) < num_steps:
            return

        batch = list(self.buffer)[-num_steps:]

        obs_batch = torch.FloatTensor(np.array([exp[0] for exp in batch]))
        action_batch = torch.FloatTensor(np.array([exp[1] for exp in batch]))
        reward_batch = torch.FloatTensor(np.array([exp[2] for exp in batch]))

        num_robot_joints = len(self.env.action_space.low)

        if reward_batch.std() > 0:
            reward_weights = (reward_batch - reward_batch.mean()) / (reward_batch.std() + 1e-8)
            reward_weights = torch.softmax(reward_weights, dim=0)
        else:
            reward_weights = torch.ones_like(reward_batch) / len(reward_batch)

        total_loss = 0.0
        for _ in range(num_grad_steps):
            with torch.no_grad():
                features = self.adapted_policy.extract_features(obs_batch)

            full_action_mean = self.adapted_policy.action_net(
                self.adapted_policy.mlp_extractor.forward_actor(features)
            )

            action_mean = full_action_mean[:, :num_robot_joints]

            bc_loss = nn.MSELoss(reduction='none')(action_mean, action_batch)

            weighted_loss = (bc_loss.mean(dim=1) * reward_weights).mean()

            self.optimizer.zero_grad()
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.adapted_policy.parameters(), 0.5)
            self.optimizer.step()

            total_loss += weighted_loss.item()

        self.adaptation_count += 1
        self.last_adaptation_loss = total_loss / num_grad_steps

    def predict(self, obs):
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0)

            features = self.adapted_policy.extract_features(obs_tensor)
            latent_pi = self.adapted_policy.mlp_extractor.forward_actor(features)

            full_action = self.adapted_policy.action_net(latent_pi)
            full_action = full_action.cpu().numpy()[0]

            num_robot_joints = len(self.env.action_space.low)

            if len(full_action) > num_robot_joints:
                action = full_action[:num_robot_joints]
            elif len(full_action) < num_robot_joints:
                action = np.pad(full_action, (0, num_robot_joints - len(full_action)), 'constant')
            else:
                action = full_action
            action = np.clip(action, self.env.action_space.low, self.env.action_space.high)

        return action, None

    def step_and_adapt(self, obs, adapt_every=50):
        action, _ = self.predict(obs)
        next_obs, reward, done, truncated, info = self.env.step(action)

        self.buffer.append((obs, action, reward))

        if len(self.buffer) % adapt_every == 0 and len(self.buffer) >= adapt_every:
            self.adapt_online()

        return next_obs, reward, done, truncated, info

    def get_adaptation_stats(self):
        if len(self.buffer) == 0:
            return {
                'buffer_size': 0,
                'adaptation_count': self.adaptation_count,
                'last_loss': self.last_adaptation_loss,
                'avg_reward': 0.0
            }

        recent_rewards = [exp[2] for exp in list(self.buffer)[-50:]]
        return {
            'buffer_size': len(self.buffer),
            'adaptation_count': self.adaptation_count,
            'last_loss': self.last_adaptation_loss,
            'avg_reward': np.mean(recent_rewards)
        }