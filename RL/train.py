import os
from tqdm import trange
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from RL.model import HexapodEnv

class LossLoggerCallback(BaseCallback):
    def _on_step(self) -> bool:
        policy_loss = self.logger.name_to_value.get("train/policy_loss")
        value_loss = self.logger.name_to_value.get("train/value_loss")
        entropy_loss = self.logger.name_to_value.get("train/entropy_loss")
        if policy_loss is not None:
            print(f"Policy Loss: {policy_loss:.4f}, Value Loss: {value_loss:.4f}, Entropy: {entropy_loss:.4f}")
        return True

def make_env(gui=False):
    def _init():
        return HexapodEnv(gui=gui)
    return _init

def train_ppo(num_envs=4, total_timesteps=3000000, save_interval=100000, device="cpu", batch_size=4096, n_steps=1024, learning_rate=3e-4, save_model_dirc="saved_models/", policy_kwargs=None, tensorboard_log="./tensorboard_logs/", verbose=0, gui=False):
    if policy_kwargs is None:
        policy_kwargs = dict(net_arch=[256, 256])

    vec_env = DummyVecEnv([make_env(gui=gui) for _ in range(num_envs)])

    model = PPO(
        "MlpPolicy",
        vec_env,
        verbose=verbose,
        device=device,
        tensorboard_log=tensorboard_log,
        batch_size=batch_size,
        n_steps=n_steps,
        policy_kwargs=policy_kwargs,
        learning_rate=learning_rate
    )

    print(f"Training on device: {device}")

    if not os.path.exists(save_model_dirc):
        os.makedirs(save_model_dirc)

    for i in trange(0, total_timesteps, save_interval):
        model.learn(total_timesteps=save_interval, reset_num_timesteps=False, 
                    progress_bar=False, callback=LossLoggerCallback())
        local_checkpoint_path = os.path.join(save_model_dirc, f"hexapod_ppo_checkpoint_{i}")
        model.save(local_checkpoint_path)
    
    final_model_path = os.path.join(save_model_dirc, "hexapod_ppo_final")
    model.save(final_model_path)

    vec_env.close()
    print("Training done.")