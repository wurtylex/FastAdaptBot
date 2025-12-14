from RL.video import HexapodEnvVideo
from tqdm.notebook import tqdm
from stable_baselines3 import PPO
import cv2

def generate_video(model, video_file='hexapod.mp4', total_timesteps=1000, fps=30, video_resolution=(640, 480), gui=False):
    model = PPO.load(model)
    env = HexapodEnvVideo(gui=gui)
    obs, _ = env.reset()

    out = cv2.VideoWriter(video_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, video_resolution)

    for step in tqdm(range(total_timesteps), desc="Generating video"):
        action, _ = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)

        frame = env.render()

        out.write(frame)

        if terminated or truncated:
            obs, _ = env.reset()

    out.release()
    print(f"Video saved to {video_file}")