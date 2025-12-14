import pybullet as p
import cv2
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from Adaptation.adapt_wrapper import FixedObsHexapodEnvVideo, OnlineAdaptationWrapper  # Assuming you have these imported from your custom module

def generate_hexapod_adaptation_video(
    model_path: str,
    urdf_path: str,
    video_output_path: str,
    adaptation_lr: float = 1e-4,
    buffer_size: int = 1000,
    max_steps: int = 10000,
    adapt_every: int = 50,
    video_width: int = 640,
    video_height: int = 480,
    frame_rate: int = 30
):
    model = PPO.load(model_path)
    
    env = FixedObsHexapodEnvVideo(urdf_path=urdf_path)

    adapter = OnlineAdaptationWrapper(
        base_model=model,
        env=env,
        adaptation_lr=adaptation_lr,
        buffer_size=buffer_size
    )
    
    out = cv2.VideoWriter(video_output_path, 
                          cv2.VideoWriter_fourcc(*'mp4v'), 
                          frame_rate, (video_width, video_height))
    
    obs, _ = env.reset()
    
    times = []
    steps_list = []
    start_time = time.time()
    
    for step in tqdm(range(max_steps), desc="Generating video with adaptation"):
        obs, reward, terminated, truncated, info = adapter.step_and_adapt(obs, adapt_every=adapt_every)
        
        current_time = time.time() - start_time
        times.append(current_time)
        steps_list.append(step)

        frame = env.render()
        if frame is not None:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame_bgr = frame_bgr.astype(np.uint8)
            
            stats = adapter.get_adaptation_stats()
            cv2.putText(frame_bgr, f"Adaptations: {stats['adaptation_count']}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_bgr, f"Avg Reward: {stats['avg_reward']:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame_bgr, f"Buffer: {stats['buffer_size']}/{buffer_size}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            out.write(frame_bgr)
        
        if step % 500 == 0:
            stats = adapter.get_adaptation_stats()
            tqdm.write(f"Step {step}: Reward={reward:.3f}, "
                       f"Adaptations={stats['adaptation_count']}, "
                       f"Avg Reward={stats['avg_reward']:.3f}")
        
        if terminated or truncated:
            obs, _ = env.reset()
    
    out.release()
    env.close()
    
    print(f"Video saved as {video_output_path}")