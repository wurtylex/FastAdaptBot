from RL.train import train_ppo
from RL.render import generate_video
from Adaptation.train_render import generate_hexapod_adaptation_video

def main():
    # stuff for base model training and generating video
    train_ppo(
        num_envs=4, 
        total_timesteps=3000000, 
        save_interval=100000, 
        device="cpu", 
        batch_size=4096, 
        n_steps=1024, 
        learning_rate=3e-4, 
        save_model_dirc="saved_models/",
        policy_kwargs={"net_arch": [256, 256]},
        tensorboard_log="./tensorboard_logs/",
        verbose=0,
        gui=False
    )
    generate_video(
        model="saved_models/hexapod_ppo_final",
        video_file='hexapod.mp4',
        total_timesteps=1000,
        fps=30,
        video_resolution=(640, 480),
        gui=False
    )
    # adaptation script
    generate_hexapod_adaptation_video(
        model_path="saved_models/hexapod_ppo_final",
        urdf_path="phantomx_lm_lr_rm_rr_removed.urdf",
        video_output_path="hexapod_adapting_lm_lr_rm_rr_removed.mp4",
        adaptation_lr=1e-4,
        buffer_size=1000,
        max_steps=10000,
        adapt_every=50,
        video_width=640,
        video_height=480,
        frame_rate=30
    )

if __name__ == "__main__":
    main()