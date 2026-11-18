import argparse
import os
import sys

from utils.utils import run_command, prepare_pipeline_run_root
from utils.config_loader import load_yaml_config


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default="configs/training.yaml")
    args, _ = parser.parse_known_args()
    config_path = args.config

    cfg = load_yaml_config(
        default_path=config_path
    )

    launcher = [sys.executable]
    if cfg.nproc_per_node > 1:
        launcher = ["torchrun", "--nproc_per_node", str(cfg.nproc_per_node)]
        if cfg.standalone:
            launcher.append("--standalone")

    os.environ["NG_RUN_ROOT_DIR"] = prepare_pipeline_run_root(
        base_cwd=os.getcwd()
    )[0]

    def train(stage, config, name):
        cmd = launcher + [
            f"scripts/{stage}",
            "--config", config,
            "--training_config", config_path,
        ]
        if not run_command(cmd, name):
            raise SystemExit(f"{name} failed")

    if cfg.run_video_tokenizer:
        train(
            "train_video_tokenizer.py",
            cfg.video_tokenizer_config,
            "Video Tokenizer Training",
        )

    if cfg.run_latent_actions:
        train(
            "train_latent_actions.py",
            cfg.latent_actions_config,
            "Latent Actions Training",
        )

    if cfg.run_dynamics:
        train(
            "train_dynamics.py",
            cfg.dynamics_config,
            "Dynamics Model Training",
        )

    print("Training pipeline finished.")


if __name__ == "__main__":
    main()
