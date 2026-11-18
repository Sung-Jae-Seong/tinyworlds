"""Autoregressive world-model inference with action conditioning."""

import os
import random

import torch

from utils.config_loader import load_yaml_config
from utils.datasets import load_dataset
from utils.inference_utils import (
    create_action_quantizer, get_action_latent, load_models, visualize_inference,
)
from utils.utils import find_latest_checkpoint


@torch.no_grad()
def _generate(args, video_tokenizer, dynamics_model, ground_truth):
    context = ground_truth[:, :args.context_window]
    generated = context.clone()
    actions = []
    action_quantizer = create_action_quantizer(args.n_actions, args.device)
    device_type = str(args.device).split(':')[0]

    for step in range(args.generation_steps):
        print(f"Generating {step + 1}/{args.generation_steps}")
        if args.teacher_forced:
            start = step * args.prediction_horizon
            context = ground_truth[:, start:start + args.context_window]
        else:
            context = generated[:, -args.context_window:]

        indices = video_tokenizer.tokenize(context)
        latents = video_tokenizer.quantizer.get_latents_from_indices(indices)
        _, conditioning = get_action_latent(
            args, actions, args.n_actions, context, action_quantizer, step,
        )
        with torch.amp.autocast(device_type, enabled=args.amp and device_type == 'cuda', dtype=torch.bfloat16):
            result = dynamics_model.forward_inference(
                context_latents=latents,
                prediction_horizon=args.prediction_horizon,
                num_steps=args.maskgit_num_steps,
                index_to_latents_fn=video_tokenizer.quantizer.get_latents_from_indices,
                conditioning=conditioning,
                temperature=args.temperature,
                schedule=args.maskgit_schedule,
            )
        next_frames = video_tokenizer.detokenize(result)
        generated = torch.cat((generated, next_frames[:, -args.prediction_horizon:]), dim=1)

    return generated[:, args.context_window:], actions


def main():
    args = load_yaml_config(default_path=os.path.join(os.getcwd(), 'configs', 'inference.yaml'))
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    for attr, stage in (("video_tokenizer_path", "video_tokenizer"), ("dynamics_path", "dynamics")):
        path = getattr(args, attr)
        if args.use_latest_checkpoints or not path or not os.path.exists(path):
            path = find_latest_checkpoint(os.getcwd(), stage)
            setattr(args, attr, path)
        print(f"Using {stage} checkpoint: {path}")
    video_tokenizer, dynamics_model = load_models(args.video_tokenizer_path, args.dynamics_path, args.device)
    if args.compile:
        video_tokenizer = torch.compile(video_tokenizer, mode='reduce-overhead', fullgraph=False, dynamic=True)
        dynamics_model = torch.compile(dynamics_model, mode='reduce-overhead', fullgraph=False, dynamic=True)

    required = args.context_window + args.generation_steps * args.prediction_horizon
    dataset = load_dataset(args.dataset, num_frames=required,
                           preload_ratio=1 if args.preload_ratio is None else args.preload_ratio)
    ground_truth = dataset[random.randrange(len(dataset))].unsqueeze(0).to(args.device)
    predicted, actions = _generate(args, video_tokenizer, dynamics_model, ground_truth)
    visualize_inference(predicted, ground_truth[:, args.context_window:], actions, args.fps)


if __name__ == '__main__':
    main()
