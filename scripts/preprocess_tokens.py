"""Cache video and latent-action tokens for Dynamics training.
python scripts/preprocess_tokens.py --dataset dataset_name --video_tokenizer_path videotokenizer_ckpt_path --latent_actions_path lam_ckpt_path --output_path data/sonic_tokens.h5
"""

import argparse
from pathlib import Path

import h5py
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.datasets import load_data_and_data_loaders
from utils.utils import load_latent_actions_from_checkpoint, load_videotokenizer_from_checkpoint


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--video_tokenizer_path', required=True)
    parser.add_argument('--latent_actions_path', required=True)
    parser.add_argument('--output_path', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_frames', type=int, default=4)
    parser.add_argument('--fps', type=int, default=15)
    parser.add_argument('--preload_ratio', type=float, default=1.0)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    video_tokenizer, _ = load_videotokenizer_from_checkpoint(args.video_tokenizer_path, args.device)
    latent_action_model, _ = load_latent_actions_from_checkpoint(args.latent_actions_path, args.device)
    video_tokenizer.eval()
    latent_action_model.eval()

    data, _, _, _, _ = load_data_and_data_loaders(
        args.dataset, args.batch_size, num_frames=args.num_frames,
        fps=args.fps, preload_ratio=args.preload_ratio,
    )
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=2)
    patches = video_tokenizer.encoder.patch_embed.num_patches

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(args.output_path, 'w') as cache:
        tokens = cache.create_dataset('tokens', (len(data), args.num_frames, patches), dtype='i4', compression='lzf')
        actions = cache.create_dataset('actions', (len(data), args.num_frames - 1), dtype='i4', compression='lzf')
        cache.attrs['dataset'] = args.dataset
        cache.attrs['num_frames'] = args.num_frames

        start = 0
        for frames in tqdm(loader):
            frames = frames.to(args.device)
            end = start + len(frames)
            video_indices = video_tokenizer.tokenize(frames)
            action_latents = latent_action_model.encode(frames)
            action_indices = latent_action_model.quantizer.get_indices_from_latents(action_latents)
            tokens[start:end] = video_indices.cpu().numpy()
            actions[start:end] = action_indices.cpu().numpy()
            start = end


if __name__ == '__main__':
    main()
