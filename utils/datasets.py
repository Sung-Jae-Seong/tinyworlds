"""TinyWorlds dataset loading, preprocessing and visualization."""

from pathlib import Path

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms import Compose, Normalize, ToTensor
from torchvision.utils import make_grid
from tqdm import tqdm


# Per-game differences are data, not separate Dataset subclasses.
# Defaults that the original load_data_and_data_loaders passed (fps=15,
# preload_ratio=1) intentionally remain in that function's signature.
DATASETS = {
    "PONG": dict(video="pong.mp4", cache="pong_frames.h5", size=(64, 64),
                 start=0, read_step=10),
    "SONIC": dict(video="sonic_frames.mp4", cache="sonic_frames.h5", size=(128, 128),
                  start=100, read_step=1),
    "POLE_POSITION": dict(video="pole_position.mp4", cache="pole_position_frames.h5",
                          size=(64, 64), start=0, read_step=1, video_slice=(1 / 50, 1 / 4)),
    "PICODOOM": dict(video="picodoom cleaned.mp4", cache="picodoom_frames.h5",
                     size=(128, 128), start=300, read_step=1),
    "ZELDA": dict(video="Zelda oot2d 1 Cut.mp4", cache="zelda_frames.h5",
                  size=(128, 128), start=1000, read_step=1),
}


class VideoDataset(Dataset):
    """Return normalized consecutive video frames as [T, C, H, W]."""

    def __init__(self, video_path, cache_path, num_frames, fps, size,
                 start=0, read_step=1, video_slice=None, preload_ratio=1,
                 train=True, split=False, transform=None):
        self.num_frames = num_frames
        self.frame_skip = max(1, 60 // fps)
        self.transform = transform or Compose([
            ToTensor(), Normalize((0.5,) * 3, (0.5,) * 3),
        ])

        if not cache_path.exists():
            frames = self._preprocess_video(video_path, size, read_step, video_slice)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(cache_path, "w") as f:
                f.create_dataset("frames", data=frames, compression="lzf")

        # One selection path for both freshly generated and existing caches.
        with h5py.File(cache_path, "r") as f:
            frames = f["frames"]
            self.data = frames[start:int(len(frames) * preload_ratio)]

        if split:
            cutoff = int(len(self.data) * 0.9)
            self.data = self.data[:cutoff] if train else self.data[cutoff:]

    @staticmethod
    def _preprocess_video(path, size, read_step, video_slice):
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise FileNotFoundError(f"Cannot open video: {path}")
        frames = []
        try:
            count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            for i in tqdm(range(0, count, max(1, read_step)), desc="Processing video"):
                capture.set(cv2.CAP_PROP_POS_FRAMES, i)
                ok, frame = capture.read()
                if not ok:
                    break
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(cv2.resize(frame, size, interpolation=cv2.INTER_AREA))
        finally:
            capture.release()

        data = np.asarray(frames, dtype=np.uint8)
        if video_slice is not None and len(data):
            a, b = video_slice
            if isinstance(a, float) or isinstance(b, float):
                a, b = int(len(data) * a), int(len(data) * b)
            a = max(0, min(len(data), int(a)))
            b = max(a, min(len(data), int(b)))
            data = data[a:b]
        return data

    def __len__(self):
        # Original length expression preserved, including its last-window exclusion.
        return max(0, len(self.data) - self.num_frames * self.frame_skip)

    def __getitem__(self, index):
        if not 0 <= index < len(self):
            raise IndexError(index)
        stop = index + self.num_frames * self.frame_skip
        frames = self.data[index:stop:self.frame_skip].astype(np.float32) / 255.0
        return torch.stack([self.transform(frame) for frame in frames])


class TokenizedVideoDataset(Dataset):
    """Precomputed video [T, P] and action [T - 1] tokens."""

    def __init__(self, path, num_frames):
        with h5py.File(path, 'r') as f:
            self.tokens = torch.from_numpy(f['tokens'][:])
            self.actions = torch.from_numpy(f['actions'][:])
        if self.tokens.shape[1] != num_frames:
            raise ValueError('Cached context_length does not match training config')

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, index):
        return self.tokens[index], self.actions[index]


def load_dataset(dataset, num_frames=1, fps=15, preload_ratio=1, cached_tokens_path=None):
    """Construct the dataset without creating data loaders."""
    if cached_tokens_path:
        return TokenizedVideoDataset(cached_tokens_path, num_frames)
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset {dataset!r}; expected one of {list(DATASETS)}")
    spec = DATASETS[dataset]
    data_dir = Path.cwd() / "data"
    options = dict(video_path=data_dir / spec["video"],
                   cache_path=data_dir / spec["cache"], num_frames=num_frames,
                   fps=30 if dataset == "PICODOOM" else fps, size=spec["size"],
                   start=spec["start"], read_step=spec["read_step"],
                   video_slice=spec.get("video_slice"), preload_ratio=preload_ratio)
    return VideoDataset(**options, train=True, split=False)


def load_data_and_data_loaders(dataset, batch_size, num_frames=1, distributed=False,
                               rank=0, world_size=1, fps=15, preload_ratio=1, cached_tokens_path=None):
    """Preserve the five-value return API; validation aliases the unsplit training data."""
    train_data = load_dataset(dataset, num_frames, fps, preload_ratio, cached_tokens_path)

    def loader(data, shuffle):
        sampler = DistributedSampler(data, num_replicas=world_size, rank=rank,
                                     shuffle=shuffle, drop_last=True) if distributed else None
        return DataLoader(data, batch_size=batch_size,
                          shuffle=shuffle if sampler is None else False,
                          sampler=sampler, num_workers=2, pin_memory=False,
                          persistent_workers=True, prefetch_factor=2, drop_last=True)

    train_loader = loader(train_data, shuffle=True)
    return train_data, train_data, train_loader, train_loader, None


def visualize_reconstruction(original, reconstruction, save_path=None):
    """Display up to four videos with up to four frames each, before/after."""
    original = original.detach().cpu().float()
    reconstruction = reconstruction.detach().cpu().float()
    if original.ndim == 4:
        original = original.unsqueeze(1)
    if reconstruction.ndim == 4:
        reconstruction = reconstruction.unsqueeze(1)

    count = min(4, original.shape[0])
    steps = min(4, original.shape[1])
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax, frames, title in zip(axes, (original, reconstruction),
                                 ("Original", "Reconstructed")):
        batch = frames[:count, :steps].reshape(-1, *frames.shape[2:])
        grid = make_grid(batch, nrow=steps, normalize=True, padding=2).clamp(0, 1)
        ax.imshow(grid.permute(1, 2, 0).numpy())
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)
