import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from models.positional_encoding import build_spatial_only_pe

class PatchEmbedding(nn.Module):
    def __init__(self, frame_size=(128, 128), patch_size=8, embed_dim=128, pos_encoding="sinusoidal"):
        super().__init__()
        H, W = frame_size
        self.frame_size = frame_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.Hp, self.Wp = H // patch_size, W // patch_size
        self.num_patches = self.Hp * self.Wp
        self.pos_encoding = pos_encoding
        assert H % patch_size == 0 and W % patch_size == 0

        self.register_buffer(
            "pos_spatial", build_spatial_only_pe(frame_size, patch_size, embed_dim),
            persistent=False
        )
        # Keep the original parameter shape for checkpoint compatibility.
        self.proj = nn.Conv2d(3 * patch_size * patch_size, embed_dim, 1)

    def forward(self, frames):
        B, T, _, _, _ = frames.shape
        x = rearrange(frames, 'b t c h w -> (b t) c h w')
        weight = self.proj.weight.reshape(self.embed_dim, 3, self.patch_size, self.patch_size)
        x = F.conv2d(x, weight, self.proj.bias, stride=self.patch_size)
        x = rearrange(x, '(b t) e hp wp -> b t (hp wp) e', b=B, t=T)
        if self.pos_encoding == "sinusoidal":
            x = x + self.pos_spatial.to(x)
        return x
