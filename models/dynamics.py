import torch
import torch.nn as nn
import math
from models.positional_encoding import build_spatial_only_pe
from models.st_transformer import STTransformer

class DynamicsModel(nn.Module):
    def __init__(self, frame_size=(128, 128), patch_size=4, embed_dim=128, num_heads=8,
                 hidden_dim=128, num_blocks=4, num_bins=4, conditioning_dim=3, latent_dim=5,
                 use_moe=False, num_experts=4, top_k_experts=2, moe_aux_loss_coeff=0.01,
                 pos_encoding="sinusoidal", use_adaln_zero=False, qk_norm=True):
        super().__init__()
        H, W = frame_size
        codebook_size = num_bins**latent_dim
        self.pos_encoding = pos_encoding

        self.latent_embed = nn.Linear(latent_dim, embed_dim)
        self.transformer = STTransformer(
            embed_dim, num_heads, hidden_dim, num_blocks, causal=True,
            conditioning_dim=conditioning_dim,
            use_moe=use_moe, num_experts=num_experts,
            top_k_experts=top_k_experts, moe_aux_loss_coeff=moe_aux_loss_coeff,
            pos_encoding=pos_encoding, grid_size=(H // patch_size, W // patch_size),
            use_adaln_zero=use_adaln_zero, qk_norm=qk_norm,
        )
        self.output_mlp = nn.Linear(embed_dim, codebook_size)

        # shared spatial-only PE (zeros in temporal tail)
        pe_spatial = build_spatial_only_pe((H, W), patch_size, embed_dim, device='cpu', dtype=torch.float32)  # [1,P,E]
        self.register_buffer("pos_spatial_dec", pe_spatial, persistent=False)

        # learnable mask token latent
        # TODO; try leanable mask embedding in embed space instead of latent space
        self.mask_token = nn.Parameter(torch.randn(1, 1, 1, latent_dim) * 0.02)  # [1, 1, 1, L]

    def forward(self, discrete_latents, training=True, conditioning=None, targets=None):
        # discrete_latents: [B, T, P, L]
        # targets: [B, T, P] indices
        # conditioning: [B, T, A]
        B, T, P, L = discrete_latents.shape

        # convert latents to float for embedding
        discrete_latents = discrete_latents.to(dtype=torch.float32)

        # Match autoregressive rollout: preserve a complete past prefix and mask the future.
        # Mix fully masked futures with partially revealed futures for MaskGIT refinement.
        if training and self.training:
            ratio = torch.rand((), device=discrete_latents.device)
            ratio = torch.where(ratio < 0.5, 1.0, ratio)
            mask_positions = (torch.arange(T, device=discrete_latents.device)[None, :, None] >=
                              torch.randint(1, T, (), device=discrete_latents.device)) & (
                                  torch.rand(B, T, P, device=discrete_latents.device) < ratio)
            mask_positions[:, -1, 0] = True  # guarantee a supervised target

            # replace selected latents with mask tokens
            discrete_latents = torch.where(mask_positions.unsqueeze(-1), self.mask_token.to(discrete_latents), discrete_latents) # [B, T, P, L]
        else:
            mask_positions = None

        embeddings = self.latent_embed(discrete_latents)  # [B, T, P, E]

        # add spatial PE (affects only first 2/3 of dimensions)
        # STTransformer adds temporal PE to last 1/3 of dimensions
        if self.pos_encoding == "sinusoidal":
            embeddings = embeddings + self.pos_spatial_dec.to(embeddings.device, embeddings.dtype)
        transformed = self.transformer(embeddings, conditioning=conditioning)  # [B, T, P, E]

        # transform to logits for each token in codebook
        predicted_logits = self.output_mlp(transformed)  # [B, T, P, L^D]

        # compute masked cross-entropy loss
        loss = None
        if training and self.training:
            assert targets is not None, "target indices are needed for training"
            Ld = predicted_logits.shape[-1] # L^D
            logits_flat = predicted_logits.reshape(-1, Ld) # [(B*T*P), L^D]
            targets_flat = targets.reshape(-1) # [(B*T*P)]
            mask_flat = mask_positions.reshape(-1).to(torch.float32) # [(B*T*P)]
            loss_per = nn.functional.cross_entropy(logits_flat, targets_flat, reduction='none')  # [(B*T*P)]
            denom = mask_flat.sum().clamp_min(1.0)
            loss = (loss_per * mask_flat).sum() / denom

        return predicted_logits, mask_positions, loss  # logits, mask, optional loss

    def exp_schedule(self, t, T, P_total, k): # accumulated unmasked tokens scheduling. other:cosine
        x = (t + 1) / T
        if k == 0:
            return P_total * x
        return P_total * math.expm1(k * x) / math.expm1(k)

    def cosine_schedule(self, t, T, P_total):
        x = (t + 1) / T
        return P_total * (1 - math.cos(math.pi * x / 2))

    @torch.no_grad()
    def forward_inference(self, context_latents, prediction_horizon, num_steps, index_to_latents_fn, conditioning=None, schedule_k=5.0, temperature: float = 0.0, schedule="exp"):
        B, T_ctx, P, _ = context_latents.shape
        H = int(prediction_horizon)
        total = H * P
        use_sampling = bool(temperature and temperature > 0)

        input_latents = torch.cat([
            context_latents,
            self.mask_token.to(context_latents).expand(B, H, P, -1)
        ], dim=1)
        mask = torch.ones(B, H, P, dtype=torch.bool, device=context_latents.device)

        def predict_probs():
            logits, _, _ = self.forward(input_latents, training=False, conditioning=conditioning, targets=None)
            return torch.softmax(logits / float(temperature) if use_sampling else logits, dim=-1)

        def fill(b, positions, indices):
            h, p = positions // P, positions % P
            token_ids = indices[b, T_ctx + h, p].reshape(1, 1, -1)
            input_latents[b, T_ctx + h, p] = index_to_latents_fn(token_ids)[0, 0]
            mask[b, h, p] = False

        for step in range(num_steps):
            if schedule == "cosine":
                target = math.ceil(self.cosine_schedule(step, num_steps, total))
            else:
                target = math.ceil(self.exp_schedule(step, num_steps, total, schedule_k))
            probs = predict_probs()
            confidence = probs.max(dim=-1).values[:, -H:, :]

            if use_sampling:
                indices = torch.distributions.Categorical(
                    probs=probs.reshape(-1, probs.shape[-1])
                ).sample().view(probs.shape[:-1])
            else:
                indices = probs.argmax(dim=-1)

            for b in range(B):
                positions = mask[b].flatten().nonzero(as_tuple=True)[0]
                remaining = positions.numel()
                if remaining == 0:
                    continue

                prev = total - remaining
                k_floor = max(total // 16, 1)
                k = max(k_floor, min(max(target - prev, 0), remaining))
                if remaining > k:
                    scores = confidence[b].contiguous().view(-1)[positions]
                    positions = positions[scores.topk(k).indices]
                fill(b, positions, indices)

            if not mask.any():
                break

        if mask.any():  # num_steps <= 0 또는 예외적인 잔여 마스크
            indices = predict_probs().argmax(dim=-1)
            for b in range(B):
                positions = mask[b].flatten().nonzero(as_tuple=True)[0]
                if positions.numel():
                    fill(b, positions, indices)

        return input_latents
