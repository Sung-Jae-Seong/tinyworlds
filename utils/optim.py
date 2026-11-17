from __future__ import annotations

import math
import torch
import torch.optim as optim


def create_optimizer(model, args):
    from torch.nn.parallel import DistributedDataParallel as DDP
    raw_model = model.module if isinstance(model, DDP) else model

    optimizer_name = getattr(args, "optimizer", "adamw")

    if optimizer_name == "muon":
        return _create_muon_split(raw_model, args)
    else:
        return _create_adamw(raw_model, args)


def _create_adamw(model, args):
    decay, no_decay = _split_decay_params(model)
    optimizer = optim.AdamW([
        {"params": decay, "weight_decay": 0.01},
        {"params": no_decay, "weight_decay": 0},
    ], lr=args.learning_rate, betas=(0.9, 0.999), eps=1e-8, fused=True)
    return [optimizer]


def _create_muon_split(model, args):
    from models.muon import Muon

    muon_params = []
    adamw_decay = []
    adamw_no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        # Muon only makes sense for 2D weight matrices (not embeddings, not biases)
        if param.ndim == 2 and "embed" not in name:
            muon_params.append(param)
        elif param.ndim == 1 or name.endswith(".bias") or "norm" in name:
            adamw_no_decay.append(param)
        else:
            adamw_decay.append(param)

    lr = args.learning_rate
    momentum = getattr(args, "muon_momentum", 0.95)
    backend_steps = getattr(args, "muon_backend_steps", 5)

    optimizers = []

    if muon_params:
        muon_opt = Muon(
            muon_params, lr=lr, momentum=momentum,
            backend_steps=backend_steps, weight_decay=0.01,
        )
        optimizers.append(muon_opt)

    # AdamW for the rest
    adamw_groups = []
    if adamw_decay:
        adamw_groups.append({"params": adamw_decay, "weight_decay": 0.01})
    if adamw_no_decay:
        adamw_groups.append({"params": adamw_no_decay, "weight_decay": 0})
    if adamw_groups:
        adamw_opt = optim.AdamW(adamw_groups, lr=lr, betas=(0.9, 0.999), eps=1e-8, fused=True)
        optimizers.append(adamw_opt)

    return optimizers


def _split_decay_params(model):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if len(param.shape) == 1 or name.endswith(".bias") or "norm" in name:
                no_decay.append(param)
            else:
                decay.append(param)
    return decay, no_decay

def create_cosine_scheduler(optimizer, total_steps, warmup_fraction=0.05, min_lr_fraction=0.01):
    warmup_steps = int(total_steps * warmup_fraction)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)

        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_fraction + (1 - min_lr_fraction) * cosine

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
