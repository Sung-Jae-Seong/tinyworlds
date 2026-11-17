import os
from typing import Dict, Iterable

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh, DeviceMesh
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import fully_shard

from models.video_tokenizer import VideoTokenizer
from models.latent_actions import LatentActionModel
from models.dynamics import DynamicsModel
from utils.config import DistributedConfig


def init_distributed_from_env() -> Dict[str, object]:
    """Initialize DeviceMesh from torchrun environment variables."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_distributed = world_size > 1 and torch.cuda.is_available()

    if is_distributed:
        device_mesh = init_device_mesh(
            "cuda", (world_size,), mesh_dim_names=("fsdp",)
        )
    else:
        device_mesh = None

    rank = device_mesh.get_rank() if is_distributed else 0

    return {
        "is_distributed": is_distributed,
        "world_size": world_size,
        "is_main": rank == 0,
        "device_mesh": device_mesh,
    }


def prepare_model_for_distributed(
    model: torch.nn.Module,
    config: DistributedConfig,
    device_mesh: DeviceMesh,
) -> torch.nn.Module:

    if config.use_ddp:
        return DDP(
            model,
            device_ids=[device_mesh.get_local_rank()],
            output_device=device_mesh.get_local_rank(),
            find_unused_parameters=False,
        )

    if config.use_fsdp:
        mp_policy = config.get_mixed_precision_policy()

        fsdp_kwargs = {
            "reshard_after_forward": config.reshard_after_forward,
            "mp_policy": mp_policy,
            "offload_policy": config.offload_policy,
        }

        def shard_layers(layers: Iterable[torch.nn.Module]) -> None:
            for layer in layers:
                fully_shard(layer, mesh=device_mesh, **fsdp_kwargs)

        # torch.compile()로 감싸진 모델도 구분할 수 있도록 처리
        original_model = getattr(model, "_orig_mod", model)

        if isinstance(original_model, (VideoTokenizer, LatentActionModel)):
            shard_layers(original_model.encoder.transformer.blocks)

            if isinstance(original_model, VideoTokenizer):
                shard_layers(original_model.encoder.latent_head)

            if isinstance(original_model, LatentActionModel):
                shard_layers(original_model.encoder.action_head)
                shard_layers(original_model.decoder.transformer.blocks)
                shard_layers(original_model.decoder.frame_head)

            fully_shard(
                original_model.encoder,
                mesh=device_mesh,
                **fsdp_kwargs,
            )
            fully_shard(
                original_model.decoder,
                mesh=device_mesh,
                **fsdp_kwargs,
            )
            fully_shard(
                original_model.quantizer,
                mesh=device_mesh,
                **fsdp_kwargs,
            )

        elif isinstance(original_model, DynamicsModel):
            shard_layers(original_model.transformer.blocks)

            fully_shard(
                original_model.latent_embed,
                mesh=device_mesh,
                **fsdp_kwargs,
            )
            fully_shard(
                original_model.output_mlp,
                mesh=device_mesh,
                **fsdp_kwargs,
            )

        else:
            raise ValueError(
                f"Unknown model type: {type(original_model).__name__}"
            )

        fully_shard(
            model,
            mesh=device_mesh,
            **fsdp_kwargs,
        )

    return model


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return model.module if isinstance(model, DDP) else model


def print_param_count_if_main(
    model: torch.nn.Module,
    model_name: str,
    is_main: bool,
) -> None:
    if not is_main:
        return

    try:
        params = sum(p.numel() for p in model.parameters())
        print(f"{model_name} parameters: {params / 1e6:.2f}M ({params})")
    except Exception:
        pass


def cleanup_distributed(is_distributed: bool) -> None:
    if is_distributed and dist.is_initialized():
        dist.destroy_process_group()
