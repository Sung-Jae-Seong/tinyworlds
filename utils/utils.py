from pathlib import Path
import time
import glob
import subprocess
import os
import re
from typing import Optional

from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    StateDictOptions,
)
from torch.distributed.fsdp import FSDPModule
from torch.nn.parallel import DistributedDataParallel as DDP

MODEL_CHECKPOINT = "model_state_dict.pt"
OPTIMIZER_CHECKPOINT = "optim_state_dict.pt"
STATE = "state.pt"

def readable_timestamp():
    """Generate a sortable timestamp for filenames (no weekday)."""
    return time.strftime("%Y_%m_%d_%H_%M_%S")

def find_latest_checkpoint(base_dir, model_name, run_root_dir: Optional[str] = None, stage_name: Optional[str] = None):
    """Select the highest-step checkpoint in the newest matching stage directory.

    Prefer the requested run root when it has matches; otherwise search results/.
    """
    aliases = {
        'video_tokenizer': ['video_tokenizer'],
        'latent_actions': ['latent_actions', 'lam', 'actions', 'action_tokenizer'],
        'dynamics': ['dynamics'],
    }.get(model_name, [model_name])

    def collect(roots):
        found = set()
        for root in roots:
            for alias in aliases:
                for pattern in (f"*{alias}_step_*", f"*{alias}_checkpoint_*"):
                    for path in glob.glob(os.path.join(root, '**', pattern), recursive=True):
                        path = os.path.normpath(path)
                        if os.path.isdir(path):
                            valid = any(os.path.isfile(os.path.join(path, f)) for f in (STATE, MODEL_CHECKPOINT))
                        else:
                            valid = os.path.splitext(path)[1] in ('.pt', '.pth')
                        if valid:
                            found.add(path)
        return list(found)

    def project_matches():
        base = os.path.join(base_dir, 'results', '**')
        stage_dirs = {
            'video_tokenizer': ['video_tokenizer'],
            'latent_actions': ['latent_actions', 'lam', 'actions', 'action_tokenizer'],
            'dynamics': ['dynamics'],
        }.get(model_name)
        roots = ([os.path.join(base, stage, 'checkpoints') for stage in stage_dirs]
                 if stage_dirs else [os.path.join(base, 'checkpoints')])
        return collect(roots) or collect([os.path.join(base, 'checkpoints')])

    if run_root_dir is not None:
        root = os.path.join(run_root_dir, stage_name, 'checkpoints') if stage_name else os.path.join(run_root_dir, '**', 'checkpoints')
        files = collect([root]) or project_matches()
    else:
        files = project_matches()

    if not files:
        raise Exception(f"No checkpoint files found for {model_name}")

    def run_dir(path):
        directory = path if os.path.isdir(path) else os.path.dirname(path)
        current = directory
        while current and os.path.basename(current) != 'checkpoints':
            parent = os.path.dirname(current)
            if parent == current:
                break
            current = parent
        return os.path.dirname(current) if os.path.basename(current) == 'checkpoints' else directory

    groups = {}
    for path in files:
        groups.setdefault(run_dir(path), []).append(path)
    newest = max(groups, key=os.path.getctime)

    def order(path):
        match = re.search(r"_step_(\d+)", os.path.basename(path))
        return (int(match.group(1)) if match else -1, os.path.getctime(path))

    return max(groups[newest], key=order)

def run_command(cmd, description):
    # empirical max dataLoader throughput settings (I used 1-6 H100s)
    env = os.environ.copy()
    env.setdefault("NG_NUM_WORKERS", str(max(2, (os.cpu_count() or 4) - 2)))
    env.setdefault("NG_PREFETCH_FACTOR", "4")
    env.setdefault("NG_PIN_MEMORY", "1")
    env["NG_PERSISTENT_WORKERS"] = "0"
    env.setdefault("TORCH_CUDNN_V8_API_ENABLED", "1")

    try:
        subprocess.run(cmd, check=True, env=env)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        return False
    except KeyboardInterrupt:
        return False

def save_training_state(model, optimizer, scheduler, config, checkpoints_dir, prefix, step):
    """Save model, optimizer, scheduler metadata, and config under a step directory."""
    import torch
    ts = readable_timestamp()
    if isinstance(model, (FSDPModule, DDP)):
        state_dict = get_model_state_dict(
            model=model,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
            ),
        )
        optimizer_state_dict = get_optimizer_state_dict(
            model=model,
            optimizers=optimizer,
            options=StateDictOptions(
                full_state_dict=True,
                cpu_offload=True,
            ),
        )
    else:
        # Avoid saving the model with _orig_mod prefix if it's compiled
        state_dict = getattr(model, '_orig_mod', model).state_dict()
        optimizer_state_dict = optimizer.state_dict()
    state = {
        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
        'config': config,
        'step': int(step) if step is not None else None,
        'timestamp': ts,
    }
    os.makedirs(checkpoints_dir, exist_ok=True)
    ckpt_path = os.path.join(checkpoints_dir, f"{prefix}_step_{int(step) if step is not None else 0}")
    os.makedirs(ckpt_path, exist_ok=True)
    torch.save(state_dict, Path(ckpt_path) / MODEL_CHECKPOINT)
    torch.save(optimizer_state_dict, Path(ckpt_path) / OPTIMIZER_CHECKPOINT)
    torch.save(state, Path(ckpt_path) / STATE)
    return ckpt_path


def _read_checkpoint(checkpoint_path):
    import torch
    ckpt = Path(checkpoint_path)
    model_sd = torch.load(ckpt if ckpt.is_file() else ckpt / MODEL_CHECKPOINT, map_location='cpu', weights_only=True)
    state = torch.load(ckpt / STATE, map_location='cpu', weights_only=False) if ckpt.is_dir() else {'config': {}}
    if ckpt.is_file() and 'model' in model_sd:
        state = {k: v for k, v in model_sd.items() if k != 'model'}
        model_sd = model_sd['model']
    return model_sd, state


def load_videotokenizer_from_checkpoint(checkpoint_path, device, model = None, is_distributed = False):
    """Instantiate VideoTokenizer from a checkpoint's saved config and load weights."""
    import torch
    from models.video_tokenizer import VideoTokenizer
    model_sd, state_cfg = _read_checkpoint(checkpoint_path)
    cfg = state_cfg.get('config', {}) or {}
    frame_size = cfg.get('frame_size', 128)
    kwargs = {
        'frame_size': (frame_size, frame_size),
        'patch_size': cfg.get('patch_size', 8),
        'embed_dim': cfg.get('embed_dim', 128),
        'num_heads': cfg.get('num_heads', 8),
        'hidden_dim': cfg.get('hidden_dim', 256),
        'num_blocks': cfg.get('num_blocks', 4),
        'latent_dim': cfg.get('latent_dim', 6),
        'num_bins': cfg.get('num_bins', 4),
    }
    if model is None:
        model = VideoTokenizer(**kwargs)
    set_model_state_dict(
        model=model,
        model_state_dict=model_sd,
        options=StateDictOptions(
            full_state_dict=True,
            broadcast_from_rank0=is_distributed,
        ),
    )
    model = model.to(device)
    return model, state_cfg


def load_latent_actions_from_checkpoint(checkpoint_path, device, model = None, is_distributed = False):
    """Instantiate LatentActionModel from a checkpoint's saved config and load weights."""
    import torch
    from models.latent_actions import LatentActionModel
    model_sd, state_cfg = _read_checkpoint(checkpoint_path)
    cfg = state_cfg.get('config', {}) or {}
    frame_size = cfg.get('frame_size', 128)
    kwargs = {
        'frame_size': (frame_size, frame_size),
        'n_actions': cfg.get('n_actions', 8),
        'patch_size': cfg.get('patch_size', 8),
        'embed_dim': cfg.get('embed_dim', 128),
        'num_heads': cfg.get('num_heads', 8),
        'hidden_dim': cfg.get('hidden_dim', 256),
        'num_blocks': cfg.get('num_blocks', 4),
    }
    if model is None:
        model = LatentActionModel(**kwargs)
    set_model_state_dict(
        model=model,
        model_state_dict=model_sd,
        options=StateDictOptions(
            full_state_dict=True,
            broadcast_from_rank0=is_distributed,
        ),
    )
    model = model.to(device)
    return model, state_cfg


def load_dynamics_from_checkpoint(checkpoint_path, device, model = None, is_distributed = False):
    """Instantiate DynamicsModel from a checkpoint's saved config and load weights."""
    import torch
    from models.dynamics import DynamicsModel
    model_sd, state_cfg = _read_checkpoint(checkpoint_path)
    cfg = state_cfg.get('config', {}) or {}
    frame_size = cfg.get('frame_size', 128)
    # Infer conditioning_dim from checkpoint if missing
    conditioning_dim = cfg.get('conditioning_dim', None)
    if conditioning_dim is None:
        cond_inferred = None
        for k, v in model_sd.items():
            # Linear weight shape: [out_features, in_features]; in_features is conditioning dim
            if k.endswith('to_gamma_beta.1.weight'):
                cond_inferred = int(v.shape[1])
                break
        conditioning_dim = cond_inferred if cond_inferred is not None else 3
    kwargs = {
        'frame_size': (frame_size, frame_size),
        'patch_size': cfg.get('patch_size', 8),
        'embed_dim': cfg.get('embed_dim', 128),
        'num_heads': cfg.get('num_heads', 8),
        'hidden_dim': cfg.get('hidden_dim', 256),
        'num_blocks': cfg.get('num_blocks', 4),
        'conditioning_dim': conditioning_dim,
        'latent_dim': cfg.get('latent_dim', 6),
        'num_bins': cfg.get('num_bins', 4),
        'use_moe': cfg.get('use_moe', False),
        'num_experts': cfg.get('num_experts', 4),
        'top_k_experts': cfg.get('top_k_experts', 2),
        'moe_aux_loss_coeff': cfg.get('moe_aux_loss_coeff', 0.01),
    }
    if model is None:
        model = DynamicsModel(**kwargs)
    set_model_state_dict(
        model=model,
        model_state_dict=model_sd,
        options=StateDictOptions(
            full_state_dict=True,
            broadcast_from_rank0=is_distributed,
        )
    )
    model = model.to(device)
    return model, state_cfg

def prepare_pipeline_run_root(run_name: Optional[str] = None, base_cwd: Optional[str] = None):
    """Create a top-level run root directory results/<timestamp_or_name>"""
    cwd = base_cwd or os.getcwd()
    ts = readable_timestamp()
    name = run_name or ts
    run_root = os.path.join(cwd, 'results', name)
    os.makedirs(run_root, exist_ok=True)
    return run_root, name


def prepare_stage_dirs(run_root_dir: str, stage_name: str):
    """Create stage subdirectories under the given run root.

    Structure:
      <run_root_dir>/<stage_name>/checkpoints
      <run_root_dir>/<stage_name>/visualizations

    Returns (stage_dir, checkpoints_dir, visualizations_dir).
    """
    stage_dir = os.path.join(run_root_dir, stage_name)
    checkpoints_dir = os.path.join(stage_dir, 'checkpoints')
    visualizations_dir = os.path.join(stage_dir, 'visualizations')
    os.makedirs(checkpoints_dir, exist_ok=True)
    os.makedirs(visualizations_dir, exist_ok=True)
    return stage_dir, checkpoints_dir, visualizations_dir
