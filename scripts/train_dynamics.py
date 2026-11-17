from contextlib import nullcontext
from pathlib import Path
import torch
import os
from tqdm import tqdm
from einops import rearrange
from models.dynamics import DynamicsModel
from utils.datasets import visualize_reconstruction, load_data_and_data_loaders
from utils.optim import create_cosine_scheduler
from utils.optim import create_optimizer
from utils.utils import (
    save_training_state,
    load_videotokenizer_from_checkpoint,
    load_latent_actions_from_checkpoint,
    load_dynamics_from_checkpoint,
    prepare_pipeline_run_root,
    prepare_stage_dirs,
)
from utils.config import DynamicsConfig, load_stage_config_merged
from utils.distributed import init_distributed_from_env, prepare_model_for_distributed, unwrap_model, print_param_count_if_main, cleanup_distributed
from torch.distributed.fsdp import FSDPModule

def main():
    # dynamics config merged with training_config.yaml (training takes priority), plus CLI overrides
    args: DynamicsConfig = load_stage_config_merged(DynamicsConfig, default_config_path=os.path.join(os.getcwd(), 'configs', 'dynamics.yaml'))
    # DDP setup
    dist_setup = init_distributed_from_env()

    # run save dir if it doesn't exist (running not from full train)
    is_main = dist_setup['is_main']
    run_root = os.environ.get('NG_RUN_ROOT_DIR')
    if not run_root:
        run_root, _ = prepare_pipeline_run_root(base_cwd=os.getcwd())
    stage_dir, checkpoints_dir, visualizations_dir = prepare_stage_dirs(run_root, 'dynamics')
    if is_main:
        print(f"Dynamics Training")
        print(f"Results will be saved in {stage_dir}")

    # load video tokenizer and latent action model
    video_tokenizer, _ = load_videotokenizer_from_checkpoint(
        checkpoint_path=args.video_tokenizer_path,
        device=args.device,
        is_distributed=dist_setup['is_distributed'],
    )
    video_tokenizer.eval().requires_grad_(False)
    latent_action_model, _ = load_latent_actions_from_checkpoint(
        checkpoint_path=args.latent_actions_path,
        device=args.device,
        is_distributed=dist_setup['is_distributed'],
    )
    unwrap_model(latent_action_model).eval().requires_grad_(False)

    # init dynamics model and optional ckpt load
    dynamics_model = DynamicsModel(
        frame_size=(args.frame_size, args.frame_size),
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        conditioning_dim=unwrap_model(latent_action_model).action_dim,
        latent_dim=args.latent_dim,
        num_bins=args.num_bins,
        use_moe=args.use_moe,
        num_experts=args.num_experts,
        top_k_experts=args.top_k_experts,
        moe_aux_loss_coeff=args.moe_aux_loss_coeff,
        pos_encoding=args.pos_encoding,
        use_adaln_zero=args.use_adaln_zero,
    ).to(args.device)
    if args.checkpoint:
        dynamics_model, _ = load_dynamics_from_checkpoint(
            checkpoint_path=args.checkpoint, 
            device=args.device, 
            model=dynamics_model,
            is_distributed=dist_setup['is_distributed'],
        )

    # optional DDP, compile, param count, tf32
    print_param_count_if_main(dynamics_model, "DynamicsModel", is_main)
    if args.compile:
        video_tokenizer = torch.compile(video_tokenizer, mode="reduce-overhead", fullgraph=False, dynamic=True)
        latent_action_model = torch.compile(latent_action_model, mode="reduce-overhead", fullgraph=False, dynamic=True)
        dynamics_model = torch.compile(dynamics_model, mode="reduce-overhead", fullgraph=False, dynamic=True)
        print("Compiled all models for training.")
    dynamics_model = prepare_model_for_distributed(
        dynamics_model, 
        args.distributed, 
        device_mesh=dist_setup['device_mesh'],
    )
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # create optimizer(s) — AdamW or Muon+AdamW split
    optimizers = create_optimizer(dynamics_model, args)

    # cosine scheduler for lr warmup and AMP grad scaler
    schedulers = [create_cosine_scheduler(opt, args.n_updates) for opt in optimizers]
    train_ctx = torch.amp.autocast(args.device, enabled=True, dtype=torch.bfloat16) if args.amp and not args.distributed.use_fsdp else nullcontext()

    loss_vals = []

    unwrap_model(dynamics_model).train()

    # dataloader
    data_overrides = {}
    if args.fps is not None:
        data_overrides['fps'] = args.fps
    if args.preload_ratio is not None:
        data_overrides['preload_ratio'] = args.preload_ratio
    _, _, training_loader, _, _ = load_data_and_data_loaders(
        dataset=args.dataset, 
        batch_size=args.batch_size_per_gpu,
        num_frames=args.context_length,
        distributed=dist_setup['is_distributed'],
        rank=dist_setup['device_mesh'].get_rank() if dist_setup['device_mesh'] is not None else 0,
        world_size=dist_setup['world_size'],
        **data_overrides,
        cached_tokens_path=args.cached_tokens_path,
    )
    using_cached_tokens = bool(args.cached_tokens_path)
    train_iter = iter(training_loader)

    for i in tqdm(range(0, args.n_updates), disable=not is_main):
        update_loss = torch.zeros((), device=args.device)
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        if isinstance(dynamics_model, FSDPModule):
            dynamics_model.set_requires_gradient_sync(False)
        if args.compile:
            torch.compiler.cudagraph_mark_step_begin()
        for micro_batch in range(args.gradient_accumulation_steps):
            try:
                x = next(train_iter)
            except StopIteration:
                train_iter = iter(training_loader)  # reset iterator when epoch ends
                x = next(train_iter)

            if using_cached_tokens:
                video_tokens, action_tokens = (t.to(args.device, non_blocking=True) for t in x)
                video_tokens = video_tokens.long()
                quantized_actions = (latent_action_model.quantizer.get_latents_from_indices(action_tokens)
                                     if args.use_actions else None)
            else:
                x = x.to(args.device, non_blocking=True)
                video_tokens = video_tokenizer.tokenize(x)
                quantized_actions = latent_action_model.encode(x) if args.use_actions else None

            video_latents = video_tokenizer.quantizer.get_latents_from_indices(video_tokens, dim=-1)

            # predict masked frame latents with dynamics model (masking in dynamics model)
            with train_ctx:
                predicted_next_logits, mask_positions, loss = dynamics_model(
                    video_latents, training=True, conditioning=quantized_actions, targets=video_tokens
                )

                # add MoE load-balancing auxiliary loss
                if args.use_moe:
                    aux_loss = unwrap_model(dynamics_model).transformer.moe_aux_loss()
                    loss = loss + aux_loss

                if isinstance(dynamics_model, FSDPModule) and micro_batch + 1 == args.gradient_accumulation_steps:
                    dynamics_model.set_requires_gradient_sync(True)

                loss /= args.gradient_accumulation_steps
                update_loss += loss.detach()
                loss.backward()

        loss_vals.append(update_loss.cpu())

        torch.nn.utils.clip_grad_norm_(unwrap_model(dynamics_model).parameters(), max_norm=1.0)
        for opt in optimizers:
            opt.step()
        for sched in schedulers:
            sched.step()

        # save model and visualize results
        if i % args.log_interval == 0:
            predicted_next_indices = torch.argmax(predicted_next_logits, dim=-1)
            predicted_next_latents = video_tokenizer.quantizer.get_latents_from_indices(predicted_next_indices, dim=-1)
            with torch.no_grad():
                predicted_frames = video_tokenizer.decoder(predicted_next_latents[:16]) # [B, T, C, H, W]

            # convert mask_positions to patch-level mask for visualization
            grid_size = args.frame_size // args.patch_size
            pixel_mask = rearrange(mask_positions.bool(), 'b t (h w) -> b t h w', h=grid_size)
            pixel_mask = pixel_mask.repeat_interleave(args.patch_size, dim=-2)
            pixel_mask = pixel_mask.repeat_interleave(args.patch_size, dim=-1)
            reference_frames = video_tokenizer.decoder(video_latents[:16]) if using_cached_tokens else x[:16]
            masked_frames = reference_frames * (~pixel_mask[:len(reference_frames)]).unsqueeze(2)
        
            ckpt_path = save_training_state(dynamics_model, optimizers[0], schedulers[0], vars(args), checkpoints_dir, prefix='dynamics', step=i)
            # save secondary optimizer/scheduler state when using split optimizers (Muon+AdamW)
            if len(optimizers) > 1:
                torch.save(
                    {f'optimizer_{i}': o.state_dict() for i, o in enumerate(optimizers)},
                    Path(ckpt_path) / 'all_optimizers.pt',
                )
                torch.save(
                    {f'scheduler_{i}': s.state_dict() for i, s in enumerate(schedulers)},
                    Path(ckpt_path) / 'all_schedulers.pt',
                )
            if is_main:
                save_path = os.path.join(visualizations_dir, f'dynamics_prediction_step_{i}.png')
                visualize_reconstruction(masked_frames[:16].cpu(), predicted_frames[:16].cpu(), save_path)

            print('\n Step', i, 'Loss:', torch.mean(torch.stack(loss_vals[-args.log_interval:])).item())

    cleanup_distributed(dist_setup['is_distributed'])

if __name__ == "__main__":
    main()
