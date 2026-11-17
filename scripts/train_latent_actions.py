from contextlib import nullcontext
import torch
import os
from models.latent_actions import LatentActionModel
from utils.datasets import load_data_and_data_loaders, visualize_reconstruction
from utils.optim import create_cosine_scheduler
from utils.optim import create_optimizer
from tqdm import tqdm
from utils.utils import save_training_state, prepare_stage_dirs, prepare_pipeline_run_root, load_latent_actions_from_checkpoint
from utils.config import LatentActionsConfig, load_stage_config_merged
from utils.distributed import init_distributed_from_env, prepare_model_for_distributed, unwrap_model, print_param_count_if_main, cleanup_distributed
from torch.distributed.fsdp import FSDPModule

def main():
    # latent actions config merged with training_config.yaml (training takes priority), plus CLI overrides
    args: LatentActionsConfig = load_stage_config_merged(LatentActionsConfig, default_config_path=os.path.join(os.getcwd(), 'configs', 'latent_actions.yaml'))

    # DDP setup
    dist_setup = init_distributed_from_env()

    # run save dir if it doesn't exist (running not from full train)
    run_root = os.environ.get('NG_RUN_ROOT_DIR')
    if not run_root:
        run_root, _ = prepare_pipeline_run_root(base_cwd=os.getcwd())
    is_main = dist_setup['is_main']
    stage_dir, checkpoints_dir, visualizations_dir = prepare_stage_dirs(run_root, 'latent_actions')
    if is_main:
        print(f"Latent Actions Training")
        print(f'Results will be saved in {stage_dir}')

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
    )

    # init model and optional ckpt load
    model = LatentActionModel(
        frame_size=(args.frame_size, args.frame_size),
        patch_size=args.patch_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        hidden_dim=args.hidden_dim,
        num_blocks=args.num_blocks,
        n_actions=args.n_actions,
    ).to(args.device)
    if args.checkpoint:
        model, _ = load_latent_actions_from_checkpoint(
            args.checkpoint, 
            args.device,
            model,
            dist_setup['is_distributed'],
        )

    # optional DDP, compile, param count, tf32
    print_param_count_if_main(model, "LatentActionModel", is_main)
    if args.compile:
        model = torch.compile(model, mode="reduce-overhead", fullgraph=False, dynamic=True)
    model = prepare_model_for_distributed(
        model, 
        args.distributed, 
        device_mesh=dist_setup['device_mesh'],
    )
    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # create optimizer(s) — AdamW or Muon+AdamW split
    optimizers = create_optimizer(model, args)

    # cosine scheduler for lr warmup and AMP
    schedulers = [create_cosine_scheduler(opt, args.n_updates) for opt in optimizers]
    train_ctx = torch.amp.autocast(args.device, enabled=True, dtype=torch.bfloat16) if args.amp and not args.distributed.use_fsdp else nullcontext()

    unwrap_model(model).train()

    train_iter = iter(training_loader)
    for i in tqdm(range(args.n_updates), disable=not is_main):
        update_loss = torch.zeros((), device=args.device)
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        if isinstance(model, FSDPModule):
            model.set_requires_gradient_sync(False)
        if args.compile:
            torch.compiler.cudagraph_mark_step_begin()
        for micro_batch in range(args.gradient_accumulation_steps):
            try:
                x = next(train_iter)
            except StopIteration:
                train_iter = iter(training_loader)
                x = next(train_iter)

            x = x.to(args.device, non_blocking=True)

            with train_ctx:
                loss, pred_frames = model(x)
                loss /= args.gradient_accumulation_steps
                if isinstance(model, FSDPModule) and micro_batch + 1 == args.gradient_accumulation_steps:
                    model.set_requires_gradient_sync(True)
                update_loss += loss.detach()
                loss.backward()

        torch.nn.utils.clip_grad_norm_(unwrap_model(model).parameters(), max_norm=1.0)
        for opt in optimizers:
            opt.step()
        for sched in schedulers:
            sched.step()

        # save model and visualize results
        if i % args.log_interval == 0:
            hyperparameters = vars(args)
            save_training_state(model, optimizers[0], None, hyperparameters, checkpoints_dir, prefix='latent_actions', step=i)
            if is_main:
                save_path = os.path.join(visualizations_dir, f'reconstructions_latent_actions_step_{i}.png')
                visualize_reconstruction(x, pred_frames, save_path)
            
                print('\n Step', i, 'Loss:', update_loss.item())

    cleanup_distributed(dist_setup['is_distributed'])

if __name__ == "__main__":
    main()
