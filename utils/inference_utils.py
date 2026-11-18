import torch
import time
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np
from utils.utils import load_videotokenizer_from_checkpoint, load_dynamics_from_checkpoint
from models.fsq import FiniteScalarQuantizer
import math

def load_models(video_tokenizer_path, dynamics_path, device):
    video_tokenizer, _ = load_videotokenizer_from_checkpoint(video_tokenizer_path, device)
    dynamics_model, _ = load_dynamics_from_checkpoint(dynamics_path, device)
    video_tokenizer.eval()
    dynamics_model.eval()

    return video_tokenizer, dynamics_model


def create_action_quantizer(n_actions, device):
    """FSQ has a fixed grid, so no latent-action checkpoint is needed for ID decoding."""
    if n_actions < 2 or n_actions & (n_actions - 1):
        raise ValueError("n_actions must be a power of two, at least 2")
    return FiniteScalarQuantizer(
        latent_dim=int(math.log2(n_actions)), num_bins=2
    ).to(device)


def visualize_inference(predicted_frames, ground_truth_frames, inferred_actions, fps):
    # Move to CPU and convert to numpy
    predicted_frames = predicted_frames.detach().cpu()
    ground_truth_frames = ground_truth_frames.detach().cpu()
    
    # Denormalize frames from [-1, 1] to [0, 1]
    predicted_frames = (predicted_frames + 1) / 2
    predicted_frames = torch.clamp(predicted_frames, 0, 1)
    ground_truth_frames = (ground_truth_frames + 1) / 2
    ground_truth_frames = torch.clamp(ground_truth_frames, 0, 1)

    # Get dimensions
    B, T, C, H, W = predicted_frames.shape

    _, num_gt_frames, _, _, _ = ground_truth_frames.shape
    
    # Create figure with ground truth and predictions side by side
    fig, axes = plt.subplots(2, T, figsize=(4 * T, 8), squeeze=False)

    # Plot ground truth frames (top row)
    for i in range(num_gt_frames):
        frame = ground_truth_frames[0, i].permute(1, 2, 0).numpy()  # [H, W, C]
        axes[0, i].imshow(frame)
        axes[0, i].set_title(f'Ground Truth {i+1}', fontsize=12, color='green')
        axes[0, i].axis('off')

    # Plot predicted frames (bottom row)
    for i in range(T):
        frame = predicted_frames[0, i].permute(1, 2, 0).numpy()  # [H, W, C]
        axes[1, i].imshow(frame)
        title = f'Predicted {i+1}'
        if i < len(inferred_actions):
            title += f'\nAction {inferred_actions[i].item()}'
        axes[1, i].set_title(title, fontsize=12, color='red')
        axes[1, i].axis('off')
    
    plt.suptitle('Ground Truth vs Predicted Frames', fontsize=16, fontweight='bold')
    
    # Save the visualization
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_dir = "inference_results"
    os.makedirs(save_dir, exist_ok=True)
    
    save_path = os.path.join(save_dir, f"inference_results_gt_vs_pred_{timestamp}.png")
    mp4_path = os.path.join(save_dir, f"inference_video_{timestamp}.mp4")
    
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to: {save_path}")

    all_frames = torch.cat([ground_truth_frames, predicted_frames], dim=1)
    save_frames_as_mp4(all_frames, mp4_path, fps)
    
    # Calculate and display reconstruction error
    mse_error = torch.mean((predicted_frames - ground_truth_frames) ** 2).item()
    print(f"\nInference stats:")
    print(f"Total frames generated: {T}")
    print(f"Mean Squared Error (GT vs Pred): {mse_error:.6f}")
    print(f"Actions used: {[action.item() for action in inferred_actions]}")


# TODO: get working mp4
def save_frames_as_mp4(frames, output_path, fps=2):
    B, T, C, H, W = frames.shape

    # OpenCV expects (W, H)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    for i in range(T):
        frame = frames[0, i].detach().cpu().permute(1, 2, 0).numpy()  # [H, W, C]
        # Ensure float32
        frame = frame.astype(np.float32)
        # Clamp and scale
        frame = np.clip(frame, 0, 1)
        frame = (frame * 255).astype(np.uint8)
        # If grayscale, convert to 3 channels
        if frame.shape[2] == 1:
            frame = np.repeat(frame, 3, axis=2)
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    print(f"MP4 video saved to: {output_path}")


def get_action_latent(args, inferred_actions, n_actions, context_frames,
                      action_quantizer, step):
    if args.use_interactive_mode:
        action_id = int(input(f"Enter action id for step {step + 1}: "))
    else:
        action_id = torch.randint(n_actions, ()).item()

    action = torch.tensor([action_id], device=args.device, dtype=torch.long)

    past_length = context_frames.shape[1] - 1
    past_ids = [a.item() for a in inferred_actions][-past_length:] if past_length else []
    past_ids = [action_id] * (past_length - len(past_ids)) + past_ids
    past_latents = action_quantizer.get_latents_from_indices(
        torch.tensor([past_ids], device=args.device, dtype=torch.long)
    )

    future_ids = action.repeat(args.prediction_horizon).unsqueeze(0)
    future_latents = action_quantizer.get_latents_from_indices(future_ids)
    inferred_actions.extend([action] * args.prediction_horizon)
    return action, torch.cat([past_latents, future_latents], dim=1)
