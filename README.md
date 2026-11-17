# TinyWorlds

TinyWorlds is a minimal autoregressive world model inspired by [Genie](https://arxiv.org/pdf/2402.15391). It learns discrete video and latent-action tokens from gameplay video without action labels, then predicts future frames conditioned on those actions.

## Getting started

From the project root, install dependencies:

```bash
python -m pip install -r requirements.txt
```

Download preprocessed gameplay `.h5` files from the [TinyWorlds datasets](https://huggingface.co/datasets/AlmondGod/tinyworlds/tree/main) and place them in `data/`. Select a dataset in `configs/training.yaml` and ensure the frame size and other model settings match it.

`configs/training.yaml` currently enables Weights & Biases logging; configure its credentials or set `use_wandb: false` if logging is not needed.

Train the video tokenizer, latent-action model, and dynamics model in sequence:

```bash
python scripts/full_train.py --config configs/training.yaml
```

For inference, configure `configs/inference.yaml` with the dataset, device, and checkpoint paths (or enable latest-checkpoint discovery), then run:

```bash
python scripts/run_inference.py --config configs/inference.yaml
```

Configuration values can be overridden with `key=value` arguments. For example:

```bash
python scripts/run_inference.py --config configs/inference.yaml dataset=SONIC device=cuda
```

The inference `n_actions` setting must match the action codebook used by the loaded checkpoint. See `configs/` for stage-specific training and inference settings.

## Architecture

| Component | Role |
| --- | --- |
| Video Tokenizer | Compresses frames into discrete tokens using a space-time transformer and finite scalar quantization (FSQ). |
| Action Tokenizer | Infers latent actions between frames without action labels and quantizes them into action tokens. |
| Dynamics Model | Predicts masked future-frame tokens conditioned on frame history and action tokens. |

The space-time transformer attends across patches within a frame and across timesteps. The action-conditioned path uses FiLM. During dynamics inference, masked tokens are filled iteratively using a MaskGIT-style schedule, and the resulting tokens are decoded into frames.

Relevant methods: [space-time attention](https://arxiv.org/pdf/2001.02908), [FSQ](https://arxiv.org/pdf/2309.15505), [FiLM](https://arxiv.org/pdf/1709.07871), and [MaskGIT](https://arxiv.org/pdf/2202.04200).

## Data

Supported datasets are `PONG`, `SONIC`, `POLE_POSITION`, `PICODOOM`, and `ZELDA`. Their video paths, HDF5 cache names, frame sizes, and preprocessing settings are defined in [`utils/datasets.py`](utils/datasets.py).

Preprocessed files are read from `data/`; when a cache is missing, the loader can preprocess the configured source video. Gameplay data and external assets are not covered by the code license.

## Training and inference options

- Training: AdamW or Muon optimizer, optional mixture-of-experts (MoE) dynamics, gradient accumulation, AMP/TF32, and `torch.compile`.
- Distributed training: DDP or FSDP, configured through the training YAML files.
- Inference: random actions, inferred ground-truth actions, interactive actions, or no action conditioning; iterative masked-token sampling with a configurable temperature and prediction horizon.

## Contributing

Keep changes backward-compatible and readable, preserve shape annotations, and include inference visualizations for model-related pull requests. Prefer small, focused implementations.

## License

The code (excluding datasets and external assets) is licensed under the [MIT License](LICENSE). TinyWorlds is an independent research project and is not affiliated with, endorsed by, or sponsored by DeepMind or Google.

Visual aesthetic inspired by [Tinygrad](https://tinygrad.org/) and [Tinygpu](https://github.com/adam-maj/tiny-gpu).
