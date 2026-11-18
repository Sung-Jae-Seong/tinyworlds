# TinyWorlds

TinyWorlds is a minimal autoregressive world model inspired by [Genie](https://arxiv.org/pdf/2402.15391). It learns discrete video and latent-action tokens from gameplay video without action labels, then predicts future frames conditioned on actions.

## Getting started

Install dependencies from the project root:

```bash
python -m pip install -r requirements.txt
```

Place preprocessed gameplay `.h5` files from the [TinyWorlds datasets](https://huggingface.co/datasets/AlmondGod/tinyworlds/tree/main) in `data/`, or provide the source videos for automatic preprocessing. Set the dataset and model parameters in `configs/`.

Train the video tokenizer, latent-action model, and dynamics model in sequence:

```bash
python scripts/full_train.py --config configs/training.yaml
```

The dynamics stage automatically finds checkpoints from earlier stages in the same run when their paths are unset.

Configure `configs/inference.yaml` and run inference:

```bash
python scripts/run_inference.py --config configs/inference.yaml
```

YAML values can be overridden using `key=value` arguments:

```bash
python scripts/run_inference.py --config configs/inference.yaml dataset=SONIC device=cuda
```

The inference `n_actions` value must match the action codebook. Checkpoint paths can be set explicitly or discovered automatically.

## Architecture

| Component | Role |
| --- | --- |
| Video Tokenizer | Encodes frames into discrete tokens using a space-time transformer and finite scalar quantization (FSQ). |
| Action Tokenizer | Learns discrete latent actions from consecutive frames without action labels. |
| Dynamics Model | Predicts masked future-frame tokens from past frames and latent actions. |

The space-time transformer uses spatial and temporal attention with FiLM conditioning. Dynamics supports optional QK normalization (`qk_norm` in `configs/dynamics.yaml`) and iteratively fills future tokens with a MaskGIT-style schedule before video decoding.

Related methods: [space-time attention](https://arxiv.org/pdf/2001.02908), [FSQ](https://arxiv.org/pdf/2309.15505), [FiLM](https://arxiv.org/pdf/1709.07871), and [MaskGIT](https://arxiv.org/pdf/2202.04200).

## Data

Supported datasets: `PONG`, `SONIC`, `POLE_POSITION`, `PICODOOM`, and `ZELDA`. Video paths, preprocessing options, and HDF5 caching are defined in [`utils/datasets.py`](utils/datasets.py).

## Options

Training supports AdamW or Muon, optional MoE dynamics, gradient accumulation, AMP/TF32, `torch.compile`, and DDP/FSDP. Inference is action-conditioned and supports random or interactive actions, configurable prediction horizon, temperature, and MaskGIT schedule. See `configs/` for options.

## License

The code (excluding datasets and external assets) is licensed under the [MIT License](LICENSE). TinyWorlds is an independent research project, not affiliated with or endorsed by DeepMind or Google.

Visual aesthetic inspired by [Tinygrad](https://tinygrad.org/) and [Tinygpu](https://github.com/adam-maj/tiny-gpu).
