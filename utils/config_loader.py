"""Read YAML configuration, optionally merging shared training options."""

import argparse
import os
from types import SimpleNamespace

from omegaconf import OmegaConf


def _namespace(value):
    if isinstance(value, dict):
        return SimpleNamespace(**{key: _namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value


def load_yaml_config(default_path, stage=False):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=default_path)
    parser.add_argument('--training_config', default='configs/training.yaml' if stage else None)
    args, overrides = parser.parse_known_args()
    cfg = OmegaConf.load(args.config)
    if args.training_config and os.path.isfile(args.training_config):
        shared = OmegaConf.load(args.training_config)
        cfg = OmegaConf.merge(cfg, {key: value for key, value in shared.items() if value is not None})
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([value.lstrip('-') for value in overrides if '=' in value]))
    return _namespace(OmegaConf.to_container(cfg, resolve=True))
