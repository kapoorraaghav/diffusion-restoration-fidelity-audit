import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def seed_everything(seed):
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def stable_seed(*parts):
    return int(hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest()[:8], 16)


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def save_image(path, tensor):
    x = tensor.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    Image.fromarray((np.clip(x, 0, 1) * 255).round().astype('uint8')).save(path)


def environment():
    return {'python': sys.version, 'torch': str(torch.__version__),
            'cuda_runtime': torch.version.cuda, 'cuda_available': torch.cuda.is_available(),
            'device': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()}


def read_config(path):
    cfg = json.loads(Path(path).read_text(encoding='utf-8'))
    required = ['seed', 'image_size', 'task', 'noise_sigma', 'diffusion_steps', 'width',
                'cnn_steps', 'denoiser_steps', 'batch_size', 'learning_rate', 'sample_seeds',
                'families', 'severities', 'checkpoint_progress', 'gate_schedule']
    missing = set(required) - cfg.keys()
    if missing:
        raise ValueError(f'Missing config fields: {sorted(missing)}')
    if cfg['image_size'] < 16 or cfg['image_size'] % 2:
        raise ValueError('image_size must be even and >=16')
    if cfg['diffusion_steps'] < 4 or cfg['width'] % 4:
        raise ValueError('diffusion_steps >=4 and width divisible by 4 required')
    if cfg['task'] not in ['gaussian_blur', 'super_resolution']:
        raise ValueError('Supported pilot tasks: gaussian_blur, super_resolution')
    if any(v <= 0 for v in cfg['severities']):
        raise ValueError('Use positive perturbation severities; zero is an automatic control')
    if not cfg['sample_seeds'] or len(set(cfg['sample_seeds'])) != len(cfg['sample_seeds']):
        raise ValueError('Sample seeds must be nonempty and unique')
    if cfg['noise_sigma'] < 0 or cfg['batch_size'] < 1:
        raise ValueError('Invalid noise or batch size')
    if cfg['cnn_steps'] < 1 or cfg['denoiser_steps'] < 1:
        raise ValueError('Both models require positive training steps')
    if cfg.get('sr_factor', 4) < 1 or cfg['image_size'] % cfg.get('sr_factor', 4):
        raise ValueError('sr_factor must divide image_size')
    if cfg['gate_schedule'] not in ['constant', 'early', 'late']:
        raise ValueError('gate_schedule must be constant, early or late')
    if any(v < 0 or v > 1 for v in cfg['checkpoint_progress']):
        raise ValueError('Checkpoint progress must lie in [0,1]')
    return cfg
