import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from .common import file_hash, stable_seed


def generate_synthetic(root, count=64, size=64, seed=123):
    """Independent procedural scenes for plumbing tests, not a natural-image benchmark."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / 'manifest.jsonl'
    if manifest.exists():
        raise FileExistsError(f'{manifest} already exists; choose a new directory')
    if count < 16:
        raise ValueError('Need at least 16 images for four disjoint splits')
    records = []
    for i in range(count):
        rng = np.random.default_rng(stable_seed(seed, i))
        canvas = Image.new('RGB', (size, size), tuple(rng.integers(20, 80, 3)))
        draw = ImageDraw.Draw(canvas)
        for j in range(7):
            x, y = rng.integers(1, max(2, size // 2), 2)
            w, h = rng.integers(max(2, size // 8), max(3, size // 2), 2)
            box = (int(x), int(y), int(min(size - 2, x + w)), int(min(size - 2, y + h)))
            color = tuple(map(int, rng.integers(70, 240, 3)))
            (draw.rectangle if j % 2 else draw.ellipse)(box, fill=color)
        path = root / f'scene_{i:05d}.png'
        canvas.save(path)
        fraction = i / count
        split = 'train' if fraction < .625 else 'calibration' if fraction < .75 else 'validation' if fraction < .875 else 'test'
        records.append({'image_id': f'scene_{i:05d}', 'group_id': f'scene_{i:05d}',
                        'path': path.name, 'split': split, 'domain': 'procedural_smoke'})
    manifest.write_text('\n'.join(json.dumps(r) for r in records) + '\n', encoding='utf-8')
    return manifest


class ImageManifest:
    def __init__(self, path, size):
        self.path = Path(path).resolve()
        self.size = size
        self.records = [json.loads(line) for line in self.path.read_text(encoding='utf-8').splitlines() if line.strip()]
        groups, ids, hashes = {}, set(), {}
        for r in self.records:
            if r['split'] not in ['train', 'calibration', 'validation', 'test']:
                raise ValueError('Invalid split')
            if r['image_id'] in ids:
                raise ValueError('Duplicate image_id')
            ids.add(r['image_id'])
            group = r['group_id']
            if group in groups and groups[group] != r['split']:
                raise ValueError(f'Group leakage: {group}')
            groups[group] = r['split']
            image_path = self.path.parent / r['path']
            digest = file_hash(image_path)
            if digest in hashes and hashes[digest] != r['split']:
                raise ValueError('Exact duplicate image across splits')
            hashes[digest] = r['split']
        for split in ['train', 'calibration', 'validation', 'test']:
            if not self.select(split):
                raise ValueError(f'Empty split: {split}')

    def select(self, split):
        return [r for r in self.records if r['split'] == split]

    def load(self, record, device='cpu'):
        with Image.open(self.path.parent / record['path']) as source:
            image = source.convert('RGB')
            # Center square crop, then a fixed bicubic resize. Record this in the run metadata.
            w, h = image.size
            side = min(w, h)
            image = image.crop(((w-side)//2, (h-side)//2, (w+side)//2, (h+side)//2))
            image = image.resize((self.size, self.size), Image.Resampling.BICUBIC)
            arr = np.array(image, dtype=np.float32) / 255
        return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)

    def fingerprint(self):
        return {'manifest_sha256': file_hash(self.path), 'images': {
            r['image_id']: file_hash(self.path.parent / r['path']) for r in self.records}}
