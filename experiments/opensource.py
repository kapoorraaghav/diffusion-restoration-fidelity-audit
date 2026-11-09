"""Adapters to the upstream NAFNet and DiffBIR architectures (no training substitutes)."""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
# Compatibility for old ncl_sae commands. Prefer installed project-environment
# packages, so a leftover target directory cannot override the pinned venv.
if importlib.util.find_spec('ftfy') is None:
    sys.path.append(str(ROOT / '.open_source_deps'))
SOURCES = {'NAFNet': '2b4af71ebe098a92a75910c233a3965a3e93ede4',
           'DiffBIR': '5c2d6c177c3f2965e6d33a32931248d242ef7b3d'}


def source_path(name):
    path = ROOT / 'third_party' / name
    actual = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != SOURCES[name]:
        raise ValueError(f'{name}: unexpected source revision {actual}')
    return path


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def ddim_class(root):
    # Upstream sampler/__init__ eagerly imports EDM with the obsolete torch.Tuple
    # annotation. Load only the unchanged DDIM submodule used by this experiment.
    name = 'diffbir.sampler'
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [str(root / 'diffbir/sampler')]
        package.__package__ = name
        sys.modules[name] = package
    from diffbir.sampler.ddim_sampler import DDIMSampler
    return DDIMSampler


def nafnet(checkpoint=None, device='cpu', width=32, blocks=None, local=True):
    """Load original classes without importing the legacy BasicSR training package.

    Only package imports are replaced; upstream architecture operations are retained.
    """
    root = source_path('NAFNet') / 'basicsr/models/archs'
    namespace = {'torch': torch, 'nn': torch.nn, '__name__': '_upstream_nafnet'}
    tree = ast.parse((root / 'arch_util.py').read_text())
    tree.body = [n for n in tree.body if isinstance(n, ast.ClassDef)
                 and n.name in {'LayerNormFunction', 'LayerNorm2d'}]
    exec(compile(tree, str(root / 'arch_util.py'), 'exec'), namespace)
    spec = importlib.util.spec_from_file_location('_naf_local', root / 'local_arch.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    namespace['Local_Base'] = module.Local_Base
    tree = ast.parse((root / 'NAFNet_arch.py').read_text())
    tree.body = [n for n in tree.body if not
                 (isinstance(n, ast.ImportFrom) and (n.module or '').startswith('basicsr'))]
    exec(compile(tree, str(root / 'NAFNet_arch.py'), 'exec'), namespace)
    model = namespace['NAFNetLocal' if local else 'NAFNet'](
        width=width, middle_blk_num=1, enc_blk_nums=blocks or [1, 1, 1, 28],
        dec_blk_nums=[1, 1, 1, 1])
    if checkpoint is not None:
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        state = state.get('params_ema', state.get('params', state))
        state = {k.removeprefix('module.'): v for k, v in state.items()}
        model.load_state_dict(state, strict=True)
    return model.eval().to(device)


def validate_sd_missing_keys(missing, clip):
    """Permit only parameters provably unused by FrozenOpenCLIPEmbedder.

    Its token encoder never uses the contrastive projection/logit scale and
    skips the last transformer block only in penultimate mode.
    """
    prefix = 'cond_stage_model.model.'
    allowed = {prefix + 'text_projection', prefix + 'logit_scale'}
    if clip.layer == 'penultimate' and clip.layer_idx == 1:
        final_index = len(clip.model.transformer.resblocks) - 1
        final_block = clip.model.transformer.resblocks[final_index]
        allowed.update(prefix + f'transformer.resblocks.{final_index}.' + key
                       for key in final_block.state_dict())
    unexpected = set(missing) - allowed
    if unexpected:
        raise ValueError(f'Stable Diffusion checkpoint is missing required keys: {sorted(unexpected)}')
    return sorted(set(missing) & allowed)


def load_diffbir(sd_checkpoint, control_checkpoint, device, version='v2'):
    import os
    os.environ['ATTN_MODE'] = 'sdp'
    root = source_path('DiffBIR')
    sys.path.insert(0, str(root))
    from omegaconf import OmegaConf
    from diffbir.model import ControlLDM
    DDIMSampler = ddim_class(root)
    config = OmegaConf.load(root / 'configs/inference/cldm.yaml')
    model = ControlLDM(**config.params)
    sd = torch.load(sd_checkpoint, map_location='cpu', weights_only=True)
    unused, missing = model.load_pretrained_sd(sd.get('state_dict', sd))
    del sd
    ignored = validate_sd_missing_keys(missing, model.clip)
    if ignored:
        print('Checkpoint omits unused CLIP parameters (not used in conditioning):', ignored)
    state = torch.load(control_checkpoint, map_location='cpu', weights_only=True)
    model.load_controlnet_from_ckpt(state)
    del state
    model.eval()
    model.cast_dtype(torch.float16 if device.startswith('cuda') else torch.float32)
    model.to(device)
    from diffbir.model.gaussian_diffusion import Diffusion
    if version not in {'v2', 'v2.1'}:
        raise ValueError('Expected DiffBIR v2 or v2.1')
    filename = 'diffusion_v2.1.yaml' if version == 'v2.1' else 'diffusion.yaml'
    diffusion_config = OmegaConf.load(root / 'configs/inference' / filename)
    diffusion = Diffusion(**diffusion_config.params)
    return model, DDIMSampler(diffusion.betas, parameterization=diffusion.parameterization,
                              rescale_cfg=False, eta=0)


@torch.inference_mode()
def refine(model, sampler, condition, steps, seed, device):
    if steps not in {10, 20, 25, 50, 100}:
        raise ValueError('Use 10, 20, 25, 50 or 100 steps for the upstream uniform schedule.')
    cond = model.prepare_condition(condition, [''])
    sampler.make_schedule(steps)
    sampler.to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    latent = torch.randn(cond['c_img'].shape, generator=generator, device=device)
    stages = {}
    for index in reversed(range(len(sampler.ddim_timesteps))):
        t = torch.full((1,), int(sampler.ddim_timesteps[index]), device=device, dtype=torch.long)
        ti = torch.full((1,), index, device=device, dtype=torch.long)
        latent = sampler.p_sample(model, latent, t, ti, cond, None, 1.0)
        if index in {0, steps // 2, steps - 1}:
            stages[str(index)] = latent.detach().cpu().clone()
    return ((model.vae_decode(latent) + 1) / 2).clamp(0, 1), stages


def image_tensor(path, device):
    array = np.array(Image.open(path).convert('RGB'), dtype=np.float32) / 255
    return torch.from_numpy(array).permute(2, 0, 1)[None].to(device)


def save_image(tensor, path):
    array = tensor[0].detach().float().cpu().permute(1, 2, 0).clamp(0, 1).numpy()
    Image.fromarray((array * 255).round().astype('uint8')).save(path)


def main():
    parser = argparse.ArgumentParser(description='Pretrained open-source restoration adapters')
    sub = parser.add_subparsers(dest='command', required=True)
    sub.add_parser('doctor')
    download = sub.add_parser('download-nafnet')
    download.add_argument('--out', default='checkpoints/opensource/NAFNet-GoPro-width32.pth')
    run = sub.add_parser('restore')
    run.add_argument('--input', required=True)
    run.add_argument('--naf-checkpoint', required=True)
    run.add_argument('--sd-checkpoint')
    run.add_argument('--control-checkpoint')
    run.add_argument('--out', required=True)
    run.add_argument('--device', default='cuda')
    run.add_argument('--steps', type=int, default=20)
    run.add_argument('--seed', type=int, default=11)
    run.add_argument('--version', choices=['v2', 'v2.1'], default='v2')
    args = parser.parse_args()
    if args.command == 'download-nafnet':
        import gdown
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(path)
        result = gdown.download(id='1Fr2QadtDCEXg6iwWX8OzeZLbHOx2t5Bj', output=str(path))
        if not result:
            raise RuntimeError('Official NAFNet checkpoint download failed')
        print(json.dumps({'checkpoint': str(path), 'sha256': sha256(path)}))
        return
    if args.command == 'doctor':
        model = nafnet(width=8, blocks=[1, 1, 1, 1], local=False)
        output = model(torch.rand(1, 3, 32, 32))
        sys.path.insert(0, str(source_path('DiffBIR')))
        from diffbir.model import ControlLDM
        DDIMSampler = ddim_class(source_path('DiffBIR'))
        print(json.dumps({'sources': SOURCES, 'naf_architecture_forward': list(output.shape),
                          'diffbir_import': True, 'cuda': torch.cuda.is_available(),
                          'pretrained_inference_verified': False}, indent=2))
        return
    if bool(args.sd_checkpoint) != bool(args.control_checkpoint):
        parser.error('Supply both --sd-checkpoint and --control-checkpoint for DiffBIR v2.')
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    x = image_tensor(args.input, args.device)
    h, w = x.shape[-2:]
    if args.sd_checkpoint and (h % 64 or w % 64 or min(h, w) < 256):
        raise ValueError('DiffBIR research inputs must be at least 256 pixels and divisible by 64; resize explicitly first.')
    cnn = nafnet(args.naf_checkpoint, args.device)
    with torch.inference_mode():
        prior = cnn(x).clamp(0, 1)
    save_image(prior, out / 'cnn_prior.png')
    del cnn
    if args.device.startswith('cuda'):
        torch.cuda.empty_cache()
    metadata = {'sources': SOURCES, 'input_sha256': sha256(args.input),
                'naf_checkpoint_sha256': sha256(args.naf_checkpoint),
                'pretrained_nafnet': True, 'diffbir_completed': False,
                'composition': 'custom NAFNet prior + DiffBIR v2; not official DiffBIR stage 1',
                'device': args.device, 'seed': args.seed, 'steps': args.steps, 'version': args.version,
                'prompt': '', 'cfg_scale': 1, 'eta': 0}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    if args.sd_checkpoint:
        model, sampler = load_diffbir(args.sd_checkpoint, args.control_checkpoint, args.device, args.version)
        result, stages = refine(model, sampler, prior, args.steps, args.seed, args.device)
        save_image(result, out / 'diffusion_restored.png')
        torch.save(stages, out / 'latent_stages.pt')
        metadata.update(diffbir_completed=True,
                        sd_checkpoint_sha256=sha256(args.sd_checkpoint),
                        control_checkpoint_sha256=sha256(args.control_checkpoint))
        (out / 'metadata.json').write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))


if __name__ == '__main__':
    main()
