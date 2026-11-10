"""Paper 2 review extension. New version; historical experiments remain immutable."""
import hashlib
import itertools
import json

VERSION = 'paper2_review_extension_v1'
SEEDS = [11, 22, 33]
CFGS = [1., 2., 4., 6.]
ALPHAS = [0., .25, .5, .75, 1.]
BETAS = [.25, .5, .75, 1.]  # beta=0 is the single shared cleaner-only candidate
METRICS = ['psnr', 'ssim', 'lpips', 'mse', 'musiq']
PREFERENCES = {'pixel': [1., 0., 0.], 'structure': [0., 1., 0.],
               'feature': [0., 0., 1.], 'balanced': [1/3, 1/3, 1/3],
               'pixel_priority': [.5, .25, .25]}
PROFILES = [dict(name='blur06', blur=.6, noise=0., jpeg=None),
            dict(name='blur12', blur=1.2, noise=0., jpeg=None),
            dict(name='blur24', blur=2.4, noise=0., jpeg=None),
            dict(name='noise05', blur=.6, noise=5/255, jpeg=None),
            dict(name='noise15', blur=.6, noise=15/255, jpeg=None),
            dict(name='jpeg80', blur=.6, noise=0., jpeg=80),
            dict(name='jpeg40', blur=.6, noise=0., jpeg=40)]
BASELINES = ['input_y', 'cleaner', 'vae_x_oracle', 'vae_y', 'vae_z']


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def strategy(cfg, alpha, beta):
    if beta == 0:
        return 'cleaner'
    if beta == 1:
        return f'cfg{cfg:g}_alpha{alpha:g}'
    if alpha == 1 and beta == .5:
        return f'blend_cfg{cfg:g}_beta0.5'
    return f'cfg{cfg:g}_alpha{alpha:g}_beta{beta:g}'


def candidates():
    return [dict(strategy=strategy(c, a, b), cfg=c, alpha=a, beta=b)
            for c, a, b in itertools.product(CFGS, ALPHAS, BETAS)]


def execution_candidates(stage, preference=None):
    all_values=candidates()
    if stage!='severity':return all_values
    legacy={'cfg1_alpha0','cfg1_alpha0.5','cfg1_alpha1','cfg6_alpha0','cfg6_alpha0.5','cfg6_alpha1',
            'blend_cfg1_beta0.5','blend_cfg6_beta0.5'}
    if preference:legacy.update(preference['choices'].values())
    return [c for c in all_values if c['strategy'] in legacy]


def frozen_protocol():
    from .workshop_design import splits
    split = splits()
    # Exact existing final inventories; severity subsets are chosen without inspecting metrics.
    severity = {d: sorted(ids, key=lambda i: hashlib.sha256(f'{VERSION}/{d}/{i}'.encode()).hexdigest())[:100]
                for d, ids in split['final'].items()}
    value = dict(version=VERSION, development=split['development'], final=split['final'],
                 severity=severity, old_split_sha256=split['sha256'], cfgs=CFGS,
                 alphas=ALPHAS, betas=BETAS, seeds=SEEDS, profiles=PROFILES,
                 candidates=candidates(), baselines=BASELINES, preferences=PREFERENCES,
                 method=dict(size=512, steps=10, sampler='edm_dpm++_3m_sde',
                             condition='z', correction_reference='z', prompts='upstream_fixed',
                             alpha_arithmetic='unclipped raw + alpha*(unclipped full - raw), then clip',
                             beta_arithmetic='clip((1-beta)*unclipped cleaner + beta*clipped q(alpha))',
                             vae='deterministic encode(sample=False) then decode; clamp RGB once',
                             musiq='pyiqa musiq/KonIQ-10k; RGB float32 [0,1]; higher is better',
                             tail_fraction=.05, bootstrap_resamples=2000, bootstrap_seed=20261102,
                             hv_references=[1.1, 2., 4.], hv_primary_reference=2., normalization='development equal-domain candidate mean loss ranges; no clipping',
                             selection='84 deployable candidates; five fixed weighted normalized losses; per-domain PSNR guard -0.2dB vs cfg6_alpha1; stable ties; VAE(x) excluded'),
                 evidence_status='review-driven extension; existing final images have already been inspected',
                 severity_rule='100/domain or entire smaller domain, hash-ranked; no metric-based replacement')
    value['sha256'] = digest(value)
    return value


def compose(raw, full, clean, alpha, beta):
    q = (raw.float() + alpha * (full.float() - raw.float())).clamp(0, 1)
    return ((1-beta)*clean.float() + beta*q).clamp(0, 1)


def degradation(reference, profile, image_id):
    """Noise uses a source-specific fixed seed; paired strengths share its noise field."""
    import io
    import numpy as np
    from PIL import Image
    import torch
    from .three_pipeline import extra_blur
    value = extra_blur(reference, profile['blur'])
    if profile['noise']:
        seed = int(hashlib.sha256(f'{VERSION}/noise/{image_id}'.encode()).hexdigest()[:8], 16)
        gen = torch.Generator(device='cpu').manual_seed(seed)
        field = torch.randn(value.shape, generator=gen, dtype=torch.float32).to(value.device)
        value = (value + profile['noise']*field).clamp(0, 1)
    if profile['jpeg']:
        # Fixed Pillow RGB JPEG: integer round-to-nearest, no chroma subsampling.
        array = (value[0].permute(1, 2, 0).cpu().numpy()*255).round().astype(np.uint8)
        stream = io.BytesIO()
        Image.fromarray(array).save(stream, format='JPEG', quality=profile['jpeg'], subsampling=0, optimize=False)
        stream.seek(0)
        with Image.open(stream) as image:
            array = np.array(image.convert('RGB'), copy=True)
        value = torch.from_numpy(array).permute(2, 0, 1)[None].to(reference.device).float()/255
    return value


def call_budget(stage, dataset=None, smoke=False):
    p = frozen_protocol()
    n = 2 if stage == 'development' and smoke else 1 if smoke else (
        100 if stage == 'development' else len(p['severity' if stage == 'severity' else 'final'][dataset]))
    # All four CFG values also retained for severity: no data-dependent call budget.
    return dict(images=n, profiles=7 if stage == 'severity' and not smoke else 1,
                sampling_calls=n*(7 if stage == 'severity' and not smoke else 1)*len(CFGS)*len(SEEDS),
                unique_candidates_per_image_profile=85)
