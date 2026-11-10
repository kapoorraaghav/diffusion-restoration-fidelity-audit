from dataclasses import dataclass

import torch

from .models import alpha_schedule


@dataclass
class SampleResult:
    image: torch.Tensor
    checkpoints: dict
    nfe: int


def random_stream(shape, steps, seed, device):
    """Explicit common random numbers, reused across every branch and intervention."""
    g = torch.Generator(device=device).manual_seed(seed)
    return (torch.randn(shape, generator=g, device=device),
            [torch.randn(shape, generator=g, device=device) for _ in range(steps)])


def schedule_weight(progress, schedule):
    if schedule == 'constant':
        return 1.0
    if schedule == 'early':
        return 1.0 - progress
    return progress


@torch.no_grad()
def sample(model, op, y, prior, stream, cfg, branch='B', q=None, resume=None,
           repaired_prior=None, repair_start=None, repair_mask=None):
    if branch not in ('A', 'B', 'C', 'global'):
        raise ValueError('Unknown branch')
    if branch == 'C' and q is None:
        raise ValueError('Branch C requires a reliability estimate')
    beta, abar = alpha_schedule(cfg['diffusion_steps'], y.device)
    y_image = op.backprojection(y).clamp(0, 1)
    x = stream[0].clone() if resume is None else resume['xt'].clone()
    start = cfg['diffusion_steps']-1 if resume is None else resume['t']
    indices = {round((1-f) * (cfg['diffusion_steps']-1)) for f in cfg['checkpoint_progress']}
    states = {}
    final = None
    calls = 0
    for t in range(start, -1, -1):
        progress = 1 - t/(cfg['diffusion_steps']-1)
        active_prior = prior
        if repaired_prior is not None and progress >= repair_start:
            active_prior = prior * (1-repair_mask) + repaired_prior * repair_mask
        weight = torch.ones_like(prior[:, :1])
        if branch == 'A':
            weight *= 0
        elif branch == 'global':
            weight *= cfg.get('global_weight', .5)
        elif branch == 'C':
            weight = (1-q) * schedule_weight(progress, cfg['gate_schedule'])
        condition = weight * active_prior + (1-weight) * y_image
        tt = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        prediction = model(x, tt, y_image, condition, weight)
        clean_raw = ((x - (1-abar[t]).sqrt() * prediction) / abar[t].sqrt()).clamp(-1, 1)
        clean_image = (clean_raw + 1) / 2
        corrected = op.correction(clean_image, y, cfg.get('measurement_step', .2)).clamp(0, 1)
        if t in indices:
            states[t] = {'t': t, 'progress': progress, 'xt': x.detach().clone(),
                         'clean_raw': clean_image.detach().clone(), 'clean': corrected.detach().clone()}
        calls += 1
        final = corrected
        if t == 0:
            break
        previous_abar = abar[t-1]
        # DDPM posterior with measurement-corrected clean estimate. This is a pilot sampler,
        # not the update equation of DPS, DDRM, DiffECC or RPU.
        coeff_clean = beta[t] * previous_abar.sqrt() / (1-abar[t])
        coeff_x = (1-beta[t]).sqrt() * (1-previous_abar) / (1-abar[t])
        variance = beta[t] * (1-previous_abar) / (1-abar[t])
        x = coeff_clean * (2*corrected-1) + coeff_x*x + variance.sqrt()*stream[1][t]
    return SampleResult(final, states, calls)
