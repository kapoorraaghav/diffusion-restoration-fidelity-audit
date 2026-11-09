import math

import numpy as np
import torch
import torch.nn.functional as F

from .operators import gaussian_kernel


def psnr(x, reference):
    mse = (x-reference).square().mean().item()
    # Finite cap makes strict JSON serialization possible; matches a 1e-12 floor.
    return -10 * math.log10(max(mse, 1e-12))


def ssim(x, reference):
    k = gaussian_kernel(1.5, x.device, x.dtype).expand(x.shape[1], 1, -1, -1)
    mu_x = F.conv2d(x, k, groups=x.shape[1])
    mu_r = F.conv2d(reference, k, groups=x.shape[1])
    vx = F.conv2d(x*x, k, groups=x.shape[1]) - mu_x.square()
    vr = F.conv2d(reference*reference, k, groups=x.shape[1]) - mu_r.square()
    cov = F.conv2d(x*reference, k, groups=x.shape[1]) - mu_x*mu_r
    value = ((2*mu_x*mu_r+.01**2)*(2*cov+.03**2) /
             ((mu_x.square()+mu_r.square()+.01**2)*(vx+vr+.03**2)))
    return value.mean().item()


def edges(x):
    gray = x.mean(1, keepdim=True)
    kx = x.new_tensor([[-1,0,1],[-2,0,2],[-1,0,1]])[None, None]/8
    ky = kx.transpose(-2, -1)
    dx = F.conv2d(F.pad(gray, (1,1,1,1), mode='replicate'), kx)
    dy = F.conv2d(F.pad(gray, (1,1,1,1), mode='replicate'), ky)
    return (dx.square()+dy.square()).sqrt()


def edge_proxy(x, reference, threshold=.12):
    pred = edges(x) > threshold
    truth = edges(reference) > threshold
    # One-pixel tolerance to avoid equating registration noise with false structure.
    truth_near = F.max_pool2d(truth.float(), 3, 1, 1).bool()
    pred_near = F.max_pool2d(pred.float(), 3, 1, 1).bool()
    false = pred & ~truth_near
    recall = ((truth & pred_near).sum() / truth.sum().clamp_min(1)).item()
    return false.float().mean().item(), recall


def metrics(x, reference, y, op, cfg):
    false, recall = edge_proxy(x, reference, cfg.get('edge_threshold', .12))
    residual = (op.forward(x)-y).square().mean().item()
    return {'psnr': psnr(x, reference), 'ssim': ssim(x, reference),
            'mse': (x-reference).square().mean().item(),
            'false_edge_proxy': false, 'valid_edge_recall': recall,
            'measurement_mse': residual,
            'normalized_measurement_mse': residual / max(cfg['noise_sigma']**2, 1e-12)}


def calibration_metrics(probability, labels, bins=10):
    p, y = probability.flatten(), labels.flatten()
    ece = 0
    for j in range(bins):
        inside = (p >= j/bins) & ((p < (j+1)/bins) if j < bins-1 else p <= 1)
        if inside.any():
            ece += inside.float().mean().item() * abs(p[inside].mean().item()-y[inside].mean().item())
    return {'brier': (p-y).square().mean().item(), 'ece': ece,
            'error_prevalence': y.mean().item()}


def bootstrap_ci(values, seed=0, repeats=2000):
    a = np.asarray(values, dtype=float)
    if not len(a):
        raise ValueError('No observations')
    rng = np.random.default_rng(seed)
    means = a[rng.integers(0, len(a), size=(repeats, len(a)))].mean(axis=1)
    low, high = np.quantile(means, [.025, .975])
    return {'mean': float(a.mean()), 'ci95_low': float(low), 'ci95_high': float(high), 'n_images': len(a)}
