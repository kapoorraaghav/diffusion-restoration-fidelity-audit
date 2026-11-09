import torch
import torch.nn.functional as F

from .operators import blur

FAMILIES = ('false_edge', 'blur', 'warp', 'spectral', 'random')


def central_mask(x):
    h, w = x.shape[-2:]
    m = torch.zeros_like(x[:, :1])
    m[..., h//4:3*h//4, w//4:3*w//4] = 1
    return m


def perturb(x, family, severity, seed):
    """Match requested global RMS before clipping; log realized RMS separately."""
    if family not in FAMILIES or severity < 0:
        raise ValueError('Invalid perturbation family or severity')
    mask = central_mask(x)
    g = torch.Generator(device=x.device).manual_seed(seed)
    if severity == 0:
        return x.clone(), torch.zeros_like(x), mask
    if family == 'false_edge':
        pattern = torch.zeros_like(x)
        mid = x.shape[-1] // 2
        pattern[..., mid-1:mid] = 1
        pattern[..., mid:mid+1] = -1
    elif family == 'blur':
        pattern = blur(x, 2.0) - x
    elif family == 'warp':
        h, w = x.shape[-2:]
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=x.device),
                                torch.linspace(-1, 1, w, device=x.device), indexing='ij')
        grid = torch.stack((xx + .08 * torch.sin(3.14159 * yy), yy), -1)[None]
        pattern = F.grid_sample(x, grid.expand(x.shape[0], -1, -1, -1),
                                mode='bilinear', padding_mode='border', align_corners=True) - x
    elif family == 'spectral':
        freq = torch.fft.rfft2(x)
        fy = torch.fft.fftfreq(x.shape[-2], device=x.device)[:, None]
        fx = torch.fft.rfftfreq(x.shape[-1], device=x.device)[None, :]
        band = ((fy.square() + fx.square()) > .15**2).to(x.dtype)
        pattern = torch.fft.irfft2(freq * band, s=x.shape[-2:])
    else:
        pattern = torch.randn(x.shape, generator=g, device=x.device)
    pattern = pattern * mask
    rms = pattern.square().mean().sqrt()
    if rms.item() < 1e-10:
        raise ValueError(f'Degenerate {family} pattern; no valid perturbation for this image')
    perturbed = (x + severity * pattern / rms).clamp(0, 1)
    return perturbed, perturbed - x, mask
