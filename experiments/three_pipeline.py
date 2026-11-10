"""Shared, testable controls for before/after-CNN Gaussian blur experiments."""
import hashlib
import json
import math

import torch
import torch.nn.functional as F


def extra_blur(x, sigma):
    """Full-frame Gaussian blur in crop pixels, reflect boundaries; exact identity at zero."""
    if not math.isfinite(sigma) or sigma < 0:
        raise ValueError('sigma must be finite and nonnegative')
    if sigma == 0:
        return x.clone()
    radius = max(1, math.ceil(3 * sigma))
    if radius >= min(x.shape[-2:]):
        raise ValueError('Blur radius must be smaller than the crop')
    axis = torch.arange(-radius, radius+1, device=x.device, dtype=x.dtype)
    kernel = torch.exp(-axis.square()/(2*sigma*sigma))
    kernel /= kernel.sum()
    channels=x.shape[1]
    horizontal=kernel.view(1,1,1,-1).expand(channels,1,1,-1)
    vertical=kernel.view(1,1,-1,1).expand(channels,1,-1,1)
    y=F.conv2d(F.pad(x,(radius,radius,0,0),mode='reflect'),horizontal,groups=channels)
    return F.conv2d(F.pad(y,(0,0,radius,radius),mode='reflect'),vertical,groups=channels)


def rms(a,b):
    return float((a-b).square().mean().sqrt())


@torch.inference_mode()
def before_prior(cnn, observation, sigma):
    return cnn(extra_blur(observation,sigma)).clamp(0,1)


@torch.inference_mode()
def match_after_sigma(prior, target_rms, max_sigma=4., absolute_tolerance=1e-4,
                      relative_tolerance=.05):
    """Bounded grid search; no assumption that CNN response is monotonic.

    Uses conditioning changes only, never sharp-reference or diffusion scores.
    Unattainable changes are flagged rather than called matched.
    """
    if target_rms < 0 or not math.isfinite(target_rms):
        raise ValueError('Invalid target RMS')
    grid=[i*max_sigma/40 for i in range(41)]
    scores=[rms(extra_blur(prior,s),prior) for s in grid]
    best=min(range(len(grid)),key=lambda i:abs(scores[i]-target_rms))
    lower=grid[max(0,best-1)]; upper=grid[min(40,best+1)]
    fine=[lower+(upper-lower)*i/20 for i in range(21)]
    candidates=list(zip(grid,scores))+[(s,rms(extra_blur(prior,s),prior)) for s in fine]
    sigma,actual=min(candidates,key=lambda pair:abs(pair[1]-target_rms))
    error=abs(actual-target_rms)
    return dict(sigma=sigma,target_rms=target_rms,actual_rms=actual,
                absolute_error=error,matched=error<=max(absolute_tolerance,relative_tolerance*target_rms),
                absolute_tolerance=absolute_tolerance,relative_tolerance=relative_tolerance,
                search_max_sigma=max_sigma)


def protocol_hash(config,records):
    """Ignore mount locations; retain image bytes, crops, sequence identities, model/settings."""
    identities=[{k:r[k] for k in ['image_id','sequence','blur_sha256','sharp_sha256','box']} for r in records]
    payload=json.dumps(dict(config=config,images=identities),sort_keys=True,separators=(',',':'))
    return hashlib.sha256(payload.encode()).hexdigest()
