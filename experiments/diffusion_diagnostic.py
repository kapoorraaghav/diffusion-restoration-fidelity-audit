"""Controlled sampler diagnostics; alternatives are not upstream reproductions."""
import numpy as np
import torch


def endpoint_schedule(steps, training_steps=1000):
    if not 2 <= steps <= training_steps:
        raise ValueError('Invalid number of steps')
    return np.rint(np.linspace(0, training_steps-1, steps)).astype(int)


def v_predictions(x, v, alpha):
    """Stable direct v-parameterization, including alpha=0 at terminal ZSNR."""
    return alpha.sqrt()*x-(1-alpha).sqrt()*v, alpha.sqrt()*v+(1-alpha).sqrt()*x


@torch.inference_mode()
def endpoint_refine(model,sampler,condition,steps,seed,device,strength=1.):
    if sampler.parameterization!='v':
        raise ValueError('Diagnostic requires v prediction')
    previous_scales=list(model.control_scales)
    try:
        model.control_scales=[strength]*len(previous_scales)
        cond=model.prepare_condition(condition,[''])
        generator=torch.Generator(device=device).manual_seed(seed)
        x=torch.randn(cond['c_img'].shape,generator=generator,device=device)
        schedule=endpoint_schedule(steps,len(sampler.training_alphas_cumprod))
        alphas=torch.as_tensor(sampler.training_alphas_cumprod,device=device,dtype=x.dtype)
        for i in reversed(range(len(schedule))):
            t=torch.full((x.shape[0],),int(schedule[i]),device=device,dtype=torch.long)
            v=model(x,t,cond)
            clean,eps=v_predictions(x,v,alphas[schedule[i]])
            # Exact clean endpoint is explicitly defined for this diagnostic variant.
            previous=alphas[schedule[i-1]] if i>0 else x.new_tensor(1.)
            x=previous.sqrt()*clean+(1-previous).sqrt()*eps
        return ((model.vae_decode(x)+1)/2).clamp(0,1)
    finally:
        model.control_scales=previous_scales
