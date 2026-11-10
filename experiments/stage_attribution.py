"""Deterministic DDIM condition-repair interventions, not semantic hallucination scores."""
import torch
from .diffusion_diagnostic import v_predictions


class CapturePrediction:
    def __init__(self,model):self.model=model;self.prediction=None
    def __call__(self,x,t,cond):
        self.prediction=self.model(x,t,cond)
        return self.prediction


@torch.inference_mode()
def rollout(model,sampler,condition,steps,seed,device,start=None,completed=0,checkpoints=()):
    if sampler.eta!=0:raise ValueError('Paired stage attribution currently supports eta=0 only')
    if not 0<=completed<=steps:raise ValueError('Invalid completed-step index')
    sampler.make_schedule(steps);sampler.to(device)
    if len(sampler.ddim_timesteps)!=steps:raise ValueError('Schedule has unexpected number of steps')
    cond=model.prepare_condition(condition,[''])
    if start is None:
        if completed:raise ValueError('Resuming requires a saved latent')
        generator=torch.Generator(device=device).manual_seed(seed)
        x=torch.randn(cond['c_img'].shape,generator=generator,device=device)
    else:x=start.to(device).clone()
    states={};predictions={}
    if completed in checkpoints:states[completed]=x.cpu().clone()
    capture=CapturePrediction(model)
    for done in range(completed,steps):
        index=steps-1-done
        t=torch.full((x.shape[0],),int(sampler.ddim_timesteps[index]),device=device,dtype=torch.long)
        ti=torch.full_like(t,index)
        old=x
        x=sampler.p_sample(capture,x,t,ti,cond,None,1.)
        if done in checkpoints:
            alpha=sampler.ddim_alphas[index].to(old)
            if sampler.parameterization=='v':clean,_=v_predictions(old,capture.prediction,alpha)
            elif sampler.parameterization=='eps':
                clean=(old-(1-alpha).sqrt()*capture.prediction)/alpha.sqrt()
            else:raise ValueError('Unsupported parameterization')
            predictions[done]=((model.vae_decode(clean)+1)/2).clamp(0,1).cpu()
        if done+1 in checkpoints:states[done+1]=x.cpu().clone()
    output=((model.vae_decode(x)+1)/2).clamp(0,1)
    return output,states,predictions


@torch.inference_mode()
def fidelity_measures(output,reference,perceptual=None):
    from .metrics import psnr,ssim,edge_proxy
    false_edge,edge_recall=edge_proxy(output,reference)
    result=dict(psnr=psnr(output,reference),ssim=ssim(output,reference),
        mse=float((output-reference).square().mean()),false_edge_proxy=false_edge,valid_edge_recall=edge_recall)
    if perceptual is not None:
        # Official LPIPS expects RGB in [-1,1]; lower means smaller learned feature distance.
        result['lpips']=float(perceptual(output*2-1,reference*2-1).mean())
    return result
