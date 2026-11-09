"""Fixed development/confirmation design and single-factor interventions."""
import copy
import hashlib
import json
import numpy as np
from .ffhq_worker import selected_ids

BASELINE=dict(sampler='edm_dpm++_3m_sde',steps=10,cfg=6.,precision='fp16',
    start='noise',noise_aug=0,strength=1.,captioner='none',prompts='upstream_fixed')
LEGACY=dict(sampler='ddim',steps=20,cfg=1.,precision='fp16',
    start='noise',noise_aug=0,strength=1.,captioner='none',prompts='empty')
INTERVENTIONS={'precision_fp32':('precision','fp32'),
               'steps20':('steps',20),'cfg1':('cfg',1.)}


def candidate(name):
    if name not in INTERVENTIONS:raise ValueError('Choose one supported single-factor intervention')
    result=copy.deepcopy(BASELINE);field,value=INTERVENTIONS[name];result[field]=value
    return result


def confirmation_ids():
    pool=np.setdiff1d(np.arange(60000,70000),selected_ids(100))
    return np.random.default_rng(20261029).permutation(pool)[:2000].tolist()


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def frozen_design(intervention,rationale,development_job):
    if len(rationale.strip())<30:raise ValueError('Record the mechanism hypothesis and supporting development evidence')
    value=dict(version=1,baseline=BASELINE,candidate=candidate(intervention),intervention=intervention,
        rationale=rationale.strip(),development_job=str(development_job),
        development_ids=selected_ids(100),confirmation_ids=confirmation_ids(),
        seeds=[11,22,33],size=512,sigma=.6,primary_postprocessing='wavelet',
        primary_metrics=['psnr','lpips'],supporting_metrics=['ssim','mse'],
        limits='DiffBIR-specific test; no causal claim about all diffusion models or hallucinations')
    value['design_sha256']=digest(value)
    return value


def validate_frozen(value):
    actual={k:v for k,v in value.items() if k!='design_sha256'}
    if digest(actual)!=value.get('design_sha256'):raise ValueError('Frozen design checksum mismatch')
    if value['baseline']!=BASELINE or value['candidate']!=candidate(value['intervention']):raise ValueError('Frozen configurations changed')
    if value['development_ids']!=selected_ids(100) or value['confirmation_ids']!=confirmation_ids():raise ValueError('Frozen split mismatch')
    return value
