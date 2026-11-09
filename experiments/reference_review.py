"""Review saved Kodak outputs against references; no automatic hallucination labels.

Uses only existing outputs, with no model downloads or new inference.
"""
import argparse
import base64
import io
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
import torch
import torch.nn.functional as F

from experiments.metrics import edges, psnr, ssim


def tensor(image):
    return torch.from_numpy(np.asarray(image.convert('RGB')).copy()).permute(2, 0, 1)[None].float()/255


def edge_masks(pred, ref, threshold):
    p, r = edges(pred)>threshold, edges(ref)>threshold
    return (p & ~F.max_pool2d(r.float(),3,1,1).bool())[0,0].numpy(), (r & ~F.max_pool2d(p.float(),3,1,1).bool())[0,0].numpy()


def panel(images, labels, width=256):
    out=Image.new('RGB',(width*len(images),width+42),'white');d=ImageDraw.Draw(out)
    for i,(im,label) in enumerate(zip(images,labels)):
        out.paste(im.resize((width,width)),(width*i,42));d.text((width*i+5,8),label,fill='black')
    return out


def heat(values):
    # Fixed 0..0.1 absolute RGB error scale across every image.
    a=np.clip(values/.1,0,1)
    return Image.fromarray(np.uint8(np.stack([a,np.minimum(a*2,1),np.zeros_like(a)],-1)*255))


def main(archive, destination):
    torch.set_num_threads(4)
    out=Path(destination);out.mkdir(parents=True,exist_ok=True)
    gallery=out/'images';gallery.mkdir(exist_ok=True)
    records=[];patches=[];checks=[];html=[]
    def embed(path):
        return 'data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()
    with zipfile.ZipFile(archive) as z:
        frame=pd.read_csv(z.open('results.csv'))
        assert len(frame)==120, 'Unexpected/incomplete source result count'
        for image_id in sorted(frame.image_id.unique()):
            ref=Image.open(io.BytesIO(z.read(f'{image_id}/sharp.png'))).convert('RGB');rt=tensor(ref)
            for sigma in [0.,.6,1.2]:
                cnn=Image.open(io.BytesIO(z.read(f'{image_id}/cnn_sigma{sigma}.png'))).convert('RGB')
                diff=Image.open(io.BytesIO(z.read(f'{image_id}/diffusion_sigma{sigma}.png'))).convert('RGB')
                ct,dt=tensor(cnn),tensor(diff)
                assert ref.size==cnn.size==diff.size==(512,512)
                rows=frame[(frame.image_id==image_id)&(frame.sigma==sigma)]
                cr=rows[rows.method=='cnn'].iloc[0];dr=rows[(rows.method=='diffusion')&(rows.seed==11)].iloc[0]
                row=dict(image_id=image_id,sigma=sigma,visualized_seed=11,complexity_group=cr.complexity_group)
                for prefix,t,source in [('cnn',ct,cr),('diffusion',dt,dr)]:
                    for key in ['psnr','ssim','lpips','mse']:
                        row[f'{prefix}_{key}_source']=float(source[key])
                    row[f'{prefix}_psnr_png']=psnr(t,rt);row[f'{prefix}_ssim_png']=ssim(t,rt)
                    checks.append(dict(image_id=image_id,sigma=sigma,method=prefix,source_psnr=float(source.psnr),png_psnr=psnr(t,rt),difference_db=psnr(t,rt)-float(source.psnr)))
                for key in ['psnr','ssim','lpips','mse']:
                    row[f'delta_{key}_diffusion_minus_cnn']=float(dr[key]-cr[key])
                masks={}
                for th in [.06,.12,.18]:
                    cf,cm=edge_masks(ct,rt,th);df,dm=edge_masks(dt,rt,th)
                    row[f'cnn_false_edge_fraction_t{th}']=cf.mean();row[f'diffusion_false_edge_fraction_t{th}']=df.mean()
                    row[f'cnn_missing_edge_fraction_t{th}']=cm.mean();row[f'diffusion_missing_edge_fraction_t{th}']=dm.mean()
                    masks[th]=(cf,cm,df,dm)
                cf,cm,df,dm=masks[.12]
                ce=(ct-rt).abs().mean(1)[0].numpy();de=(dt-rt).abs().mean(1)[0].numpy()
                cmap=(ct-rt).square().mean(1)[0].numpy();dmap=(dt-rt).square().mean(1)[0].numpy()
                candidates=[]
                for y in range(0,512,128):
                    for x in range(0,512,128):
                        sl=np.s_[y:y+128,x:x+128]
                        candidates.append(dict(image_id=image_id,sigma=sigma,seed=11,x=x,y=y,width=128,height=128,
                            cnn_mse=float(cmap[sl].mean()),diffusion_mse=float(dmap[sl].mean()),
                            delta_mse=float((dmap[sl]-cmap[sl]).mean()),
                            added_false_edge_fraction=float((df & ~cf)[sl].mean()),
                            removed_false_edge_fraction=float((cf & ~df)[sl].mean())))
                # Two largest MSE increases plus largest newly unsupported-edge area.
                ordered=sorted(candidates,key=lambda p:p['delta_mse'],reverse=True)[:2]
                edge_top=max(candidates,key=lambda p:p['added_false_edge_fraction'])
                if (edge_top['x'],edge_top['y']) not in [(p['x'],p['y']) for p in ordered]:ordered.append(edge_top)
                for i,p in enumerate(ordered):
                    x,y=p['x'],p['y'];box=(x,y,x+128,y+128)
                    name=f'{image_id}_s{sigma}_patch{i+1}.png'
                    panel([ref.crop(box),cnn.crop(box),diff.crop(box)],['Sharp reference','CNN','Diffusion seed 11']).save(gallery/name)
                    p.update(patch_file='images/'+name,reviewer='',label='',attribution='',notes='')
                    patches.append(p)
                falsemap=np.zeros((512,512,3),dtype=np.uint8)
                falsemap[cf & df]=[220,160,0];falsemap[df & ~cf]=[255,30,80];falsemap[cf & ~df]=[0,190,200]
                name=f'{image_id}_s{sigma}_comparison.png'
                panel([ref,cnn,diff,heat(ce),heat(de),Image.fromarray(falsemap)],
                    ['Sharp reference',f'CNN PSNR {cr.psnr:.2f}',f'Diffusion PSNR {dr.psnr:.2f}',
                     'CNN absolute error','Diffusion absolute error','Unsupported edge candidates']).save(gallery/name)
                records.append(row)
                html.append(f'<section><h2>{image_id}, blur sigma={sigma}</h2><p>Seed 11: PSNR {cr.psnr:.2f} → {dr.psnr:.2f}; SSIM {cr.ssim:.3f} → {dr.ssim:.3f}; LPIPS {cr.lpips:.3f} → {dr.lpips:.3f}.</p><img src="{embed(gallery/name)}"><details><summary>Inspect candidate patches</summary>'+''.join(f'<img src="{embed(gallery/Path(p["patch_file"]).name)}">' for p in patches if p['image_id']==image_id and p['sigma']==sigma)+'</details></section>')
    pd.DataFrame(records).to_csv(out/'reference_metrics_seed11.csv',index=False)
    pd.DataFrame(patches).to_csv(out/'candidate_patch_review.csv',index=False)
    pd.DataFrame(checks).to_csv(out/'png_metric_checks.csv',index=False)
    frame.to_csv(out/'original_results_all_seeds.csv',index=False)
    results=pd.DataFrame(records)
    summary=results.groupby('sigma')[[f'delta_{k}_diffusion_minus_cnn' for k in ['psnr','ssim','lpips']]].mean()
    description='''<h1>Kodak: reference-based error review</h1><p>24 image–blur cases, 8 images. Visuals and paired metrics below use diffusion seed 11 only; the original CSV preserves all three seeds. PSNR/SSIM higher is better; LPIPS lower is better. Source metrics use original float tensors. Spatial maps use saved 8-bit PNGs and can differ slightly.</p><p><b>These are error candidates, not confirmed hallucinations.</b> Edge novelty includes ordinary sharpening, noise, shifts and numerical corruption. An edge missing from the CNN but correct in the reference is not flagged as unsupported. Edge masks use RGB-mean Sobel magnitude threshold 0.12 and one-pixel matching tolerance; CSV also includes thresholds 0.06 and 0.18. Colours: pink = unsupported edge newly present after diffusion; gold = present in both stages; cyan = CNN unsupported edge absent after diffusion. These are positional overlaps, not causal proof. Error heatmaps share a fixed scale: black=0, yellow=absolute RGB error at least 0.1.</p><p>Patch selection is exploratory: two largest 128px-patch MSE increases and one largest new unsupported-edge fraction per case, deduplicated. It is deliberately biased toward errors; it cannot estimate hallucination prevalence or detector accuracy. Review all full images and add random/control patches before formal validation.</p><p>Suggested labels: invented structure, altered structure, missing structure, blur/detail loss, colour/brightness, numerical corruption, no clear error, uncertain. Attribution: CNN-origin/preserved, amplified, corrected, new after diffusion, uncertain. Leave labels blank until reference-based human review. Stable errors across stages are not proof of CNN causation.</p>'''
    (out/'Kodak_Reference_Review.html').write_text('<!doctype html><meta charset="utf-8"><title>Kodak reference review</title><style>body{font:16px system-ui;max-width:1600px;margin:30px auto;padding:20px;background:#f4f6fa;color:#182333}img{max-width:100%;height:auto}section{background:white;padding:18px;margin:24px 0;border:1px solid #ccd4e0}summary{cursor:pointer;padding:12px}table{border-collapse:collapse}td,th{padding:8px;border:1px solid #ccc}</style>'+description+'<h2>Mean changes, seed 11 (diffusion minus CNN)</h2>'+summary.to_html()+''.join(html),encoding='utf-8')
    (out/'README.txt').write_text('Open Kodak_Reference_Review.html in a browser. All images are embedded, so this HTML works alone.\nCSV definitions and limitations are described in the HTML.\nNo new inference, downloads, automated hallucination labels or image alignment were applied.\nSource archive: '+str(Path(archive).resolve()),encoding='utf-8')
    print(summary.to_string());print('cases',len(records),'candidate patches',len(patches),'max PNG PSNR difference',max(abs(c['difference_db']) for c in checks))
    bundle=out.parent/(out.name+'.zip')
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as z:
        for path in out.rglob('*'):
            if path.is_file():z.write(path,path.relative_to(out))
    print('Report:',out/'Kodak_Reference_Review.html');print('Bundle:',bundle)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();main(a.archive,a.output)
