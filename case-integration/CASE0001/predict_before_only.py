"""Future checkpoint-backed, before-only entry. Default: validate weights, no forward.

This wrapper does not change TADPM, conditioning, U-ViT, or DDIM equations.
It does not compute ADD against the dataset's explicitly marked after placeholders.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace


def checkpoint_state(path):
    import torch
    try:
        obj = torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    except RuntimeError as e:
        if 'mmap' not in str(e):
            raise
        obj = torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(obj,dict) or not isinstance(obj.get('base_model'),dict):
        raise ValueError('A full official TADPM checkpoint with base_model is required; smoke outputs/TGNet weights are not checkpoints')
    state = {k.removeprefix('module.'):v for k,v in obj['base_model'].items()}
    if len(state) != len(obj['base_model']):
        raise ValueError('Duplicate checkpoint keys after DataParallel prefix removal')
    return state


def model_config(repo):
    import yaml
    from easydict import EasyDict
    config = yaml.safe_load((repo/'config/TADPM.yaml').read_text(encoding='utf-8'))['model']
    config['args'] = SimpleNamespace(encoder_ckpts='')
    return EasyDict(config)


def check_checkpoint(repo, path):
    import torch
    sys.path.insert(0,str(repo))
    import models
    from utils import builder
    state = checkpoint_state(path)
    # Meta construction validates official key/shape compatibility without allocating a GPU model.
    with torch.device('meta'):
        model = builder.model_builder(model_config(repo))
    expected = model.state_dict()
    missing,unexpected = sorted(expected.keys()-state.keys()),sorted(state.keys()-expected.keys())
    wrong = {k:[list(state[k].shape) if torch.is_tensor(state[k]) else 'not a tensor',list(expected[k].shape)]
             for k in expected.keys() & state.keys()
             if not torch.is_tensor(state[k]) or state[k].shape != expected[k].shape}
    if missing or unexpected or wrong:
        raise ValueError(f'Not a complete matching TADPM checkpoint: missing={missing}, unexpected={unexpected}, shapes={wrong}')
    if not all(torch.isfinite(t).all().item() for t in state.values() if t.is_floating_point()):
        raise ValueError('Checkpoint contains non-finite values')
    with path.open('rb') as f:
        digest = hashlib.file_digest(f,'sha256').hexdigest()
    return dict(provided=True,path=str(path),sha256=digest,full_key_shape_match=True,
        tensor_count=len(state),training_provenance_verified=False,
        trained_weights_confirmed=False,
        note='Compatible full weights; training origin and validity must be verified with the provider')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--case',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--run',action='store_true',help='Explicitly execute unchanged 100-step DDIM on a CUDA GPU')
    p.add_argument('--out',type=Path,help='Required with --run; must not already exist')
    p.add_argument('--seed',type=int,default=0)
    a = p.parse_args()
    repo,case,checkpoint = [x.resolve() for x in [a.repo,a.case,a.checkpoint]]
    status = check_checkpoint(repo,checkpoint)
    print(json.dumps(status,ensure_ascii=False,indent=2))
    if not a.run:
        print('Checkpoint shape check only; no model forward or DDIM executed.')
        return
    if not a.out or a.out.exists():
        raise ValueError('--run requires a new --out directory; existing results are never overwritten')
    import torch
    from utils import builder
    from read_case import load_dataset,FIELDS
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required for this optional prediction; no CPU fallback')
    torch.manual_seed(a.seed)
    loader,batch,config = load_dataset(repo,case)
    d = dict(zip(FIELDS,batch))
    assert d['masks'].sum().item() == 27
    model = builder.model_builder(model_config(repo))
    model.load_state_dict(checkpoint_state(checkpoint),strict=True)
    model = model.to('cuda').eval()
    names = ['faces','feats','center','Fs','cordinates','centroid','before_points']
    inputs = [d[k].to(device='cuda',dtype=torch.float32) for k in names]
    with torch.inference_mode():
        outputs = model(*inputs,gt_6dof=None)
    assert tuple(outputs.shape)==(1,32,9) and torch.isfinite(outputs).all()
    a.out.mkdir(parents=True,exist_ok=False)
    torch.save(dict(outputs=outputs.cpu(),masks=d['masks'],case_ids=d['index'],
                    checkpoint=status,clinical_confirmed=False),a.out/'slot_parameters.pt')
    report = dict(case_id='CASE0001',checkpoint=status,seed=a.seed,
        output_shape=list(outputs.shape),all_finite=True,
        ddim_steps=model.regressor.sampling_timesteps,after_data_used_by_model=False,
        add_metric_computed=False,clinical_confirmed=False,
        empty_slots_ignored=[0,3,15,16,31],
        parameters='first3: predicted centroid in common normalized frame; last6: first two rows of rotation',
        source_transform=json.loads((case/'reports/preparation.json').read_text(encoding='utf-8'))['normalization'],
        note='Raw per-slot model output only; do not interpret absent slots as generated teeth')
    (a.out/'prediction_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'Checkpoint-backed raw parameters saved in {a.out}; ADD not evaluated.')


if __name__ == '__main__':
    main()
