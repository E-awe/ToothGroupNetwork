"""Read one prepared case using TADPM's actual Dataset and DataLoader; no forward."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
import uuid
from pathlib import Path
from types import SimpleNamespace


FIELDS = ['index', 'feats', 'center', 'cordinates', 'faces', 'Fs',
          'before_points', 'after_points', 'centroid', 'after_centroid', 'gt_params', 'masks']
SHAPES = [(1,), (1,32,10,256,64), (1,32,256,64,3), (1,32,256,64,9),
          (1,32,256,64,3), (1,32), (1,32,512,3), (1,32,512,3),
          (1,32,3), (1,32,3), (1,32,9), (1,32)]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_dataset(repo, case):
    import torch
    from easydict import EasyDict
    sys.path.insert(0, str(repo))
    import datasets
    from utils import builder
    assert Path(datasets.__file__).resolve().is_relative_to(repo), datasets.__file__
    assert Path(builder.__file__).resolve().is_relative_to(repo), builder.__file__
    config = json.loads((case/'dataset_config.json').read_text())
    for key in ['dataroot','paramroot','before_path','after_path','file']:
        config[key] = str(case/config[key])
    assert config['train'] is False and config['bs'] == 1
    _, loader = builder.dataset_builder(SimpleNamespace(num_workers=0), EasyDict(config))
    assert len(loader.dataset) == 1 and len(loader) == 1
    return loader, next(iter(loader)), config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', type=Path, required=True)
    p.add_argument('--case', type=Path, required=True)
    p.add_argument('--report', type=Path)
    p.add_argument('--checkpoint', type=Path, help='Optional: strictly validate a trained full TADPM checkpoint without running forward')
    a = p.parse_args()
    repo, case = a.repo.resolve(), a.case.resolve()
    report = a.report or case/'reports'/('input_read_'+uuid.uuid4().hex[:8]+'.json')
    if report.exists():
        raise FileExistsError(f'Report already exists: {report}')
    import numpy as np
    import open3d as o3d
    import torch
    import trimesh
    loader, batch, config = load_dataset(repo,case)
    d = dict(zip(FIELDS,batch))
    for key, shape in zip(FIELDS, SHAPES):
        assert tuple(d[key].shape) == shape, (key,d[key].shape)
        assert torch.isfinite(d[key]).all(), key
    assert d['index'].item() == 1
    mask = d['masks'][0].numpy()
    assert np.array_equal(mask,np.load(case/'mask.npy',allow_pickle=False))
    assert int(mask[:16].sum()) == 13 and int(mask[16:].sum()) == 14
    assert np.flatnonzero(mask==0).tolist() == [0,3,15,16,31]
    for key in ['feats','center','cordinates','faces','Fs','before_points','after_points','centroid','after_centroid']:
        assert torch.count_nonzero(d[key][0,mask==0]).item() == 0, key
    assert np.all(d['Fs'][0].numpy()[mask==1] == 16384)
    assert torch.equal(d['before_points'],d['after_points'])
    assert torch.equal(d['centroid'],d['after_centroid'])
    identity = np.tile(np.eye(3),(32,1,1))
    assert np.array_equal(np.load(case/'param_PLACEHOLDER/1.npy'),identity)
    assert np.array_equal(d['gt_params'][0,:,3:].numpy(),np.tile([1,0,0,0,1,0],(32,1)))
    assert torch.equal(d['gt_params'][0,:,:3],d['centroid'][0])
    prep = json.loads((case/'reports/preparation.json').read_text(encoding='utf-8'))
    assert prep['after_ground_truth_available'] is False
    geometries = []
    for tooth in prep['teeth']:
        slot = tooth['slot']
        mesh = trimesh.load_mesh(case/'remesh_before'/f'1_{slot}.obj',process=False)
        single = trimesh.load_mesh(case/'single_before'/f'1_{slot}.obj')
        points = np.asarray(o3d.io.read_point_cloud(str(case/'pcd_before512'/f'1_{slot}.ply')).points)
        assert len(mesh.faces) == 16384 and mesh.is_watertight and mesh.is_winding_consistent
        assert np.isfinite(mesh.vertices).all() and np.isfinite(mesh.face_normals).all()
        assert np.min(mesh.area_faces)>0
        assert single.is_watertight and single.is_winding_consistent
        assert points.shape == (513,3)
        assert np.allclose(points[-1],single.centroid,rtol=0,atol=1e-7)
        sampling = np.load(case/'sampling_indices'/f'1_{slot}.npy',allow_pickle=False)
        assert sampling.shape == (512,) and len(np.unique(sampling)) == 512
        assert np.allclose(points[:512],single.vertices[sampling],rtol=0,atol=1e-7)
        # The official face-patch layout must reconstruct the exact saved face order.
        reordered = np.arange(16384).reshape(64,256).T
        assert np.array_equal(d['faces'][0,slot].numpy(),mesh.faces[reordered])
        geometries.append(dict(slot=slot,fdi_candidate=tooth['fdi_candidate'],faces=len(mesh.faces),
            watertight=True,winding_consistent=True,ply_rows=len(points),sampling_unique=512))
    assert len(geometries) == 27
    for name in ['component_fdi_candidates.csv','component_fdi_candidates.json','input_manifest.json',
                 'source_geometry.json','checkpoint_manifest.json','predictions/CASE0001_upper.json',
                 'predictions/CASE0001_lower.json']:
        assert sha(case/'original_reference'/name) == prep['source_sha256'][name], name
    with (case/'tadpm_slot_manifest.csv').open(encoding='utf-8-sig',newline='') as f:
        rows = list(csv.DictReader(f))
    lower = {r['component']:r for r in rows if r['jaw']=='lower' and r['component']}
    assert [(c,lower[c]['original_prediction_fdi'],lower[c]['fdi_candidate']) for c in ['C10','C11','C12']] == [('C10','43','43'),('C11','43','44'),('C12','44','45')]
    # Import the actual inference/model/diffusion entry points, without model construction or forward.
    import models
    import runners.test_tadpm
    from models.diffusion import diffusion
    assert hasattr(diffusion,'ddim_sample')
    checkpoint_status = {'provided':False,'trained_weights_confirmed':False}
    if a.checkpoint:
        from predict_before_only import check_checkpoint
        checkpoint_status = check_checkpoint(repo,a.checkpoint.resolve())
    versions = {}
    for name in ['torch','open3d','numpy','trimesh','timm','einops','scipy','easydict']:
        versions[name] = importlib.metadata.version(name)
    result = dict(status='PASS',case_id='CASE0001',test='native_DentalDataset_and_official_DataLoader',
        python=sys.version,interpreter=sys.executable,platform=platform.platform(),versions=versions,
        cuda_available=torch.cuda.is_available(),torch_cuda_version=torch.version.cuda,
        gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        num_workers=0,batch_size=1,teeth_upper=13,teeth_lower=14,mask=mask.tolist(),
        empty_slots=[0,3,15,16,31],empty_fdi=[18,15,28,48,38],
        shapes={k:list(v.shape) for k,v in d.items()},dtypes={k:str(v.dtype) for k,v in d.items()},
        normalized_common_center=prep['normalization']['center'],scale=40,
        geometry_checks=geometries,raw_predictions_preserved=True,clinical_confirmed=False,
        after_is_placeholder=True,metrics_allowed=False,trained_checkpoint=checkpoint_status,
        model_entry_import_passed=True,model_forward_executed=False,ddim_executed=False,
        dataset_config=config)
    try:
        result['repo_commit'] = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    except (FileNotFoundError,subprocess.CalledProcessError):
        result['repo_commit'] = None
    report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('PASS: official Dataset/DataLoader, 27/32 slots (upper13 + lower14).')
    print('Empty slots: 0,3,15,16,31. No model forward or DDIM inference executed.')
    print('After/parameters are loader placeholders, not treatment ground truth.')
    print(f'Report: {report}')


if __name__ == '__main__':
    main()
