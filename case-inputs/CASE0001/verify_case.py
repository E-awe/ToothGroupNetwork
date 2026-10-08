"""Read real before-only cases through official DentalDataset/DataLoader; no network."""
import argparse, csv, hashlib, importlib.abc, importlib.metadata, json, os, platform, sys, uuid
from pathlib import Path
from types import SimpleNamespace

SHAPES=dict(index=(1,),feats=(1,32,10,256,64),center=(1,32,256,64,3),
    cordinates=(1,32,256,64,9),faces=(1,32,256,64,3),Fs=(1,32),
    before_points=(1,32,512,3),centroid=(1,32,3),masks=(1,32))


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--case',type=Path,default=Path(__file__).resolve().parent/'case_1')
    parser.add_argument('--report',type=Path)
    parser.add_argument('--apply-dataset-patch',action='store_true')
    a=parser.parse_args();repo=a.repo.resolve();case=a.case.resolve()
    if a.apply_dataset_patch:
        from apply_dataset_patch import apply
        apply(repo)
    report=a.report or case/'reports'/f'input_read_{uuid.uuid4().hex[:8]}.json'
    if report.exists():raise FileExistsError('Existing reports are never overwritten')
    # Prevent accidental main/runners/model imports: this task is Dataset loading only.
    class NoNetwork(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if fullname.split('.')[0] in ['models','runners','main']:
                raise ImportError('Input-only verification refuses network or runner import: '+fullname)
    sys.meta_path.insert(0,NoNetwork())
    cfg=json.loads((case/'dataset_config.json').read_text())
    assert cfg['NAME']=='DentalDataset' and cfg['before_only'] and not cfg['train'] and cfg['bs']==1
    assert not set(cfg)&{'after_path','paramroot'}
    for key in ['dataroot','before_path','file']:
        cfg[key]=str(case/cfg[key]);assert Path(cfg[key]).resolve().is_relative_to(case)
    # No after, target/param folder, model weights or old smoke outputs in this dataset.
    forbidden=('after','placeholder','gt_params','param_placeholder','case_0','smoke_batch')
    for path in case.rglob('*'):
        relative=str(path.relative_to(case)).lower()
        assert not any(token in relative for token in forbidden),relative
    read_paths=set()
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):return
        path=Path(os.fsdecode(args[0])).resolve()
        if path.is_relative_to(case):
            read_paths.add(str(path.relative_to(case)))
        parts={part.lower() for part in path.parts}
        if parts & {'case_0','pcd_after512','pcd_after512_placeholder','param','param_placeholder'}:
            raise RuntimeError('Forbidden historical/after/GT access: '+str(path))
    sys.addaudithook(audit)
    import numpy as np
    import torch
    import trimesh
    import open3d as o3d
    from easydict import EasyDict
    sys.path.insert(0,str(repo))
    import datasets
    from utils import builder
    assert Path(datasets.__file__).resolve().is_relative_to(repo)
    assert Path(builder.__file__).resolve().is_relative_to(repo)
    _,loader=builder.dataset_builder(SimpleNamespace(num_workers=0),EasyDict(cfg))
    assert hasattr(loader.dataset,'before_only') and loader.dataset.before_only
    manifest=json.loads((case/'case_manifest.json').read_text(encoding='utf-8'))
    expected={m['dataset_id']:m for m in manifest}
    assert len(loader)==len(loader.dataset)==len(expected)
    results=[]
    for batch in loader:
        assert set(batch)==set(SHAPES),batch.keys()
        for field,shape in SHAPES.items():
            assert tuple(batch[field].shape)==shape,(field,batch[field].shape)
            assert torch.isfinite(batch[field]).all(),field
        index=int(batch['index'].item());spec=expected[index]; name=spec['case_name']
        prep=json.loads((case/spec['preparation_report']).read_text(encoding='utf-8'))
        mask=batch['masks'][0].numpy()
        assert np.array_equal(mask,np.load(case/'masks'/f'{index}.npy',allow_pickle=False))
        assert np.array_equal(mask,spec['mask'])
        assert np.all(batch['Fs'][0].numpy()[mask==1]==16384)
        for field in SHAPES.keys()-{'index','masks'}:
            assert not torch.count_nonzero(batch[field][0,mask==0]).item(),field
        reference=case/'original_reference'/name
        originals_verified=0
        for path in reference.rglob('*'):
            if not path.is_file():continue
            relative=path.relative_to(reference).as_posix()
            digest=prep['reviewed_mapping_sha256'] if relative=='reviewed_component_fdi.csv' else prep['source_file_sha256'][relative]
            assert sha(path)==digest,relative; originals_verified+=1
        geometry=[]
        matrix=np.asarray(prep['normalization']['matrix'])
        whole_meshes={jaw:trimesh.load_mesh(reference/f'{jaw}_predicted_source_coords.ply',process=False)
                      for jaw in ['upper','lower']}
        component_ids={jaw:np.load(reference/f'{jaw}_vertex_components.npy',allow_pickle=False)
                       for jaw in whole_meshes}
        for tooth in prep['teeth']:
            slot=tooth['slot'];base=f'{index}_{slot}'
            mesh=trimesh.load_mesh(case/'remesh_before'/f'{base}.obj',process=False)
            single=trimesh.load_mesh(case/'single_before'/f'{base}.obj')
            original=trimesh.load_mesh(case/'source_components'/name/f"{tooth['jaw']}_{tooth['component']}.obj",process=False)
            jaw=tooth['jaw'];cid=int(tooth['component'][1:])
            source_indices=np.load(case/'source_vertex_indices'/name/f'{jaw}_C{cid}.npy',allow_pickle=False)
            assert np.array_equal(source_indices,np.flatnonzero(component_ids[jaw]==cid))
            assert np.allclose(original.vertices,whole_meshes[jaw].vertices[source_indices],rtol=0,atol=1e-12)
            source_faces=whole_meshes[jaw].faces[component_ids[jaw][whole_meshes[jaw].faces[:,0]]==cid]
            assert np.array_equal(source_indices[original.faces],source_faces)
            unprocessed=trimesh.load_mesh(case/'single_before'/f'{base}.obj',process=False)
            assert np.array_equal(unprocessed.faces,original.faces)
            assert np.allclose(unprocessed.vertices,original.vertices@matrix[:3,:3].T+matrix[:3,3],rtol=0,atol=1e-11)
            assert mesh.is_watertight and mesh.is_winding_consistent and np.min(mesh.area_faces)>0
            assert np.isfinite(mesh.vertex_normals).all() and np.isfinite(mesh.face_normals).all()
            points=np.asarray(o3d.io.read_point_cloud(str(case/'pcd_before512'/f'{base}.ply')).points)
            selected=np.load(case/'sampling_indices'/f'{base}.npy',allow_pickle=False)
            assert points.shape==(513,3) and selected.shape==(512,) and len(np.unique(selected))==512
            assert np.allclose(points[:512],single.vertices[selected],rtol=0,atol=1e-11)
            assert np.allclose(points[-1],single.centroid,rtol=0,atol=1e-11)
            assert np.array_equal(batch['before_points'][0,slot].numpy(),points[:512])
            assert np.array_equal(batch['centroid'][0,slot].numpy(),points[-1])
            order=np.arange(16384).reshape(64,256).T
            assert np.array_equal(batch['faces'][0,slot].numpy(),mesh.faces[order])
            # Same official feature routine, checked here against the saved mesh.
            from datasets.DentalDataset import load_mesh_shape
            values=load_mesh_shape(mesh,request=loader.dataset.feats)
            for key,value in zip(['feats','center','cordinates','faces','Fs'],values):
                assert np.array_equal(batch[key][0,slot].numpy(),value),key
            geometry.append(dict(slot=slot,jaw=tooth['jaw'],component=tooth['component'],
                original_prediction_fdi=tooth['original_prediction_fdi'],fdi_candidate=tooth['fdi_candidate'],
                watertight=True,winding_consistent=True,source_vertex_face_correspondence_verified=True,
                mesh_faces=16384,points=512,ply_rows=513))
        if name=='CASE0001':
            assert mask[:16].sum()==13 and mask[16:].sum()==14
            assert np.flatnonzero(mask==0).tolist()==[0,3,15,16,31]
            lower={t['component']:t for t in prep['teeth'] if t['jaw']=='lower'}
            assert [(lower[c]['original_prediction_fdi'],lower[c]['fdi_candidate']) for c in ['C10','C11','C12']]==[(43,43),(43,44),(44,45)]
        assert not prep['historical_cache_used'] and not prep['after_data_used'] and not prep['gt_data_used']
        results.append(dict(case_name=name,dataset_id=index,upper_teeth=int(mask[:16].sum()),
            lower_teeth=int(mask[16:].sum()),mask=mask.tolist(),empty_slots=prep['empty_slots'],
            shapes={k:list(v.shape) for k,v in batch.items()},dtypes={k:str(v.dtype) for k,v in batch.items()},
            originals_verified=originals_verified,geometry_checks=geometry,clinical_confirmed=False))
        print(f'PASS {name}: upper{mask[:16].sum()} + lower{mask[16:].sum()} real teeth, no after/GT.',flush=True)
    output=dict(status='PASS',test='native_DentalDataset_before_only_and_official_DataLoader',
        python=sys.version,interpreter=sys.executable,platform=platform.platform(),
        versions={n:importlib.metadata.version(n) for n in ['torch','numpy','open3d','trimesh','timm','scipy']},
        dataset_python=str(Path(sys.modules['datasets.DentalDataset'].__file__).resolve()),
        dataset_source_sha256=sha(repo/'datasets/DentalDataset.py'),batch_size=1,num_workers=0,
        cases=results,dataset_read_paths=sorted(read_paths),after_or_gt_fields_returned=False,
        after_or_gt_files_read=False,historical_case_data_read=False,network_imported=False,
        model_forward_executed=False,ddim_executed=False,gpu_inference_executed=False,
        real_after_available=False,checkpoint_required_for_this_test=False)
    report.parent.mkdir(parents=True,exist_ok=True)
    report.write_text(json.dumps(output,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(f'Report: {report}')


if __name__=='__main__':main()
