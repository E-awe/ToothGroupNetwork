"""Convert TGNet case exports to TADPM BEFORE ONLY; never read historical datasets.

Single: --source DIR --reviewed-csv CSV --repo TADPM --out NEW_DIR
Batch: --batch-manifest JSON --repo TADPM --out NEW_DIR
JSON list entries: source, reviewed_csv, case_name, dataset_id (unique integer).
"""
import argparse, ast, csv, hashlib, json, random, shutil, sys
from pathlib import Path
import numpy as np
import open3d as o3d
import pyfqmr
import trimesh

SLOTS=[18,17,16,15,14,13,12,11,21,22,23,24,25,26,27,28,
       48,47,46,45,44,43,42,41,31,32,33,34,35,36,37,38]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(root):
    return {p.relative_to(root).as_posix():sha(p) for p in root.rglob('*') if p.is_file()}


def official_fps(repo):
    path=repo/'data_preprocess/get_pointcloud.py'
    function=next(n for n in ast.parse(path.read_text()).body
                  if isinstance(n,ast.FunctionDef) and n.name=='farthest_point_sample')
    # Importing that official CLI would parse argv and launch its paired-after pool.
    scope={'np':np}
    exec(compile(ast.Module(body=[function],type_ignores=[]),str(path),'exec'),scope)
    return scope['farthest_point_sample']


def prepare_case(spec,repo,out,seed,fps,MAPS):
    source=Path(spec['source']).resolve(); reviewed=Path(spec['reviewed_csv']).resolve()
    name=spec['case_name']; index=spec['dataset_id']
    before_hash=inventory(source); reviewed_hash=sha(reviewed)
    with reviewed.open(encoding='utf-8-sig',newline='') as f:
        rows=[r for r in csv.DictReader(f) if r['case_id']==name]
    teeth=[r for r in rows if r['record_type']=='component' and r['role']=='tooth']
    if not teeth:
        raise ValueError(f'{name}: no matching reviewed tooth rows')
    mapping={(r['jaw'],int(r['component_id'][1:])):r for r in teeth}
    if len(mapping)!=len(teeth):
        raise ValueError(f'{name}: duplicate component records')
    manifest=json.loads((source/'input_manifest.json').read_text(encoding='utf-8'))
    meshes={jaw:trimesh.load_mesh(source/f'{jaw}_predicted_source_coords.ply',process=False)
            for jaw in ['upper','lower']}
    ids={jaw:np.load(source/f'{jaw}_vertex_components.npy',allow_pickle=False) for jaw in meshes}
    frame_checks={}
    for jaw,mesh in meshes.items():
        model_path=source/'input'/name/f'{name}_{jaw}.obj'
        model=trimesh.load_mesh(model_path,process=False)
        matrix=np.asarray(manifest[jaw]['rotation_source_to_model'])
        transformed=mesh.vertices@matrix[:3,:3].T+matrix[:3,3]
        assert np.array_equal(mesh.faces,model.faces)
        error=float(np.max(np.abs(transformed-model.vertices)))
        assert error<1e-7 and sha(model_path)==manifest[jaw]['obj_sha256']
        assert ids[jaw].shape==(len(mesh.vertices),)
        prediction=json.loads((source/'predictions'/f'{name}_{jaw}.json').read_text())
        assert len(prediction['labels'])==len(prediction['instances'])==len(mesh.vertices)
        frame_checks[jaw]=dict(source_to_tgnet_model=matrix.tolist(),max_coordinate_error=error)
    center=np.asarray(trimesh.util.concatenate(list(meshes.values())).centroid)
    transform=np.eye(4); transform[:3,:3]/=40.; transform[:3,3]=-center/40.
    case_ref=out/'original_reference'/name
    # Only originals from THIS CASE; no historical case, no generated after or target.
    case_ref.mkdir(parents=True)
    for jaw in meshes:
        for filename in [f'{jaw}_predicted_source_coords.ply',f'{jaw}_vertex_components.npy']:
            shutil.copy2(source/filename,case_ref/filename)
    for filename in ['component_fdi_candidates.csv','component_fdi_candidates.json',
                     'input_manifest.json','source_geometry.json']:
        shutil.copy2(source/filename,case_ref/filename)
    shutil.copytree(source/'predictions',case_ref/'predictions')
    shutil.copytree(source/'input',case_ref/'input')
    shutil.copy2(reviewed,case_ref/'reviewed_component_fdi.csv')
    for folder in ['source_components','source_vertex_indices','background']:
        (out/folder/name).mkdir(parents=True)
    mask=np.zeros(32,dtype=np.int32); records=[]; backgrounds=[]
    for jaw,whole in meshes.items():
        components=ids[jaw]
        for cid in np.unique(components):
            cid=int(cid)
            selected_faces=np.flatnonzero(components[whole.faces[:,0]]==cid)
            assert np.all(components[whole.faces[selected_faces]]==cid)
            tooth=whole.submesh([selected_faces],append=True,repair=False)
            assert tooth.is_watertight and tooth.is_winding_consistent
            tooth.visual=trimesh.visual.ColorVisuals(mesh=tooth)
            component=f'{jaw}_C{cid}.obj'
            if (jaw,cid) not in mapping:
                roles=[r for r in rows if r['jaw']==jaw and r['component_id']==f'C{cid}' and r['role']=='gingiva']
                if len(roles)!=1:
                    raise ValueError(f'{name}/{jaw}/C{cid}: unclassified mesh component')
                tooth.export(out/'background'/name/component,digits=17,include_color=False)
                backgrounds.append(dict(jaw=jaw,component=f'C{cid}',vertices=len(tooth.vertices)))
                continue
            row=mapping[(jaw,cid)]; fdi=int(row['reviewed_candidate_fdi']);slot=SLOTS.index(fdi)
            assert (slot<16)==(jaw=='upper') and not mask[slot]
            assert len(tooth.vertices)==int(row['vertex_count'])
            tooth.export(out/'source_components'/name/component,digits=17,include_color=False)
            selected_vertices=np.flatnonzero(components==cid)
            np.save(out/'source_vertex_indices'/name/f'{jaw}_C{cid}.npy',selected_vertices)
            tooth.vertices=(tooth.vertices-center)/40.
            basename=f'{index}_{slot}.obj'
            tooth.export(out/'single_before'/basename,digits=12,include_color=False)
            single=trimesh.load_mesh(out/'single_before'/basename)
            np.random.seed(seed+index*32+slot);random.seed(seed+index*32+slot)
            assert len(single.vertices)>=512
            points,selected=fps(single.vertices,512)
            xyz=np.concatenate([points,np.asarray(single.centroid)[None,:]])
            cloud=o3d.geometry.PointCloud(); cloud.points=o3d.utility.Vector3dVector(xyz)
            assert o3d.io.write_point_cloud(str(out/'pcd_before512'/f'{index}_{slot}.ply'),cloud)
            np.save(out/'sampling_indices'/f'{index}_{slot}.npy',selected.astype(np.int64))
            simplifier=pyfqmr.Simplify(); simplifier.setMesh(single.vertices,single.faces)
            simplifier.simplify_mesh(target_count=256,aggressiveness=7,preserve_border=True,verbose=False)
            v,f,_=simplifier.getMesh(); assert len(f)==256
            base=trimesh.Trimesh(v,f,process=False); assert base.is_watertight
            base.export(out/'simplify_before'/basename,digits=12,include_color=False)
            remesh=MAPS(v,f,base_size=256,timeout=20).mesh_upsampling(depth=3)
            assert len(remesh.faces)==16384 and np.isfinite(remesh.vertices).all()
            remesh.export(out/'remesh_before'/basename,digits=12,include_color=False)
            mask[slot]=1
            records.append(dict(jaw=jaw,component=f'C{cid}',slot=slot,fdi_candidate=fdi,
                original_prediction_fdi=int(row['original_prediction_fdi']),clinical_confirmed=False,
                source_component_vertices=len(selected_vertices),source_faces=len(selected_faces),
                before_centroid=xyz[-1].tolist(),seed=seed+index*32+slot,
                preprocessing='fresh_this_case_geometry_FPS512_QEM256_official_MAPS_depth3',
                reused_historical_geometry=False))
            print(f'{name}/{jaw}/C{cid}: candidateFDI{fdi}, slot{slot}; new FPS/MAPS',flush=True)
    assert len(records)==len(teeth)
    assert inventory(source)==before_hash and sha(reviewed)==reviewed_hash
    report=dict(case_name=name,dataset_id=index,teeth=records,backgrounds=backgrounds,
        mask=mask.tolist(),empty_slots=np.flatnonzero(mask==0).tolist(),
        unrepresented_fdi=[SLOTS[s] for s in np.flatnonzero(mask==0)],
        source_file_sha256=before_hash,reviewed_mapping_sha256=reviewed_hash,
        original_files_unchanged=True,clinical_confirmed=False,
        after_files_created=False,after_data_used=False,gt_files_created=False,gt_data_used=False,
        historical_cache_used=False,model_forward_executed=False,normalization=dict(
            formula='(source_vertex - merged_upper_lower_surface_area_centroid) / 40.0',
            center=center.tolist(),scale=40.,matrix=transform.tolist(),inverse_matrix=np.linalg.inv(transform).tolist(),
            includes_gingiva_in_global_center=True,gingiva_excluded_from_network_input=True,
            extra_rotation_applied=False,coordinate_frame_checks=frame_checks),
        manifold='not run: every component watertight and winding-consistent',
        raw_prediction_policy='preserved bytes; only independent reviewed candidates choose slots')
    np.save(out/'masks'/f'{index}.npy',mask)
    (out/'reports'/f'{name}_preparation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path);p.add_argument('--reviewed-csv',type=Path)
    p.add_argument('--case-name');p.add_argument('--dataset-id',type=int,default=1)
    p.add_argument('--batch-manifest',type=Path)
    p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,default=2026)
    a=p.parse_args();repo=a.repo.resolve();out=a.out.resolve()
    if a.batch_manifest:
        if a.source or a.reviewed_csv: p.error('Use single-case OR batch arguments')
        specs=json.loads(a.batch_manifest.read_text(encoding='utf-8'))
        for spec in specs:
            for key in ['source','reviewed_csv']:
                spec[key]=str((a.batch_manifest.resolve().parent/Path(spec[key])).resolve())
    else:
        if not a.source or not a.reviewed_csv: p.error('--source and --reviewed-csv required')
        specs=[dict(source=str(a.source),reviewed_csv=str(a.reviewed_csv),
                    case_name=a.case_name or a.source.name,dataset_id=a.dataset_id)]
    if not specs or len({s['dataset_id'] for s in specs})!=len(specs) or len({s['case_name'] for s in specs})!=len(specs):
        p.error('Case names and integer IDs must be nonempty and unique')
    for s in specs:
        if type(s['dataset_id']) is not int or s['dataset_id']<0 or Path(s['case_name']).name!=s['case_name'] or s['case_name'] in ['.','..']:
            p.error('Invalid case name or nonnegative integer dataset_id')
    if out.exists(): raise FileExistsError('Existing output is never overwritten; choose a new directory')
    for s in specs:
        source=Path(s['source']).resolve()
        if out.is_relative_to(source) or source.is_relative_to(out):
            p.error('Output must be separate from source')
    for folder in ['single_before','simplify_before','remesh_before','pcd_before512','sampling_indices',
                   'source_components','source_vertex_indices','background','original_reference','reports','masks']:
        (out/folder).mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(repo/'data_preprocess'))
    from maps import MAPS
    fps=official_fps(repo)
    reports=[prepare_case(s,repo,out,a.seed,fps,MAPS) for s in specs]
    (out/'case_ids.txt').write_text(''.join(str(s['dataset_id'])+'\n' for s in specs))
    config=dict(NAME='DentalDataset',before_only=True,train=False,bs=1,npoint=512,
                dataroot='remesh_before',before_path='pcd_before512',file='case_ids.txt')
    (out/'dataset_config.json').write_text(json.dumps(config,indent=2)+'\n')
    manifest=[dict(case_name=r['case_name'],dataset_id=r['dataset_id'],mask=r['mask'],
                   preparation_report=f"reports/{r['case_name']}_preparation.json") for r in reports]
    (out/'case_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (out/'slot_manifest.csv').open('w',encoding='utf-8-sig',newline='') as f:
        columns=['case_name','dataset_id','slot','jaw','fdi_candidate','component','mask','original_prediction_fdi','clinical_confirmed']
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for r in reports:
            for slot,fdi in enumerate(SLOTS):
                tooth=next((t for t in r['teeth'] if t['slot']==slot),{})
                writer.writerow(dict(case_name=r['case_name'],dataset_id=r['dataset_id'],slot=slot,
                    jaw='upper' if slot<16 else 'lower',fdi_candidate=fdi,mask=r['mask'][slot],
                    component=tooth.get('component',''),original_prediction_fdi=tooth.get('original_prediction_fdi',''),clinical_confirmed=False))
    print(f'Prepared {len(specs)} real before-only cases; no after/GT/historical cache.',flush=True)


if __name__=='__main__':main()
