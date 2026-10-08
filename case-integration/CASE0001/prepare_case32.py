"""Prepare CASE0001's independent reviewed candidates; reuse historical upper MAPS/FPS."""
import argparse, ast, csv, hashlib, json, random, shutil, sys
from pathlib import Path
import numpy as np
import open3d as o3d
import pyfqmr
import trimesh
from scipy.spatial import cKDTree

UPPER = [18,17,16,15,14,13,12,11,21,22,23,24,25,26,27,28]
LOWER = [48,47,46,45,44,43,42,41,31,32,33,34,35,36,37,38]
SLOTS = UPPER + LOWER


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def inventory(root):
    return {str(p.relative_to(root)): sha(p) for p in root.rglob('*') if p.is_file()}


def read_cloud(path):
    return np.asarray(o3d.io.read_point_cloud(str(path)).points)


def write_cloud(path, xyz):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    assert o3d.io.write_point_cloud(str(path), pcd)


def face_set(faces):
    faces = np.sort(faces, axis=1)
    return faces[np.lexsort(faces.T[::-1])]


def equivalent_mesh(a, b, tol=1e-6):
    assert len(a.vertices) == len(b.vertices) and len(a.faces) == len(b.faces)
    distances, indices = cKDTree(b.vertices).query(a.vertices)
    assert distances.max() < tol and len(np.unique(indices)) == len(b.vertices)
    assert np.array_equal(face_set(indices[a.faces]), face_set(b.faces))
    return float(distances.max())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['source', 'reviewed-csv', 'history', 'repo', 'out']:
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    source, history, repo, out = [x.resolve() for x in [a.source,a.history,a.repo,a.out]]
    if out.exists():
        raise FileExistsError('Use a new case directory; existing cases are not overwritten')
    protected_source, protected_history = inventory(source), inventory(history)
    with a.reviewed_csv.open(encoding='utf-8-sig', newline='') as f:
        reviewed = list(csv.DictReader(f))
    teeth = [r for r in reviewed if r['record_type']=='component' and r['role']=='tooth']
    assert len(teeth)==27 and sum(r['jaw']=='upper' for r in teeth)==13
    mapping = {(r['jaw'],int(r['component_id'][1:])):r for r in teeth}
    meshes = {jaw:trimesh.load_mesh(source/f'{jaw}_predicted_source_coords.ply',process=False)
              for jaw in ['upper','lower']}
    ids = {jaw:np.load(source/f'{jaw}_vertex_components.npy',allow_pickle=False)
           for jaw in meshes}
    combined = trimesh.util.concatenate(list(meshes.values()))
    center = np.asarray(combined.centroid)
    old_report = json.loads((history/'reports/preparation.json').read_text())
    old_center = np.asarray(old_report['normalization']['center'])
    delta = (old_center-center)/40.0
    old_upper = trimesh.load_mesh(history/'original_coordinates/initial_upper.obj',process=False)
    assert np.array_equal(old_upper.faces,meshes['upper'].faces)
    upper_delta = np.linalg.norm(old_upper.vertices-meshes['upper'].vertices,axis=1).max()
    assert upper_delta < 1e-6
    matrix = np.eye(4); matrix[:3,:3]/=40.0; matrix[:3,3]=-center/40.0
    dirs = ['single_before','simplify_before','remesh_before','pcd_before512',
            'pcd_after512_PLACEHOLDER','param_PLACEHOLDER','sampling_indices',
            'source_components','source_vertex_indices','background','original_reference','reports']
    for name in dirs:
        (out/name).mkdir(parents=True,exist_ok=False)
    for name in ['component_fdi_candidates.csv','component_fdi_candidates.json',
                 'input_manifest.json','source_geometry.json','checkpoint_manifest.json']:
        shutil.copy2(source/name,out/'original_reference'/name)
    shutil.copytree(source/'predictions',out/'original_reference/predictions')
    shutil.copy2(a.reviewed_csv,out/'original_reference/reviewed_component_fdi.csv')
    for jaw in meshes:
        shutil.copy2(source/f'{jaw}_predicted_source_coords.ply',out/'original_reference'/f'{jaw}_source_coords.ply')
    sys.path.insert(0,str(repo/'data_preprocess'))
    from maps import MAPS
    fps_source=repo/'data_preprocess/get_pointcloud.py'
    node=next(n for n in ast.parse(fps_source.read_text()).body
              if isinstance(n,ast.FunctionDef) and n.name=='farthest_point_sample')
    env={'np':np};exec(compile(ast.Module(body=[node],type_ignores=[]),str(fps_source),'exec'),env)
    fps=env['farthest_point_sample']
    mask=np.zeros(32,dtype=np.int32); records=[]
    for jaw,whole in meshes.items():
        labels=ids[jaw]
        assert labels.shape==(len(whole.vertices),)
        for c in np.unique(labels):
            c=int(c)
            face_indices=np.flatnonzero(labels[whole.faces[:,0]]==c)
            assert np.all(labels[whole.faces[face_indices]]==c)
            tooth=whole.submesh([face_indices],append=True,repair=False)
            assert tooth.is_watertight and tooth.is_winding_consistent
            name=f'{jaw}_C{c}.obj'
            tooth.visual=trimesh.visual.ColorVisuals(mesh=tooth)
            if (jaw,c) not in mapping:
                tooth.export(out/'background'/name,digits=17,include_color=False)
                continue
            r=mapping[jaw,c]; fdi=int(r['reviewed_candidate_fdi']);slot=SLOTS.index(fdi)
            assert (slot<16)==(jaw=='upper') and mask[slot]==0
            tooth.export(out/'source_components'/name,digits=17,include_color=False)
            np.save(out/'source_vertex_indices'/f'{jaw}_C{c}.npy',np.flatnonzero(labels==c))
            basename=f'1_{slot}.obj'; audit={}
            if jaw=='upper':
                original=trimesh.load_mesh(history/f'original_coordinates/component_{c-1}.stl',process=True)
                audit['original_component_max_vertex_distance']=equivalent_mesh(original,tooth)
                for sub in ['single_before','simplify_before','remesh_before']:
                    old=trimesh.load_mesh(history/sub/f'0_{slot}.obj',process=False)
                    new=old.copy();new.vertices=new.vertices+delta
                    new.export(out/sub/basename,digits=12,include_color=False)
                    reloaded=trimesh.load_mesh(out/sub/basename,process=False)
                    assert np.array_equal(reloaded.faces,old.faces)
                    assert np.allclose(reloaded.vertices,old.vertices+delta,atol=1e-11,rtol=0)
                    audit[sub+'_reused_sha256']=sha(history/sub/f'0_{slot}.obj')
                cloud=read_cloud(history/'pcd_before512'/f'0_{slot}.ply')+delta
                audit['pointcloud_reused_sha256']=sha(history/'pcd_before512'/f'0_{slot}.ply')
                mode='historical_upper_reused_plus_common_center_translation'
            else:
                tooth.vertices=(tooth.vertices-center)/40.0
                tooth.export(out/'single_before'/basename,digits=12,include_color=False)
                single=trimesh.load_mesh(out/'single_before'/basename)
                np.random.seed(slot); random.seed(slot)
                points,indices=fps(single.vertices,512)
                cloud=np.concatenate([points,single.centroid[None,:]])
                simplifier=pyfqmr.Simplify(); simplifier.setMesh(single.vertices,single.faces)
                simplifier.simplify_mesh(target_count=256,aggressiveness=7,preserve_border=True,verbose=False)
                v,f,_=simplifier.getMesh(); assert len(f)==256
                base=trimesh.Trimesh(v,f,process=False); assert base.is_watertight
                base.export(out/'simplify_before'/basename,digits=12,include_color=False)
                remesh=MAPS(v,f,base_size=256,timeout=20).mesh_upsampling(depth=3)
                assert len(remesh.faces)==16384 and np.isfinite(remesh.vertices).all()
                remesh.export(out/'remesh_before'/basename,digits=12,include_color=False)
                mode='new_lower_official_fps512_qem256_maps_depth3'
            assert cloud.shape==(513,3)
            single=trimesh.load_mesh(out/'single_before'/basename)
            distances,indices=cKDTree(single.vertices).query(cloud[:512])
            assert distances.max()<1e-7 and len(np.unique(indices))==512
            assert np.allclose(cloud[-1],single.centroid,atol=1e-7,rtol=0)
            normalized_source_center=(whole.vertices[labels==c].mean(0)-center)/40.
            assert np.isfinite(normalized_source_center).all()
            np.save(out/'sampling_indices'/f'1_{slot}.npy',indices.astype(np.int64))
            ply=out/'pcd_before512'/f'1_{slot}.ply';write_cloud(ply,cloud)
            shutil.copy2(ply,out/'pcd_after512_PLACEHOLDER'/ply.name)
            mask[slot]=1
            records.append(dict(jaw=jaw,component=f'C{c}',fdi_candidate=fdi,slot=slot,
                original_prediction_fdi=int(r['original_prediction_fdi']),clinical_confirmed=False,
                preprocessing=mode,source_vertex_count=int((labels==c).sum()),
                original_faces=len(face_indices),remesh_faces=16384,ply_rows=513,
                centroid=cloud[-1].tolist(),point_to_single_vertex_max_error=float(distances.max()),reuse_audit=audit))
            print(f'{jaw} C{c}: FDI candidate{fdi}, slot{slot}, {mode}',flush=True)
    assert mask.sum()==27 and np.flatnonzero(mask==0).tolist()==[0,3,15,16,31]
    np.save(out/'mask.npy',mask);np.save(out/'param_PLACEHOLDER/1.npy',np.tile(np.eye(3),(32,1,1)))
    (out/'case_ids.txt').write_text('1\n')
    cfg=dict(NAME='DentalDataset',train=False,bs=1,npoint=512,dataroot='remesh_before',
             paramroot='param_PLACEHOLDER',before_path='pcd_before512',
             after_path='pcd_after512_PLACEHOLDER',file='case_ids.txt')
    (out/'dataset_config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    rows=[]
    for slot,fdi in enumerate(SLOTS):
        item=next((r for r in records if r['slot']==slot),None)
        rows.append(dict(slot=slot,jaw='upper' if slot<16 else 'lower',fdi_candidate=fdi,
            component='' if item is None else item['component'],mask=int(mask[slot]),
            original_prediction_fdi='' if item is None else item['original_prediction_fdi'],
            clinical_confirmed=False,status='unrepresented_in_input' if item is None else '3d_review_candidate',
            user_designated_missing_position=fdi==15))
    with (out/'tadpm_slot_manifest.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    assert protected_source==inventory(source) and protected_history==inventory(history)
    report=dict(case_id='CASE0001',dataset_integer_id=1,source_files_unchanged=True,
        history_files_unchanged=True,source_sha256=protected_source,history_sha256=protected_history,
        historical_upper_geometry_max_distance=float(upper_delta),upper_reused_count=13,lower_new_count=14,
        mapping_policy='independent_user_approved_3d_review_candidates_not_clinical_ground_truth',
        raw_tgnet_duplicate_conflict='lower C10 and C11 both43; raw mapping cannot occupy32 fixed slots uniquely',
        lower_candidate_overrides={'C11':44,'C12':45},mask=mask.tolist(),unrepresented_slots=[0,3,15,16,31],
        unrepresented_fdi=[18,15,28,48,38],user_designated_missing_position_candidate=15,
        normalization=dict(formula='(p_source - merged_upper_lower_surface_centroid) / 40.0',
            center=center.tolist(),matrix=matrix.tolist(),inverse_matrix=np.linalg.inv(matrix).tolist(),
            historical_upper_center=old_center.tolist(),upper_cache_translation=delta.tolist(),
            scale=40.,extra_rotation_applied=False,includes_background_in_global_center=True,
            background_excluded_from_tooth_input=True,source_frame_preserves_relative_jaw_pose=True),
        after_ground_truth_available=False,after_policy='before-copy loader placeholder; not an after scan',
        params_policy='32 identity rotations; loader placeholders, not registered treatment parameters',
        training_allowed=False,evaluation_allowed=False,model_forward_executed=False,teeth=records)
    (out/'reports/preparation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('Prepared27 candidate teeth in32 fixed slots; upper cache reused13; original files unchanged.')


if __name__=='__main__':
    main()
