
import json
import random
from pathlib import Path
import numpy as np
import os
import torch
import torch.utils.data as data
import trimesh
from scipy.spatial.transform import Rotation
from utils.builder import DATASETS
import copy
import open3d as o3d
from utils.pointcloud import read_pointcloud
from utils.rotation6d import matrix_to_rotation_6d

def  randomize_mesh_orientation(mesh: trimesh.Trimesh):
    mesh1 = copy.deepcopy(mesh)
    axis_seq = ''.join(random.sample('xyz', 3))
    angles = [random.uniform(-30,30) for _ in range(3)]
    rotation = Rotation.from_euler(axis_seq, angles, degrees=True)
    translation = np.array([random.uniform(-0.04,0.04),random.uniform(-0.04,0.04),random.uniform(-0.01,0.01)])
    centroid = mesh.centroid
    mesh1.vertices = rotation.apply(mesh1.vertices - centroid) + centroid + translation
    return mesh1, rotation, translation

def load_mesh_shape(mesh, augments=[], request=[], seed=None):

    F = mesh.faces
    V = mesh.vertices

    Fs = mesh.faces.shape[0]
    face_coordinate = V[F.flatten()].reshape(-1, 9)

    face_center = V[F.flatten()].reshape(-1, 3, 3).mean(axis=1)
    vertex_normals = mesh.vertex_normals
    face_normals = mesh.face_normals
    face_curvs = np.vstack([
        (vertex_normals[F[:, 0]] * face_normals).sum(axis=1),
        (vertex_normals[F[:, 1]] * face_normals).sum(axis=1),
        (vertex_normals[F[:, 2]] * face_normals).sum(axis=1),
    ])

    feats = []
    if 'area' in request:
        feats.append(mesh.area_faces)
    if 'normal' in request:
        feats.append(face_normals.T)
    if 'center' in request:
        feats.append(face_center.T)
    if 'face_angles' in request:
        feats.append(np.sort(mesh.face_angles, axis=1).T)
    if 'curvs' in request:
        feats.append(np.sort(face_curvs, axis=0))

    feats = np.vstack(feats)
    patch_num = Fs // 4 // 4 // 4
    allindex = np.array(list(range(0, Fs)))
    indices = allindex.reshape(-1, patch_num).transpose(1, 0)

    feats_patch = feats[:, indices]
    center_patch = face_center[indices]
    cordinates_patch = face_coordinate[indices]
    faces_patch = mesh.faces[indices]

    feats_patch = feats_patch
    center_patch = center_patch
    cordinates_patch = cordinates_patch
    faces_patch = faces_patch

    feats_patcha = np.concatenate((feats_patch, np.zeros((10, 256 - patch_num, 64), dtype=np.float32)), 1)
    center_patcha = np.concatenate((center_patch, np.zeros((256 - patch_num, 64, 3), dtype=np.float32)), 0)
    cordinates_patcha = np.concatenate((cordinates_patch, np.zeros((256 - patch_num, 64, 9), dtype=np.float32)), 0)
    faces_patcha = np.concatenate((faces_patch, np.zeros((256 - patch_num, 64, 3), dtype=int)), 0)
    Fs_patcha = np.array(Fs)

    return feats_patcha, center_patcha, cordinates_patcha, faces_patcha, Fs_patcha

@DATASETS.register_module()
class DentalDataset(data.Dataset):
    def __init__(self, config):
        super().__init__()
        self.feats = ['area', 'face_angles', 'curvs', 'normal']
        self.before_only = bool(getattr(config, 'before_only', False))
        self.dataroot = config.dataroot
        self.paramroot = None if self.before_only else config.paramroot
        self.before_path = config.before_path
        self.after_path = None if self.before_only else config.after_path
        self.npoint = config.npoint
        with open(config.file) as f:
            self.indexes = [int(i.strip()) for i in f.readlines()]
        self.train = config.train
        if self.before_only:
            if self.train:
                raise ValueError('before_only is for input reading/prediction, not supervised training')
            if getattr(config, 'after_path', None) is not None or getattr(config, 'paramroot', None) is not None:
                raise ValueError('before_only must not receive after_path or paramroot')
    
    def __getitem__(self, idx):
        if self.before_only:
            return self._getitem_before_only(idx)
        point_num = self.npoint
        feats = np.zeros((32,10,256,64))
        center = np.zeros((32,256,64,3))
        cordinates = np.zeros((32,256,64,9))
        faces = np.zeros((32,256,64,3))
        Fs = np.zeros(32)
        before_points = np.zeros((32,point_num,3))
        after_points = np.zeros((32,point_num,3))
        centroid = np.zeros((32,3))
        after_centroid = np.zeros((32,3))
        index = self.indexes[idx]
        masks = np.zeros((32),dtype=np.int32)
        ### get gt rotation and translation params
        matrix = np.load(os.path.join(self.paramroot,f'{index}.npy'))
        # gt_params = torch.cat([torch.from_numpy(after_centroid),rot6d],dim=-1)
        for i in range(32):
            obj_path = os.path.join(self.dataroot,f'{index}_{i}.obj')
            before_path = os.path.join(self.before_path,f'{index}_{i}.ply')
            after_path = os.path.join(self.after_path,f'{index}_{i}.ply')
            if os.path.exists(obj_path) and os.path.exists(before_path) and os.path.exists(after_path):
                mesh = trimesh.load_mesh(obj_path, process=False)
                masks[i] = 1
                before = read_pointcloud(before_path)
                before_points[i] = before[:point_num]
                centroid[i] = before[point_num]
                after = read_pointcloud(after_path)
                after_points[i] = after[:point_num]
                after_centroid[i] = after[point_num] 
                # if self.train and np.random.rand() < 0.3:
                #     mesh, rotation, translation = randomize_mesh_orientation(mesh)
                #     rot_matrix = rotation.as_matrix()
                #     before_points[i] = (before_points[i] - np.expand_dims(centroid[i],0)) @ rot_matrix + np.expand_dims(centroid[i] + translation,0)
                #     centroid[i] = centroid[i] + translation
                #     matrix[i] = rot_matrix.T @ matrix[i]

                feats[i], center[i], cordinates[i], faces[i], Fs[i]= load_mesh_shape(mesh, 
                                                                    request=self.feats)
        rot6d = matrix_to_rotation_6d(torch.from_numpy(matrix).float())
        gt_params = torch.cat([torch.from_numpy(after_centroid),rot6d],dim=-1)
        return   index,feats,center,cordinates,faces,Fs,before_points,after_points,centroid,after_centroid,gt_params,masks

    def _getitem_before_only(self, idx):
        """Keep official before tensor shapes, with no after/GT files or targets.

        Paired training/evaluation still uses the unchanged tuple above.
        This opt-in mode returns a dict for a dedicated input reader/predictor.
        """
        sample = dict(
            index=self.indexes[idx],
            feats=np.zeros((32,10,256,64)),
            center=np.zeros((32,256,64,3)),
            cordinates=np.zeros((32,256,64,9)),
            faces=np.zeros((32,256,64,3)),
            Fs=np.zeros(32),
            before_points=np.zeros((32,self.npoint,3)),
            centroid=np.zeros((32,3)),
            masks=np.zeros(32,dtype=np.int32),
        )
        index = sample['index']
        for slot in range(32):
            obj = Path(self.dataroot) / f'{index}_{slot}.obj'
            ply = Path(self.before_path) / f'{index}_{slot}.ply'
            if not obj.exists() and not ply.exists():
                continue
            if not obj.is_file() or not ply.is_file():
                raise FileNotFoundError(f'Incomplete before-only slot {slot}: {obj}, {ply}')
            mesh = trimesh.load_mesh(obj,process=False)
            points = read_pointcloud(str(ply))
            if points.shape != (self.npoint+1,3) or not np.isfinite(points).all():
                raise ValueError(f'{ply}: expected {self.npoint} finite points + one centroid')
            if not np.isfinite(mesh.vertices).all() or len(mesh.faces) != 16384:
                raise ValueError(f'{obj}: expected finite vertices and 16384 ordered MAPS faces')
            sample['before_points'][slot] = points[:self.npoint]
            sample['centroid'][slot] = points[self.npoint]
            values = load_mesh_shape(mesh,request=self.feats)
            for key,value in zip(['feats','center','cordinates','faces','Fs'],values):
                sample[key][slot] = value
            sample['masks'][slot] = 1
        return sample


    def __len__(self):
        return len(self.indexes)
