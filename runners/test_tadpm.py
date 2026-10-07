import torch
import torch.nn as nn
from utils import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter
from utils.pointcloud import write_pointcloud
import numpy as np
from einops import rearrange
from utils.rotation6d import rotation_6d_to_matrix
import os
import trimesh
from utils.smoke import configure_smoke_dataset, SmokeRecorder

def transform_mesh(index, outputs):
    """
    You can use this function to visualize the transformation results of the meshes.
    """
    before_meshes = []
    after_meshes = []
    for j in range(32):
        before_path = os.path.join('*/single_before',f'{index}_{j}.obj')
        after_path = os.path.join('*/single_after',f'{index}_{j}.obj')
        if os.path.exists(before_path) and os.path.exists(after_path):
            before_mesh = trimesh.load_mesh(before_path)
            after_mesh = trimesh.load_mesh(before_path)
            predicted_centroid = outputs[j,:3]
            rot6d = outputs[j,3:]
            rot_matrix = rotation_6d_to_matrix(rot6d)
            after_vertices = (torch.matmul(torch.from_numpy(after_mesh.vertices - after_mesh.centroid).float().cuda(), rot_matrix.transpose(0,1)) + predicted_centroid.unsqueeze(0))
            after_mesh.vertices = after_vertices.cpu().numpy()
            before_meshes.append(before_mesh)
            after_meshes.append(after_mesh)
    before_meshes = trimesh.util.concatenate(before_meshes)
    after_meshes = trimesh.util.concatenate(after_meshes)


def test_tadpm(args, config, logger):
    smoke = not args.ckpts
    if smoke:
        configure_smoke_dataset(args, config)
        logger.warning('No TADPM checkpoint: random-weight forward smoke test; no clinical prediction or ADD evaluation.')
    device = torch.device(getattr(args, 'device', 'cuda'))
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in this interpreter. Use your CUDA-enabled Python; --device cpu is only for CPU compatibility checks.')
    # build dataset
    config.model.args = args
    (_, test_dataloader)= builder.dataset_builder(args, config.dataset.test)
    # build model
    base_model = builder.model_builder(config.model)
    

    # resume ckpts
    # base_model.load_model_from_ckpt(args.ckpts)
    if args.ckpts:
        builder.load_model_from_ckpt(base_model,args.ckpts)
 
    logger.info('Using Data parallel ...' , logger = logger)
    base_model = base_model.to(device)
    recorder = SmokeRecorder(base_model, args) if smoke else None
    if device.type == 'cuda':
        base_model = nn.DataParallel(base_model)
    # optimizer & scheduler
    logger.info(f"[TEST] Start test")
    base_model.eval()  # set model to eval mode
    losses = AverageMeter(['loss'])
    with torch.no_grad():
        for idx, (index,feats,center,cordinates,faces,Fs,before_points,after_points,centroid,after_centroid,gt_params,masks) in enumerate(test_dataloader):
            faces = faces.to(device=device, dtype=torch.float32)
            feats = feats.to(device=device, dtype=torch.float32)
            center = center.to(device=device, dtype=torch.float32)
            gt_params = gt_params.to(device=device, dtype=torch.float32)
            Fs = Fs.to(device)
            cordinates = cordinates.to(device)
            centroid = centroid.to(device=device, dtype=torch.float32)
            after_centroid = after_centroid.to(device=device, dtype=torch.float32)
            before_points = before_points.to(device=device, dtype=torch.float32)
            after_points = after_points.to(device=device, dtype=torch.float32)
            masks = masks.to(device)
            if smoke and not masks.any():
                raise RuntimeError('No valid teeth loaded. Check case_0 OBJ/before/after paths.')
            outputs = base_model(faces, feats, center, Fs, cordinates, centroid, before_points, gt_params).to(torch.float32)
            if smoke:
                recorder.save(index, masks, outputs)
                continue

            predicted_centroid = outputs[:,:,:3]
            rot6d = rearrange(outputs[:,:,3:],'b n c -> (b n) c')
            criterion = nn.MSELoss(reduction='none')
            rot_matrix = rotation_6d_to_matrix(rot6d)
            predicted_points = rearrange(before_points - centroid.unsqueeze(2),'b n p c -> (b n) p c')
            predicted_points = torch.bmm(predicted_points,rot_matrix)
            predicted_points = predicted_points + rearrange(predicted_centroid,'b n c->(b n) c').unsqueeze(1)
            after_points = rearrange(after_points,'b n p c -> (b n) p c')
            loss = criterion(predicted_points,after_points).sum(dim=-1).sqrt()
            loss = 40*(loss * masks.flatten().unsqueeze(1)).mean()
            losses.update([loss.item()])
    if smoke:
        recorder.finish()
    else:
        print(f"Calculated ADD metric is: {losses.avg()[0]:.4f}")

