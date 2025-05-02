import taichi as ti
from utils.general_utils import safe_state
from train_gs_fixed_pcd import train_gs_with_fixed_pcd
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args, OptimizationParams
from simulator import Simulator
import time
import torch
import os
import subprocess
from scene import GaussianModel
from scene.cameras import Camera
from gaussian_renderer import render
from utils.reg_utils import mini_batch_knn
from argparse import ArgumentParser, Namespace
import numpy as np
import trimesh as tm
import torchvision
import json
from utils.system_utils import mkdir_p, write_particles
from tqdm import tqdm
from utils.vis_utils import save_pointcloud_video
from new_trajectory import load_pcd_file

def read_estimation_result(dataset: ModelParams, phys_args):
    files = os.listdir(dataset.model_path)
    pred_json = [f for f in files if "-pred.json" in f]
    result_json = None
    result = None
    if len(pred_json) == 0:
        print('Cannot find estimation result')

    with open(os.path.join(dataset.model_path, "debug.json".format(str(phys_args.config_id))), 'r') as f:
        print(f'Load file: {result_json}')
        result = json.load(f)
    return result

if __name__ == "__main__":
    start_time = time.time()

    parser = ArgumentParser(description="Generate new trajectory")
    parser.add_argument('-vid', '--view_id', type=int, default=0)
    parser.add_argument('-knn', '--use_knn', type=bool, default=False)
    parser.add_argument('-cid', '--config_id', type=int, default=0)
    model = ModelParams(parser)#, sentinel=True)
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    gs_args, phys_args = get_combined_args(parser)
    phys_args.view_id = gs_args.view_id
    # setattr(phys_args, "use_knn", gs_args.use_knn)
    # setattr(phys_args, "config_id", gs_args.config_id)
    print(phys_args)
    safe_state(gs_args.quiet)
    dataset = model.extract(gs_args)
    
    ti.init(arch=ti.cuda, debug=False, fast_math=False, device_memory_fraction=0.4)

    import h5py
    # 0. Load trained pcd
    # vol = load_pcd_file(dataset.model_path, gs_args.iteration)

    model_metas = h5py.File(os.path.join('/mnt/kostas-graid/datasets/chenwang/traj/ObjaverseXL_sketchfab/raw/hf-objaverse-v1/outputs_v6', "118731_001.h5"), 'r')
    model_pcls = torch.from_numpy(np.array(model_metas['x']))
    vol = ((model_pcls[0] - 5)).to(device='cuda', dtype=torch.float32).contiguous()
    # estimation_params = Namespace(**read_estimation_result(dataset, phys_args))

    estimation_params = phys_args
    estimation_params = vars(estimation_params)
    estimation_params['mat_params']['E'] = np.array(model_metas['E']).item()
    estimation_params['mat_params']['nu'] = np.array(model_metas['nu']).item()
    estimation_params['voxel_size'] = 0.08
    estimation_params['bc']['ground'][0][1] = (np.array(model_metas['floor_height']).item() - 5)
    phys_args = Namespace(**estimation_params)
    print(phys_args)
    
    simulator = Simulator(phys_args, vol)
    max_f = phys_args.predict_frames
    pos = [vol.clone().cpu().numpy()]
    # idx = np.random.choice(116279, 2048, replace=False)
    with torch.no_grad():
        simulator.initialize(vol=torch.from_numpy(np.array(model_metas['vol'])).to(device='cuda', dtype=torch.float32).contiguous())
        # simulator.initialize()
        for f in tqdm(range(max_f)):
            xyz = simulator.forward(f)
            # print(xyz)
            # xyz_vis = xyz[idx, :].cpu().numpy()
            pos.append(xyz.clone().cpu().numpy())

    pos = np.array(pos)
    save_pointcloud_video(pos, ((model_pcls-5)).cpu().numpy(), os.path.join('./output/debug', "video.gif"), fps=24, point_color='blue', vis_flag='objaverse')