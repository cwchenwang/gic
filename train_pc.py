import torch
import taichi as ti
import torch.nn as nn
import time, os, json
from tqdm import tqdm, trange
from train_gs import training
from argparse import ArgumentParser, Namespace
from gaussian_renderer import render
from scene import Scene, DeformModel
from utils.general_utils import safe_state
from gaussian_renderer import GaussianModel
from simulator import MPMSimulator
from simulator import Estimator
from train_gs_fixed_pcd import train_gs_with_fixed_pcd, assign_gs_to_pcd
from utils.system_utils import check_gs_model, draw_curve, write_particles
from utils.vis_utils import save_pointcloud_video
from arguments import ModelParams, PipelineParams, OptimizationParams, get_combined_args
import h5py
import numpy as np

def forward(estimator: Estimator, img_backward=True, vol=None):
    dt = estimator.simulator.dt_ori[None]
    pos = []
    while True:
        for idx in range(estimator.max_f):
            if idx == 0:
                estimator.initialize(vol=vol)
                estimator.simulator.set_dt(dt)
            x = estimator.forward(idx, img_backward)
            pos.append(x.detach().cpu().numpy()) if isinstance(x, torch.Tensor) else pos.append(x)
        if not estimator.succeed():
            pos = []
            dt /= 2
            print('cfl condition dissatisfy, shrink dt {}, step cnt {}'.format(dt, estimator.simulator.n_substeps[None] * 2))
        else:
            break
    pos = np.array(pos)
    return pos
    # save_pointcloud_video(pos, pos, os.path.join('./output/debug', "sim_0.08_grad.gif"), fps=24, point_color='blue', vis_flag='objaverse')

def backward(estimator: Estimator):
    print('Geometry loss {}, image loss {}, step {}'.format(estimator.loss[None], estimator.image_loss, estimator.simulator.n_substeps[None]))
    max_f = estimator.max_f
    pbar = trange(max_f)
    pbar.set_description(f"[Backward]")
    
    estimator.loss.grad[None] = 1
    estimator.clear_grads()
    
    for ri in pbar:
        i = max_f - 1 - ri
        if i > 0:
            estimator.backward(i)
        else:
            pos_grad, velocity_grad, mu_grad, lam_grad, \
            yield_stress_grad, viscosity_grad, \
            friction_alpha_grad, cohesion_grad, rho_grad = estimator.backward(i)
            estimator.init_velocities.backward(retain_graph=True, gradient=velocity_grad)
            estimator.init_rhos.backward(retain_graph=True, gradient=rho_grad)
            estimator.init_pos.backward(retain_graph=True, gradient=pos_grad)
            estimator.init_mu.backward(retain_graph=True, gradient=mu_grad)
            estimator.init_lam.backward(retain_graph=True, gradient=lam_grad)
            estimator.yield_stress.backward(retain_graph=True, gradient=yield_stress_grad)
            estimator.plastic_viscosity.backward(retain_graph=True, gradient=viscosity_grad)
            estimator.friction_alpha.backward(retain_graph=True, gradient=friction_alpha_grad)
            estimator.cohesion.backward(gradient=cohesion_grad)

def train(estimator: Estimator, phys_args, max_f=None, vol=None):
    losses = []
    estimated_params = []
    if estimator.stage[None] == Estimator.velocity_stage:
        iter_cnt = phys_args.vel_iter_cnt
    elif estimator.stage[None] == Estimator.physical_params_stage:
        iter_cnt = phys_args.iter_cnt

    if max_f is not None:
        estimator.max_f = max_f
    
    for stage, train_param in enumerate(zip([max_f], [iter_cnt])):
        max_f, iter_cnt = train_param
        if max_f is not None:
            estimator.max_f = max_f
        for i in range(iter_cnt):
            # 1. record current params
            d = {}
            param_groups = estimator.get_optimizer().param_groups
            report_msg = ''
            report_msg += f'iter {i}'
            report_msg += f'\nvelocity: {estimator.init_vel.cpu().detach().tolist()}'
            for params in param_groups:
                name = params['name']
                p = params['params'][0].detach().cpu()
                if name == 'Poisson ratio':
                    p = estimator.get_nu().detach().cpu()
                    report_msg += f'\n{name}: {p}'
                elif name in ['Youngs modulus', 'Yield stress', 'plastic viscosity', 'shear modulus', 'bulk modulus']:
                    p = 10**p
                    report_msg += f'\n{name}: {p}'
                #TODO: optimization
                if name != 'velocity':
                    d.update({name: p.item()})
                else:
                    d.update({name: p})
            print(report_msg)
            estimated_params.append(d)

            # 2. forward, backward, and update
            estimator.zero_grad()
            estimator.loss[None] = 0.0
            pos = forward(estimator, vol=vol)
            if i % 10 == 0:
                save_pointcloud_video(pos, estimator.gts.detach().cpu().numpy(), os.path.join('./output/debug', f"{i:03d}_{estimator.dt}.gif"), fps=24, point_color='blue', vis_flag='objaverse')
            losses.append(estimator.loss[None] + estimator.image_loss)
            backward(estimator)
            estimator.step(i)
            
            # 3. record loss and save best params
            min_idx = losses.index(min(losses))
            best_params = estimated_params[min_idx]
            print("Best params: ", best_params, 'in {} iteration'.format(min_idx))
            print("Min loss: {}".format(losses[min_idx]))
    
    if estimator.stage[None] == Estimator.velocity_stage and len(losses) > 0:
        min_idx = losses.index(min(losses))
        best_params = estimated_params[min_idx]
        estimator.init_vel = nn.Parameter(best_params['velocity'].to(estimator.device))

    return losses, estimated_params

if __name__ == "__main__":
    start_time = time.time()

    parser = ArgumentParser(description="Physical parameter estimation")
    model = ModelParams(parser)#, sentinel=True)
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--config_file", default='config/torus.json', type=str)
    gs_args, phys_args = get_combined_args(parser)
    config_id = phys_args.id
    print(phys_args)
    safe_state(gs_args.quiet)

    ti.init(arch=ti.cuda, debug=False, fast_math=False, device_memory_fraction=0.5)

    obj_id = '34002_002'
    model_metas = h5py.File(os.path.join(f'/mnt/kostas-graid/datasets/chenwang/traj/ObjaverseXL_sketchfab/raw/hf-objaverse-v1/outputs_v6/34002_002.h5'), 'r')
    # model_pcls = torch.from_numpy(np.array(model_metas['x'])) - 5
    # model_pcls[:, :, 1] = model_pcls[:, :, 1] - (np.array(model_metas['floor_height']).item() - 5)
    # vol = model_pcls[0].to(device='cuda', dtype=torch.float32).contiguous()
    # gts = model_pcls[:-1]
    voxel_size = 0.08
    path = f'./output/{obj_id}-ourvol-200.npz'
    data = np.load(f'{path}.npz')
    print(f'loading {path}')
    vol = torch.from_numpy(data['pos'][0]).to(device='cuda', dtype=torch.float32).contiguous()
    gts = torch.from_numpy(data['pos']).to(device='cuda', dtype=torch.float32).contiguous()
    # save_pointcloud_video(gts[:48], gts1, os.path.join('./output/debug', "sim_0.08_comp.gif"), fps=24, point_color='blue', vis_flag='objaverse')

    estimation_params = phys_args
    estimation_params = vars(estimation_params)
    # estimation_params['init_E'] = np.log10(np.array(model_metas['E'])).item()
    # estimation_params['init_nu'] = np.array(model_metas['nu']).item()
    estimation_params['voxel_size'] = voxel_size
    # estimation_params['bc']['ground'][0][1] = (np.array(model_metas['floor_height']).item() - 5)
    phys_args = Namespace(**estimation_params)
    print(phys_args)

    estimator = Estimator(phys_args, 'float32', gts, surface_index=None, init_vol=vol, dynamic_scene=None, pipeline=pipeline.extract(gs_args), image_op=op.extract(gs_args))
    estimator.set_stage(Estimator.physical_params_stage)
    # max_f = len(gts) - 1
    max_f = len(gts)
    losses, e_s = train(estimator, phys_args, max_f, vol=torch.from_numpy(np.array(model_metas['vol'])).to(device='cuda', dtype=torch.float32).contiguous())
    # losses, e_s = train(estimator, phys_args, max_f, vol=None)
