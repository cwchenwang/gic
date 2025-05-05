import numpy as np
import matplotlib.pyplot as plt
import os
import json

from PIL import Image
from io import BytesIO
import sys, pathlib, html
import torch

def camera_view_dir_y(elev, azim):
    """Unit vector for camera direction with Y as 'up'."""
    elev_rad = np.radians(elev)
    azim_rad = np.radians(azim)
    dx = np.sin(azim_rad) * np.cos(elev_rad)
    dy = np.sin(elev_rad)
    dz = np.cos(azim_rad) * np.cos(elev_rad)
    return np.array([dx, dy, dz])

def compute_depth(points, elev, azim):
    """Project points onto the camera's view direction (Y as 'up')."""
    view_dir = camera_view_dir_y(elev, azim)
    # depth = p · view_dir
    depth = points @ view_dir
    return depth

def save_pointcloud_video(points_pred, points_gt, save_path, drag_mask=None, fps=48, point_color='blue', vis_flag='pacnerf'):
    
    # Configure the figure
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_box_aspect([1, 1, 1])
    
    if 'objaverse' in vis_flag:
        x_max, y_max, z_max = 3, 3, 3
        x_min, y_min, z_min = -3, -3, -3
    elif 'pacnerf' in vis_flag:
        x_max, y_max, z_max = 0.5, 1.2, 0.5
        x_min, y_min, z_min = -0.5, 0.1, -0.5
    
    if 'shapenet' or 'objaverse' in vis_flag:
        elev, azim = 45, 225
        
    ax.view_init(elev=elev, azim=azim, vertical_axis='y')
        
    # Plot and save each frame
    cmap_1 = plt.colormaps.get_cmap('cool')
    cmap_2 = plt.colormaps.get_cmap('autumn')
    frames_pred = []
    frames_gt = []

    if drag_mask is not None and drag_mask.sum() == 0:
        drag_mask = None
    
    for label, points in [('pred', points_pred), ('gt', points_gt)]:
        
        for i in range(len(points)):
            
            frame_points = points[i].cpu().numpy() if isinstance(points[i], torch.Tensor) else points[i]
            if drag_mask is not None and not (drag_mask == True).all():
                drag_mask = (drag_mask == 1.0)
                drag_points = frame_points[drag_mask]
                frame_points = frame_points[~drag_mask]
                
            depth_frame_points = compute_depth(frame_points, elev=elev, azim=azim)
            depth_frame_points_normalized = (depth_frame_points - depth_frame_points.min()) / \
                (depth_frame_points.max() - depth_frame_points.min())
            color_frame_points = cmap_1(depth_frame_points_normalized)

            if drag_mask is not None and not (drag_mask == True).all():
                frame_points_drag = drag_points
                depth_frame_points_drag = compute_depth(frame_points_drag, elev=elev, azim=azim)
                depth_frame_points_drag_normalized = (depth_frame_points_drag - depth_frame_points_drag.min()) / \
                    (depth_frame_points_drag.max() - depth_frame_points_drag.min())
                color_frame_points_drag = cmap_2(np.ones_like(depth_frame_points_drag_normalized) * -10)
                all_points = np.concatenate([frame_points, frame_points_drag], axis=0)
                all_color = np.concatenate([color_frame_points, color_frame_points_drag], axis=0)
            else:
                all_points, all_color = frame_points, color_frame_points
                
            
            ax.clear()
            ax.scatter(all_points[:, 0], all_points[:, 1], all_points[:, 2], c=all_color, s=1, depthshade=False)
            
            ax.axis('off')  # Turn off the axes
            ax.grid(False)  # Hide the grid
            
            # Set equal aspect ratio
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
            ax.set_zlim(z_min, z_max)
            
            # Adjust margins for tight layout
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            
            # Save frame
            buf = BytesIO()
            plt.savefig(buf, bbox_inches='tight', pad_inches=0.0, dpi=300)
            buf.seek(0)
            
            if label == 'pred':
                frames_pred.append(Image.open(buf))
            else:
                frames_gt.append(Image.open(buf))
                
    plt.close()
    frames = []
    for i in range(len(frames_pred)):
        frame = np.concatenate([np.array(frames_pred[i]), np.array(frames_gt[i])], axis=1)
        frames.append(Image.fromarray(frame))
    frames[0].save(save_path, save_all=True, append_images=frames[1:], fps=fps, loop=0)