import math
import re
import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import open3d as o3d
import copy
import cv2  # <--- 1. 重新导入cv2库
import argparse

def matrix_x(angle):
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    return np.array([[1, 0, 0, 0],
                     [0, cos_angle, -sin_angle, 0],
                     [0, sin_angle, cos_angle, 0],
                     [0, 0, 0, 1]])

def adjust_view_to_fit_screen(vis, o3d_pc_r, screen_width, screen_height):
    out_screen = True
    max_iter = 100
    count = 0
    while out_screen and count < max_iter:
        out_screen = False
        count += 1
        
        view_control = vis.get_view_control()
        cam_params = view_control.convert_to_pinhole_camera_parameters()
        
        extrinsic = np.array(cam_params.extrinsic)
        view_point = np.array(extrinsic[0:3, 3])
        extrinsic[:3, 3] = view_point + np.array([0, 0, 0.005])
        cam_params.extrinsic = extrinsic
        
        view_control.convert_from_pinhole_camera_parameters(cam_params, allow_arbitrary=True)
        
        param = view_control.convert_to_pinhole_camera_parameters()
        intrinsic_matrix = param.intrinsic.intrinsic_matrix
        extrinsic_matrix = param.extrinsic

        r = cv2.Rodrigues(extrinsic_matrix[:3, :3])[0]
        t = extrinsic_matrix[:3, 3:]
        points_2d, _ = cv2.projectPoints(np.asarray(o3d_pc_r.points), r, t, intrinsic_matrix, None)
    
        for point in points_2d:
            x, y = point[0]
            if x < 0 or x >= screen_width or y < 0 or y >= screen_height:
                out_screen = True
                break
    if count >= max_iter:
        print("Warning: adjust_view_to_fit_screen reached max iterations.")


def render_and_save_point_cloud_with_depth(o3d_pc_r, view_idx, depth_save_path, point_size):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1036, height=1036, visible=False)
    vis.add_geometry(o3d_pc_r)
    
    adjust_view_to_fit_screen(vis, o3d_pc_r, 1036, 1036)

    render_option = vis.get_render_option()
    render_option.background_color = np.asarray([1, 1, 1])
    render_option.light_on = True
    render_option.point_size = point_size
    render_option.mesh_show_back_face = True
    
    depth = vis.capture_depth_float_buffer(do_render=True)
    depth = np.asarray(depth)
    vis.destroy_window()
    
    depth[depth == 1.0] = 0
    depth[depth == np.inf] = 0
    depth[np.isnan(depth)] = 0
    
    if np.max(depth) > 0:
        valid_mask = depth > 0
        min_depth = np.min(depth[valid_mask])
        max_depth = np.max(depth[valid_mask])
        if max_depth > min_depth:
            depth_normalized = np.ones_like(depth, dtype=np.float32)
            depth_normalized[valid_mask] = (depth[valid_mask] - min_depth) / (max_depth - min_depth)
            depth_normalized[valid_mask] = 1.0 - depth_normalized[valid_mask]
        else:
            depth_normalized = np.ones_like(depth, dtype=np.float32)
    else:
        depth_normalized = np.ones_like(depth, dtype=np.float32)
    
    depth_image = (depth_normalized * 255).astype(np.uint8)
    depth_image = Image.fromarray(depth_image, mode='L')
    post_transform = transforms.Resize((336, 336), interpolation=transforms.InterpolationMode.BICUBIC)
    depth_image = post_transform(depth_image)
    
    depth_image_path = f"{depth_save_path}/view_{view_idx}_depth.png"
    depth_image.save(depth_image_path)

def get_mv_depth_maps_eyecandy(pcd_path, point_size, depth_save_path, file_id):
    print(f"--- Processing: {pcd_path}")
    o3d_pc = o3d.io.read_point_cloud(pcd_path)
    if not o3d_pc.has_points():
        print(f"Warning: No points found in {pcd_path}. Skipping.")
        return
    o3d_pc.colors = o3d.utility.Vector3dVector(np.asarray(o3d_pc.points) * 0 + 0.7)
    o3d_pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=50))
    o3d_pc.transform(matrix_x(0.25 * np.pi))
    view_list_x = [-90, -60, -45, -30, -0]
    view_list_y = [-30, -15, 15, 30]
    view_idx = 0
    for v_x in view_list_x:
        o3d_pc_o = copy.deepcopy(o3d_pc)
        R = o3d_pc.get_rotation_matrix_from_axis_angle([np.radians(v_x), 0, 0])
        o3d_pc_r = o3d_pc_o.rotate(R, center=o3d_pc.get_center())
        render_and_save_point_cloud_with_depth(o3d_pc_r, view_idx, depth_save_path, point_size)
        view_idx += 1
    for v_y in view_list_y:
        o3d_pc_o = copy.deepcopy(o3d_pc)
        R_x = o3d_pc.get_rotation_matrix_from_axis_angle([np.radians(-45), 0, 0])
        o3d_pc_o.rotate(R_x, center=o3d_pc.get_center())
        R_y = o3d_pc.get_rotation_matrix_from_axis_angle([0, np.radians(v_y), 0])
        o3d_pc_o.rotate(R_y, center=o3d_pc.get_center())
        o3d_pc_r = o3d_pc_o
        render_and_save_point_cloud_with_depth(o3d_pc_r, view_idx, depth_save_path, point_size)
        view_idx += 1
    print(f"--- Finished. Generated {view_idx} depth maps in {depth_save_path}")

def process_directory(directory_path, point_size, data_root_name="Eyecandies"):
    for root, dirs, files in os.walk(directory_path):
        files.sort()
        for file in files:
            if file.endswith(".pcd"):
                pcd_path = os.path.join(root, file)
                file_id = os.path.splitext(file)[0]
                output_root = root.replace(data_root_name, f"{data_root_name}_rendered")
                if output_root.endswith('/pcd'):
                    output_root = output_root[:-4] + '/2d_depth'
                else:
                    output_root = output_root.replace(os.path.join(os.sep, 'pcd'), os.path.join(os.sep, '2d_depth'))
                depth_save_path = os.path.join(output_root, file_id)
                os.makedirs(depth_save_path, exist_ok=True)
                get_mv_depth_maps_eyecandy(pcd_path, point_size, depth_save_path, file_id)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Batch Generate Multi-View Depth Maps for Eyecandies from PCD", add_help=True)
    parser.add_argument("--data_path", type=str, default="./data/Eyecandies", help="path to Eyecandies dataset root")
    args = parser.parse_args()
    cls_list = ["Lollipop"]
    data_root_name = os.path.basename(os.path.normpath(args.data_path))
    if not os.path.exists(args.data_path):
        exit(1)
    for cls in cls_list:
        point_size = 10
        base_cls_path = os.path.join(args.data_path, cls)
        for subfolder in ["test"]:
            for condition in ["good", "anomaly"]:
                directory_to_process = os.path.join(base_cls_path, subfolder, condition, "pcd")
                if os.path.exists(directory_to_process):
                    print(f"Processing: {directory_to_process}")
                    process_directory(directory_to_process, point_size, data_root_name)
                else:
                    print(f"\nInfo: Directory not found, skipping: {directory_to_process}")