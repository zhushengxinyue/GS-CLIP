import numpy as np
import tifffile as tiff
from PIL import Image
import torch
from torchvision import transforms
import open3d as o3d
import copy
import os
import argparse

def read_tiff_organized_pc(path):
    tiff_img = tiff.imread(path)
    return tiff_img

def resize_organized_pc(organized_pc, target_height=336, target_width=336, tensor_out=False, mode="nearest"):
    torch_organized_pc = torch.tensor(organized_pc).permute(2, 0, 1).unsqueeze(dim=0).float()
    torch_resized_organized_pc = torch.nn.functional.interpolate(torch_organized_pc, size=(target_height, target_width), mode=mode)
    if tensor_out:
        return torch_resized_organized_pc.squeeze(dim=0)
    else:
        return torch_resized_organized_pc.squeeze(dim=0).permute(1, 2, 0).numpy()

def organized_pc_to_unorganized_pc(organized_pc):
    return organized_pc.reshape(organized_pc.shape[0] * organized_pc.shape[1], organized_pc.shape[2])

def matrix_x(angle):
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    return np.array([[1, 0, 0, 0],
                     [0, cos_angle, -sin_angle, 0],
                     [0, sin_angle, cos_angle, 0],
                     [0, 0, 0, 1]])

def render_and_save_point_cloud_with_depth(o3d_pc_r, view_idx, depth_save_path, point_size):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1024, height=1024, visible=False)
    vis.add_geometry(o3d_pc_r)
    
    render_option = vis.get_render_option()
    render_option.background_color = np.asarray([1, 1, 1]) 
    render_option.light_on = True
    render_option.point_size = point_size
    render_option.mesh_show_back_face = True
    
    control = vis.get_view_control()
    param = control.convert_to_pinhole_camera_parameters()
    
    depth = vis.capture_depth_float_buffer(do_render=True)
    depth = np.asarray(depth)
    
    depth[depth == np.inf] = 0
    depth[np.isnan(depth)] = 0
    
    if np.max(depth) > 0:
        valid_mask = depth > 0
        if np.sum(valid_mask) > 0:
            min_depth = np.min(depth[valid_mask])
            max_depth = np.max(depth[valid_mask])
            if max_depth > min_depth:
                depth_normalized = np.ones_like(depth)  
                depth_normalized[valid_mask] = (depth[valid_mask] - min_depth) / (max_depth - min_depth)
                depth_normalized[valid_mask] = 1.0 - depth_normalized[valid_mask]
            else:
                depth_normalized = np.ones_like(depth)
        else:
            depth_normalized = np.ones_like(depth)
    else:
        depth_normalized = np.ones_like(depth)
    
    depth_image = (depth_normalized * 255).astype(np.uint8)
    depth_image = Image.fromarray(depth_image, mode='L')
    post_transform = transforms.Resize((336, 336), interpolation=transforms.InterpolationMode.BICUBIC)
    depth_image = post_transform(depth_image)
    
    depth_image_path = f"{depth_save_path}/view_{view_idx}_depth.png"
    depth_image.save(depth_image_path)
    
    vis.destroy_window()

def get_mv_depth_maps(tiff_path, point_size, depth_save_path):
    print(f"--- Processing: {tiff_path}")
    os.makedirs(depth_save_path, exist_ok=True)
    
    # 数据处理流程
    organized_pc = read_tiff_organized_pc(tiff_path)
    organized_pc = resize_organized_pc(organized_pc, 336, 336)
    unorganized_pc = organized_pc_to_unorganized_pc(organized_pc)
    
    nonzero_indices = np.nonzero(unorganized_pc[:, 2])[0]
    unorganized_pc_no_zeros = unorganized_pc[nonzero_indices, :]

    if len(unorganized_pc_no_zeros) == 0:
        print(f"Warning: No valid points found in {tiff_path}. Skipping.")
        return

    # 创建点云对象
    o3d_pc = o3d.geometry.PointCloud()
    o3d_pc.points = o3d.utility.Vector3dVector(unorganized_pc_no_zeros)
    o3d_pc.colors = o3d.utility.Vector3dVector(np.asarray(o3d_pc.points) * 0 + 0.7)
    o3d_pc.transform(matrix_x(np.pi))
    o3d_pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=50))
    
    # 视角设置 (与您提供的代码一致)
    # view_list_x = [-45.0,-15.0, 0, 15.0, 45.0]       # 9
    # view_list_y = [-30 ,-15, 15, 30]   
    view_list_x = [-144, -108, -72.0, -36.0, 0, 36.0, 72.0, 108.0, 144]
    view_list_y = []

    view_idx = 0
    # X轴旋转
    for v_x in view_list_x:
        o3d_pc_o = copy.deepcopy(o3d_pc)
        R = o3d_pc.get_rotation_matrix_from_axis_angle([np.radians(v_x), np.radians(0), np.radians(0)])
        o3d_pc_r = o3d_pc_o.rotate(R, center=o3d_pc.get_center())
        render_and_save_point_cloud_with_depth(o3d_pc_r, view_idx, depth_save_path, point_size)
        view_idx += 1
    
    # Y轴旋转
    for v_y in view_list_y:
        o3d_pc_o = copy.deepcopy(o3d_pc)
        R = o3d_pc.get_rotation_matrix_from_axis_angle([np.radians(0), np.radians(v_y), np.radians(0)])
        o3d_pc_r = o3d_pc_o.rotate(R, center=o3d_pc.get_center())
        render_and_save_point_cloud_with_depth(o3d_pc_r, view_idx, depth_save_path, point_size)
        view_idx += 1
    
    print(f"--- Finished. Generated {view_idx} depth maps in {depth_save_path}")

def process_directory(directory_path, point_size):
    for root, dirs, files in os.walk(directory_path):
        dirs.sort()
        files.sort()
        for file in files:
            if file.endswith(".tiff"):
                tiff_path = os.path.join(root, file)
                file_id = os.path.splitext(file)[0]
                
                # 定义输出路径，将 "xyz" 替换为 "depth_maps"
                # 您可以根据需要修改 "depth_maps"
                depth_save_path = os.path.join(root.replace("xyz", "2d_depth"), file_id)
                
                # 调用主逻辑函数处理单个文件
                get_mv_depth_maps(tiff_path, point_size, depth_save_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser("Batch Generate Multi-View Depth Maps", add_help=True)
    parser.add_argument("--data_path", type=str, default="./data/mvtec_3d", help="path to test dataset")
    args = parser.parse_args()
    cls_list = ['cable_gland', 'peach', 'foam', 'potato', 'tire']
    point_sizes = [12, 7, 8, 8, 8]
    
    if not os.path.exists(args.data_path):
        print(f"Error: Data path '{args.data_path}' does not exist!")
        exit(1)
    
    for cls, point_size in zip(cls_list, point_sizes):
        base_directory = f"{args.data_path}/{cls}"
        for folder in ["test"]:
            directory_path = os.path.join(base_directory, folder)
            if os.path.exists(directory_path):
                print(f"Processing class: '{cls}', folder: '{folder}'")
                process_directory(directory_path, point_size)
            else:
                print(f"\nWarning: Directory not found, skipping: {directory_path}")
