import os
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import open3d as o3d
import copy
from sklearn.neighbors import NearestNeighbors
import argparse

def matrix_x(angle):
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    return np.array([[1, 0, 0, 0], [0, cos_angle, -sin_angle, 0], [0, sin_angle, cos_angle, 0], [0, 0, 0, 1]])

def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points

def farthest_point_sample(xyz, npoint):
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def interpolation_points(points, target_size):
    num_points = len(points)
    num_new_points = target_size - num_points
    if num_new_points <= 0:
        return points
    neigh = NearestNeighbors(n_neighbors=4, algorithm='auto').fit(points)
    distances, indices = neigh.kneighbors(points)
    new_points = []
    points_to_generate_per_neighbor_pair = max(1, num_new_points // (num_points * 3) + 1)
    for idx in range(num_points):
        point = points[idx]
        for neighbor_idx in indices[idx, 1:]:
            neighbor_point = points[neighbor_idx]
            for ratio in np.linspace(0.2, 0.8, num=points_to_generate_per_neighbor_pair):
                new_point = point * (1 - ratio) + neighbor_point * ratio
                new_points.append(new_point)
    new_points = np.array(new_points)
    if len(new_points) > num_new_points:
        selected_indices = np.random.choice(len(new_points), size=num_new_points, replace=False)
        new_points = new_points[selected_indices]
    return np.vstack((points, new_points))

def render_and_save_point_cloud_with_depth(o3d_pc_r, view_idx, depth_save_path, point_size):
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1024, height=1024, visible=False)
    vis.add_geometry(o3d_pc_r)
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

def get_mv_depth_maps_real3d(pcd_path, txt_path, point_size, depth_save_path, file_id):
    print(f"--- Processing: {pcd_path}")
    os.makedirs(depth_save_path, exist_ok=True)
    target_size = 336 * 336
    if os.path.exists(txt_path):
        txt_data = np.genfromtxt(txt_path, delimiter=',')
        points_o = txt_data[:, :3]
    else:
        o3d_pc_o = o3d.io.read_point_cloud(pcd_path)
        points_o = np.asarray(o3d_pc_o.points)
    if len(points_o) == target_size:
        print(f"    Points ({len(points_o)}) == target ({target_size}). Skipping sampling.")
        points = points_o
    elif len(points_o) > target_size:
        print(f"    Points ({len(points_o)}) >= target ({target_size}). Downsampling...")
        points_tensor = torch.from_numpy(points_o).float().unsqueeze(0)
        if torch.cuda.is_available():
            points_tensor = points_tensor.to('cuda')
        points_idx = farthest_point_sample(points_tensor, target_size)
        sampled_points_tensor = index_points(points_tensor, points_idx)
        points = sampled_points_tensor.squeeze(0).cpu().numpy()
    else:
        print(f"    Points ({len(points_o)}) < target ({target_size}). Upsampling...")
        points = interpolation_points(points_o, target_size)
    o3d_pc = o3d.geometry.PointCloud()
    o3d_pc.points = o3d.utility.Vector3dVector(points)
    o3d_pc.colors = o3d.utility.Vector3dVector(np.asarray(o3d_pc.points) * 0 + 0.7)
    o3d_pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.8, max_nn=30))
    o3d_pc.transform(matrix_x(np.pi))
    
    view_list_x = [-45.0,-15.0, 0, 15.0, 45.0]       # 9 views
    view_list_y = [-30 ,-15, 15, 30]
    view_idx = 0

    for v_x in view_list_x:
        o3d_pc_o = copy.deepcopy(o3d_pc)

        R_x = o3d_pc.get_rotation_matrix_from_axis_angle([np.radians(180), 0, 0])
        o3d_pc_o.rotate(R_x, center=o3d_pc.get_center())

        R = o3d_pc.get_rotation_matrix_from_axis_angle([np.radians(v_x), 0, 0])
        o3d_pc_r = o3d_pc_o.rotate(R, center=o3d_pc.get_center())
        render_and_save_point_cloud_with_depth(o3d_pc_r, view_idx, depth_save_path, point_size)
        view_idx += 1
        
    for v_y in view_list_y:
        o3d_pc_o = copy.deepcopy(o3d_pc)
        
        R_x = o3d_pc.get_rotation_matrix_from_axis_angle([np.radians(180), 0, 0])
        o3d_pc_o.rotate(R_x, center=o3d_pc.get_center())
        
        R_y = o3d_pc.get_rotation_matrix_from_axis_angle([0, np.radians(v_y), 0])
        o3d_pc_o.rotate(R_y, center=o3d_pc.get_center()) # 再次修改 o3d_pc_o
        
        o3d_pc_r = o3d_pc_o
        
        render_and_save_point_cloud_with_depth(o3d_pc_r, view_idx, depth_save_path, point_size)
        view_idx += 1
    print(f"--- Finished. Generated {view_idx} depth maps in {depth_save_path}")


def process_directory(directory_path, point_size, data_root_path, cls):
    for root, dirs, files in os.walk(directory_path):
        files.sort()
        for file in files:
            if file.endswith(".pcd"):
                pcd_path = os.path.join(root, file)
                file_id = os.path.splitext(file)[0]
                
                txt_path = os.path.join(data_root_path, cls, "GT", f"{file_id}.txt")

                data_root_name = os.path.basename(os.path.normpath(data_root_path))
                output_root = root.replace(data_root_name, f"{data_root_name}_rendered")
                
                if output_root.endswith('/pcd'):
                    output_root = output_root[:-4] + '/2d_depth'
                else:
                    output_root = output_root.replace(os.path.join(os.sep, 'pcd'), os.path.join(os.sep, '2d_depth'))
                
                depth_save_path = os.path.join(output_root, file_id)
                os.makedirs(depth_save_path, exist_ok=True)
                
                get_mv_depth_maps_real3d(pcd_path, txt_path, point_size, depth_save_path, file_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Batch Generate Multi-View Depth Maps for Real3D-AD", add_help=True)
    parser.add_argument("--data_path", type=str, default="./data/Anomaly-ShapeNet-rendered", help="path to Real3D-AD dataset root")
    args = parser.parse_args()

    cls_list = ['ashtray0', 'bag0', 'bottle0', 'bottle1', 'bottle3', 'bowl0', 'bowl1', 'bowl2', 'bowl3', 'bowl4',
                    'bowl5', 'bucket0', 'bucket1', 'cap0', 'cap3', 'cap4', 'cap5', 'cup0', 'cup1', 'eraser0', 'headset0',
                  'headset1', 'helmet0', 'helmet1', 'helmet2', 'helmet3', 'jar0', 'microphone0', 'shelf0', 'tap0', 'tap1',
                   'vase0', 'vase1', 'vase2', 'vase3', 'vase4', 'vase5', 'vase7', 'vase8', 'vase9']
    
    if not os.path.exists(args.data_path):
        print(f"错误: 数据路径 '{args.data_path}' 不存在!")
        exit(1)

    for cls in cls_list:
        point_size = 12
        
        base_cls_path = os.path.join(args.data_path, cls)
        
        for subfolder in ["test"]:
            for condition in ["good", "anomaly"]:
                directory_to_process = os.path.join(base_cls_path, subfolder, condition, "pcd")
                
                if os.path.exists(directory_to_process):
                    print(f"Processing: {directory_to_process}")
                    process_directory(directory_to_process, point_size, args.data_path, cls)
                else:
                    print(f"\nInfo: Directory not found, skipping: {directory_to_process}")
