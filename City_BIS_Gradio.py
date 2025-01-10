import argparse
import os
import sys
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d
import gradio as gr
import plotly.graph_objects as go
import json
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import LocalOutlierFactor
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

def parse_args(args):

    parser = argparse.ArgumentParser(description="City-BIS")
    # For pipeline 
    parser.add_argument("--RGB_vis_path", default="", type=str)
    parser.add_argument("--semantic_results_path", default="", type=str)

    return parser.parse_args(args)

def classify_points_by_grid(points, step=0.3, threshold=10):
    xyz = points[:, :3]
    categories_pred = points[:, 3]
    categories_gt = points[:, 3]

    xy = xyz[:, :2]

    grid_x = np.floor(xy[:, 0] / step).astype(int)
    grid_y = np.floor(xy[:, 1] / step).astype(int)

    grid_coords = np.vstack((grid_x, grid_y)).T

    unique_grids, counts = np.unique(grid_coords, axis=0, return_counts=True)

    grid_count_dict = {tuple(grid): count for grid, count in zip(unique_grids, counts)}

    points_with_classification = np.zeros((points.shape[0], 5))
    points_with_classification[:, :4] = points

    for i in range(points.shape[0]):
        grid = (grid_x[i], grid_y[i])
        count = grid_count_dict.get(grid, 0)
        if count > threshold:
            points_with_classification[i, 4] = 1
        else:
            points_with_classification[i, 4] = 2

    return points_with_classification

def compute_k_nearest_neighbors_average_distance(points, k=10):
    xyz = points[:, :3]
    
    nbrs = NearestNeighbors(n_neighbors=k+1)
    nbrs.fit(xyz)
    distances, indices = nbrs.kneighbors(xyz)
    avg_distances = np.mean(distances[:, 1:], axis=1)
    
    return avg_distances

def detect_lof_noise(data, n_neighbors=20, contamination=0.1):
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
    labels = lof.fit_predict(data)

    lof.fit(data)
    lof_values = lof.negative_outlier_factor_

    return lof_values, labels

def clean_and_merge_point_clouds(wall_cloud, roof_cloud):
    wall_cloud_cleaned = wall_cloud[wall_cloud[:, 5] != -1]
    roof_cloud_cleaned = roof_cloud[roof_cloud[:, 5] != -1]

    max_wall_instance_id = np.max(wall_cloud_cleaned[:, 5])

    roof_cloud_cleaned[:, 5] += (max_wall_instance_id + 1)

    merged_cloud = np.vstack([wall_cloud_cleaned, roof_cloud_cleaned])

    return merged_cloud

def extract_instances(point_cloud):
    instances = {}
    for i in range(point_cloud.shape[0]):
        instance_id = point_cloud[i, 5]
        if instance_id not in instances:
            instances[instance_id] = []
        instances[instance_id].append(point_cloud[i, :5])
    return {key: np.array(value) for key, value in instances.items()}

def compute_min_distances(instances):
    instance_ids = list(instances.keys())
    n_instances = len(instance_ids)
    min_distances = np.full((n_instances, n_instances), np.inf)

    for i in range(n_instances):
        for j in range(i + 1, n_instances):
            instance_i_points = instances[instance_ids[i]]
            instance_j_points = instances[instance_ids[j]]

            dist = cdist(instance_i_points[:, :3], instance_j_points[:, :3], 'euclidean')
            min_dist = np.min(dist)

            min_distances[i, j] = min_distances[j, i] = min_dist

    return min_distances, instance_ids

def merge_instances(instances, min_distances, instance_ids, threshold):
    merged_instances = {i: instances[instance_ids[i]] for i in range(len(instance_ids))}
    to_merge = set()

    for i in range(len(instance_ids)):
        for j in range(i + 1, len(instance_ids)):
            if min_distances[i, j] < threshold:
                merged_instances[i] = np.vstack([merged_instances[i], instances[instance_ids[j]]])
                to_merge.add(j)

    merged_instances = {i: merged_instances[i] for i in merged_instances if i not in to_merge}

    return merged_instances

def save_to_txt(merged_instances, filename):
    with open(filename, 'w') as f:
        i = 0
        for instance_id, points in merged_instances.items():

            for point in points:
                f.write(' '.join(map(str, point)) + ' ' + str(i) + '\n')

            i = i + 1
    print(f"Save in {filename}")

def create_bbox_line(bbox):
    x, y, z = bbox[:3]
    w, l, h = bbox[3:6]
    
    corners = np.array([
        [x-w/2, y-l/2, z-h/2],
        [x+w/2, y-l/2, z-h/2],
        [x+w/2, y+l/2, z-h/2],
        [x-w/2, y+l/2, z-h/2],
        [x-w/2, y-l/2, z+h/2],
        [x+w/2, y-l/2, z+h/2],
        [x+w/2, y+l/2, z+h/2],
        [x-w/2, y+l/2, z+h/2]
    ])
    
    lines = []
    for i, j in [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]:
        lines.append(go.Scatter3d(x=corners[[i,j],0], y=corners[[i,j],1], z=corners[[i,j],2],
              mode='lines', line=dict(color='red', width=2),showlegend=False))
    return lines

def pc_norm(pc):
    """ pc: NxC, return NxC """
    xyz = pc[:, :3]
    other_feature = pc[:, 3:]

    centroid = np.mean(xyz, axis=0)
    xyz = xyz - centroid
    m = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
    xyz = xyz / m

    pc = np.concatenate((xyz, other_feature), axis=1)
    return pc

def downsample_point_cloud(points, voxel_size = 2):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[:,:3])
    pcd.colors = o3d.utility.Vector3dVector(points[:,3:])
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
    downsampled_points = np.asarray(downsampled_pcd.points)
    downsampled_colors = np.asarray(downsampled_pcd.colors)
    return downsampled_points,downsampled_colors


def pc_vis(file_input, Tar_object_id = None):

    try:
        mesh_vertices = np.loadtxt(file_input.name)
    except ValueError:
        return None, f"Error: File '{file_input.name}' has invalid format. Ensure it contains only numeric data."
    except Exception as e:
        return None, f"Error: Unable to read the file '{file_input.name}'. {e}"
  
    coords = mesh_vertices[:, 0:3]
    colors = mesh_vertices[:, 3:6]

    colors = colors.astype(np.float32) / 255
    mesh_vertices = np.concatenate([coords, colors], axis=1) # total point cloud for this scene
    points, colors = downsample_point_cloud(mesh_vertices, 0.025)
    color_strings = ['rgb({},{},{})'.format(r, g, b) for r, g, b in colors]

    fig = go.Figure(
            data=[
                go.Scatter3d(
                    x = points[:, 0], y = points[:, 1], z = points[:, 2],
                    mode='markers',
                    marker=dict(
                        size=1,
                        color=color_strings,  # Use the list of RGB strings for the marker colors
                    )
                )
            ],
            layout=dict(
                scene=dict(
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    zaxis=dict(visible=False),
                    aspectratio=dict(x=1, y=1, z=0.05)
                ),
                paper_bgcolor='rgb(255,255,255)'  # Set the background color to dark gray 50, 50, 50
            ),
          )
        
    return fig

def SI_DVDC(file_input_semantic, Tar_object_id = None):

    Building_Semantic_Load_Path = file_input_semantic.name

    print(Building_Semantic_Load_Path)

    # Building_Semantic_Load_Path = "/content/drive/MyDrive/City-BIS/semantic_results/Building_Semantic.txt"
    
    ### SI-DVDC
    data = np.loadtxt(Building_Semantic_Load_Path)

    rows_with_1 = data[data[:, 4] != 1]
    Ground_rows_with_1 = data[data[:, 4] != 0]
    rows_without_1 = data[data[:, 3] != 1]

    Building_points_GT = np.zeros((rows_with_1.shape[0], 4))
    Building_points_Pred = np.zeros((rows_without_1.shape[0], 4))

    Ground_points_GT = np.zeros((Ground_rows_with_1.shape[0], 6))
    Ground_points_GT[:, :3] = Ground_rows_with_1[:, :3]
    Ground_points_GT[:, 3:6] = 0.7

    Building_points_GT[:, :3] = rows_with_1[:, :3]
    Building_points_Pred[:, :3] = rows_without_1[:, :3]

    Building_points_GT[:, 3] = 1
    Building_points_Pred[:, 3] = 1

    building_points = classify_points_by_grid(Building_points_GT, DoPP_threshold, DoPP_point_number)

    wall_points = building_points[building_points[:, 4] == 1]
    roof_points = building_points[building_points[:, 4] == 2]

    ## 2D DVDC for wall points
    coordinates = wall_points[:, :2]

    wall_points_with_labels = np.zeros((wall_points.shape[0], 11))
    wall_points_with_labels[:, :5] = wall_points

    wall_points_k_lof = np.zeros((wall_points.shape[0], 6))
    wall_points_k_dis = np.zeros((wall_points.shape[0], 6))
    wall_points_k_dlof = np.zeros((wall_points.shape[0], 5))
    wall_points_k_ddis = np.zeros((wall_points.shape[0], 5))

    k_order = 0
    for k in k_value_list:
        lof_values_k, label_k = detect_lof_noise(coordinates, n_neighbors=k)
        dis_values_k = compute_k_nearest_neighbors_average_distance(coordinates, k)
        wall_points_k_lof[:, k_order] = lof_values_k
        wall_points_k_dis[:, k_order] = dis_values_k
        if k_order > 0:
            wall_points_k_dlof[:, k_order - 1] = wall_points_k_lof[:, k_order] - wall_points_k_lof[:, k_order - 1]
            wall_points_k_ddis[:, k_order - 1] = wall_points_k_dis[:, k_order] - wall_points_k_dis[:, k_order - 1]
        k_order = k_order + 1

    wall_dis_w = np.ones((wall_points_k_ddis.shape[0], wall_points_k_ddis.shape[1]))
    wall_dis_min = np.min(wall_points_k_ddis)
    wall_dis_max = np.max(wall_points_k_ddis)

    wall_dis_w[wall_points_k_ddis < r] = 1.0
    wall_dis_w[wall_points_k_ddis >= r] = 1.0 - ((wall_points_k_ddis[wall_points_k_ddis >= r] - wall_dis_min) / (wall_dis_max - wall_dis_min))
    wall_points_k_dlof_w = wall_points_k_dlof + (1.0 - wall_dis_w) * wall_points_k_dlof
    wall_exception_points = np.any(wall_points_k_dlof_w > DVDC_threshold, axis=1)
    wall_filter_points = wall_points[~wall_exception_points]

    dbscan = DBSCAN(eps=0.08, min_samples=30)
    wall_labels = dbscan.fit_predict(wall_filter_points[:, :2])

    wall_points_with_labels = np.zeros((wall_filter_points.shape[0], 6))
    wall_points_with_labels[:, :5] = wall_filter_points
    wall_points_with_labels[:, 5] = wall_labels

    ## 3D DVDC for roof points
    coordinates = roof_points[:, :3]

    roof_points_with_labels = np.zeros((roof_points.shape[0], 11))
    roof_points_with_labels[:, :5] = roof_points

    roof_points_k_lof = np.zeros((roof_points.shape[0], 6))
    roof_points_k_dis = np.zeros((roof_points.shape[0], 6))
    roof_points_k_dlof = np.zeros((roof_points.shape[0], 5))
    roof_points_k_ddis = np.zeros((roof_points.shape[0], 5))

    k_order = 0
    for k in k_value_list:
        lof_values_k, label_k = detect_lof_noise(coordinates, n_neighbors=k)
        dis_values_k = compute_k_nearest_neighbors_average_distance(coordinates, k)
        roof_points_k_lof[:, k_order] = lof_values_k
        roof_points_k_dis[:, k_order] = dis_values_k
        if k_order > 0:
            roof_points_k_dlof[:, k_order - 1] = roof_points_k_lof[:, k_order] - roof_points_k_lof[:, k_order - 1]
            roof_points_k_ddis[:, k_order - 1] = roof_points_k_dis[:, k_order] - roof_points_k_dis[:, k_order - 1]
        k_order = k_order + 1

    roof_dis_w = np.ones((roof_points_k_ddis.shape[0], roof_points_k_ddis.shape[1]))
    roof_dis_min = np.min(roof_points_k_ddis)
    roof_dis_max = np.max(roof_points_k_ddis)

    roof_dis_w[roof_points_k_ddis < r] = 1.0
    roof_dis_w[roof_points_k_ddis >= r] = 1.0 - ((roof_points_k_ddis[roof_points_k_ddis >= r] - roof_dis_min) / (roof_dis_max - roof_dis_min))
    roof_points_k_dlof_w = roof_points_k_dlof + (1.0 - roof_dis_w) * roof_points_k_dlof
    roof_exception_points = np.any(roof_points_k_dlof_w > DVDC_threshold, axis=1)
    roof_filter_points = roof_points[~roof_exception_points]

    dbscan = DBSCAN(eps=0.1, min_samples=30)
    roof_labels = dbscan.fit_predict(roof_filter_points[:, :3])

    roof_points_with_labels = np.zeros((roof_filter_points.shape[0], 6))
    roof_points_with_labels[:, :5] = roof_filter_points
    roof_points_with_labels[:, 5] = roof_labels

    ## Merge wall points and roof points
    print(f"Filtered wall points: {wall_points.shape}")
    print(f"Filtered roof points: {roof_points.shape}")

    merged_building_points = clean_and_merge_point_clouds(wall_points_with_labels, roof_points_with_labels)

    print(f"Filtered building points: {merged_building_points.shape}")

    ## Extract building instances
    instances = extract_instances(merged_building_points)

    print("Number of building primitives before merging:", len(instances))
    '''
    for inst_id, points in instances.items():
        print(f"Instance {inst_id}: {points.shape[0]} points")
    '''

    min_distances, instance_ids = compute_min_distances(instances)
    merged_instances = merge_instances(instances, min_distances, instance_ids, merged_threshold)

    random_colors = np.array([
            [31, 119, 180],
            [255, 127, 14],
            [46, 160, 44],
            [214, 40, 39],
            [148, 103, 189],
            [140, 86, 75],
            [227, 119, 194],
            [126, 126, 126],
            [188, 189, 32],
            [26, 190, 207]
        ]) / 255.
    
    print("Number of building instances after merging:", len(merged_instances))

    total_array = []

    for inst_id, points in merged_instances.items():
        coords = points[:, :3]
        random_color = random_colors[np.random.randint(0, len(random_colors))]
        color_array = np.tile(random_color, (coords.shape[0], 1))
        instance_array = np.hstack((coords, color_array))
        total_array.append(instance_array)
    
    total_array = np.vstack(total_array)
    total_array = np.vstack((total_array, Ground_points_GT))

    points, colors = downsample_point_cloud(total_array, 0.025)
    color_strings = ['rgb({},{},{})'.format(r, g, b) for r, g, b in colors]

    fig = go.Figure(
            data=[
                go.Scatter3d(
                    x = points[:, 0], y = points[:, 1], z = points[:, 2],
                    mode='markers',
                    marker=dict(
                        size=1,
                        color=color_strings,  # Use the list of RGB strings for the marker colors
                    )
                )
            ],
            layout=dict(
                scene=dict(
                    xaxis=dict(visible=False),
                    yaxis=dict(visible=False),
                    zaxis=dict(visible=False),
                    aspectratio=dict(x=1, y=1, z=0.05)
                ),
                paper_bgcolor='rgb(255,255,255)'  # Set the background color to dark gray 50, 50, 50
            ),
          )
        
    return fig




if __name__ == "__main__":

  global args, DoPP_threshold, DoPP_point_number, k_value_list, DVDC_threshold, r, merged_threshold

  args = parse_args(sys.argv[1:])
  DoPP_threshold = 0.015
  DoPP_point_number = 10
  k_value_list = [10, 20, 30, 40, 50, 60]
  DVDC_threshold = 0.5
  r = 0.01
  merged_threshold = 0.02

  RGB_vis_path = args.RGB_vis_path
  semantic_results_path = args.semantic_results_path

  print('loading model...')

  with gr.Blocks(
    css="#pc_scene { height: 400px; width: 100%; } .button { height: 300px; width: 100px; }"
  ) as demo:

    gr.Markdown(
            """
            ## City-BIS: City-scale Building Instance Segmentation from LiDAR Point Clouds via Structure-aware Method. 🚀
            ## If you think this demo interesting, please consider starring 🌟 our github repo.)
            """
            )

    gr.Markdown(
            """
            ## Usage:
            1. Select one of the city-scale point clouds.
            2. Click **Visualize for Point Clouds**. 
            3. Provide the semantic results to preform Building Instance Extraction.
            4. Click **Building Instance Extraction via SI-DVDC**.
            """)
  

    with gr.Row():
        file_input_RGB = gr.File(label="Upload a TXT file", file_types=[".txt"], elem_id = "button")
        file_input_semantic = gr.File(label="Upload a TXT file", file_types=[".txt"], elem_id = "button")
    
    with gr.Row():
        pc_scene = gr.Plot(label = f"City Scene",elem_id = "pc_scene")
        building_instances = gr.Plot(label = f"Building Instances",elem_id = "pc_scene")
    
    with gr.Row():
        pc_vis_button = gr.Button("Visualize for Point Clouds!")
        BIS_button = gr.Button("Building Instance Extraction via SI-DVDC!")

    pc_vis_button.click(fn = pc_vis, inputs = [file_input_RGB], outputs = pc_scene)
    BIS_button.click(fn = SI_DVDC, inputs = [file_input_semantic], outputs = building_instances)

    demo.queue()
    demo.launch(share = True)
