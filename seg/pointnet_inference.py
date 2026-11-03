import os
import argparse
import numpy as np
import nibabel as nib
import open3d as o3d
import matplotlib.pyplot as plt
import torch
import plotly.graph_objects as go
from pointnet_train import SegmentationPointNet


def nii_to_pointcloud(nii_path, num_points=5120, apply_rotation=True):
    """
    Converts a NIfTI volume to a normalized point cloud.
    """
    nii = nib.load(nii_path)
    volume = nii.get_fdata()
    affine = nii.affine

    coords = np.argwhere(volume > 0)
    if coords.size == 0:
        raise ValueError("No foreground voxels found in the NIfTI volume.")

    coords_hom = np.hstack((coords, np.ones((coords.shape[0], 1))))
    world_coords = (affine @ coords_hom.T).T[:, :3]

    # Rotation matrix for 90° rotation about the x-axis
    rotation_matrix = np.array([
        [1, 0,  0],
        [0, 0, 1],
        [0, -1,  0]
    ])

    if apply_rotation:
        world_coords = (rotation_matrix @ world_coords.T).T

    vol_mean = np.mean(world_coords, axis=0)
    vol_scale = np.max(np.abs(world_coords - vol_mean), axis=0)
    normalized_coords = (world_coords - vol_mean) / vol_scale

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(normalized_coords)
    if len(normalized_coords) > num_points:
        pcd = pcd.farthest_point_down_sample(num_points)

    return vol_mean, vol_scale, np.asarray(pcd.points), rotation_matrix


def visualize_pointcloud_interactive(points, title="Interactive Point Cloud"):
    """
    Displays a 3D interactive visualization of a point cloud.
    """
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode='markers',
        marker=dict(size=3, color=points[:, 2], colorscale='jet', opacity=1.0)
    )])
    fig.update_layout(title=title, scene=dict(aspectmode="data"))
    fig.show()


def load_model(model_path):
    """
    Loads a pre-trained PointNet segmentation model and prepares it for inference.
    """
    model = SegmentationPointNet(num_classes=6)
    state_dict = torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(state_dict)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    return model


def run_model_on_pointcloud(model, points):
    """
    Runs the segmentation model on a point cloud and returns predicted class labels.
    """
    model.eval()
    with torch.no_grad():
        points_tensor = torch.tensor(points, dtype=torch.float32).unsqueeze(0)
        if torch.cuda.is_available():
            points_tensor = points_tensor.cuda()
        preds, _ = model(points_tensor)
        return preds.argmax(dim=2).cpu().numpy().squeeze()


def visualize_predictions(points, labels, title="Segmentation Result"):
    """
    Visualizes a 3D point cloud with color-coded segmentation labels.
    """
    fig = go.Figure(data=[go.Scatter3d(
        x=points[:, 0], y=points[:, 1], z=points[:, 2],
        mode='markers',
        marker=dict(size=3, color=labels, colorscale='jet', opacity=0.8)
    )])
    fig.update_layout(title=title, scene=dict(aspectmode="data"))
    fig.show()


def save_points_by_label(points, labels, label, output_dir, filename):
    """
    Saves a subset of points from a point cloud based on a specific label.
    """
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[labels == label])
    o3d.io.write_point_cloud(save_path, pcd)


def create_bounding_box_and_save(points, labels, label, output_dir, filename, padding=5):
    """
    Saves points within a bounding box around labeled regions of a point cloud.
    """
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, filename)
    target_points = points[labels == label]

    min_coords = np.min(target_points, axis=0) - padding
    max_coords = np.max(target_points, axis=0) + padding

    box_mask = np.all((points >= min_coords) & (points <= max_coords), axis=1)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points[box_mask])
    o3d.io.write_point_cloud(save_path, pcd)


def main(args):

    apply_rotation= True

    # Step 1: Load NIfTI and create point cloud
    mean, scale, normalized_points, rotation_matrix = nii_to_pointcloud(args.nii_path, args.num_points, apply_rotation= apply_rotation)

    # Optional: 
    #visualize_pointcloud_interactive(normalized_points, title="Input Point Cloud")

    # Step 2: Load model and run inference
    model = load_model(args.model_path)
    predictions = run_model_on_pointcloud(model, normalized_points)

    # Optional: 
    #visualize_predictions(normalized_points, predictions)

    # Step 3: Denormalize points back to world coordinates
    denorm_points = normalized_points * scale + mean

    # Step 4: Undo rotation to align points with original world coordinate system
    rot180_y = np.array([
        [-1, 0,  0],
        [ 0, -1,  0],
        [ 0, 0, 1]
    ])

    if apply_rotation:
        rotated_back = (rotation_matrix.T @ denorm_points.T).T
        aligned_points = (rot180_y @ rotated_back.T).T

    else:
        aligned_points= (rot180_y @ denorm_points.T).T

    # Step 5: Save full point cloud
    full_pcd = o3d.geometry.PointCloud()
    full_pcd.points = o3d.utility.Vector3dVector(aligned_points)
    o3d.io.write_point_cloud(args.output_pcd, full_pcd)

    # Step 6: Save seperate labels
    for label in np.unique(predictions):
        mapped = 25 - label
        save_points_by_label(
            aligned_points, predictions, label,
            output_dir=args.output_separate,
            filename=f"verLev{mapped}.pcd"
        )

    # Step 7: Save bounding boxes
    for label in np.unique(predictions):
        mapped = 25 - label
        create_bounding_box_and_save(
            aligned_points, predictions, label,
            output_dir=args.output_bbox,
            filename=f"verLev{mapped}.pcd",
            padding=args.bbox_padding
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NIfTI to Point Cloud Segmentation Pipeline")
    parser.add_argument("--nii_path", type=str, default='volumes/unet_reconstructed_volume.nii', help="Path to input NIfTI volume")
    parser.add_argument("--model_path", type=str, default='weights/pointnet_best_model.pth', help="Path to trained PointNet model (.pth)")
    parser.add_argument("--num_points", type=int, default=20480, help="Number of points for downsampling")
    parser.add_argument("--output_pcd", type=str, default="volumes/output_pointcloud.pcd", help="Path to save full output point cloud")
    parser.add_argument("--output_separate", type=str, default="volumes/separate", help="Directory to save labeled point clouds")
    parser.add_argument("--output_bbox", type=str, default="volumes/bbox", help="Directory to save bounding box outputs")
    parser.add_argument("--bbox_padding", type=float, default=2, help="Padding for bounding boxes")

    args = parser.parse_args()
    main(args)
