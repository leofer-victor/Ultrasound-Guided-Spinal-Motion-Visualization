#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   compounding_angle_margin.py
@Time    :   2026/11/03 13:47:41
@Author  :   Feng Li 
@Contact :   feng.li@tum.de
@Desc    :   
'''

import os
import sys
import math
import gc
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable

from PySide6 import QtCore
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QFileDialog, QLabel, QDoubleSpinBox,
    QProgressBar, QMessageBox, QSpinBox
)

import pyvista as pv
from pyvistaqt import QtInteractor

try:
    import imageio.v3 as iio
except Exception:
    import imageio as iio

try:
    import nibabel as nib
except Exception:
    nib = None


# ============================================================
# Math utilities
# ============================================================

def quat_to_rotmat(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Quaternion (x, y, z, w) -> 3x3 rotation matrix."""
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    n = np.linalg.norm(q)
    if n == 0:
        return np.eye(3, dtype=np.float64)

    q /= n
    x, y, z, w = q

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    return np.array([
        [1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy)],
        [2 * (xy + wz),     1 - 2 * (xx + zz), 2 * (yz - wx)],
        [2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy)],
    ], dtype=np.float64)


def rotation_about_local_z(angle_deg: float) -> np.ndarray:
    """Rotation around the flange/tool local Z axis."""
    a = np.deg2rad(float(angle_deg))
    c, s = np.cos(a), np.sin(a)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)


@dataclass
class Pose:
    t_mm: np.ndarray
    R: np.ndarray


def apply_mount_angle_compensation(
    poses: List[Pose],
    mount_angle_deg: float,
) -> List[Pose]:
    """
    Pose.R is interpreted as R_base_flange.
    Mount correction is in the flange local frame, therefore right-multiply:

        R_base_image = R_base_flange @ Rz(mount_angle)
    """
    R_mount = rotation_about_local_z(mount_angle_deg)
    return [Pose(t_mm=p.t_mm.copy(), R=p.R @ R_mount) for p in poses]


# ============================================================
# IO
# ============================================================

def parse_data_txt(txt_path: str) -> List[Pose]:
    """
    Supported pose formats:

    1) Current, 7 values:
       tx, ty, tz, qw, qx, qy, qz

    2) Legacy, >=9 values:
       timestamp, index, tx, ty, tz, qx, qy, qz, qw

    Translation is assumed to be in metres and converted to millimetres.
    """
    poses: List[Pose] = []

    with open(txt_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.replace(",", " ").split()

            try:
                values = [float(v) for v in parts]
            except ValueError as exc:
                raise ValueError(
                    f"Invalid numeric value in pose file {txt_path!r}, "
                    f"line {line_number}: {line!r}"
                ) from exc

            if len(values) == 7:
                tx_m, ty_m, tz_m, qw, qx, qy, qz = values
            elif len(values) >= 9:
                tx_m, ty_m, tz_m = values[2:5]
                qx, qy, qz, qw = values[5:9]
            else:
                raise ValueError(
                    f"Unsupported pose format in {txt_path!r}, line {line_number}: "
                    f"expected 7 or >=9 values, got {len(values)}."
                )

            t_mm = np.array([tx_m, ty_m, tz_m], dtype=np.float64) * 1000.0
            R = quat_to_rotmat(qx, qy, qz, qw)
            poses.append(Pose(t_mm=t_mm, R=R))

    if not poses:
        raise ValueError(f"No valid poses found in {txt_path!r}.")

    return poses


def _numeric_sort_key(path: str):
    """Sort timestamp/numeric image filenames numerically when possible."""
    stem = os.path.splitext(os.path.basename(path))[0]
    try:
        return (0, float(stem))
    except ValueError:
        return (1, stem.lower())


def collect_image_paths(folder: str) -> List[str]:
    """Collect common ultrasound image formats."""
    valid_ext = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    paths = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in valid_ext
    ]
    paths.sort(key=_numeric_sort_key)
    return paths


def load_ultrasound_image(path: str) -> np.ndarray:
    """
    Load one ultrasound image as a 2D float32 intensity image.

    Grayscale images preserve their original pixel values. RGB/RGBA images
    are converted to luminance. For normal grayscale ultrasound stored as
    RGB with equal channels, this gives the same intensity.
    """
    img = np.asarray(iio.imread(path))

    if img.ndim == 2:
        gray = img
    elif img.ndim == 3:
        if img.shape[2] == 1:
            gray = img[..., 0]
        else:
            rgb = img[..., :3].astype(np.float32)
            gray = (
                0.2126 * rgb[..., 0]
                + 0.7152 * rgb[..., 1]
                + 0.0722 * rgb[..., 2]
            )
    else:
        raise ValueError(
            f"Unsupported image shape {img.shape} for {path!r}. "
            "Expected a 2D grayscale or RGB/RGBA image."
        )

    return np.asarray(gray, dtype=np.float32)


def locate_pose_file(folder: str, pose_file: Optional[str]) -> str:
    if pose_file:
        if not os.path.isfile(pose_file):
            raise FileNotFoundError(f"Pose file does not exist: {pose_file}")
        return pose_file

    data_txt = os.path.join(folder, "data.txt")
    if os.path.isfile(data_txt):
        return data_txt

    txts = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".txt")
    ])

    if not txts:
        raise FileNotFoundError(
            "Pose file not found. Please select a .txt file in the UI."
        )

    return txts[0]


# ============================================================
# Geometry / compounding
# ============================================================

def create_local_pixel_coordinates(
    height: int,
    width: int,
    pixel_spacing_mm: float,
) -> np.ndarray:
    """
    Create every ultrasound pixel coordinate in the image/tool frame.

    Pose origin = top-center of the ultrasound image.

        local X = image horizontal direction
        local Y = 0
        local Z = image depth direction

    Output shape: (H*W, 3), float32.
    """
    cx = (width - 1) / 2.0

    x = (
        (np.arange(width, dtype=np.float32) - np.float32(cx))
        * np.float32(pixel_spacing_mm)
    )
    z = np.arange(height, dtype=np.float32) * np.float32(pixel_spacing_mm)

    xx, zz = np.meshgrid(x, z, indexing="xy")

    local_pts = np.empty((height * width, 3), dtype=np.float32)
    local_pts[:, 0] = xx.ravel()
    local_pts[:, 1] = 0.0
    local_pts[:, 2] = zz.ravel()

    return local_pts


def compute_volume_bounds(
    poses: List[Pose],
    height: int,
    width: int,
    pixel_spacing_mm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute world-space bounds using only four image corners per pose."""
    cx = (width - 1) / 2.0

    corners_px = np.array([
        [0.0, 0.0],
        [width - 1.0, 0.0],
        [0.0, height - 1.0],
        [width - 1.0, height - 1.0],
    ], dtype=np.float64)

    x_diff = (corners_px[:, 0] - cx) * pixel_spacing_mm
    z_diff = corners_px[:, 1] * pixel_spacing_mm

    local_corners = np.column_stack([
        x_diff,
        np.zeros_like(x_diff),
        z_diff,
    ])

    mins = np.full(3, np.inf, dtype=np.float64)
    maxs = np.full(3, -np.inf, dtype=np.float64)

    for pose in poses:
        world = (pose.R @ local_corners.T).T + pose.t_mm
        mins = np.minimum(mins, world.min(axis=0))
        maxs = np.maximum(maxs, world.max(axis=0))

    return mins, maxs


def estimate_memory_bytes(nx: int, ny: int, nz: int) -> int:
    # float32 intensity sum + uint32 count = 8 persistent bytes / voxel.
    return int(nx) * int(ny) * int(nz) * 8


def human_bytes(n: int) -> str:
    value = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TB"


def compound_ultrasound_to_volume(
    folder: str,
    pixel_spacing_mm: float,
    voxel_size_mm: float,
    frame_step: int = 1,
    pose_file: Optional[str] = None,
    mount_angle_deg: float = 0.0,
    progress_cb: Optional[Callable[[int, int], None]] = None,
) -> Tuple[pv.ImageData, np.ndarray, np.ndarray, dict]:
    """
    Compound original ultrasound intensities into a 3D volume.

    Pairing is done first. Frame skipping is applied afterwards:
        frame_step=10 -> pairs 0, 10, 20, ...

    For each voxel:
        output intensity = sum(contributing pixels) / number of pixels
    """
    if pixel_spacing_mm <= 0:
        raise ValueError("Pixel spacing must be > 0.")
    if voxel_size_mm <= 0:
        raise ValueError("Voxel size must be > 0.")

    frame_step = max(1, int(frame_step))

    image_paths = collect_image_paths(folder)
    if not image_paths:
        raise RuntimeError(
            "No ultrasound images found. Supported formats: "
            "PNG, JPG, JPEG, BMP, TIF, TIFF."
        )

    pose_file = locate_pose_file(folder, pose_file)

    poses = parse_data_txt(pose_file)
    poses = apply_mount_angle_compensation(poses, mount_angle_deg)

    n_pairs = min(len(image_paths), len(poses))
    if n_pairs == 0:
        raise RuntimeError("No image/pose pairs found.")

    image_paths = image_paths[:n_pairs]
    poses = poses[:n_pairs]

    # Important: skip after pairing so image-pose correspondence is preserved.
    selected_indices = list(range(0, n_pairs, frame_step))
    selected_images = [image_paths[i] for i in selected_indices]
    selected_poses = [poses[i] for i in selected_indices]

    # Read one image for geometry. It is reused as frame 0 below.
    first_img = load_ultrasound_image(selected_images[0])
    height, width = first_img.shape

    vol_min_mm, vol_max_mm = compute_volume_bounds(
        selected_poses,
        height,
        width,
        pixel_spacing_mm,
    )

    pad = 2.0 * voxel_size_mm
    vol_min_mm -= pad
    vol_max_mm += pad

    size_mm = vol_max_mm - vol_min_mm

    nx = int(math.ceil(size_mm[0] / voxel_size_mm)) + 1
    ny = int(math.ceil(size_mm[1] / voxel_size_mm)) + 1
    nz = int(math.ceil(size_mm[2] / voxel_size_mm)) + 1

    if nx <= 0 or ny <= 0 or nz <= 0:
        raise RuntimeError(f"Invalid volume dimensions: {(nx, ny, nz)}")

    n_voxels = int(nx) * int(ny) * int(nz)
    estimated_bytes = estimate_memory_bytes(nx, ny, nz)

    # Old-PC safety guard against accidental enormous volumes.
    if estimated_bytes > 12 * 1024**3:
        raise MemoryError(
            "Requested volume is extremely large.\n\n"
            f"Dimensions: {nx} x {ny} x {nz}\n"
            f"Voxels: {n_voxels:,}\n"
            f"Estimated persistent compounding arrays: {human_bytes(estimated_bytes)}\n\n"
            "Please increase Voxel size and/or Frame step."
        )

    intensity_sum = np.zeros(n_voxels, dtype=np.float32)
    count_flat = np.zeros(n_voxels, dtype=np.uint32)

    # Geometry does not change from frame to frame, so compute it once.
    local_pts = create_local_pixel_coordinates(
        height,
        width,
        pixel_spacing_mm,
    )

    vol_min_f32 = vol_min_mm.astype(np.float32)
    voxel_f32 = np.float32(voxel_size_mm)
    total = len(selected_images)

    for frame_i, (image_path, pose) in enumerate(zip(selected_images, selected_poses)):
        image = first_img if frame_i == 0 else load_ultrasound_image(image_path)

        if image.shape != (height, width):
            raise ValueError(
                f"Image size mismatch:\n{image_path}\n"
                f"expected {(height, width)}, got {image.shape}"
            )

        intensities = image.ravel().astype(np.float32, copy=False)

        R = pose.R.astype(np.float32, copy=False)
        t = pose.t_mm.astype(np.float32, copy=False)

        # local_pts are row vectors: world = local @ R.T + t
        world_pts = local_pts @ R.T
        world_pts += t

        rel = (world_pts - vol_min_f32) / voxel_f32

        ix = np.floor(rel[:, 0]).astype(np.int32)
        iy = np.floor(rel[:, 1]).astype(np.int32)
        iz = np.floor(rel[:, 2]).astype(np.int32)

        valid = (
            (ix >= 0) & (iy >= 0) & (iz >= 0)
            & (ix < nx) & (iy < ny) & (iz < nz)
        )

        if np.any(valid):
            ixv = ix[valid].astype(np.int64, copy=False)
            iyv = iy[valid].astype(np.int64, copy=False)
            izv = iz[valid].astype(np.int64, copy=False)
            values = intensities[valid]

            # Flat index for C-order array shaped (nx, ny, nz).
            flat_idx = (ixv * ny + iyv) * nz + izv

            # Group repeated voxel hits within this frame, then update global sum/count.
            unique_idx, inverse, counts = np.unique(
                flat_idx,
                return_inverse=True,
                return_counts=True,
            )

            grouped_sum = np.bincount(
                inverse,
                weights=values,
                minlength=unique_idx.size,
            ).astype(np.float32, copy=False)

            intensity_sum[unique_idx] += grouped_sum
            count_flat[unique_idx] += counts.astype(np.uint32, copy=False)

            del ixv, iyv, izv, values, flat_idx
            del unique_idx, inverse, counts, grouped_sum

        del image, intensities, world_pts, rel, ix, iy, iz, valid

        if progress_cb:
            progress_cb(frame_i + 1, total)

    # Reuse intensity_sum as the final average-intensity buffer.
    occupied_flat = count_flat > 0
    np.divide(
        intensity_sum,
        count_flat,
        out=intensity_sum,
        where=occupied_flat,
    )
    intensity_sum[~occupied_flat] = 0.0

    volume = intensity_sum.reshape((nx, ny, nz), order="C")
    count_volume = count_flat.reshape((nx, ny, nz), order="C")

    if hasattr(pv, "ImageData"):
        grid = pv.ImageData(
            dimensions=(nx, ny, nz),
            spacing=(voxel_size_mm, voxel_size_mm, voxel_size_mm),
            origin=tuple(vol_min_mm.tolist()),
        )
    elif hasattr(pv, "UniformGrid"):
        grid = pv.UniformGrid()
        grid.origin = tuple(vol_min_mm.tolist())
        grid.spacing = (voxel_size_mm, voxel_size_mm, voxel_size_mm)
        grid.dimensions = (nx, ny, nz)
    else:
        import vtk
        vtk_grid = vtk.vtkImageData()
        vtk_grid.SetDimensions(nx, ny, nz)
        vtk_grid.SetSpacing(voxel_size_mm, voxel_size_mm, voxel_size_mm)
        vtk_grid.SetOrigin(*vol_min_mm.tolist())
        grid = pv.wrap(vtk_grid)

    grid.point_data.clear()
    grid["intensity"] = volume.ravel(order="F")
    grid["count"] = count_volume.ravel(order="F")

    occupied_values = intensity_sum[occupied_flat]
    if occupied_values.size:
        min_intensity = float(occupied_values.min())
        max_intensity = float(occupied_values.max())
        mean_intensity = float(occupied_values.mean())
    else:
        min_intensity = max_intensity = mean_intensity = 0.0

    info = {
        "image_files": len(image_paths),
        "poses": len(poses),
        "paired_frames": n_pairs,
        "frame_step": frame_step,
        "used_frames": len(selected_images),
        "image_shape": (height, width),
        "volume_shape": (nx, ny, nz),
        "voxel_size_mm": voxel_size_mm,
        "estimated_array_memory": human_bytes(estimated_bytes),
        "occupied_voxels": int(np.count_nonzero(count_flat)),
        "min_intensity": min_intensity,
        "max_intensity": max_intensity,
        "mean_intensity": mean_intensity,
    }

    return grid, volume, count_volume, info


# ============================================================
# Qt worker
# ============================================================

class CompoundWorker(QtCore.QObject):
    progressed = QtCore.Signal(int, int)
    finished = QtCore.Signal(object, object, object, object, object)
    # grid, volume, count_volume, info, error

    def __init__(
        self,
        folder: str,
        pixel_spacing: float,
        voxel_size: float,
        frame_step: int,
        pose_file: Optional[str],
        mount_angle_deg: float,
    ):
        super().__init__()
        self.folder = folder
        self.pixel_spacing = float(pixel_spacing)
        self.voxel_size = float(voxel_size)
        self.frame_step = int(frame_step)
        self.pose_file = pose_file
        self.mount_angle_deg = float(mount_angle_deg)

    @QtCore.Slot()
    def run(self):
        try:
            grid, volume, count_volume, info = compound_ultrasound_to_volume(
                folder=self.folder,
                pixel_spacing_mm=self.pixel_spacing,
                voxel_size_mm=self.voxel_size,
                frame_step=self.frame_step,
                pose_file=self.pose_file,
                mount_angle_deg=self.mount_angle_deg,
                progress_cb=lambda c, t: self.progressed.emit(c, t),
            )
            self.finished.emit(grid, volume, count_volume, info, None)
        except Exception as exc:
            self.finished.emit(None, None, None, None, exc)


# ============================================================
# Main window
# ============================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Ultrasound Intensity Compounding")
        self.resize(1300, 850)

        self._worker_thread: Optional[QtCore.QThread] = None
        self._worker: Optional[CompoundWorker] = None

        self._last_grid: Optional[pv.ImageData] = None
        self._last_volume: Optional[np.ndarray] = None
        self._last_count_volume: Optional[np.ndarray] = None
        self._last_info: Optional[dict] = None

        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Image folder row
        folder_row = QHBoxLayout()
        layout.addLayout(folder_row)

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Select folder containing ultrasound images")

        self.btn_browse = QPushButton("Browse...")
        self.btn_browse.clicked.connect(self.on_browse)

        folder_row.addWidget(QLabel("Image folder:"))
        folder_row.addWidget(self.path_edit, 1)
        folder_row.addWidget(self.btn_browse)

        # Pose row
        pose_row = QHBoxLayout()
        layout.addLayout(pose_row)

        self.pose_edit = QLineEdit()
        self.pose_edit.setPlaceholderText(
            "Select pose .txt (optional; auto-detect data.txt if empty)"
        )

        self.btn_pose = QPushButton("Pose...")
        self.btn_pose.clicked.connect(self.on_browse_pose)

        pose_row.addWidget(QLabel("Pose file:"))
        pose_row.addWidget(self.pose_edit, 1)
        pose_row.addWidget(self.btn_pose)

        # Parameters row
        params_row = QHBoxLayout()
        layout.addLayout(params_row)

        self.spacing_spin = QDoubleSpinBox()
        self.spacing_spin.setDecimals(9)
        self.spacing_spin.setRange(1e-6, 1000.0)
        self.spacing_spin.setValue(0.083333333)
        self.spacing_spin.setSuffix(" mm")
        self.spacing_spin.setToolTip("Physical spacing of one ultrasound image pixel.")

        self.voxel_spin = QDoubleSpinBox()
        self.voxel_spin.setDecimals(4)
        self.voxel_spin.setRange(0.01, 20.0)
        self.voxel_spin.setSingleStep(0.05)
        self.voxel_spin.setValue(0.30)
        self.voxel_spin.setSuffix(" mm")
        self.voxel_spin.setToolTip(
            "3D isotropic voxel size. Larger values are faster and use less memory."
        )

        self.frame_step_spin = QSpinBox()
        self.frame_step_spin.setRange(1, 1000)
        self.frame_step_spin.setValue(1)
        self.frame_step_spin.setSuffix(" frame(s)")
        self.frame_step_spin.setToolTip(
            "Use every Nth image-pose pair. Example: 10 uses 0, 10, 20, ..."
        )

        self.mount_angle_spin = QDoubleSpinBox()
        self.mount_angle_spin.setDecimals(1)
        self.mount_angle_spin.setRange(-360.0, 360.0)
        self.mount_angle_spin.setSingleStep(45.0)
        self.mount_angle_spin.setValue(0.0)
        self.mount_angle_spin.setSuffix(" deg")
        self.mount_angle_spin.setToolTip(
            "Probe mounting compensation around flange/tool local Z."
        )

        self.btn_run = QPushButton("Run Compounding")
        self.btn_run.clicked.connect(self.on_run)

        self.btn_reset = QPushButton("Reset")
        self.btn_reset.clicked.connect(self.on_reset)

        params_row.addWidget(QLabel("Pixel spacing:"))
        params_row.addWidget(self.spacing_spin)
        params_row.addWidget(QLabel("Voxel size:"))
        params_row.addWidget(self.voxel_spin)
        params_row.addWidget(QLabel("Use every:"))
        params_row.addWidget(self.frame_step_spin)
        params_row.addWidget(QLabel("Mount Z:"))
        params_row.addWidget(self.mount_angle_spin)
        params_row.addStretch(1)
        params_row.addWidget(self.btn_run)
        params_row.addWidget(self.btn_reset)

        # Export row
        export_row = QHBoxLayout()
        layout.addLayout(export_row)

        self.btn_save_nifti = QPushButton("Save as NIfTI...")
        self.btn_save_nifti.clicked.connect(self.on_save_nifti)

        export_row.addStretch(1)
        export_row.addWidget(self.btn_save_nifti)

        # Progress / status
        self.prog = QProgressBar()
        self.prog.setRange(0, 100)
        self.prog.setValue(0)
        layout.addWidget(self.prog)

        self.status_label = QLabel(
            "Ready. Intensity compounding uses voxel-wise averaging."
        )
        layout.addWidget(self.status_label)

        # PyVista
        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter.interactor)
        self.plotter.enable_depth_peeling()
        self.plotter.add_axes()

    def set_running(self, running: bool):
        self.btn_run.setEnabled(not running)
        self.btn_reset.setEnabled(not running)
        self.btn_browse.setEnabled(not running)
        self.btn_pose.setEnabled(not running)
        self.spacing_spin.setEnabled(not running)
        self.voxel_spin.setEnabled(not running)
        self.frame_step_spin.setEnabled(not running)
        self.mount_angle_spin.setEnabled(not running)

    @QtCore.Slot()
    def on_browse(self):
        typed = self.path_edit.text().strip()
        if typed and os.path.isdir(typed):
            start_dir = typed
        elif os.path.isdir("/home/leofer/experiment"):
            start_dir = "/home/leofer/experiment"
        else:
            start_dir = os.getcwd()

        folder = QFileDialog.getExistingDirectory(
            self,
            "Select ultrasound image folder",
            start_dir,
        )
        if folder:
            self.path_edit.setText(folder)

    @QtCore.Slot()
    def on_browse_pose(self):
        start_dir = self.path_edit.text().strip() or os.getcwd()
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select pose .txt",
            start_dir,
            "Text Files (*.txt);;All Files (*)",
        )
        if path:
            self.pose_edit.setText(path)

    @QtCore.Slot()
    def on_run(self):
        folder = self.path_edit.text().strip()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(
                self,
                "Input",
                "Please select a valid ultrasound image folder.",
            )
            return

        pose_path = self.pose_edit.text().strip() or None
        pixel_spacing = float(self.spacing_spin.value())
        voxel_size = float(self.voxel_spin.value())
        frame_step = int(self.frame_step_spin.value())
        mount_angle = float(self.mount_angle_spin.value())

        self.clear_result_data()
        self.prog.setValue(0)
        self.status_label.setText("Compounding...")
        self.set_running(True)

        thread = QtCore.QThread(self)
        worker = CompoundWorker(
            folder=folder,
            pixel_spacing=pixel_spacing,
            voxel_size=voxel_size,
            frame_step=frame_step,
            pose_file=pose_path,
            mount_angle_deg=mount_angle,
        )

        self._worker_thread = thread
        self._worker = worker
        worker.moveToThread(thread)

        thread.started.connect(worker.run)
        worker.progressed.connect(self.on_progress)
        worker.finished.connect(self.on_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self.on_thread_finished)

        thread.start()

    @QtCore.Slot(int, int)
    def on_progress(self, current: int, total: int):
        if total > 0:
            pct = int(round(100.0 * current / total))
            self.prog.setValue(max(0, min(100, pct)))
            self.status_label.setText(f"Compounding frame {current}/{total}...")

    @QtCore.Slot(object, object, object, object, object)
    def on_finished(self, grid, volume, count_volume, info, error):
        if error is not None:
            self.prog.setValue(0)
            self.status_label.setText("Failed.")
            QMessageBox.critical(self, "Compounding Error", str(error))
            return

        self._last_grid = grid
        self._last_volume = volume
        self._last_count_volume = count_volume
        self._last_info = info

        self.render_volume()
        self.prog.setValue(100)

        self.status_label.setText(
            f"Done | used {info['used_frames']}/{info['paired_frames']} paired frames | "
            f"volume {info['volume_shape']} | voxel {info['voxel_size_mm']:.3f} mm | "
            f"occupied {info['occupied_voxels']:,} voxels | "
            f"arrays ~{info['estimated_array_memory']}"
        )

    @QtCore.Slot()
    def on_thread_finished(self):
        self._worker = None
        self._worker_thread = None
        self.set_running(False)

    def render_volume(self):
        grid = self._last_grid
        volume = self._last_volume
        count_volume = self._last_count_volume

        if grid is None or volume is None or count_volume is None:
            return

        self.plotter.clear()
        self.plotter.add_axes()

        occupied = count_volume > 0
        if np.any(occupied):
            vmax = float(volume[occupied].max())
        else:
            vmax = 1.0

        if vmax <= 0:
            vmax = 1.0

        try:
            # Gray colormap: low intensity dark gray/black, highest intensity white.
            self.plotter.add_volume(
                grid,
                scalars="intensity",
                cmap="gray",
                clim=(0.0, vmax),
                opacity="linear",
                shade=False,
            )
        except Exception as exc:
            QMessageBox.critical(
                self,
                "VTK Display Error",
                f"Failed to render volume:\n{exc}",
            )
            return

        self.plotter.reset_camera()
        self.plotter.render()

    def clear_result_data(self):
        """Clear rendering and release large arrays; preserve UI input values."""
        try:
            self.plotter.clear()
            self.plotter.add_axes()
            self.plotter.render()
        except Exception:
            pass

        self._last_grid = None
        self._last_volume = None
        self._last_count_volume = None
        self._last_info = None
        gc.collect()

    @QtCore.Slot()
    def on_reset(self):
        if self._worker_thread is not None:
            QMessageBox.information(
                self,
                "Reset",
                "Compounding is currently running. Reset is available after it finishes.",
            )
            return

        self.clear_result_data()
        self.prog.setValue(0)
        self.status_label.setText(
            "Reset complete. Ready for a new compounding run."
        )

    def _require_result(self) -> bool:
        if self._last_grid is None or self._last_volume is None:
            QMessageBox.information(
                self,
                "Export",
                "Please run compounding first.",
            )
            return False
        return True

    @QtCore.Slot()
    def on_save_nifti(self):
        if not self._require_result():
            return

        if nib is None:
            QMessageBox.critical(
                self,
                "NIfTI",
                "nibabel is not installed.\nInstall it with: pip install nibabel",
            )
            return

        default_dir = self.path_edit.text().strip() or os.getcwd()
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save compounded ultrasound volume",
            os.path.join(default_dir, "compounded_ultrasound.nii.gz"),
            "NIfTI (*.nii *.nii.gz)",
        )

        if not path:
            return

        try:
            grid = self._last_grid
            volume = self._last_volume

            sx, sy, sz = grid.spacing
            ox, oy, oz = grid.origin

            affine = np.array([
                [sx, 0.0, 0.0, ox],
                [0.0, sy, 0.0, oy],
                [0.0, 0.0, sz, oz],
                [0.0, 0.0, 0.0, 1.0],
            ], dtype=np.float64)

            img = nib.Nifti1Image(
                volume.astype(np.float32, copy=False),
                affine,
            )

            try:
                img.header.set_zooms((float(sx), float(sy), float(sz)))
            except Exception:
                pass

            nib.save(img, path)
            QMessageBox.information(
                self,
                "NIfTI",
                f"Saved compounded intensity volume to:\n{path}",
            )

        except Exception as exc:
            QMessageBox.critical(self, "NIfTI Save Error", str(exc))


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())