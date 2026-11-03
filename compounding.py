import os
import sys
import math
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional

from PySide6 import QtCore, QtWidgets
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLineEdit, QFileDialog, QLabel, QDoubleSpinBox,
    QProgressBar, QMessageBox, QCheckBox, QComboBox, QSpinBox
)

# PyVista / VTK
import pyvista as pv
from pyvistaqt import QtInteractor

# Image IO
try:
    import imageio.v3 as iio
except Exception:
    import imageio as iio

# Optional NIfTI writer
try:
    import nibabel as nib
except Exception:
    nib = None


# -----------------------------
# Math Utilities
# -----------------------------

def quat_to_rotmat(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """
    Convert quaternion (x, y, z, w) to a 3x3 rotation matrix.
    Ensures normalization for numerical stability.
    """
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    n = np.linalg.norm(q)
    if n == 0:
        return np.eye(3)
    q /= n
    x, y, z, w = q
    xx, yy, zz = x*x, y*y, z*z
    xy, xz, yz = x*y, x*z, y*z
    wx, wy, wz = w*x, w*y, w*z
    R = np.array([
        [1 - 2*(yy + zz),     2*(xy - wz),         2*(xz + wy)],
        [2*(xy + wz),         1 - 2*(xx + zz),     2*(yz - wx)],
        [2*(xz - wy),         2*(yz + wx),         1 - 2*(xx + yy)]
    ], dtype=np.float64)
    return R


@dataclass
class Pose:
    t_mm: np.ndarray   # translation in millimeters, shape (3,)
    R: np.ndarray      # 3x3 rotation matrix


# -----------------------------
# IO: parse poses and masks
# -----------------------------

def parse_data_txt(txt_path: str) -> List[Pose]:
    poses: List[Pose] = []
    with open(txt_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 9:
                continue
            # Format per line: [timestamp, index, tx, ty, tz, qx, qy, qz, qw]
            # translations are in meters in the file; convert to mm
            tx_m, ty_m, tz_m = map(float, parts[2:5])
            qx, qy, qz, qw = map(float, parts[5:9])
            t_mm = np.array([tx_m, ty_m, tz_m], dtype=np.float64) * 1000.0
            R = quat_to_rotmat(qx, qy, qz, qw)
            poses.append(Pose(t_mm=t_mm, R=R))
    return poses


def load_mask(path: str) -> np.ndarray:
    """Load a mask image (png) and return a binary numpy array of shape (H, W).
    Non-zero pixels are treated as foreground (value=1)."""
    img = iio.imread(path)
    if img.ndim == 3:
        # Use first channel if RGB/RGBA
        img = img[..., 0]
    mask = (img > 0).astype(np.uint8)
    return mask


# -----------------------------
# Compounding core
# -----------------------------

def compute_volume_bounds(
    masks_paths: List[str],
    poses: List[Pose],
    pixel_spacing_mm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    mins = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    maxs = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)

    for mpath, pose in zip(masks_paths, poses):
        mask = load_mask(mpath)
        H, W = mask.shape
        cx = (W - 1) / 2.0
        cy = (H - 1) / 2.0

        # --- image origin of the pose---
        ox, oy = cx, 0.0

        corners_px = np.array([[0,0],[W-1,0],[0,H-1],[W-1,H-1]], dtype=np.float64)

        x_diff = (corners_px[:, 0] - ox) * pixel_spacing_mm
        y_diff = (corners_px[:, 1] - oy) * pixel_spacing_mm
        
        # X_local=0, Y_local=-x_diff, Z_local=-y_diff
        local_pts = np.column_stack([
            x_diff,  
            np.zeros_like(x_diff),              
            y_diff                 
        ])
        world_pts = (pose.R @ local_pts.T).T + pose.t_mm

        mins = np.minimum(mins, world_pts.min(axis=0))
        maxs = np.maximum(maxs, world_pts.max(axis=0))

    return mins, maxs


def compound_masks_to_volume(
    folder: str,
    pixel_spacing_mm: float = 0.083333333,
    voxel_size_mm: Optional[float] = None,
    pose_file: Optional[str] = None,
    progress_cb: Optional[callable] = None,
    write_mode: str = "count",           # "count" or "binary"
    thickness_vox: int = 1                # splat thickness along volume-z
) -> Tuple[pv.ImageData, np.ndarray]:
    """
    Read masks 0000.png.. and corresponding poses from data.txt in the same folder,
    compound into a 3D volume, and return a PyVista ImageData and the raw 3D numpy array.

    """
    if voxel_size_mm is None:
        voxel_size_mm = pixel_spacing_mm

    # Collect mask file paths sorted by zero-padded name
    masks_paths = sorted([
        os.path.join(folder, f) for f in os.listdir(folder)
        if f.lower().endswith('.png') and f[:4].isdigit()
    ])

    # Locate pose file: prefer provided path; else try data.txt; else first .txt in folder
    if pose_file is None:
        data_txt = os.path.join(folder, 'data.txt')
        if os.path.isfile(data_txt):
            pose_file = data_txt
        else:
            txts = [os.path.join(folder, f) for f in os.listdir(folder) if f.lower().endswith('.txt')]
            if not txts:
                raise FileNotFoundError("Pose file not found. Please select a .txt via the UI.")
            pose_file = sorted(txts)[0]

    poses = parse_data_txt(pose_file)

    n = min(len(masks_paths), len(poses))
    if n == 0:
        raise RuntimeError("No matching masks/poses found.")
    masks_paths = masks_paths[:n]
    poses = poses[:n]

    # Compute bounds
    vol_min_mm, vol_max_mm = compute_volume_bounds(masks_paths, poses, pixel_spacing_mm)

    # Pad a bit; also pad for thickness
    pad = voxel_size_mm * max(2, thickness_vox)
    vol_min_mm -= pad
    vol_max_mm += pad

    # Dimensions
    size_mm = vol_max_mm - vol_min_mm
    nx = int(math.ceil(size_mm[0] / voxel_size_mm)) + 1
    ny = int(math.ceil(size_mm[1] / voxel_size_mm)) + 1
    nz = int(math.ceil(size_mm[2] / voxel_size_mm)) + 1

    # Volume array in (nx, ny, nz)
    dtype = np.uint16 if write_mode == "count" else np.uint8
    vol = np.zeros((nx, ny, nz), dtype=dtype)

    half_t = max(0, (int(thickness_vox) - 1) // 2)

    total = n
    for i, (mpath, pose) in enumerate(zip(masks_paths, poses)):
        mask = load_mask(mpath)
        H, W = mask.shape
        ys, xs = np.nonzero(mask)
        if xs.size == 0:
            if progress_cb:
                progress_cb(i + 1, total)
            continue

        cx = (W - 1) / 2.0
        cy = (H - 1) / 2.0

        # top center
        ox, oy = cx, 0.0

        x_diff = (xs - ox) * pixel_spacing_mm
        y_diff = (ys - oy) * pixel_spacing_mm

        local_pts = np.stack([
            x_diff,
            np.zeros_like(x_diff),                 
            y_diff                
        ], axis=1)
        world_pts = (pose.R @ local_pts.T).T + pose.t_mm  # (K,3)

        # Map to voxel indices
        rel = (world_pts - vol_min_mm) / voxel_size_mm
        ix = np.floor(rel[:, 0]).astype(np.int64)
        iy = np.floor(rel[:, 1]).astype(np.int64)
        iz = np.floor(rel[:, 2]).astype(np.int64)

        # Clip to bounds
        valid = (
            (ix >= 0) & (iy >= 0) & (iz >= 0) &
            (ix < nx) & (iy < ny) & (iz < nz)
        )
        ix, iy, iz = ix[valid], iy[valid], iz[valid]

        if ix.size == 0:
            if progress_cb:
                progress_cb(i + 1, total)
            continue

        if half_t == 0:
            if write_mode == "count":
                np.add.at(vol, (ix, iy, iz), 1)
            else:  # binary
                vol[ix, iy, iz] = 1
        else:
            # splat thickness along z
            for dz in range(-half_t, half_t + 1):
                iz2 = iz + dz
                inb = (iz2 >= 0) & (iz2 < nz)
                if not np.any(inb):
                    continue
                if write_mode == "count":
                    np.add.at(vol, (ix[inb], iy[inb], iz2[inb]), 1)
                else:
                    vol[ix[inb], iy[inb], iz2[inb]] = 1

        if progress_cb:
            progress_cb(i + 1, total)

    # Create VTK ImageData/UniformGrid (version-compatible)
    if hasattr(pv, "ImageData"):
        grid = pv.ImageData(
            dimensions=(nx, ny, nz),
            spacing=(voxel_size_mm, voxel_size_mm, voxel_size_mm),
            origin=tuple(vol_min_mm.tolist()),
        )
    elif hasattr(pv, "UniformGrid"):
        grid = pv.UniformGrid()
        grid.origin = tuple(vol_min_mm.tolist())  # in mm
        grid.spacing = (voxel_size_mm, voxel_size_mm, voxel_size_mm)
        grid.dimensions = (nx, ny, nz)
    else:
        import vtk
        vtk_grid = vtk.vtkImageData()
        vtk_grid.SetDimensions(nx, ny, nz)
        vtk_grid.SetSpacing(voxel_size_mm, voxel_size_mm, voxel_size_mm)
        vtk_grid.SetOrigin(*vol_min_mm.tolist())
        grid = pv.wrap(vtk_grid)

    # Attach scalars; PyVista expects Fortran ordering for 3D arrays
    grid.point_data.clear()
    grid["values"] = vol.ravel(order='F')

    return grid, vol


# -----------------------------
# Qt Worker Thread
# -----------------------------
class CompoundWorker(QtCore.QObject):
    progressed = QtCore.Signal(int, int)  # current, total
    finished = QtCore.Signal(object, object, object)  # grid, vol, error

    def __init__(self, folder: str, pixel_spacing: float, voxel: float,
                 pose_file: Optional[str] = None, write_mode: str = "count", thickness_vox: int = 1):
        super().__init__()
        self.folder = folder
        self.pixel_spacing = pixel_spacing
        self.voxel = voxel
        self.pose_file = pose_file
        self.write_mode = write_mode
        self.thickness_vox = int(thickness_vox)

    @QtCore.Slot()
    def run(self):
        try:
            grid, vol = compound_masks_to_volume(
                self.folder,
                pixel_spacing_mm=self.pixel_spacing,
                voxel_size_mm=self.voxel,
                pose_file=self.pose_file,
                write_mode=self.write_mode,
                thickness_vox=self.thickness_vox,
                progress_cb=lambda c, t: self.progressed.emit(c, t)
            )
            self.finished.emit(grid, vol, None)
        except Exception as e:
            self.finished.emit(None, None, e)


# -----------------------------
# Main Window
# -----------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mask Compounding to Volume")
        self.resize(1200, 800)

        central = QWidget(self)
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)

        # Controls
        ctrl_layout = QHBoxLayout()
        layout.addLayout(ctrl_layout)

        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Select folder containing 0000.png.. and data.txt")
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self.on_browse)

        ctrl_layout.addWidget(QLabel("Folder:"))
        ctrl_layout.addWidget(self.path_edit, 1)
        ctrl_layout.addWidget(btn_browse)

        # Pose file chooser row
        ctrl_layout2 = QHBoxLayout()
        layout.addLayout(ctrl_layout2)
        self.pose_edit = QLineEdit()
        self.pose_edit.setPlaceholderText("Select pose .txt (optional – auto-detect if empty)")
        btn_pose = QPushButton("Pose…")
        btn_pose.clicked.connect(self.on_browse_pose)
        ctrl_layout2.addWidget(QLabel("Pose file:"))
        ctrl_layout2.addWidget(self.pose_edit, 1)
        ctrl_layout2.addWidget(btn_pose)

        # Params row
        params_layout = QHBoxLayout()
        layout.addLayout(params_layout)

        self.spacing_spin = QDoubleSpinBox()
        self.spacing_spin.setDecimals(9)
        self.spacing_spin.setRange(1e-6, 1000.0)
        self.spacing_spin.setValue(0.083333333)
        self.spacing_spin.setSuffix(" mm (pixel)")

        self.voxel_spin = QDoubleSpinBox()
        self.voxel_spin.setDecimals(6)
        self.voxel_spin.setRange(1e-3, 100.0)
        self.voxel_spin.setValue(0.083333)
        self.voxel_spin.setSuffix(" mm (voxel)")

        self.write_mode_combo = QComboBox()
        self.write_mode_combo.addItems(["count", "binary"])  # count:累加  binary:置1
        self.thick_spin = QSpinBox()
        self.thick_spin.setRange(1, 21)
        self.thick_spin.setSingleStep(1)
        self.thick_spin.setValue(1)
        self.thick_spin.setSuffix(" vox")

        btn_run = QPushButton("Run Compounding")
        btn_run.clicked.connect(self.on_run)

        params_layout.addWidget(QLabel("Pixel spacing:"))
        params_layout.addWidget(self.spacing_spin)
        params_layout.addWidget(QLabel("Voxel size:"))
        params_layout.addWidget(self.voxel_spin)
        params_layout.addWidget(QLabel("Write:"))
        params_layout.addWidget(self.write_mode_combo)
        params_layout.addWidget(QLabel("Thickness:"))
        params_layout.addWidget(self.thick_spin)
        params_layout.addStretch(1)
        params_layout.addWidget(btn_run)

        # --- Export buttons row ---
        export_layout = QHBoxLayout()
        layout.addLayout(export_layout)
        self.btn_save_nifti = QPushButton("Save as NIfTI…")
        self.btn_save_nifti.clicked.connect(self.on_save_nifti)
        self.btn_export_stl = QPushButton("Export STL…")
        self.btn_export_stl.clicked.connect(self.on_export_stl)
        self.btn_save_ply = QPushButton("Save as PLY…")
        self.btn_save_ply.clicked.connect(self.on_save_ply)
        export_layout.addStretch(1)
        export_layout.addWidget(self.btn_save_nifti)
        export_layout.addWidget(self.btn_export_stl)
        export_layout.addWidget(self.btn_save_ply)

        # Progress
        self.prog = QProgressBar()
        self.prog.setRange(0, 100)
        self.prog.setValue(0)
        layout.addWidget(self.prog)

        # PyVista scene
        self.plotter = QtInteractor(self)
        layout.addWidget(self.plotter.interactor)
        self.plotter.enable_depth_peeling()
        self.plotter.add_axes()

        # Status / data
        self._worker_thread: Optional[QtCore.QThread] = None
        self._worker: Optional[CompoundWorker] = None  # keep a strong reference to worker
        self._last_grid: Optional[pv.ImageData] = None
        self._last_vol: Optional[np.ndarray] = None

    # ---------- slots ----------
    def on_browse(self):
        folder = QFileDialog.getExistingDirectory(self, "Select folder with masks + pose file", dir=directory)
        if folder:
            self.path_edit.setText(folder)

    def on_browse_pose(self):
        start_dir = self.path_edit.text().strip() or os.getcwd()
        path, _ = QFileDialog.getOpenFileName(self, "Select pose .txt", start_dir, "Text Files (*.txt);;All Files (*)")
        if path:
            self.pose_edit.setText(path)

    def on_run(self):
        folder = self.path_edit.text().strip()
        if not folder or not os.path.isdir(folder):
            QMessageBox.warning(self, "Input", "Please select a valid folder.")
            return

        pixel_spacing = float(self.spacing_spin.value())
        voxel = float(self.voxel_spin.value())
        write_mode = self.write_mode_combo.currentText()
        thickness_vox = int(self.thick_spin.value())

        # Prepare worker
        self.prog.setValue(0)
        self._worker_thread = QtCore.QThread(self)
        pose_path = self.pose_edit.text().strip() or None
        worker = CompoundWorker(folder, pixel_spacing, voxel, pose_path, write_mode, thickness_vox)
        self._worker = worker  # prevent GC
        worker.moveToThread(self._worker_thread)

        worker.progressed.connect(self.on_progress)
        worker.finished.connect(self.on_finished)
        self._worker_thread.started.connect(worker.run)
        worker.finished.connect(self._worker_thread.quit)
        worker.finished.connect(worker.deleteLater)
        self._worker_thread.finished.connect(self._worker_thread.deleteLater)
        self._worker_thread.finished.connect(lambda: setattr(self, "_worker", None))

        self._worker_thread.start()

    @QtCore.Slot(int, int)
    def on_progress(self, current: int, total: int):
        if total > 0:
            pct = int(100 * current / total)
            self.prog.setValue(pct)

    @QtCore.Slot(object, object, object)
    def on_finished(self, grid: Optional[pv.ImageData], vol: Optional[np.ndarray], error: Optional[Exception]):
        if error is not None:
            QMessageBox.critical(self, "Error", str(error))
            return

        self.plotter.clear()
        # Volume rendering; tune opacity as needed
        try:
            self.plotter.add_volume(grid, scalars="values", cmap="jet")
        except Exception:
            # Fallback to isosurface if volume rendering unsupported
            try:
                threshold = np.percentile(grid["values"], 75)
                surf = grid.threshold(value=threshold).extract_surface()
                self.plotter.add_mesh(surf, opacity=0.5)
            except Exception as e:
                QMessageBox.critical(self, "VTK Display Error", f"Failed to display volume: {e}")
                return

        self.plotter.reset_camera()
        self.plotter.render()
        self.plotter._last_grid = grid  # keep reference alive for plotter
        self._last_grid = grid
        self._last_vol = vol
        self.prog.setValue(100)

    # ---------- export helpers ----------
    def _require_grid(self) -> Optional[pv.ImageData]:
        grid = self._last_grid or getattr(self.plotter, "_last_grid", None)
        if grid is None:
            QMessageBox.information(self, "Export", "Please run compounding first.")
            return None
        return grid

    @QtCore.Slot()
    def on_save_nifti(self):
        grid = self._require_grid()
        if grid is None:
            return
        if nib is None:
            QMessageBox.critical(self, "NIfTI", "nibabel is not installed. Please install with 'pip install nibabel'.")
            return
        # Choose path
        path, _ = QFileDialog.getSaveFileName(self, "Save NIfTI", '/home/leofer/experiment/loop_x/ipcai2026/us_compounding', "NIfTI (*.nii *.nii.gz)")
        if not path:
            return
        try:
            # Get volume array. Prefer cached numpy volume; else reconstruct from grid scalars.
            if self._last_vol is not None:
                vol = self._last_vol
            else:
                # Rebuild ndarray from grid scalars (Fortran order)
                dims = grid.dimensions
                nx, ny, nz = dims
                scal = grid["values"]
                vol = np.asanyarray(scal).reshape((nx, ny, nz), order='F')

            sx, sy, sz = grid.spacing
            ox, oy, oz = grid.origin
            # Affine mapping voxel(i,j,k) -> world mm
            affine = np.array([
                [sx, 0,  0,  ox],
                [0,  sy, 0,  oy],
                [0,  0,  sz, oz],
                [0,  0,  0,  1 ],
            ], dtype=float)
            img = nib.Nifti1Image(vol, affine)
            # Set zooms in header for good measure
            try:
                hdr = img.header
                hdr.set_zooms((float(sx), float(sy), float(sz)))
            except Exception:
                pass
            nib.save(img, path)
            QMessageBox.information(self, "NIfTI", f"Saved NIfTI to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "NIfTI Save Error", str(e))

    @QtCore.Slot()
    def on_export_stl(self):
        grid = self._require_grid()
        if grid is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export STL", '/home/leofer/experiment/loop_x/ipcai2026/us_compounding', "STL Mesh (*.stl)")
        if not path:
            return
        try:
            # Choose a reasonable threshold: include all voxels > 0.5
            # Works for both binary and count volumes (any non-zero is kept)
            ugrid = grid.threshold(value=0.5)
            surf = ugrid.extract_surface()
            if surf.n_points == 0:
                QMessageBox.warning(self, "STL Export", "No surface found at threshold 0.5. Try a thicker splat or 'binary' mode.")
                return
            surf.save(path)
            QMessageBox.information(self, "STL Export", f"Saved STL to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "STL Export Error", str(e))
    
    @QtCore.Slot()
    def on_save_ply(self):
        grid = self._require_grid()
        if grid is None:
            return
        
        path, _ = QFileDialog.getSaveFileName(
            self, 
            "Save Point Cloud", 
            '/home/leofer/experiment/loop_x/ipcai2026/us_compounding', 
            "PLY Point Cloud (*.ply);;All Files (*)"
        )
        if not path:
            return

        try:
            ugrid = grid.threshold(value=0.5)
            
            
            point_cloud = ugrid.cell_centers()

            if point_cloud.n_points == 0:
                QMessageBox.warning(self, "PLY Export", "Please use splat or 'binary' model。")
                return
            
            point_cloud.save(path)
            QMessageBox.information(self, "PLY Export", f"Save PLY as:\n{path}")
        
        except Exception as e:
            QMessageBox.critical(self, "PLY Export Error", str(e))


# -----------------------------
# main
# -----------------------------
if __name__ == '__main__':
    directory = '/home/leofer/experiment/loop_x/ipcai2026/compounding'
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
