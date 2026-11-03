#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   spine_motion.py
@Time    :   2026/11/03 13:30:31
@Author  :   Feng Li 
@Contact :   feng.li@tum.de
@Desc    :   
'''

import sys
import os
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple
import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
import vtk

from PySide6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QSlider, QLabel, QFileDialog, QLineEdit, QFormLayout, QGroupBox, QMessageBox,
    QComboBox, QListWidget, QListWidgetItem, QCheckBox,
    QDoubleSpinBox
)
from PySide6.QtCore import Qt

# ==========================
# Data Model
# ==========================
@dataclass
class Vertebra:
    name: str
    mesh: pv.PolyData
    actor: object
    pivot: np.ndarray  # 3-vector
    parent: Optional[str] = None
    cur_rot_deg: float = 0.0
    cur_trans_val: float = 0.0


# ==========================
# Main Widget
# ==========================
class SpineViewer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spine Viewer")
        self.resize(1500, 950)

        # Scene
        self.plotter = QtInteractor(self)
        self.plotter.show_axes()
        self.plotter.add_axes()

        # Data
        self.vertebrae: Dict[str, Vertebra] = {}
        self.driver_name: Optional[str] = None
        self.ply_mesh: Optional[pv.PolyData] = None
        self.ply_actor: Optional[object] = None

        # Ground-truth PLY (for evaluation)
        self.gt_ply_mesh: Optional[pv.PolyData] = None
        self.gt_ply_actor: Optional[object] = None

        # Kinematic params (tunable)
        self.att_factors = [1.0, 0.7, 0.45, 0.05, 0.0]  # L1..L5 [1.0, 0.7, 0.45, 0.05, 0.0] [1.0, 0.4, 0.3, 0.07, 0.0]
        self.max_angle_deg = 30.0 # 30
        self.max_trans_mm  = 40.0 # 40

        # ICP step clamp (per-iteration refine)
        self.icp_step_max_angle_deg = 5
        self.icp_step_max_trans_mm  = 8
        # shrink each accepted ICP step to be conservative (0<scale<=1)
        self.icp_step_scale = 0.3  # e.g., only take 30% of each ICP delta 0.3
        # per-vertebra total cap per run (trust region)
        self.icp_total_cap_angle_deg = 5.0   # base cap per refine call (angle) 5.0
        self.icp_total_cap_trans_mm  = 5.0   # base cap per refine call (translation) 5.0
        # NEW: per-level multipliers (L1 largest → L4 smallest)
        # index = hierarchy depth from driver (0=L1, 1=L2, 2=L3, 3=L4, 4=L5)
        self.icp_level_step_mult   = [1.0, 0.8, 0.5, 0.2, 0.0]   # scales step clamp & step shrink [1.0, 0.8, 0.4, 0.2, 0.0]
        self.icp_level_total_mult  = [1.0, 0.8, 0.5, 0.2, 0.0]    # scales total trust-region caps [1.0, 0.8, 0.5, 0.1, 0.0] 

        # ICP axis bias (prefer these axes during optimization)
        self.icp_rot_axis_idx = 0   # 同 on_driver_motion 的 ROTATION_AXIS_IDX=1 (Y轴)  0=X, 1=Y, 2=Z   1
        self.icp_trans_axis_idx = 1 # 同 on_driver_motion 的 TRANSLATION_AXIS_IDX=0 (X轴)  0

        # 轴向偏置权重：平行分量权重大，正交分量可轻微保留(0~1)
        self.icp_rot_parallel_weight = 1.0
        self.icp_rot_orth_weight     = 0.2   # 设为 0 表示只允许绕主轴旋转 0.2
        self.icp_trans_parallel_weight = 1.0
        self.icp_trans_orth_weight     = 0.2 # 可留一点正交平移，避免局部极值 0.2

        # Undo stack for ICP runs (list of snapshots)
        self._icp_undo_stack = []

        # Subsampling params (tunable)
        self.voxel_size_mm = 0.5            # PLY voxel
        self.src_fps_k_l1 = 600             # Level-1 FPS for source
        self.tar_fps_k_l1 = 1200            # Level-1 FPS for target
        self.src_fps_k_l2 = 1500            # Level-2 FPS for source
        self.tar_fps_k_l2 = 3000            # Level-2 FPS for target

        # ICP iteration / stopping
        self.max_iter_l1 = 8
        self.max_iter_l2 = 10
        self.converge_eps_mm = 0.3

        # Picking (optional, disabled in UI)
        self.pick_marker = None
        self.picking_on = False
        self._picker_callback_ref = None

        # UI Controls
        self.btn_load_multi = QPushButton("Load STL(s)...")
        self.btn_load_multi.clicked.connect(self.load_multiple_stls)
        self.list_v = QListWidget()
        self.list_v.currentTextChanged.connect(self.on_select_vertebra)
        self.btn_set_driver = QPushButton("Set as Driver")
        self.btn_set_driver.clicked.connect(self.set_as_driver)

        # Driver controls
        self.lbl_motion = QLabel("Motion: 0.0")
        self.slider_motion = QSlider(Qt.Horizontal)
        self.slider_motion.setRange(-60, 60)
        self.slider_motion.setValue(0)
        self.slider_motion.valueChanged.connect(self.on_driver_motion)
        self.chk_propagate = QCheckBox("Propagate driver's delta to descendants")
        self.chk_propagate.setChecked(True)

        self.pivot_edit = QLineEdit("")
        self.pivot_edit.setPlaceholderText("x,y,z  (leave empty for model center)")
        self.btn_apply_pivot = QPushButton("Apply Pivot to Selected")
        self.btn_apply_pivot.clicked.connect(self.apply_pivot_to_selected)
        self.btn_pivot_center = QPushButton("Set Pivot = Mesh Center")
        self.btn_pivot_center.clicked.connect(self.set_pivot_center_for_selected)

        self.parent_combo = QComboBox()
        self.parent_combo.currentTextChanged.connect(self.on_parent_changed)
        self.lbl_parent = QLabel("Parent:")

        # PLY Controls
        self.btn_load_ply = QPushButton("Load PLY")
        self.btn_load_ply.clicked.connect(self.load_ply_file)
        self.ply_parent_combo = QComboBox()
        self.ply_parent_combo.currentTextChanged.connect(self.on_ply_pose_changed)
        self.ply_tx_spin = QDoubleSpinBox(decimals=3, minimum=-1000, maximum=1000, singleStep=0.1)
        self.ply_ty_spin = QDoubleSpinBox(decimals=3, minimum=-1000, maximum=1000, singleStep=0.1)
        self.ply_tz_spin = QDoubleSpinBox(decimals=3, minimum=-1000, maximum=1000, singleStep=0.1)
        self.ply_rx_spin = QDoubleSpinBox(decimals=3, minimum=-360, maximum=360, singleStep=0.5)
        self.ply_ry_spin = QDoubleSpinBox(decimals=3, minimum=-360, maximum=360, singleStep=0.5)
        self.ply_rz_spin = QDoubleSpinBox(decimals=3, minimum=-360, maximum=360, singleStep=0.5)
        for spin in [self.ply_tx_spin, self.ply_ty_spin, self.ply_tz_spin,
                     self.ply_rx_spin, self.ply_ry_spin, self.ply_rz_spin]:
            spin.valueChanged.connect(self.on_ply_pose_changed)
        self.btn_ply_reset = QPushButton("Reset PLY Pose")
        self.btn_ply_reset.clicked.connect(self.on_ply_reset)

        # NEW: Ground-truth 6D pose controls (side-by-side with PLY in the same rows)
        self.btn_load_gt = QPushButton("Load GT PLY...")
        self.btn_load_gt.clicked.connect(self.load_gt_ply_file)
        self.gt_parent_combo = QComboBox()
        self.gt_parent_combo.currentTextChanged.connect(self.on_gt_pose_changed)
        self.gt_tx_spin = QDoubleSpinBox(decimals=3, minimum=-1000, maximum=1000, singleStep=0.1)
        self.gt_ty_spin = QDoubleSpinBox(decimals=3, minimum=-1000, maximum=1000, singleStep=0.1)
        self.gt_tz_spin = QDoubleSpinBox(decimals=3, minimum=-1000, maximum=1000, singleStep=0.1)
        self.gt_rx_spin = QDoubleSpinBox(decimals=3, minimum=-360, maximum=360, singleStep=0.5)
        self.gt_ry_spin = QDoubleSpinBox(decimals=3, minimum=-360, maximum=360, singleStep=0.5)
        self.gt_rz_spin = QDoubleSpinBox(decimals=3, minimum=-360, maximum=360, singleStep=0.5)
        for spin in [self.gt_tx_spin, self.gt_ty_spin, self.gt_tz_spin,
                     self.gt_rx_spin, self.gt_ry_spin, self.gt_rz_spin]:
            spin.valueChanged.connect(self.on_gt_pose_changed)
        self.btn_gt_reset = QPushButton("Reset GT Pose")
        self.btn_gt_reset.clicked.connect(self.on_gt_reset)

        # ICP trigger
        self.btn_run_icp = QPushButton("Run Fine Registration (ICP)")
        self.btn_run_icp.clicked.connect(self.run_icp_registration)
        # NEW: Undo button
        self.btn_undo_icp = QPushButton("Undo Last ICP")
        self.btn_undo_icp.setEnabled(False)
        self.btn_undo_icp.clicked.connect(self.undo_last_icp)

        # ---- Print poses button ----
        self.btn_print_poses = QPushButton("Print All Vertebra Poses")
        self.btn_print_poses.clicked.connect(self.on_print_all_poses)
        

        left = QVBoxLayout()
        left.addWidget(self.btn_load_multi)
        left.addWidget(QLabel("Vertebrae:"))
        left.addWidget(self.list_v, 1)
        hl = QHBoxLayout()
        hl.addWidget(self.btn_set_driver)
        hl.addWidget(self.chk_propagate)
        left.addLayout(hl)

        par_box = QGroupBox("Hierarchy")
        par_form = QFormLayout()
        par_form.addRow(self.lbl_parent, self.parent_combo)
        par_box.setLayout(par_form)
        left.addWidget(par_box)

        ply_box = QGroupBox("Point Cloud Controls (PLY & GT side-by-side)")
        ply_form = QFormLayout()
        # Row: Load buttons
        row_load = QHBoxLayout(); row_load.addWidget(self.btn_load_ply); row_load.addWidget(self.btn_load_gt)
        ply_form.addRow(row_load)

        # NEW: Row: Remove buttons
        self.btn_remove_ply = QPushButton("Remove PLY")
        self.btn_remove_ply.clicked.connect(self.remove_ply)
        self.btn_remove_gt = QPushButton("Remove GT")
        self.btn_remove_gt.clicked.connect(self.remove_gt)
        row_remove = QHBoxLayout(); row_remove.addWidget(self.btn_remove_ply); row_remove.addWidget(self.btn_remove_gt)
        ply_form.addRow(row_remove)

        # Row: Parent combos (PLY | GT)
        row_parent = QHBoxLayout(); row_parent.addWidget(self.ply_parent_combo); row_parent.addWidget(self.gt_parent_combo)
        ply_form.addRow("Parent (PLY | GT):", row_parent)
        # Row: Trans X
        row_tx = QHBoxLayout(); row_tx.addWidget(self.ply_tx_spin); row_tx.addWidget(self.gt_tx_spin)
        ply_form.addRow("Trans X (PLY | GT):", row_tx)
        # Row: Trans Y
        row_ty = QHBoxLayout(); row_ty.addWidget(self.ply_ty_spin); row_ty.addWidget(self.gt_ty_spin)
        ply_form.addRow("Trans Y (PLY | GT):", row_ty)
        # Row: Trans Z
        row_tz = QHBoxLayout(); row_tz.addWidget(self.ply_tz_spin); row_tz.addWidget(self.gt_tz_spin)
        ply_form.addRow("Trans Z (PLY | GT):", row_tz)
        # Row: Rot X
        row_rx = QHBoxLayout(); row_rx.addWidget(self.ply_rx_spin); row_rx.addWidget(self.gt_rx_spin)
        ply_form.addRow("Rot X (PLY | GT):", row_rx)
        # Row: Rot Y
        row_ry = QHBoxLayout(); row_ry.addWidget(self.ply_ry_spin); row_ry.addWidget(self.gt_ry_spin)
        ply_form.addRow("Rot Y (PLY | GT):", row_ry)
        # Row: Rot Z
        row_rz = QHBoxLayout(); row_rz.addWidget(self.ply_rz_spin); row_rz.addWidget(self.gt_rz_spin)
        ply_form.addRow("Rot Z (PLY | GT):", row_rz)
        # Row: Reset buttons
        row_reset = QHBoxLayout(); row_reset.addWidget(self.btn_ply_reset); row_reset.addWidget(self.btn_gt_reset)
        ply_form.addRow(row_reset)
        ply_box.setLayout(ply_form)
        left.addWidget(ply_box)

        # NEW: Ground-truth controls box (keep evaluation controls only)
        gt_box = QGroupBox("Ground Truth Evaluation")
        gt_form = QHBoxLayout()
        self.btn_eval_gt = QPushButton("Evaluate vs GT")
        self.btn_eval_gt.clicked.connect(self.evaluate_against_gt)
        self.chk_auto_eval = QCheckBox("Auto-evaluate after ICP")
        self.chk_auto_eval.setChecked(True)
        gt_form.addWidget(self.btn_eval_gt)
        gt_form.addWidget(self.chk_auto_eval)
        gt_box.setLayout(gt_form)

        # NEW: Auto-tune att_factors toggle
        self.chk_auto_tune_att = QCheckBox("Auto-tune att_factors")
        self.chk_auto_tune_att.setChecked(True)  # 默认开启
        gt_form.addWidget(self.chk_auto_tune_att)

        left.addWidget(gt_box)

        ctrl_box = QGroupBox("Driver Controls (Lateral Bend Model)")
        ctrl_v = QVBoxLayout()
        ctrl_v.addWidget(self.lbl_motion)
        ctrl_v.addWidget(self.slider_motion)
        ctrl_box.setLayout(ctrl_v)
        left.addWidget(ctrl_box)

        left.addWidget(self.btn_run_icp)
        left.addWidget(self.btn_undo_icp)

        left.addWidget(self.btn_print_poses)

        left.addStretch(1)

        root = QHBoxLayout(self)
        root.addLayout(left, 0)
        root.addWidget(self.plotter.interactor, 1)

    # ---------- Utilities ----------
    def _descendants_of(self, name: str) -> List[str]:
        desc = []
        queue = [name]
        while queue:
            cur = queue.pop(0)
            for child_name, v in self.vertebrae.items():
                if v.parent == cur and child_name not in desc and child_name != name:
                    desc.append(child_name)
                    queue.append(child_name)
        return desc

    def _set_actor_origin(self, v: Vertebra):
        v.actor.SetOrigin(*v.pivot.tolist())

    # ---------- Loading ----------
    def load_multiple_stls(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Select STL files", stl_directory, "STL (*.stl)")
        if not paths:
            return
        for path in paths:
            try:
                mesh = pv.read(path)
                if mesh.n_faces == 0:
                    QMessageBox.warning(self, "Load warning", f"Empty mesh: {os.path.basename(path)}")
                    continue
                name = os.path.splitext(os.path.basename(path))[0]
                actor = self.plotter.add_mesh(mesh, color="lightgray", smooth_shading=True, show_edges=False, pickable=True)
                pivot = np.array(mesh.center)

                # TODO: pivot setup
                # pivot[1] = pivot[1] - 20
                v = Vertebra(name=name, mesh=mesh, actor=actor, pivot=pivot)
                self._set_actor_origin(v)
                self.vertebrae[name] = v
                self.list_v.addItem(QListWidgetItem(name))
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load {path}:\n{e}")
                return
        if self.driver_name is None and self.list_v.count() > 0:
            self.list_v.setCurrentRow(0)
            self.set_as_driver()
        self.update_parent_combo()
        self.update_ply_parent_combo()
        self.update_gt_parent_combo()
        self.plotter.reset_camera()
        self.plotter.render()

    def update_parent_combo(self):
        sel = self.list_v.currentItem().text() if self.list_v.currentItem() else None
        self.parent_combo.blockSignals(True)
        self.parent_combo.clear()
        self.parent_combo.addItem("<None>")
        for name in self.vertebrae.keys():
            if name != sel:
                self.parent_combo.addItem(name)
        if sel and sel in self.vertebrae:
            cur_parent = self.vertebrae[sel].parent
            if cur_parent is None:
                self.parent_combo.setCurrentText("<None>")
            else:
                self.parent_combo.setCurrentText(cur_parent)
        self.parent_combo.blockSignals(False)

    def load_ply_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select PLY file", ply_directory, "PLY (*.ply)")
        if not path:
            return
        if self.ply_actor:
            self.plotter.remove_actor(self.ply_actor)
        try:
            self.ply_mesh = pv.read(path)
            self.ply_actor = self.plotter.add_mesh(
                self.ply_mesh,
                style='points',
                color='cyan',
                render_points_as_spheres=True,
                point_size=5.0,
                pickable=False
            )
            self.on_ply_reset()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load {path}:\n{e}")
            self.ply_mesh = None
            self.ply_actor = None
        
        self.plotter.reset_camera()
        self.plotter.render()

    def load_gt_ply_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Ground-Truth PLY file", gt_directory, "PLY (*.ply)")
        if not path:
            return
        if self.gt_ply_actor:
            self.plotter.remove_actor(self.gt_ply_actor)
        try:
            self.gt_ply_mesh = pv.read(path)
            # Visualize in magenta for contrast
            self.gt_ply_actor = self.plotter.add_mesh(
                self.gt_ply_mesh,
                style='points',
                color='magenta',
                render_points_as_spheres=True,
                point_size=4.0,
                pickable=False
            )
            self.update_gt_parent_combo()
            self.on_gt_reset()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load GT {path}:{e}")
            self.gt_ply_mesh = None
            self.gt_ply_actor = None
        self.plotter.render()
    
    # ---------- Remove PLY / GT ----------
    def remove_ply(self):
        """Remove the currently loaded PLY (target) from the scene and clear state."""
        try:
            if self.ply_actor is not None:
                # remove
                self.plotter.remove_actor(self.ply_actor)
            # clear data
            self.ply_actor = None
            self.ply_mesh = None
            self.ply_parent_combo.blockSignals(True)
            self.ply_parent_combo.setCurrentText("<None>")
            self.ply_parent_combo.blockSignals(False)
            # render
            self.plotter.render()
        except Exception as e:
            QMessageBox.warning(self, "Remove PLY", f"移除 PLY 失败：\n{e}")

    def remove_gt(self):
        """Remove the currently loaded Ground-Truth PLY from the scene and clear state."""
        try:
            if self.gt_ply_actor is not None:
                self.plotter.remove_actor(self.gt_ply_actor)
            self.gt_ply_actor = None
            self.gt_ply_mesh = None
            if hasattr(self, 'gt_parent_combo'):
                self.gt_parent_combo.blockSignals(True)
                self.gt_parent_combo.setCurrentText("<None>")
                self.gt_parent_combo.blockSignals(False)
            self.plotter.render()
        except Exception as e:
            QMessageBox.warning(self, "Remove GT", f"移除 GT 失败：\n{e}")

    def update_ply_parent_combo(self):
        current_parent = self.ply_parent_combo.currentText()
        self.ply_parent_combo.blockSignals(True)
        self.ply_parent_combo.clear()
        self.ply_parent_combo.addItem("<None>")
        all_vertebrae = sorted(self.vertebrae.keys())
        self.ply_parent_combo.addItems(all_vertebrae)
        if current_parent in all_vertebrae:
            self.ply_parent_combo.setCurrentText(current_parent)
        else:
            self.ply_parent_combo.setCurrentText("<None>")
        self.ply_parent_combo.blockSignals(False)

    def update_gt_parent_combo(self):
        current_parent = self.gt_parent_combo.currentText() if hasattr(self, 'gt_parent_combo') else None
        if not hasattr(self, 'gt_parent_combo'):
            return
        self.gt_parent_combo.blockSignals(True)
        self.gt_parent_combo.clear()
        self.gt_parent_combo.addItem("<None>")
        all_vertebrae = sorted(self.vertebrae.keys())
        self.gt_parent_combo.addItems(all_vertebrae)
        if current_parent in all_vertebrae:
            self.gt_parent_combo.setCurrentText(current_parent)
        else:
            self.gt_parent_combo.setCurrentText("<None>")
        self.gt_parent_combo.blockSignals(False)

    # ---------- Selection & Params ----------
    def on_select_vertebra(self, name: str):
        if not name or name not in self.vertebrae:
            return
        v = self.vertebrae[name]
        self.pivot_edit.setText(f"{v.pivot[0]:.3f}, {v.pivot[1]:.3f}, {v.pivot[2]:.3f}")
        self.update_parent_combo()

    def set_as_driver(self):
        item = self.list_v.currentItem()
        if not item:
            return
        self.driver_name = item.text()
        v = self.vertebrae[self.driver_name]
        v.cur_rot_deg = 0.0
        v.cur_trans_val = 0.0
        self.slider_motion.blockSignals(True)
        self.slider_motion.setValue(0)
        self.slider_motion.blockSignals(False)
        self.lbl_motion.setText("Motion: 0.0")

    def on_parent_changed(self, parent_text: str):
        sel = self.list_v.currentItem().text() if self.list_v.currentItem() else None
        if not sel:
            return
        if parent_text == "<None>":
            self.vertebrae[sel].parent = None
        else:
            if parent_text == sel:
                return
            self.vertebrae[sel].parent = parent_text

    # ---------- Pivot ops ----------
    def apply_pivot_to_selected(self):
        item = self.list_v.currentItem()
        if not item:
            return
        name = item.text()
        v = self.vertebrae[name]
        txt = self.pivot_edit.text().strip()
        if not txt:
            v.pivot = np.array(v.mesh.center)
        else:
            try:
                parts = [float(x) for x in txt.replace(",", " ").split()]
                if len(parts) != 3:
                    raise ValueError("Need 3 numbers")
                v.pivot = np.array(parts, dtype=float)
            except Exception:
                QMessageBox.warning(self, "Parse error", "Pivot must be 'x,y,z'")
                return
        self._set_actor_origin(v)
        self.plotter.render()

    def set_pivot_center_for_selected(self):
        item = self.list_v.currentItem()
        if not item:
            return
        name = item.text()
        v = self.vertebrae[name]
        v.pivot = np.array(v.mesh.center)
        self._set_actor_origin(v)
        self.pivot_edit.setText(f"{v.pivot[0]:.3f}, {v.pivot[1]:.3f}, {v.pivot[2]:.3f}")
        self.plotter.render()

    # ---------- Driver transforms ----------
    def _propagate_motion_recursive(self, parent_name: str, delta_angle: float, delta_trans_vec: np.ndarray,
                                    rotation_axis_idx: int, attenuation_factors: List[float], level: int):
        for child_name, v in self.vertebrae.items():
            if v.parent == parent_name:
                factor = 0.0
                if (level - 1) < len(attenuation_factors):
                    factor = attenuation_factors[level - 1]
                if factor <= 0.0:
                    continue
                scaled_delta_angle = delta_angle * factor
                scaled_delta_trans_vec = delta_trans_vec * factor
                # TODO1: Inverse rotation and translation for specific levels.
                # if level == 2:
                #     scaled_delta_angle = - 2 * scaled_delta_angle
                #     scaled_delta_trans_vec = 3.5 * scaled_delta_trans_vec
                # if level == 1:
                #     scaled_delta_trans_vec = 3.5 * scaled_delta_trans_vec
                # if level == 0:
                #     scaled_delta_trans_vec = 2 * scaled_delta_trans_vec
                c = v
                self._set_actor_origin(c)
                if rotation_axis_idx == 0:
                    c.actor.RotateX(scaled_delta_angle)
                elif rotation_axis_idx == 1:
                    c.actor.RotateY(scaled_delta_angle)
                else:
                    c.actor.RotateZ(scaled_delta_angle)
                c.actor.AddPosition(*scaled_delta_trans_vec)
                self._propagate_motion_recursive(child_name, delta_angle, delta_trans_vec,
                                                 rotation_axis_idx, attenuation_factors, level + 1)

    def on_driver_motion(self, value: int):
        if not self.driver_name:
            return
        v = self.vertebrae[self.driver_name]
        self._set_actor_origin(v)
        # --- TODO： Define your kinetimic model ---
        ROTATION_AXIS_IDX = 1  # 0=X, 1=Y, 2=Z
        TRANSLATION_AXIS_IDX = 0 # 0=X, 1=Y, 2=Z
        SLIDER_MAX = 60.0
        MAX_ANGLE_DEG = float(self.max_angle_deg)
        MAX_TRANS_MM  = float(self.max_trans_mm)
        ATTENUATION_FACTORS = list(self.att_factors[1:])

        new_slider_val = float(value)
        old_slider_val = v.cur_trans_val
        delta_slider_val = new_slider_val - old_slider_val
        v.cur_trans_val = new_slider_val
        self.lbl_motion.setText(f"Motion: {new_slider_val:.1f}")
        if delta_slider_val == 0:
            self.on_ply_pose_changed()
            return
        delta_angle = (delta_slider_val / SLIDER_MAX) * MAX_ANGLE_DEG
        delta_trans = (delta_slider_val / SLIDER_MAX) * MAX_TRANS_MM
        delta_trans_vec = np.zeros(3)
        delta_trans_vec[TRANSLATION_AXIS_IDX] = delta_trans
        if ROTATION_AXIS_IDX == 0:
            v.actor.RotateX(delta_angle)
        elif ROTATION_AXIS_IDX == 1:
            v.actor.RotateY(delta_angle)
        else:
            v.actor.RotateZ(delta_angle)
        v.actor.AddPosition(*delta_trans_vec)
        if self.chk_propagate.isChecked():
            self._propagate_motion_recursive(self.driver_name, delta_angle, delta_trans_vec,
                                             ROTATION_AXIS_IDX, ATTENUATION_FACTORS, level=1)
        self.on_ply_pose_changed()

    # ---------- PLY Pose (6D) ----------
    def on_ply_reset(self):
        for spin in [self.ply_tx_spin, self.ply_ty_spin, self.ply_tz_spin,
                     self.ply_rx_spin, self.ply_ry_spin, self.ply_rz_spin]:
            spin.blockSignals(True)
        # TODO
        self.ply_tx_spin.setValue(-23.700) # -23.700 -392.498 -522.475 -90 -90 0
        self.ply_ty_spin.setValue(-392.498) # -39.700  -390.798 -530.475 -90 -88.600 0
        self.ply_tz_spin.setValue(-522.475)  
        self.ply_rx_spin.setValue(-90)  
        self.ply_ry_spin.setValue(-90)  
        self.ply_rz_spin.setValue(0.0)  
        for spin in [self.ply_tx_spin, self.ply_ty_spin, self.ply_tz_spin,
                     self.ply_rx_spin, self.ply_ry_spin, self.ply_rz_spin]:
            spin.blockSignals(False)
        self.on_ply_pose_changed()

    def on_gt_reset(self):
        # Default GT pose: identity (0 translation/rotation)
        for spin in [self.gt_tx_spin, self.gt_ty_spin, self.gt_tz_spin,
                     self.gt_rx_spin, self.gt_ry_spin, self.gt_rz_spin]:
            spin.blockSignals(True)
        self.gt_tx_spin.setValue(44.9)  # 44.9 -60.4 150.7 0.0 180 0.0    44.9 -58.4 152.7
        self.gt_ty_spin.setValue(-58.4) # 46.900  -60.4 150.7 0.0 180 -3.000
        self.gt_tz_spin.setValue(150.7)
        self.gt_rx_spin.setValue(0.0)
        self.gt_ry_spin.setValue(180.000)
        self.gt_rz_spin.setValue(0.0)
        for spin in [self.gt_tx_spin, self.gt_ty_spin, self.gt_tz_spin, 
                     self.gt_rx_spin, self.gt_ry_spin, self.gt_rz_spin]:
            spin.blockSignals(False)
        self.on_gt_pose_changed()

    def _vtk_matrix_to_ndarray(self, vtk_matrix) -> np.ndarray:
        arr = np.eye(4)
        if vtk_matrix is None:
            return arr
        for i in range(4):
            for j in range(4):
                arr[i, j] = vtk_matrix.GetElement(i, j)
        return arr

    def on_ply_pose_changed(self):
        if self.ply_actor is None:
            if self.plotter:
                self.plotter.render()
            return
        T_parent_world_vtk = vtk.vtkTransform()
        parent_name = self.ply_parent_combo.currentText()
        if parent_name != "<None>" and parent_name in self.vertebrae:
            parent_v = self.vertebrae[parent_name]
            T_parent_world_vtk.SetMatrix(parent_v.actor.GetMatrix())
        tx = self.ply_tx_spin.value(); ty = self.ply_ty_spin.value(); tz = self.ply_tz_spin.value()
        rx = self.ply_rx_spin.value(); ry = self.ply_ry_spin.value(); rz = self.ply_rz_spin.value()
        T_local_vtk = vtk.vtkTransform(); T_local_vtk.Translate(tx, ty, tz); T_local_vtk.RotateZ(rz); T_local_vtk.RotateY(ry); T_local_vtk.RotateX(rx)
        T_final_vtk = vtk.vtkTransform(); T_final_vtk.DeepCopy(T_parent_world_vtk); T_final_vtk.Concatenate(T_local_vtk)
        self.ply_actor.SetUserTransform(T_final_vtk)
        self.plotter.render()

    def on_gt_pose_changed(self):
        if self.gt_ply_actor is None:
            if self.plotter:
                self.plotter.render()
            return
        T_parent_world_vtk = vtk.vtkTransform()
        parent_name = self.gt_parent_combo.currentText() if hasattr(self, 'gt_parent_combo') else "<None>"
        if parent_name != "<None>" and parent_name in self.vertebrae:
            parent_v = self.vertebrae[parent_name]
            T_parent_world_vtk.SetMatrix(parent_v.actor.GetMatrix())
        tx = self.gt_tx_spin.value(); ty = self.gt_ty_spin.value(); tz = self.gt_tz_spin.value()
        rx = self.gt_rx_spin.value(); ry = self.gt_ry_spin.value(); rz = self.gt_rz_spin.value()
        T_local_vtk = vtk.vtkTransform(); T_local_vtk.Translate(tx, ty, tz); T_local_vtk.RotateZ(rz); T_local_vtk.RotateY(ry); T_local_vtk.RotateX(rx)
        T_final_vtk = vtk.vtkTransform(); T_final_vtk.DeepCopy(T_parent_world_vtk); T_final_vtk.Concatenate(T_local_vtk)
        self.gt_ply_actor.SetUserTransform(T_final_vtk)
        self.plotter.render()

    # ======= Utils: matrices/points =======
    def _axis_from_idx(self, idx: int) -> np.ndarray:
        if idx == 0: return np.array([1.0, 0.0, 0.0])
        if idx == 1: return np.array([0.0, 1.0, 0.0])
        return np.array([0.0, 0.0, 1.0])

    def _rotmat_from_axis_angle(self, axis: np.ndarray, angle: float) -> np.ndarray:
        axis = np.asarray(axis, dtype=float)
        n = np.linalg.norm(axis)
        if n < 1e-9 or abs(angle) < 1e-12:
            return np.eye(3)
        ux, uy, uz = axis / n
        c, s = np.cos(angle), np.sin(angle)
        return np.array([
            [c+ux*ux*(1-c),    ux*uy*(1-c)-uz*s, ux*uz*(1-c)+uy*s],
            [uy*ux*(1-c)+uz*s, c+uy*uy*(1-c),    uy*uz*(1-c)-ux*s],
            [uz*ux*(1-c)-uy*s, uz*uy*(1-c)+ux*s, c+uz*uz*(1-c)]
        ])

    def _bias_delta_to_axes(self, Delta: np.ndarray,
                            rot_axis_idx: int, trans_axis_idx: int,
                            rot_par_w: float, rot_orth_w: float,
                            trans_par_w: float, trans_orth_w: float) -> np.ndarray:
        """
        将刚体增量 Delta 的旋转/平移按指定主轴进行偏置。
        - 旋转：分解为主轴和平面正交分量，分别按权重缩放角度后再组合。
        - 平移：按主轴方向分解并加权。
        """
        R = Delta[:3, :3].copy()
        t = Delta[:3, 3].copy()

        # --- translation bias ---
        a_t = self._axis_from_idx(trans_axis_idx)
        a_t = a_t / np.linalg.norm(a_t)
        t_par = np.dot(t, a_t) * a_t
        t_orth = t - t_par
        t_biased = trans_par_w * t_par + trans_orth_w * t_orth

        # --- rotation bias ---
        # 提取轴角
        tr = (np.trace(R) - 1.0) * 0.5
        tr = float(np.clip(tr, -1.0, 1.0))
        angle = float(np.arccos(tr))
        if angle < 1e-12:
            R_biased = np.eye(3)
        else:
            rx = R[2,1] - R[1,2]
            ry = R[0,2] - R[2,0]
            rz = R[1,0] - R[0,1]
            axis = np.array([rx, ry, rz])
            n = np.linalg.norm(axis)
            if n < 1e-12:
                R_biased = np.eye(3)
            else:
                u = axis / n
                a_r = self._axis_from_idx(rot_axis_idx)
                a_r = a_r / np.linalg.norm(a_r)

                # u 在主轴方向上的分量强度
                coef_p = abs(np.dot(u, a_r))
                coef_o = float(np.sqrt(max(0.0, 1.0 - coef_p*coef_p)))

                # 平行分量：绕主轴 a_r 旋转 angle_p
                sign_par = np.sign(np.dot(u, a_r)) if coef_p > 1e-9 else 1.0
                angle_p = angle * rot_par_w * coef_p
                R_par = self._rotmat_from_axis_angle(sign_par * a_r, angle_p)

                # 正交分量：绕 u 的正交方向旋转 angle_o
                if coef_o < 1e-9 or rot_orth_w <= 0.0:
                    R_orth = np.eye(3)
                else:
                    u_perp = u - np.dot(u, a_r) * a_r
                    u_perp_n = np.linalg.norm(u_perp)
                    if u_perp_n < 1e-12:
                        R_orth = np.eye(3)
                    else:
                        u_perp /= u_perp_n
                        angle_o = angle * rot_orth_w * coef_o
                        R_orth = self._rotmat_from_axis_angle(u_perp, angle_o)

                # 组合（先平行后正交，或反之差别很小）
                R_biased = R_orth @ R_par

        Out = np.eye(4)
        Out[:3, :3] = R_biased
        Out[:3, 3] = t_biased
        return Out
    
    def on_print_all_poses(self):
        """Print current 4x4 world pose of each vertebra."""
        print("\n========== Current Vertebra World Poses ==========")
        for name, v in self.vertebrae.items():
            M = self._actor_world_mat_4x4(v.actor)
            tx, ty, tz = M[0, 3], M[1, 3], M[2, 3]

            # 提取欧拉角 (XYZ 顺序)
            sy = np.sqrt(M[0, 0]**2 + M[1, 0]**2)
            singular = sy < 1e-8
            if not singular:
                rx = np.degrees(np.arctan2(M[2, 1], M[2, 2]))
                ry = np.degrees(np.arctan2(-M[2, 0], sy))
                rz = np.degrees(np.arctan2(M[1, 0], M[0, 0]))
            else:
                rx = np.degrees(np.arctan2(-M[1, 2], M[1, 1]))
                ry = np.degrees(np.arctan2(-M[2, 0], sy))
                rz = 0.0

            print(f"[{name}]")
            print(f"  Translation (mm): [{tx:8.3f}, {ty:8.3f}, {tz:8.3f}]")
            print(f"  Rotation (deg XYZ): [{rx:7.3f}, {ry:7.3f}, {rz:7.3f}]")
            print("  4x4 Matrix:")
            print(np.array2string(M, formatter={'float_kind': lambda x: f'{x:8.3f}'}))
            print()
        print("==================================================\n")
    
    def _actor_world_mat_4x4(self, actor) -> np.ndarray:
        m = actor.GetMatrix(); M = np.eye(4)
        for i in range(4):
            for j in range(4):
                M[i, j] = m.GetElement(i, j)
        return M

    def _poly_points(self, poly: pv.PolyData) -> np.ndarray:
        return np.asarray(poly.points)

    def _np_points_to_vtk_poly(self, pts_np: np.ndarray):
        poly = vtk.vtkPolyData()
        if pts_np is None or pts_np.size == 0:
            poly.SetPoints(vtk.vtkPoints()); return poly
        vtk_pts = vtk.vtkPoints(); vtk_pts.SetNumberOfPoints(len(pts_np))
        for i, p in enumerate(pts_np):
            vtk_pts.SetPoint(i, float(p[0]), float(p[1]), float(p[2]))
        poly.SetPoints(vtk_pts)
        verts = vtk.vtkCellArray()
        for i in range(len(pts_np)):
            verts.InsertNextCell(1); verts.InsertCellPoint(i)
        poly.SetVerts(verts)
        return poly

    def _transform_points(self, pts: np.ndarray, M4: np.ndarray) -> np.ndarray:
        if pts.size == 0:
            return pts
        homo = np.c_[pts, np.ones((pts.shape[0], 1))]
        out = (M4 @ homo.T).T[:, :3]
        return out

    # ======= Sampling helpers =======
    def _voxel_downsample(self, pts: np.ndarray, voxel: float) -> np.ndarray:
        if pts.shape[0] == 0: return pts
        key = np.floor(pts / float(voxel))
        _, idx = np.unique(key, axis=0, return_index=True)
        return pts[np.sort(idx)]

    def _farthest_point_sample(self, pts: np.ndarray, k: int) -> np.ndarray:
        N = pts.shape[0]
        if N == 0 or k >= N:
            return pts
        sel = np.zeros(k, dtype=int)
        sel[0] = np.random.randint(0, N)
        d2 = np.full(N, np.inf, dtype=float)
        last = pts[sel[0]]
        for i in range(1, k):
            diff = pts - last
            d2 = np.minimum(d2, np.einsum('ij,ij->i', diff, diff))
            sel[i] = int(np.argmax(d2))
            last = pts[sel[i]]
        return pts[sel]

    # ======= PLY subset with cache =======
    def _ply_world_and_tree(self) -> Tuple[np.ndarray, object]:
        # Build PLY world coords from actor's UserTransform and cache KDTree
        T = np.eye(4)
        if self.ply_actor is not None and self.ply_actor.GetUserTransform() is not None:
            M = self.ply_actor.GetUserTransform().GetMatrix()
            for i in range(4):
                for j in range(4):
                    T[i, j] = M.GetElement(i, j)
        ply_local = np.asarray(self.ply_mesh.points) if self.ply_mesh is not None else np.empty((0,3))
        ply_world_full = self._transform_points(ply_local, T)
        # Voxel downsample for speed
        ply_world = self._voxel_downsample(ply_world_full, voxel=self.voxel_size_mm)
        from scipy.spatial import cKDTree
        tree = cKDTree(ply_world) if ply_world.shape[0] > 0 else None
        return ply_world, tree

    def _gt_world_points(self) -> np.ndarray:
        if self.gt_ply_mesh is None:
            return np.empty((0,3))
        # Assume GT actor has no extra user transform (already in world). If it does, respect it.
        T = np.eye(4)
        if self.gt_ply_actor is not None and self.gt_ply_actor.GetUserTransform() is not None:
            M = self.gt_ply_actor.GetUserTransform().GetMatrix()
            for i in range(4):
                for j in range(4):
                    T[i, j] = M.GetElement(i, j)
        gt_local = np.asarray(self.gt_ply_mesh.points)
        return self._transform_points(gt_local, T)

    def _ply_subset_near_mesh(self, mesh_world_pts: np.ndarray, tree, ply_world: np.ndarray,
                              radius_mm: float, k_limit: int) -> np.ndarray:
        from scipy.spatial import cKDTree
        if tree is None or ply_world.shape[0] == 0:
            return np.empty((0,3))
        center = mesh_world_pts.mean(axis=0)
        idxs = tree.query_ball_point(center, r=float(radius_mm))
        subset = ply_world[idxs] if len(idxs) > 0 else np.empty((0, 3))
        if subset.shape[0] > k_limit:
            # Use FPS for better coverage
            subset = self._farthest_point_sample(subset, k_limit)
        return subset

    # ======= VTK rigid ICP =======
    def _vtk_icp_rigid_delta(self, source_np: np.ndarray, target_np: np.ndarray) -> np.ndarray:
        if source_np.shape[0] < 50 or target_np.shape[0] < 50:
            return np.eye(4)
        src_poly = self._np_points_to_vtk_poly(source_np)
        tar_poly = self._np_points_to_vtk_poly(target_np)
        icp = vtk.vtkIterativeClosestPointTransform()
        icp.SetSource(src_poly)
        icp.SetTarget(tar_poly)
        icp.GetLandmarkTransform().SetModeToRigidBody()
        icp.SetMaximumNumberOfIterations(25)
        icp.SetMaximumMeanDistance(0.1)
        icp.CheckMeanDistanceOn()
        icp.StartByMatchingCentroidsOn()
        icp.Modified(); icp.Update()
        M = icp.GetMatrix(); Delta = np.eye(4)
        for i in range(4):
            for j in range(4):
                Delta[i, j] = M.GetElement(i, j)
        return Delta

    def _clamp_delta_rigid(self, Delta: np.ndarray, max_angle_deg: float, max_trans_mm: float) -> np.ndarray:
        R = Delta[:3, :3]; t = Delta[:3, 3]
        tr = (np.trace(R) - 1.0) * 0.5; tr = np.clip(tr, -1.0, 1.0)
        angle = np.degrees(np.arccos(tr))
        if angle > 1e-6:
            rx = R[2,1] - R[1,2]; ry = R[0,2] - R[2,0]; rz = R[1,0] - R[0,1]
            axis = np.array([rx, ry, rz], dtype=float); n = np.linalg.norm(axis)
            axis = axis / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
        else:
            axis = np.array([1.0, 0.0, 0.0]); angle = 0.0
        angle_c = float(np.clip(angle, -max_angle_deg, max_angle_deg))
        tnorm = np.linalg.norm(t)
        t_c = t if tnorm < 1e-9 else t * min(1.0, max_trans_mm / tnorm)
        th = np.radians(angle_c); ux, uy, uz = axis; c, s = np.cos(th), np.sin(th)
        Rc = np.array([[c+ux*ux*(1-c),    ux*uy*(1-c)-uz*s, ux*uz*(1-c)+uy*s],
                       [uy*ux*(1-c)+uz*s, c+uy*uy*(1-c),    uy*uz*(1-c)-ux*s],
                       [uz*ux*(1-c)-uy*s, uz*uy*(1-c)+ux*s, c+uz*uz*(1-c)]])
        Out = np.eye(4); Out[:3, :3] = Rc; Out[:3, 3] = t_c
        return Out

    # NEW: delta utilities (metrics + shrink)
    def _delta_metrics(self, Delta: np.ndarray) -> tuple[float, float]:
        """Return (angle_deg, trans_mm) of a rigid 4x4."""
        R = Delta[:3,:3]; t = Delta[:3,3]
        tr = (np.trace(R) - 1.0) * 0.5
        tr = np.clip(tr, -1.0, 1.0)
        angle = float(np.degrees(np.arccos(tr)))
        tnorm = float(np.linalg.norm(t))
        return angle, tnorm

    def _shrink_delta(self, Delta: np.ndarray, scale: float) -> np.ndarray:
        """Exponentially shrink a rigid transform: angle*=scale, translation*=scale."""
        scale = float(np.clip(scale, 0.0, 1.0))
        R = Delta[:3,:3]; t = Delta[:3,3]
        # axis-angle from R
        tr = (np.trace(R) - 1.0) * 0.5
        tr = np.clip(tr, -1.0, 1.0)
        angle = np.arccos(tr)
        if angle > 1e-6:
            rx = R[2,1] - R[1,2]; ry = R[0,2] - R[2,0]; rz = R[1,0] - R[0,1]
            axis = np.array([rx, ry, rz]); n = np.linalg.norm(axis)
            axis = axis / n if n > 1e-9 else np.array([1.0,0.0,0.0])
        else:
            axis = np.array([1.0,0.0,0.0]); angle = 0.0
        th = angle * scale
        ux, uy, uz = axis; c, s = np.cos(th), np.sin(th)
        Rs = np.array([[c+ux*ux*(1-c),    ux*uy*(1-c)-uz*s, ux*uz*(1-c)+uy*s],
                       [uy*ux*(1-c)+uz*s, c+uy*uy*(1-c),    uy*uz*(1-c)-ux*s],
                       [uz*ux*(1-c)-uy*s, uz*uy*(1-c)+ux*s, c+uz*uz*(1-c)]])
        Out = np.eye(4)
        Out[:3,:3] = Rs
        Out[:3, 3] = t * scale
        return Out

    def _depth_from_driver(self, node_name: str) -> int:
        if self.driver_name is None or node_name not in self.vertebrae:
            return 0
        d, cur = 0, node_name
        while self.vertebrae[cur].parent is not None:
            cur = self.vertebrae[cur].parent; d += 1
        return d

    # ======= MAIN: fast auto-subset ICP =======
    def _snapshot_world_mats(self) -> dict:
        snap = {}
        for name, v in self.vertebrae.items():
            M = self._actor_world_mat_4x4(v.actor)
            snap[name] = M.copy()
        return snap

    def _apply_world_mats(self, snap: dict):
        for name, M in snap.items():
            if name in self.vertebrae:
                Tmat = vtk.vtkMatrix4x4()
                for i in range(4):
                    for j in range(4):
                        Tmat.SetElement(i, j, M[i, j])
                self.vertebrae[name].actor.PokeMatrix(Tmat)
        self.plotter.render()

    def undo_last_icp(self):
        if not self._icp_undo_stack:
            return
        snap = self._icp_undo_stack.pop()
        self._apply_world_mats(snap)
        if not self._icp_undo_stack:
            self.btn_undo_icp.setEnabled(False)

    def run_icp_registration(self):
        if self.ply_mesh is None or self.ply_actor is None:
            QMessageBox.warning(self, "ICP Error", "Please load PLY."); return
        if len(self.vertebrae) == 0:
            QMessageBox.warning(self, "ICP Error", "Please load STL."); return

        # Only driver & descendants for speed (change to list(self.vertebrae.keys()) if needed)
        names = [self.driver_name] + self._descendants_of(self.driver_name) if self.driver_name else list(self.vertebrae.keys())

        # Snapshot for undo
        snap = self._snapshot_world_mats()

        # Build PLY cache once (world coords + KDTree), with voxel downsample
        ply_world, tree = self._ply_world_and_tree()
        from scipy.spatial import cKDTree

        for name in names:
            v = self.vertebrae[name]
            M_world = self._actor_world_mat_4x4(v.actor)
            src_full = self._transform_points(self._poly_points(v.mesh), M_world)
            # Multi-scale FPS on source
            src_l1 = self._farthest_point_sample(src_full, k=self.src_fps_k_l1)
            src_l2 = self._farthest_point_sample(src_full, k=self.src_fps_k_l2)

            # Neighborhood radius from bbox diagonal (tighter)
            b = v.mesh.bounds
            diag = np.linalg.norm([b[1]-b[0], b[3]-b[2], b[5]-b[4]])
            radius = max(5.0, 1.0 * float(diag))

            # Target subsets via cache + FPS cap
            tar_l1 = self._ply_subset_near_mesh(src_l1, tree, ply_world, radius_mm=radius, k_limit=self.tar_fps_k_l1)
            tar_l2 = self._ply_subset_near_mesh(src_l2, tree, ply_world, radius_mm=radius, k_limit=self.tar_fps_k_l2)
            if tar_l2.shape[0] < 200 or src_l2.shape[0] < 200:
                continue

            # Hierarchy-based scaling: L1 largest → L4 smallest
            level = self._depth_from_driver(name)
            step_mult = self.icp_level_step_mult[level] if level < len(self.icp_level_step_mult) else self.icp_level_step_mult[-1]
            total_mult = self.icp_level_total_mult[level] if level < len(self.icp_level_total_mult) else self.icp_level_total_mult[-1]

            # Per-step clamp (base from att_factors + per-level step multiplier)
            att = self.att_factors[level] if level < len(self.att_factors) else 0.0
            base_max_a = max(0.2, att) * float(self.icp_step_max_angle_deg)
            base_max_t = max(0.2, att) * float(self.icp_step_max_trans_mm)
            max_a = base_max_a * step_mult
            max_t = base_max_t * step_mult

            # Step shrink & total caps
            step_scale = float(self.icp_step_scale) * step_mult
            total_cap_a = float(self.icp_total_cap_angle_deg) * total_mult
            total_cap_t = float(self.icp_total_cap_trans_mm) * total_mult

            cur_M = M_world.copy()
            cum_angle, cum_trans = 0.0, 0.0

            # Level-1: quick coarse with tiny sets
            for _ in range(self.max_iter_l1):
                Delta = self._vtk_icp_rigid_delta(src_l1, tar_l1)
                Delta_c = self._clamp_delta_rigid(Delta, max_a, max_t)

                # NEW: 轴向偏置（优先沿选定轴优化）
                Delta_b = self._bias_delta_to_axes(
                    Delta_c,
                    self.icp_rot_axis_idx, self.icp_trans_axis_idx,
                    self.icp_rot_parallel_weight, self.icp_rot_orth_weight,
                    self.icp_trans_parallel_weight, self.icp_trans_orth_weight
                )

                Delta_s = self._shrink_delta(Delta_b, step_scale)
                inc_a, inc_t = self._delta_metrics(Delta_s)
                rem_a = max(0.0, total_cap_a - cum_angle)
                rem_t = max(0.0, total_cap_t - cum_trans)
                if inc_a > 0 or inc_t > 0:
                    s_a = 1.0 if inc_a <= rem_a else (rem_a / inc_a if inc_a > 1e-9 else 0.0)
                    s_t = 1.0 if inc_t <= rem_t else (rem_t / inc_t if inc_t > 1e-9 else 0.0)
                    s = min(s_a, s_t)
                    if s < 1.0:
                        Delta_s = self._shrink_delta(Delta_s, s)
                        inc_a *= s; inc_t *= s
                cur_M = Delta_s @ cur_M
                cum_angle += inc_a; cum_trans += inc_t
                if cum_angle >= total_cap_a or cum_trans >= total_cap_t:
                    break
                dists, _ = cKDTree(tar_l1).query(src_l1, k=1, workers=-1)
                if float(np.mean(np.clip(dists, 0, 5.0))) < self.converge_eps_mm:
                    break

            # Level-2: fine
            for _ in range(self.max_iter_l2):
                Delta = self._vtk_icp_rigid_delta(src_l2, tar_l2)
                Delta_c = self._clamp_delta_rigid(Delta, max_a, max_t)
                Delta_s = self._shrink_delta(Delta_c, step_scale)
                inc_a, inc_t = self._delta_metrics(Delta_s)
                rem_a = max(0.0, total_cap_a - cum_angle)
                rem_t = max(0.0, total_cap_t - cum_trans)
                if inc_a > 0 or inc_t > 0:
                    s_a = 1.0 if inc_a <= rem_a else (rem_a / inc_a if inc_a > 1e-9 else 0.0)
                    s_t = 1.0 if inc_t <= rem_t else (rem_t / inc_t if inc_t > 1e-9 else 0.0)
                    s = min(s_a, s_t)
                    if s < 1.0:
                        Delta_s = self._shrink_delta(Delta_s, s)
                        inc_a *= s; inc_t *= s
                cur_M = Delta_s @ cur_M
                cum_angle += inc_a; cum_trans += inc_t
                if cum_angle >= total_cap_a or cum_trans >= total_cap_t:
                    break
                dists, _ = cKDTree(tar_l2).query(src_l2, k=1, workers=-1)
                if float(np.mean(np.clip(dists, 0, 5.0))) < self.converge_eps_mm:
                    break

            # Apply final matrix
            Tmat = vtk.vtkMatrix4x4()
            for i in range(4):
                for j in range(4):
                    Tmat.SetElement(i, j, cur_M[i, j])
            v.actor.PokeMatrix(Tmat)

        # Push undo snapshot and enable button
        self._icp_undo_stack.append(snap)
        self.btn_undo_icp.setEnabled(True)

        self.plotter.render()

        # Auto-evaluate against GT if requested
        if self.chk_auto_eval.isChecked() and self.gt_ply_mesh is not None:
            metrics = self.evaluate_against_gt()  # returns dict
            if self.chk_auto_tune_att.isChecked() and metrics is not None:
                self._auto_tune_att_factors(metrics)

    # ======= EVALUATION vs GROUND TRUTH =======
    def _merged_vertebra_world_points(self, voxel_mm: float = 1.5, fps_k: int = 40000) -> np.ndarray:
        """Collect all vertebra meshes in WORLD coordinates, then downsample."""
        all_pts = []
        for v in self.vertebrae.values():
            M_world = self._actor_world_mat_4x4(v.actor)
            pts = self._transform_points(self._poly_points(v.mesh), M_world)
            all_pts.append(pts)
        if not all_pts:
            return np.empty((0,3))
        merged = np.concatenate(all_pts, axis=0)
        if merged.shape[0] == 0:
            return merged
        merged = self._voxel_downsample(merged, voxel=voxel_mm)
        # cap by FPS to avoid excessive memory
        if merged.shape[0] > fps_k:
            merged = self._farthest_point_sample(merged, fps_k)
        return merged

    def _dist_stats(self, d: np.ndarray) -> dict:
        d = np.asarray(d).astype(float)
        if d.size == 0:
            return {"mean": np.nan, "median": np.nan, "p95": np.nan, "max": np.nan}
        return {
            "mean": float(np.mean(d)),
            "median": float(np.median(d)),
            "p95": float(np.percentile(d, 95)),
            "max": float(np.max(d)),
        }

    def _compute_eval_metrics(self) -> Optional[dict]:
        """Return evaluation metrics dict or None if not computable.
        Keys:
          d_s2g, d_g2s: np.ndarray of one-sided NN distances
          stats_s2g, stats_g2s: dict(mean, median, p95, max)
          chamfer_mean: float
          hd95: float  (symmetric, = max(p95_s2g, p95_g2s))
        """
        if self.gt_ply_mesh is None:
            return None
        src_world = self._merged_vertebra_world_points(voxel_mm=1.5, fps_k=60000)
        gt_world  = self._gt_world_points()
        if src_world.shape[0] == 0 or gt_world.shape[0] == 0:
            return None
        from scipy.spatial import cKDTree
        tree_gt  = cKDTree(gt_world)
        tree_src = cKDTree(src_world)
        d_s2g, _ = tree_gt.query(src_world, k=1, workers=-1)
        d_g2s, _ = tree_src.query(gt_world,  k=1, workers=-1)
        stats_s2g = self._dist_stats(d_s2g)
        stats_g2s = self._dist_stats(d_g2s)
        chamfer_mean = float(np.mean([stats_s2g["mean"], stats_g2s["mean"]]))
        hd95 = float(max(stats_s2g["p95"], stats_g2s["p95"]))
        return {
            "d_s2g": d_s2g, "d_g2s": d_g2s,
            "stats_s2g": stats_s2g, "stats_g2s": stats_g2s,
            "chamfer_mean": chamfer_mean, "hd95": hd95
        }
    
    def evaluate_against_gt(self):
        metrics = self._compute_eval_metrics()
        if metrics is None:
            QMessageBox.warning(self, "Evaluate", "点数量不足或未加载 GT。")
            return

        stats_s2g = metrics["stats_s2g"]
        stats_g2s = metrics["stats_g2s"]
        chamfer_mean = metrics["chamfer_mean"]
        hd95 = metrics["hd95"]
        hausdorff = max(stats_s2g["max"], stats_g2s["max"])  # 你的“最大 HD”保持不变

        msg = (
            f"Chamfer (mean of means): {chamfer_mean:.2f} mm\n"
            f"S→GT  mean/median/p95/max: {stats_s2g['mean']:.2f} / {stats_s2g['median']:.2f} / {stats_s2g['p95']:.2f} / {stats_s2g['max']:.2f} mm\n"
            f"GT→S  mean/median/p95/max: {stats_g2s['mean']:.2f} / {stats_g2s['median']:.2f} / {stats_g2s['p95']:.2f} / {stats_g2s['max']:.2f} mm\n"
            f"HD95 (symmetric): {hd95:.2f} mm\n"
            f"Hausdorff (max of two): {hausdorff:.2f} mm"
        )
        print("\n[Evaluation vs GT]\n" + msg)
        self.plotter.add_text("Eval vs GT (mm)\n" + msg.replace(" mm", ""), position='upper_left', font_size=10, name='eval_text')
        QMessageBox.information(self, "Evaluation vs Ground Truth", msg)
        return metrics  # NEW: 便于 run_icp_registration 后拿到指标
    
    def _auto_tune_att_factors(self, metrics: dict):
        """
        根据评估结果自适应微调 att_factors（主要调 L2/L3，轻调 L4；L1/L5固定）。
        - 目标/阈值可按需要调整：US↔CT/CBCT 的经验阈值见注释。
        - 使用小步长 + EMA 平滑，避免震荡。
        """
        if not isinstance(metrics, dict):
            return

        chamfer = float(metrics.get("chamfer_mean", np.nan))
        hd95    = float(metrics.get("hd95", np.nan))
        if not np.isfinite(chamfer) or not np.isfinite(hd95):
            return

        # --- 超参数（可按项目微调） ---
        # 目标区间（US↔CT/CBCT 常见情况）
        CH_GOOD = 1.5   # mm 以内认为对齐较好
        CH_BAD  = 3.0   # mm 以上需要放宽
        HD_GOOD = 4.0   # mm 以内较好
        HD_BAD  = 6.0   # mm 以上需要放宽

        # 学习率/平滑
        lr = 0.12       # 单次调整步长（小）
        ema = 0.7       # 指数平滑：new = ema*old + (1-ema)*proposed

        # 每层权重（调参力度），L1/L5 不动
        layer_weights = [0.0, 0.9, 0.6, 0.3, 0.0]  # L2最大，其次 L3，L4很小

        # 计算“放宽/收紧”信号：[-1, +1]
        def score(x, good, bad):
            if x <= good: return -1.0  # 收紧
            if x >= bad:  return +1.0  # 放宽
            # 线性插值
            return (x - good) / max(1e-6, (bad - good)) * 2.0 - 1.0  # good→bad 映射到 [-1, +1]

        s_ch = score(chamfer, CH_GOOD, CH_BAD)
        s_hd = score(hd95,    HD_GOOD, HD_BAD)
        s = 0.6 * s_ch + 0.4 * s_hd  # 综合打分（可改权重）

        # 当前向量
        att = list(self.att_factors)
        att0 = att.copy()

        # 对 L2/L3/L4 做小幅增减，并限幅到 [0,1]
        for i in range(len(att)):
            w = layer_weights[i] if i < len(layer_weights) else 0.0
            if w <= 0.0:
                continue
            delta = lr * w * s
            proposed = att[i] + delta
            proposed = float(np.clip(proposed, 0.0, 1.0))
            # EMA 平滑
            att[i] = float(ema * att[i] + (1.0 - ema) * proposed)

        # 强制 L1=1.0, L5=0.0 不变（若有 L5）
        if len(att) >= 1:
            att[0] = 1.0
        if len(att) >= 5:
            att[4] = 0.0

        self.att_factors = att

        print(f"[Auto-tune] Chamfer={chamfer:.2f} mm, HD95={hd95:.2f} mm")
        print(f"[Auto-tune] att_factors: {att0}  ->  {self.att_factors}")

# ==========================
# Main
# ==========================
def main():
    pv.set_jupyter_backend(None)
    app = QApplication(sys.argv)
    w = SpineViewer()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    # You can edit these two defaults for your environment
    stl_directory = '/home/leofer/experiment/loop_x/ipcai2026/stl/stl/v4_seperate/simplification'
    ply_directory = "/home/leofer/experiment/loop_x/ipcai2026/us_compounding"
    gt_directory = "/home/leofer/experiment/loop_x/ipcai2026/us_compounding_gt"
    main()
