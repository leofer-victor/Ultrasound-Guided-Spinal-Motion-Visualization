#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   ultrasound_pose_match_viewer.py
@Time    :   2026/11/03 13:30:39
@Author  :   Feng Li 
@Contact :   feng.li@tum.de
@Desc    :   
'''

import os
import shutil
import sys
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

from PySide6.QtWidgets import (
    QApplication, QWidget, QLabel, QLineEdit, QPushButton, QVBoxLayout, QHBoxLayout,
    QFileDialog, QMessageBox, QGroupBox, QFormLayout
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt

# ----------------------------
# Data structures
# ----------------------------
@dataclass
class PoseRecord:
    timestamp: float
    index: int
    tx: float
    ty: float
    tz: float
    qx: float
    qy: float
    qz: float
    qw: float

    def as_vec(self) -> Tuple[Tuple[float,float,float], Tuple[float,float,float,float]]:
        return (self.tx, self.ty, self.tz), (self.qx, self.qy, self.qz, self.qw)


# ----------------------------
# Utilities
# ----------------------------
def read_points_and_poses_from_name1(name1_dir: str) -> Tuple[Optional[Tuple[float,float,float]], Optional[Tuple[float,float,float,float,float,float,float]]]:
    """Read points.txt and poses.txt from the given 'name1' directory.
    - points.txt: one line with 3 numbers (x y z)
    - poses.txt: one line with 7 numbers (tx ty tz qx qy qz qw)
    Returns (point_xyz or None, pose_7 or None)
    """
    point = None
    pose = None

    points_path = os.path.join(name1_dir, 'points.txt')
    poses_path = os.path.join(name1_dir, 'poses.txt')

    if os.path.exists(points_path):
        try:
            with open(points_path, 'r', encoding='utf-8') as f:
                nums = f.read().strip().replace(',', ' ').split()
                if len(nums) >= 3:
                    point = tuple(map(float, nums[:3]))
        except Exception as e:
            print(f"Failed to read {points_path}: {e}")
    else:
        print(f"[Info] {points_path} not found.")

    if os.path.exists(poses_path):
        try:
            with open(poses_path, 'r', encoding='utf-8') as f:
                nums = f.read().strip().replace(',', ' ').split()
                if len(nums) >= 7:
                    pose = tuple(map(float, nums[:7]))
        except Exception as e:
            print(f"Failed to read {poses_path}: {e}")
    else:
        print(f"[Info] {poses_path} not found.")

    return point, pose


def parse_center_txt(center_txt_path: str) -> List[PoseRecord]:
    """Parse a center.txt file.
    Each line: timestamp, index, tx, ty, tz, qx, qy, qz, qw
    Returns a list of PoseRecord.
    """
    records: List[PoseRecord] = []
    with open(center_txt_path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = line.replace(',', ' ').split()
            if len(parts) < 9:
                print(f"[Warn] line {line_no} has < 9 tokens, skipping: {line}")
                continue
            try:
                vals = list(map(float, parts[:9]))
                rec = PoseRecord(
                    timestamp=vals[0],
                    index=int(vals[1]),
                    tx=vals[2],
                    ty=vals[3],
                    tz=vals[4],
                    qx=vals[5],
                    qy=vals[6],
                    qz=vals[7],
                    qw=vals[8],
                )
                records.append(rec)
            except Exception as e:
                print(f"[Warn] failed to parse line {line_no}: {e}")
    return records


def quaternion_angle(q1: Tuple[float,float,float,float], q2: Tuple[float,float,float,float]) -> float:
    """Return the smallest angle (radians) between two unit quaternions q1, q2."""
    # Ensure dot is in [-1, 1] and consider double cover (q and -q are identical)
    dot = abs(q1[0]*q2[0] + q1[1]*q2[1] + q1[2]*q2[2] + q1[3]*q2[3])
    dot = max(-1.0, min(1.0, dot))
    # The angular difference between orientations represented by unit quaternions:
    # angle = 2 * arccos(dot)
    return 2.0 * math.acos(dot)


def pose_distance(p1: PoseRecord, p2: PoseRecord, rot_weight: float = 1.0) -> float:
    """A simple pose distance: ||t1 - t2|| + rot_weight * angle(q1, q2).
    Units: translation (same as your dataset, often meters) + radians * rot_weight.
    You may tune rot_weight to balance translation vs rotation.
    """
    (tx1, ty1, tz1), q1 = p1.as_vec()
    (tx2, ty2, tz2), q2 = p2.as_vec()
    dt = math.sqrt((tx1-tx2)**2 + (ty1-ty2)**2 + (tz1-tz2)**2)
    da = quaternion_angle(q1, q2)
    return dt


def zero_pad(num: int, width: int = 4) -> str:
    return str(num).zfill(width)


# ----------------------------
# Main UI
# ----------------------------
class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Ultrasound Pose Match & Viewer (PySide6)")
        self.resize(1200, 700)

        # Inputs
        self.name1_dir_edit = QLineEdit('connected_point')
        self.root_a_edit = QLineEdit('../../img_path1')  # dataset A root (has center/ and center.txt)
        self.root_b_edit = QLineEdit('../../img_path2')  # dataset B root (has center/ and center.txt)
        self.folder_a_name_edit = QLineEdit('data1')  # 默认名字 center
        self.folder_b_name_edit = QLineEdit('data1')
        self.index_edit = QLineEdit('5')  # user inputs a number like 5 -> '0005.png'

        self.save_dir_edit = QLineEdit()
        self.save_dir_edit.setPlaceholderText('Optional: Enter path to save matched images...')
        self.btn_browse_save = QPushButton('Browse...')

        # Buttons
        self.btn_browse_name1 = QPushButton('Browse name1...')
        self.btn_browse_a = QPushButton('Browse Root A...')
        self.btn_browse_b = QPushButton('Browse Root B...')
        self.btn_run = QPushButton('Load && Match')
        self.btn_save = QPushButton('Save')
        self.btn_save.setEnabled(False)

        self.btn_browse_name1.clicked.connect(self._browse_name1)
        self.btn_browse_a.clicked.connect(lambda: self._browse_dir(self.root_a_edit))
        self.btn_browse_b.clicked.connect(lambda: self._browse_dir(self.root_b_edit))
        self.btn_run.clicked.connect(self._run)
        self.btn_browse_save.clicked.connect(lambda: self._browse_res_dir(self.save_dir_edit))
        self.btn_save.clicked.connect(self._save_images)

        # Image views
        self.img_label_1 = QLabel('Image A will appear here')
        self.img_label_1.setAlignment(Qt.AlignCenter)
        self.img_label_1.setMinimumSize(512, 512)
        self.img_label_1.setStyleSheet("border: 1px solid #aaa;")

        self.img_label_2 = QLabel('Image B (closest pose) will appear here')
        self.img_label_2.setAlignment(Qt.AlignCenter)
        self.img_label_2.setMinimumSize(512, 512)
        self.img_label_2.setStyleSheet("border: 1px solid #aaa;")

        # Layouts
        form_box = QGroupBox('Inputs')
        form = QFormLayout()
        form.addRow('name1 folder:', self._h(self.name1_dir_edit, self.btn_browse_name1))
        form.addRow('Dataset Root A:', self._h(self.root_a_edit, self.btn_browse_a, self.folder_a_name_edit))
        form.addRow('Dataset Root B:', self._h(self.root_b_edit, self.btn_browse_b, self.folder_b_name_edit))
        # form.addRow('A folder name (e.g. data1):', self.folder_a_name_edit)
        # form.addRow('B folder name (e.g. data1):', self.folder_b_name_edit)
        form.addRow('Image index (int):', self.index_edit)
        form.addRow('Save Path (Optional):', self._h(self.save_dir_edit, self.btn_browse_save))
        form_box.setLayout(form)

        img_layout = QHBoxLayout()
        img_layout.addWidget(self.img_label_1, 1)
        img_layout.addWidget(self.img_label_2, 1)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self.btn_run, 1)  
        button_layout.addWidget(self.btn_save, 1) 

        main = QVBoxLayout(self)
        main.addWidget(form_box)
        main.addLayout(button_layout)  
        main.addLayout(img_layout, 1)

    def _h(self, *widgets):
        w = QWidget()
        lay = QHBoxLayout(w)
        for wd in widgets:
            if isinstance(wd, QLineEdit):
                lay.addWidget(wd, stretch=3) 
            else:
                lay.addWidget(wd)
        lay.addStretch(1)
        lay.setContentsMargins(0, 0, 0, 0)
        return w

    def _browse_name1(self):
        d = QFileDialog.getExistingDirectory(self, 'Select name1 directory', '/home/leofer/experiment/loop_x/ipcai2026/data/connect_point/movement')
        if d:
            self.name1_dir_edit.setText(d)

    def _browse_dir(self, line_edit: QLineEdit):
        d = QFileDialog.getExistingDirectory(self, 'Select directory', '/home/leofer/experiment/loop_x/ipcai2026/extracted_imgs/movement')
        if d:
            line_edit.setText(d)

    def _browse_res_dir(self, line_edit: QLineEdit):
        d = QFileDialog.getExistingDirectory(self, 'Select directory', '/home/leofer/experiment/loop_x/ipcai2026/optical_flow')
        if d:
            line_edit.setText(d)

    def _run(self):
        # 运行开始时，禁用保存按钮并清除旧路径
        self.btn_save.setEnabled(False)
        self.img_path_a_to_save = None
        self.img_path_b_to_save = None
        
        try:
            idx = int(self.index_edit.text().strip())
        except ValueError:
            QMessageBox.warning(self, 'Invalid input', 'Please enter a valid integer index.')
            return

        name1_dir = self.name1_dir_edit.text().strip()
        root_a = self.root_a_edit.text().strip()
        root_b = self.root_b_edit.text().strip()

        # 1) Read points.txt and poses.txt from name1 (optional usage)
        point_xyz, pose_7 = read_points_and_poses_from_name1(name1_dir)
        if point_xyz is not None:
            print(f"[name1] points.txt -> (x,y,z) = {point_xyz}")
        else:
            print("[name1] points.txt not found or invalid.")
        if pose_7 is not None:
            print(f"[name1] poses.txt -> (tx,ty,tz,qx,qy,qz,qw) = {pose_7}")
        else:
            print("[name1] poses.txt not found or invalid.")

        # 2) Load A/center.txt and A image by index
        folder_a_name = self.folder_a_name_edit.text().strip()
        folder_b_name = self.folder_b_name_edit.text().strip()

        center_a_path = os.path.join(root_a, f'{folder_a_name}.txt')
        center_b_path = os.path.join(root_b, f'{folder_b_name}.txt')
        imgs_a_dir = os.path.join(root_a, folder_a_name)
        imgs_b_dir = os.path.join(root_b, folder_b_name)

        if not os.path.exists(center_a_path):
            QMessageBox.critical(self, 'Error', f'center.txt not found at: {center_a_path}')
            return
        if not os.path.exists(center_b_path):
            QMessageBox.critical(self, 'Error', f'center.txt not found at: {center_b_path}')
            return
        if not os.path.isdir(imgs_a_dir):
            QMessageBox.critical(self, 'Error', f'Image folder not found at: {imgs_a_dir}')
            return
        if not os.path.isdir(imgs_b_dir):
            QMessageBox.critical(self, 'Error', f'Image folder not found at: {imgs_b_dir}')
            return

        records_a = parse_center_txt(center_a_path)
        records_b = parse_center_txt(center_b_path)
        if not records_a:
            QMessageBox.critical(self, 'Error', f'No valid entries in {center_a_path}')
            return
        if not records_b:
            QMessageBox.critical(self, 'Error', f'No valid entries in {center_b_path}')
            return

        # Find pose in A by exact index
        rec_a = next((r for r in records_a if r.index == idx), None)
        if rec_a is None:
            QMessageBox.critical(self, 'Error', f'Index {idx} not found in A center.txt')
            return

        # Load image A
        img_name = f"{zero_pad(idx)}.png"
        img_path_a = os.path.join(imgs_a_dir, img_name)
        if not os.path.exists(img_path_a):
            QMessageBox.critical(self, 'Error', f'Image not found: {img_path_a}')
            return

        self._set_image(self.img_label_1, img_path_a)
        self.img_path_a_to_save = img_path_a  # <-- 存储路径

        # 3) In B, find the closest pose to rec_a
        best_rec_b = None
        best_dist = float('inf')
        # Weight can be tuned; start with 1.0
        ROT_WEIGHT = 1.0
        for r in records_b:
            d = pose_distance(rec_a, r, rot_weight=ROT_WEIGHT)
            if d < best_dist:
                best_dist = d
                best_rec_b = r

        if best_rec_b is None:
            QMessageBox.critical(self, 'Error', 'Failed to find closest pose in B.')
            return

        # Load corresponding image in B by its index
        img_name_b = f"{zero_pad(best_rec_b.index)}.png"
        img_path_b = os.path.join(imgs_b_dir, img_name_b)
        if not os.path.exists(img_path_b):
            QMessageBox.critical(self, 'Error', f'Closest pose found (index {best_rec_b.index}), but image missing: {img_path_b}')
            return

        self._set_image(self.img_label_2, img_path_b)
        self.img_path_b_to_save = img_path_b  # <-- 存储路径

        # Provide a brief info message
        print(f"Loaded A index {idx} and matched B index {best_rec_b.index}.\nDistance = {best_dist:.6f}")
        
        # 运行成功，启用保存按钮
        self.btn_save.setEnabled(True)
    
    def _save_images(self):
        """
        Saves the currently displayed images to the specified save directory
        with the names 0000.png and 0001.png.
        """
        save_dir = self.save_dir_edit.text().strip()
        
        # 检查1：是否指定了保存路径
        if not save_dir:
            QMessageBox.warning(self, 'No Save Path', 'Please enter a save path in the "Save Path" box.')
            return

        # 检查2：是否已成功加载图像
        if not self.img_path_a_to_save or not self.img_path_b_to_save:
            QMessageBox.critical(self, 'Error', 'No images have been loaded. Please run "Load && Match" first.')
            return
            
        # 检查3：源文件是否仍然存在
        if not os.path.exists(self.img_path_a_to_save):
             QMessageBox.critical(self, 'Error', f'Source file missing: {self.img_path_a_to_save}')
             return
        if not os.path.exists(self.img_path_b_to_save):
             QMessageBox.critical(self, 'Error', f'Source file missing: {self.img_path_b_to_save}')
             return

        # 创建保存目录
        try:
            os.makedirs(save_dir, exist_ok=True)
        except OSError as e:
            QMessageBox.critical(self, 'Error', f'Failed to create save directory: {save_dir}\nError: {e}')
            return

        # 定义目标路径（根据您的命名要求）
        dest_path_a = os.path.join(save_dir, "0000.png")
        dest_path_b = os.path.join(save_dir, "0001.png")

        # 复制文件
        try:
            shutil.copy2(self.img_path_a_to_save, dest_path_a)
            shutil.copy2(self.img_path_b_to_save, dest_path_b)
            
            info_msg = f"Successfully saved images:\n{dest_path_a}\n{dest_path_b}"
            print(info_msg)
            QMessageBox.information(self, 'Save Successful', info_msg)

        except Exception as e:
            QMessageBox.critical(self, 'Error', f'Failed to save images.\nError: {e}')

    def _set_image(self, label: QLabel, path: str):
        pix = QPixmap(path)
        if pix.isNull():
            label.setText(f"Failed to load image: {path}")
        else:
            label.setPixmap(pix.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        # Keep images scaled responsively
        if isinstance(self.img_label_1.pixmap(), QPixmap):
            self.img_label_1.setPixmap(self.img_label_1.pixmap().scaled(self.img_label_1.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        if isinstance(self.img_label_2.pixmap(), QPixmap):
            self.img_label_2.setPixmap(self.img_label_2.pixmap().scaled(self.img_label_2.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        return super().resizeEvent(event)


def main():
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    op_path = '/home/leofer/experiment/loop_x/ipcai2026/optical flow/res'
    main()
