#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   generate_gt.py
@Time    :   2026/11/03 13:47:47
@Author  :   Feng Li 
@Contact :   feng.li@tum.de
@Desc    :   
'''

import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog, QMessageBox, QLabel, QDoubleSpinBox, QFormLayout
)
from PySide6.QtCore import Qt

import vtk
from vtkmodules.vtkIOImage import vtkDICOMImageReader
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkCommonCore import vtkUnsignedCharArray
from vtkmodules.vtkRenderingCore import (
    vtkRenderer,
    vtkRenderWindow,
    vtkVolume,
    vtkVolumeProperty,
    vtkColorTransferFunction,
)
from vtkmodules.vtkRenderingVolumeOpenGL2 import vtkSmartVolumeMapper
from vtkmodules.vtkFiltersCore import vtkFlyingEdges3D, vtkPolyDataNormals
from vtkmodules.vtkFiltersModeling import vtkPolyDataPointSampler
from vtkmodules.vtkFiltersCore import vtkProbeFilter
from vtkmodules.vtkIOPLY import vtkPLYWriter
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor


class DICOMViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("DICOM 可视化与点云导出 (PySide6 + VTK)")
        self.resize(1200, 800)

        # Data holders
        self.ct_data: vtkImageData | None = None
        self.dicom_dir: str | None = None

        # --- UI Layout ---
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QHBoxLayout(central)

        # Left controls
        left = QVBoxLayout()
        root.addLayout(left, 0)

        self.btn_load = QPushButton("Load DICOM")
        self.btn_load.clicked.connect(self.load_dicom_folder)
        left.addWidget(self.btn_load)

        self.btn_export = QPushButton("Export point cloud(.ply)")
        self.btn_export.clicked.connect(self.export_ply_pointcloud)
        self.btn_export.setEnabled(False)
        left.addWidget(self.btn_export)

        form = QFormLayout()
        left.addLayout(form)

        # Iso-value 控件（提取骨表面的等值）
        self.iso_spin = QDoubleSpinBox()
        self.iso_spin.setDecimals(1)
        self.iso_spin.setRange(-2000.0, 4000.0)
        self.iso_spin.setSingleStep(10.0)
        self.iso_spin.setValue(150.0)  # 骨阈值示例
        form.addRow("Isosurface HU:", self.iso_spin)

        # 采样间距（控制导出的点密度）
        self.sample_dist_spin = QDoubleSpinBox()
        self.sample_dist_spin.setDecimals(2)
        self.sample_dist_spin.setRange(0.1, 10.0)
        self.sample_dist_spin.setSingleStep(0.1)
        self.sample_dist_spin.setValue(1)
        form.addRow("Sampling interval: ", self.sample_dist_spin)

        self.lbl_notes = QLabel("Status: Not yet loaded")
        self.lbl_notes.setWordWrap(True)
        left.addWidget(self.lbl_notes)
        left.addStretch(1)

        # Right: VTK view
        self.vtk_widget = QVTKRenderWindowInteractor(self)
        root.addWidget(self.vtk_widget, 1)

        # VTK rendering setup
        self.renderer = vtkRenderer()
        self.vtk_widget.GetRenderWindow().AddRenderer(self.renderer)
        self.iren = self.vtk_widget.GetRenderWindow().GetInteractor()

        # Volume/mapper placeholders
        self.volume = None
        self.mapper = None
        self.volume_property = None

        # Default transfer functions (可按需要调整/替换)
        self.color_transfer_function = vtkColorTransferFunction()
        self.color_transfer_function.AddRGBPoint(-30, 0, 0, 0)
        self.color_transfer_function.AddRGBPoint(0, 0.25, 0.8, 0.65)  # water
        self.color_transfer_function.AddRGBPoint(30, 0, 0, 0)         # water
        self.color_transfer_function.AddRGBPoint(100, 0, 0, 0)
        self.color_transfer_function.AddRGBPoint(150, 1, 0.8, 0.2)    # spine
        self.color_transfer_function.AddRGBPoint(200, 0, 0, 0)
        self.color_transfer_function.AddRGBPoint(1500, 0, 0, 0)
        self.color_transfer_function.AddRGBPoint(2000, 0, 0, 0)
        self.color_transfer_function.AddRGBPoint(3000, 0, 0, 0)       # vessels

        self.opacity_transfer_function = vtk.vtkPiecewiseFunction()
        self.opacity_transfer_function.AddPoint(-30, 0.0)
        self.opacity_transfer_function.AddPoint(0, 0.0)   # water
        self.opacity_transfer_function.AddPoint(30, 0.0)  # water
        self.opacity_transfer_function.AddPoint(100, 0.0)
        self.opacity_transfer_function.AddPoint(150, 0.9) # spine
        self.opacity_transfer_function.AddPoint(200, 0.0)
        self.opacity_transfer_function.AddPoint(1500, 0.0)
        self.opacity_transfer_function.AddPoint(2000, 0.0)
        self.opacity_transfer_function.AddPoint(3000, 0.0)

        # Start interactor
        self.iren.Initialize()
        self.iren.Start()

    # ------------------------ Load & Visualize ------------------------

    def load_dicom_folder(self):
        directory = QFileDialog.getExistingDirectory(self, "选择含DICOM序列的文件夹", "/home/leofer/experiment/loop_x/data/ct/ipcai2026")
        if not directory:
            return

        reader = vtkDICOMImageReader()
        reader.SetDirectoryName(directory)
        reader.Update()

        image = reader.GetOutput()
        if image is None or image.GetNumberOfPoints() == 0:
            QMessageBox.warning(self, "加载失败", "未能从该目录读取到有效的DICOM体数据。")
            self.lbl_notes.setText("Status: Loading failed")
            return

        self.ct_data = image
        self.dicom_dir = directory
        self.lbl_notes.setText("Status: Loading successful")

        self.setup_volume_rendering()
        self.btn_export.setEnabled(True)

    def setup_volume_rendering(self):
        # 移除旧的 volume
        if self.volume is not None:
            self.renderer.RemoveViewProp(self.volume)
            self.volume = None

        # Mapper
        self.mapper = vtkSmartVolumeMapper()
        self.mapper.SetInputData(self.ct_data)
        self.mapper.SetBlendModeToComposite()  # 体渲染

        # Volume property
        self.volume_property = vtkVolumeProperty()
        self.volume_property.SetIndependentComponents(True)
        self.volume_property.ShadeOn()
        self.volume_property.SetInterpolationTypeToLinear()
        self.volume_property.SetColor(self.color_transfer_function)
        self.volume_property.SetScalarOpacity(self.opacity_transfer_function)

        self.volume = vtkVolume()
        self.volume.SetMapper(self.mapper)
        self.volume.SetProperty(self.volume_property)

        # Renderer
        self.renderer.AddVolume(self.volume)
        self.renderer.ResetCamera()
        self.vtk_widget.GetRenderWindow().Render()

    # ------------------------ Export Point Cloud ------------------------

    def export_ply_pointcloud(self):
        if self.ct_data is None:
            QMessageBox.warning(self, "导出失败", "尚未加载CT/CBCT体数据。")
            return

        iso_value = float(self.iso_spin.value())
        sample_dist = float(self.sample_dist_spin.value())

        # 1) 等值面（骨表面等）
        fe = vtkFlyingEdges3D()
        fe.SetInputData(self.ct_data)
        fe.SetValue(0, iso_value)
        fe.Update()
        surface = fe.GetOutput()

        if surface.GetNumberOfPoints() == 0:
            QMessageBox.warning(self, "导出失败", f"在等值 {iso_value} 处没有提取到表面点。请调整阈值。")
            return

        # 2) 法向（可改为光顺+法向，提升采样质量）
        normals = vtkPolyDataNormals()
        normals.SetInputData(surface)
        normals.SplittingOff()
        normals.SetFeatureAngle(60.0)
        normals.Update()
        smooth_surface = normals.GetOutput()

        # 3) 表面采样为点云
        sampler = vtkPolyDataPointSampler()
        sampler.SetInputData(smooth_surface)
        sampler.SetDistance(sample_dist)
        sampler.Update()
        pts_poly = sampler.GetOutput()

        if pts_poly.GetNumberOfPoints() == 0:
            QMessageBox.warning(self, "导出失败", "采样后点数为0，请调小采样间距。")
            return

        # 4) 从原始体采样 HU（用于着色）
        probe = vtkProbeFilter()
        probe.SetInputData(pts_poly)
        probe.SetSourceData(self.ct_data)
        probe.Update()
        probed_pts = probe.GetOutput()

       # 5) 用 color transfer function 生成 RGB 顶点颜色（uchar）
        scalars = probed_pts.GetPointData().GetScalars()
        if scalars is not None:
            n = probed_pts.GetNumberOfPoints()
            rgb = vtk.vtkUnsignedCharArray()
            rgb.SetName("RGB")
            rgb.SetNumberOfComponents(3)
            rgb.SetNumberOfTuples(n)
            ctf = self.color_transfer_function
            for i in range(n):
                hu = scalars.GetTuple1(i)
                r, g, b = ctf.GetColor(hu)  # [0..1]
                rgb.SetTuple3(i, int(255 * r), int(255 * g), int(255 * b))
            probed_pts.GetPointData().SetScalars(rgb)

        # 6) 手动为点创建 Verts（每个点一个顶点单元），避免依赖 vtkVertexGlyphFilter
        verts_poly = vtk.vtkPolyData()
        verts_poly.SetPoints(probed_pts.GetPoints())
        verts_poly.GetPointData().ShallowCopy(probed_pts.GetPointData())

        verts = vtk.vtkCellArray()
        npts = probed_pts.GetNumberOfPoints()
        verts.AllocateExact(npts, npts)  # 预分配：n 个单元，每单元 1 个点
        for i in range(npts):
            verts.InsertNextCell(1)
            verts.InsertCellPoint(i)
        verts_poly.SetVerts(verts)

        # 7) 写 PLY
        filename, _ = QFileDialog.getSaveFileName(
            self, "保存点云为 .ply", "/home/leofer/experiment/loop_x/ipcai2026/gt", "PLY Files (*.ply)"
        )
        if not filename:
            return

        writer = vtk.vtkPLYWriter()
        writer.SetFileName(filename)
        writer.SetInputData(verts_poly)
        writer.SetFileTypeToBinary()
        if not writer.Write():
            QMessageBox.warning(self, "导出失败", "写入 PLY 失败。请检查路径或磁盘权限。")
            return

        QMessageBox.information(self, "完成", f"已导出点云：{filename}")


def main():
    app = QApplication(sys.argv)
    w = DICOMViewer()
    w.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
