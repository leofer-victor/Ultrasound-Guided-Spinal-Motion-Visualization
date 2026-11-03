# Ultrasound-Guided Spinal Motion Visualization

Research code accompanying the article:

> F. Li, Y. Bi, T. Song, Z. Jiang, and N. Navab, "Ultrasound-guided real-time spinal motion visualization for spinal instability assessment," *International Journal of Computer Assisted Radiology and Surgery*, vol. 21, pp. 989-996, 2026. [https://doi.org/10.1007/s11548-026-03627-x](https://doi.org/10.1007/s11548-026-03627-x)

This repository contains prototype tools used in a robotic ultrasound-CBCT workflow for radiation-reduced, real-time 3D visualization of lumbar spinal motion. The pipeline combines ultrasound segmentation and 3D compounding, a CBCT-derived vertebral model, kinematic initialization, and ICP-based registration.

## Demonstration video

<video controls src="media/IPCAI2026_video.mp4" title="Title"></video>
The video demonstrates the ultrasound-guided spinal motion visualization workflow presented in the accompanying paper.

## Repository contents

| File | Description |
| --- | --- |
| `generate_gt.py` | PySide6/VTK viewer for loading a DICOM series and exporting a sampled bone-surface point cloud. |
| `compounding.py` | GUI for compounding tracked binary ultrasound masks into a 3D volume and exporting NIfTI, STL, or PLY data. |
| `compounding_angle_margin.py` | GUI for intensity compounding with probe-mount angle compensation and configurable sampling. |
| `spine_motion.py` | Interactive lumbar-spine kinematic viewer with coarse alignment, ICP refinement, and ground-truth evaluation. |
| `ultrasound_pose_match_viewer.py` | Utility for matching ultrasound images by tracked probe pose and reviewing/saving matched data. |
| `seg/u_net_inference.py` | U-Net inference and mask restoration for ultrasound image sequences, with NIfTI export. |
| `seg/u_net_inference_v1.py` | Experimental U-Net inference variant retained for reproducibility. |
| `seg/pointnet_inference.py` | PointNet-based point-cloud segmentation and per-label export. |

## Installation

Python 3.10 or newer is recommended. Create an isolated environment and install the main runtime dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  numpy scipy pillow imageio nibabel opencv-python \
  PySide6 vtk pyvista pyvistaqt \
  tensorflow torch open3d matplotlib plotly
```

For systems without a display, the GUI applications require an appropriate Qt/VTK display backend or a virtual display.

## External assets

The repository does not include patient/phantom data or trained model weights. Supply these assets locally:

- A trained Keras U-Net model for `seg/u_net_inference.py`.
- A trained PointNet checkpoint for `seg/pointnet_inference.py`.
- The PointNet model definition imported as `pointnet_train.SegmentationPointNet`.
- DICOM, ultrasound image, tracking-pose, NIfTI, STL, and PLY inputs as required by each tool.

Do not commit clinical data, personally identifiable information, large model checkpoints, or generated volumes to a public repository.

## Usage

Run the GUI applications from the repository root:

```bash
python generate_gt.py
python compounding.py
python compounding_angle_margin.py
python spine_motion.py
python ultrasound_pose_match_viewer.py
```

Example U-Net inference:

```bash
python seg/u_net_inference.py \
  --scan_mode linear \
  --input_dir /path/to/sweep_images \
  --reference_nii_path /path/to/reference.nii.gz \
  --model_path /path/to/model.keras
```

Example PointNet inference:

```bash
python seg/pointnet_inference.py \
  --nii_path volumes/unet_reconstructed_volume.nii \
  --model_path /path/to/pointnet_best_model.pth
```

Use `--help` to inspect all command-line options:

```bash
python seg/u_net_inference.py --help
python seg/pointnet_inference.py --help
```

## Input conventions

Several tools expect image sequences and corresponding tracked poses. Depending on the script, a pose line is represented as either:

```text
timestamp index tx ty tz qx qy qz qw
```

or:

```text
tx ty tz qw qx qy qz
```

Translations are interpreted in metres by the compounding tools and converted internally to millimetres. Confirm image ordering, calibration, coordinate frames, quaternion order, and physical spacing before processing new acquisitions.

## Research-code notice

This is research software and is not a medical device. It has not been validated for diagnosis, treatment planning, or other clinical use. Parameters and data conventions may need to be adapted to a new acquisition setup.

## Citation

If this code supports your research, please cite:

```bibtex
@article{li2026ultrasound,
  title   = {Ultrasound-guided real-time spinal motion visualization for spinal instability assessment},
  author  = {Li, Feng and Bi, Yuan and Song, Tianyu and Jiang, Zhongliang and Navab, Nassir},
  journal = {International Journal of Computer Assisted Radiology and Surgery},
  volume  = {21},
  pages   = {989--996},
  year    = {2026},
  doi     = {10.1007/s11548-026-03627-x}
}
```

## License

No software license has been selected yet. Before making the repository public, add a `LICENSE` file that reflects the intended terms of reuse. The article's Creative Commons license does not automatically determine the license of the source code.
