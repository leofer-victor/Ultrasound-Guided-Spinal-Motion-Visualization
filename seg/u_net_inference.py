import os
import argparse
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import nibabel as nib
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import get_custom_objects
from nibabel.orientations import io_orientation, ornt_transform, apply_orientation
import shutil

# --------------------- Loss Functions ---------------------

@tf.keras.utils.register_keras_serializable()
def focal_loss(gamma=2.0, alpha=0.75):
    """
    Focal Loss function for handling class imbalance.
    """
    def focal_loss_fixed(y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        loss = -y_true * alpha * tf.pow(1 - y_pred, gamma) * tf.math.log(y_pred)
        loss -= (1 - y_true) * (1 - alpha) * tf.pow(y_pred, gamma) * tf.math.log(1 - y_pred)
        return tf.reduce_mean(loss)
    return focal_loss_fixed

@tf.keras.utils.register_keras_serializable()
def dice_loss(y_true, y_pred, smooth=1e-6):
    """
    Dice loss function for overlap between prediction and ground truth.
    """
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1 - ((2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth))

@tf.keras.utils.register_keras_serializable()
def combined_loss(alpha=0.75, gamma=2.0):
    """
    Combined focal and dice loss.
    """
    def loss(y_true, y_pred):
        focal = focal_loss(gamma=gamma, alpha=alpha)(y_true, y_pred)
        dice = dice_loss(y_true, y_pred)
        return focal + dice
    return loss

@tf.keras.utils.register_keras_serializable()
class IoUMetric(tf.keras.metrics.Metric):
    """
    Custom IoU (Jaccard) metric for segmentation.
    """
    def __init__(self, name="iou", threshold=0.5, **kwargs):
        super(IoUMetric, self).__init__(name=name, **kwargs)
        self.threshold = threshold
        self.intersection = self.add_weight(name="intersection", initializer="zeros")
        self.union = self.add_weight(name="union", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true > self.threshold, tf.float32)
        y_pred = tf.cast(y_pred > self.threshold, tf.float32)
        intersection = tf.reduce_sum(y_true * y_pred)
        union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) - intersection
        self.intersection.assign_add(intersection)
        self.union.assign_add(union)

    def result(self):
        return (self.intersection + tf.keras.backend.epsilon()) / (self.union + tf.keras.backend.epsilon())

    def reset_state(self):
        self.intersection.assign(0)
        self.union.assign(0)

# --------------------- Preprocessing Functions ---------------------

def get_crop_box(image_size, crop_width, crop_height):
    """
    Calculate a centered crop box for an image of a given size.
    """
    img_w, img_h = image_size
    crop_w = min(crop_width, img_w)
    crop_h = min(crop_height, img_h)
    left = (img_w - crop_w) // 2
    #upper = 0
    upper = 70
    #left+=10
    return (left, upper, left + crop_w, upper + crop_h)

def preprocess_image(img_path, target_size=(256, 256), crop_width=400, crop_height=400):
    """
    Load, crop, normalize, and resize a grayscale image for model input.
    Returns the processed image, resized array, crop box, and original size.
    """
    img = Image.open(img_path).convert('L')
    original_size = img.size[::-1]  # (height, width)
    crop_box = get_crop_box(img.size, crop_width, crop_height)
    img_cropped = img.crop(crop_box)
    img_array = np.array(img_cropped, dtype=np.float32) / 255.0
    img_resized = tf.image.resize(img_array[..., np.newaxis], target_size)
    return np.expand_dims(img_resized, axis=0), img_resized, crop_box, original_size

# --------------------- Postprocessing Functions ---------------------

def restore_original_size(segmented_slices, crop_box, original_size):
    """
    Resize each predicted mask back to original size using the crop box.
    """
    left, upper, right, lower = crop_box
    original_h, original_w = original_size
    restored_slices = []

    for seg_img in segmented_slices:
        crop_w, crop_h = right - left, lower - upper
        resized_seg = cv2.resize(seg_img, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
        restored_img = np.zeros((original_h, original_w), dtype=seg_img.dtype)
        restored_img[upper:lower, left:right] = resized_seg
        restored_slices.append(restored_img)

    return restored_slices

# --------------------- Segmentation Pipeline ---------------------

def reset_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)  
    os.makedirs(path)
    
def rotate_images_in_directory(directory_path, degrees= -90):
    """
    Rotates all images in the given directory by a given angle.
    """
    if not os.path.exists(directory_path):
        print(f"Error: Directory '{directory_path}' does not exist.")
        return

    files = os.listdir(directory_path)
    image_extensions = {".jpg", ".jpeg", ".png", ".gif"}

    for file in files:
        file_path = os.path.join(directory_path, file)

        if os.path.isfile(file_path) and os.path.splitext(file)[1].lower() in image_extensions:
            try:
                with Image.open(file_path) as img:
                    # Rotate the image
                    rotated_img = img.rotate(degrees, expand=True)

                    # Save the rotated image
                    rotated_img.save(file_path)
            except Exception as e:
                print(f"Error rotating file '{file}': {e}")

def unet_segmentation(
    scan_mode,
    input_dir,
    output_dir_cropped,
    output_dir_masks,
    output_dir_restored,
    nii_output_path,
    reference_nii_path,
    crop_width=400,
    crop_height=400,
    target_size=(256, 256),
    threshold=0.5,
    model_path='model.keras'
):
    """
    Perform segmentation on a directory of images using a trained U-Net model,
    restore masks to original size, and save a NIfTI volume.
    """
    # Load reference affine and spacing
    reference_nii = nib.load(reference_nii_path)
    affine = reference_nii.affine

    # # Load reference NIfTI to get spacing
    # reference_nii = nib.load(reference_nii_path)
    # spacing = reference_nii.header.get_zooms()

    # # Create a new affine matrix using spacing and default orientation
    # affine = np.diag(spacing + (1.0,))  # 4x4 diagonal matrix


    # Load model with custom losses
    get_custom_objects().update({
        'loss': combined_loss,
        'IoUMetric': IoUMetric,
        'dice_loss': dice_loss,
        'focal_loss': focal_loss
    })
    model = load_model(model_path)

    # Ensure output directories exist
    # os.makedirs(output_dir_cropped, exist_ok=True)
    # os.makedirs(output_dir_masks, exist_ok=True)
    # os.makedirs(output_dir_restored, exist_ok=True)

    reset_dir(output_dir_cropped)
    reset_dir(output_dir_masks)
    reset_dir(output_dir_restored)

    if scan_mode == "linear":
        rotate_images_in_directory(input_dir, degrees= -90)

    restored_slices = []

    for img_name in sorted(os.listdir(input_dir)):
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        img_path = os.path.join(input_dir, img_name)

        # Preprocess image
        preprocessed_img, resized_img, crop_box, original_size = preprocess_image(
            img_path, target_size=target_size, crop_width=crop_width, crop_height=crop_height)

        # Save cropped image
        Image.fromarray((np.squeeze(resized_img) * 255).astype(np.uint8)).save(
            os.path.join(output_dir_cropped, img_name))

        # Predict mask
        predicted_mask = model.predict(preprocessed_img)[0]
        binary_mask = (predicted_mask > threshold).astype(np.uint8) * 255

        # Save predicted mask
        Image.fromarray(np.squeeze(binary_mask)).convert("L").save(
            os.path.join(output_dir_masks, img_name))

        # Restore to original size
        restored_img = restore_original_size([np.squeeze(binary_mask)], crop_box, original_size)[0]
        if scan_mode == "linear":
            restored_img= np.rot90(restored_img, k=1)
        restored_slices.append(restored_img)

        # Save restored image
        Image.fromarray(restored_img).convert("L").save(
            os.path.join(output_dir_restored, img_name))

    # Create and save NIfTI volume
    #restored_slices= restored_slices[::-1]
    volume = np.stack(restored_slices, axis=0)

    if scan_mode == "linear":
        volume= np.transpose(volume, (0, 2, 1))
        rotate_images_in_directory(input_dir, degrees= 90)
        rotate_images_in_directory(output_dir_cropped, degrees= 90)
        rotate_images_in_directory(output_dir_masks, degrees= 90)
    else:
        volume= np.transpose(volume, (2, 0, 1))
    
    volume=np.flip(volume, axis=2)
    nii_image = nib.Nifti1Image(volume.astype(np.uint8), affine)
    nib.save(nii_image, nii_output_path)
    print(f"3D volume saved to {nii_output_path}")

# --------------------- CLI Entry Point ---------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="U-Net segmentation with mask restoration and NIfTI export.")

    parser.add_argument("--scan_mode", choices=["linear", "u", "zigzag"], default="linear", help="Type of scan pattern used for the recording. Default is 'linear'.")
    parser.add_argument("--input_dir", type=str, default="sweep_images", help="Directory of input grayscale images.")
    parser.add_argument("--output_dir_cropped", type=str, default="unet_output/cropped_images", help="Directory to save cropped images.")
    parser.add_argument("--output_dir_masks", type=str, default="unet_output/masks", help="Directory to save predicted masks.")
    parser.add_argument("--output_dir_restored", type=str, default="unet_output/restored_masks", help="Directory to save restored masks.")
    parser.add_argument("--nii_output_path", type=str, default="volumes/unet_reconstructed_volume.nii", help="Path to save NIfTI volume.")
    parser.add_argument("--reference_nii_path", type=str, required=True, help="Reference NIfTI for spacing and affine.")
    parser.add_argument("--model_path", type=str, default="weights/27_02_2025_best_model.keras", help="Path to trained Keras model.")
    parser.add_argument("--crop_width", type=int, default=250, help="Crop width (horizontal).")
    parser.add_argument("--crop_height", type=int, default=200, help="Crop height (vertical).")
    parser.add_argument("--threshold", type=float, default=0.5, help="Segmentation threshold.")

    args = parser.parse_args()

    unet_segmentation(
        scan_mode=args.scan_mode,
        input_dir=args.input_dir,
        output_dir_cropped=args.output_dir_cropped,
        output_dir_masks=args.output_dir_masks,
        output_dir_restored=args.output_dir_restored,
        nii_output_path=args.nii_output_path,
        reference_nii_path=args.reference_nii_path,
        crop_width=args.crop_width,
        crop_height=args.crop_height,
        threshold=args.threshold,
        model_path=args.model_path
    )