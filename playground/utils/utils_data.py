import concurrent.futures
import gc
import os
from random import randrange
from typing import Callable, List, Union

import torchvision.transforms.functional as TF
from PIL.Image import Image as PILImage

from utils.distortions import *

distortion_groups = {
    "orientation": ["perspective"],
    "focus": ["gaublur", "lensblur", "motionblur"],
    "resolution": ["jpeg2000", "jpeg"],
    "lighting": ["brighten", "darken", "meanshift"],
    "background": ["colorblock"],
    "color_calibration": ["colorshift", "colorsat1", "colorsat2"],
}

distortion_groups_mapping = {
    "gaublur": "focus",
    "lensblur": "focus",
    "motionblur": "focus",
    "colordiff": "color_calibration",
    "colorshift": "color_calibration",
    "colorsat1": "color_calibration",
    "colorsat2": "color_calibration",
    "jpeg2000": "resolution",
    "jpeg": "resolution",
    "brighten": "lighting",
    "darken": "lighting",
    "meanshift": "lighting",
    "colorblock": "background",
    "perspective": "orientation",
}

distortion_range = {
    "gaublur": [0.1, 0.5, 1, 2, 5],
    "lensblur": [1, 2, 4, 6, 8],
    "motionblur": [1, 2, 4, 6, 10],
    "colordiff": [1, 3, 6, 8, 12],
    "colorshift": [1, 3, 6, 8, 12],
    "colorsat1": [0.4, 0.2, 0.1, 0, -0.4],
    "colorsat2": [1, 2, 3, 6, 9],
    "jpeg2000": [16, 32, 45, 120, 170],
    "jpeg": [43, 36, 24, 7, 4],
    "brighten": [0.1, 0.2, 0.4, 0.7, 1.1],
    "darken": [0.05, 0.1, 0.2, 0.4, 0.8],
    "meanshift": [0, 0.08, -0.08, 0.15, -0.15],
    "colorblock": [2, 4, 6, 8, 10],
    "perspective": [0, 1, 2, 3],
}

distortion_functions = {
    "gaublur": gaussian_blur,
    "lensblur": lens_blur,
    "motionblur": motion_blur,
    "colordiff": color_diffusion,
    "colorshift": color_shift,
    "colorsat1": color_saturation1,
    "colorsat2": color_saturation2,
    "jpeg2000": jpeg2000,
    "jpeg": jpeg,
    "brighten": brighten,
    "darken": darken,
    "meanshift": mean_shift,
    "colorblock": color_block,
    "perspective": perspective,
}

def apply_distortion(img_path, distorted_image_path, method, range_val, transform, distortion_functions, counter):
    """
    Apply a specific distortion to an image and save the distorted version.

    Args:
        img_path (str): Path to the input image file.
        distortion_type (str): Type of distortion.
        method (str): Distortion method.
        range_val (float): Value of the distortion range.
        transform (torchvision.transforms.Compose): Image transformation pipeline.
        distortion_functions (dict): Dictionary containing distortion methods and their corresponding functions.
    """
    orig_image = Image.open(img_path).convert('RGB')
    image = transform(orig_image)
    distort_function = distortion_functions.get(method)
    if distort_function:
        image = distort_function(image, range_val)
        image_pil = transforms.ToPILImage()(image)
        #print(f'Distortion path: {distorted_image_path}')    
        image_pil.save(distorted_image_path)
    del orig_image, image_pil, image  # Explicitly delete variables to free up memory
    gc.collect()  # Trigger garbage collection

def create_distortions_batch(img_path, folder_path, batch_size=10, counter=None):
    """
    Apply various distortions to the input image and save the distorted versions in different folders.

    Args:
        img_path (str): Path to the input image file.
        folder_path (str): Path to the main folder where distorted images will be saved.
        distortion_groups (dict): Dictionary containing distortion types and their associated methods.
        distortion_range (dict): Dictionary containing distortion methods and their ranges.
        distortion_functions (dict): Dictionary containing distortion methods and their corresponding functions.
        batch_size (int): Number of images to process in each batch.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for distortion_type, methods in distortion_groups.items():
            distortion_type_folder = os.path.join(folder_path, distortion_type)
            os.makedirs(distortion_type_folder, exist_ok=True)
            for method in methods:
                method_folder = os.path.join(distortion_type_folder, method)
                os.makedirs(method_folder, exist_ok=True)
                ranges = distortion_range.get(method, [])
                for range_val in ranges:
                    distorted_image_path = os.path.join(folder_path, distortion_type, method, f"{counter}_{method}_{range_val}.png")
                    future = executor.submit(
                        apply_distortion, img_path, distorted_image_path, method, range_val, transform, distortion_functions, counter
                    )
                    futures.append(future)
                    if len(futures) >= batch_size:
                        concurrent.futures.wait(futures)  # Wait for batch processing to complete
                        futures.clear()  # Clear the list for the next batch
        # Process any remaining futures
        concurrent.futures.wait(futures)

def center_corners_crop(img: PILImage, crop_size: int = 224) -> List[PILImage]:
    """
    Return the center crop and the four corners of the image.

    Args:
        img (PIL.Image): image to crop
        crop_size (int): size of each crop

    Returns:
        crops (List[PIL.Image]): list of the five crops
    """
    width, height = img.size

    # Calculate the coordinates for the center crop and the four corners
    cx = width // 2
    cy = height // 2
    crops = [
        TF.crop(img, cy - crop_size // 2, cx - crop_size // 2, crop_size, crop_size),  # Center
        TF.crop(img, 0, 0, crop_size, crop_size),  # Top-left corner
        TF.crop(img, height - crop_size, 0, crop_size, crop_size),  # Bottom-left corner
        TF.crop(img, 0, width - crop_size, crop_size, crop_size),  # Top-right corner
        TF.crop(img, height - crop_size, width - crop_size, crop_size, crop_size)  # Bottom-right corner
    ]

    return crops

def resize_crop(img: PILImage, crop_size: int = 224, downscale_factor: int = 1) -> PILImage:
    """
    Resize the image with the desired downscale factor and optionally crop it to the desired size. The crop is randomly
    sampled from the image. If crop_size is None, no crop is applied. If the crop is out of bounds, the image is
    automatically padded with zeros.

    Args:
        img (PIL Image): image to resize and crop
        crop_size (int): size of the crop. If None, no crop is applied
        downscale_factor (int): downscale factor to apply to the image

    Returns:
        img (PIL Image): resized and/or cropped image
    """
    w, h = img.size
    if downscale_factor > 1:
        img = img.resize((w // downscale_factor, h // downscale_factor))
        w, h = img.size

    if crop_size is not None:
        top = randrange(0, max(1, h - crop_size))
        left = randrange(0, max(1, w - crop_size))
        img = TF.crop(img, top, left, crop_size, crop_size)     # Automatically pad with zeros if the crop is out of bounds

    return img