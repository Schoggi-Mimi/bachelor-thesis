import concurrent.futures
import gc
import os
from random import randrange
from typing import List, Tuple

import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL.Image import Image as PILImage
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.distortions import *
from utils.utils_distortions import skin_segmentation

distortion_groups = {
    "background": ["color_block"],
    "lighting": ["brighten", "darken"],
    "focus": ["gaussian_blur", "lens_blur", "motion_blur"],
    "orientation": ["perspective_top", "perspective_bottom", "perspective_left", "perspective_right"],
    "color_calibration": ["color_saturation1", "color_saturation2"],
    #"background": ["color_block"],
    "resolution": ["change_resolution"],
    "field_of_view": ["crop_image"],
}

distortion_groups_mapping = {
    "color_block": "background",
    "gaussian_blur": "focus",
    "lens_blur": "focus",
    "motion_blur": "focus",
    "color_saturation1": "color_calibration",
    "color_saturation2": "color_calibration",
    "brighten": "lighting",
    "darken": "lighting",
    "perspective_top": "orientation",
    "perspective_bottom": "orientation",
    "perspective_left": "orientation",
    "perspective_right": "orientation",
    "crop_image": "field_of_view",
    "change_resolution": "resolution",
}

distortion_range = {
    "gaussian_blur": [0, 1, 2, 3, 5],
    "lens_blur": [0, 2, 4, 6, 8],
    "motion_blur": [0, 2, 4, 6, 8],
    "color_saturation1": [0, 0.2, 0.4, 0.6, 0.8],
    "color_saturation2": [0, 1, 2, 3, 4],
    "brighten": [0.0, 0.2, 0.4, 0.7, 1.1],
    "darken": [0.0, 0.2, 0.4, 0.6, 0.8],
    "perspective_top": [0.0, 0.2, 0.4, 0.6, 0.8],
    "perspective_bottom": [0.0, 0.2, 0.4, 0.6, 0.8],
    "perspective_left": [0.0, 0.2, 0.4, 0.6, 0.8],
    "perspective_right": [0.0, 0.2, 0.4, 0.6, 0.8],
    "color_block": [0.0, 0.5, 1.0, 1.5, 2.0],
    "change_resolution": [0.0, 0.2, 0.4, 0.6, 0.8],
    "crop_image": [0, 1, 2, 3, 4],
}

distortion_functions = {
    "color_block": color_block,
    "gaussian_blur": gaussian_blur,
    "lens_blur": lens_blur,
    "motion_blur": motion_blur,
    "color_saturation1": color_saturation1,
    "color_saturation2": color_saturation2,
    "brighten": brighten,
    "darken": darken,
    "perspective_top": perspective_top,
    "perspective_bottom": perspective_bottom,
    "perspective_left": perspective_left,
    "perspective_right": perspective_right,
    "crop_image": crop_image,
    "change_resolution": change_resolution,
}

def apply_distortion(img_path, distorted_image_path, method, range_val, distortion_functions):
    """
    Apply a specific distortion to an image and save the distorted version.

    Args:
        img_path (str): Path to the input image file.
        distortion_type (str): Type of distortion.
        method (str): Distortion method.
        range_val (float): Value of the distortion range.
        distortion_functions (dict): Dictionary containing distortion methods and their corresponding functions.
    """
    orig_image = Image.open(img_path).convert('RGB')
    image = transforms.ToTensor()(orig_image)
    distort_function = distortion_functions.get(method)
    image = distort_function(image, range_val)
    #image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image)
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
                        apply_distortion, img_path, distorted_image_path, method, range_val, distortion_functions
                    )
                    futures.append(future)
                    if len(futures) >= batch_size:
                        concurrent.futures.wait(futures)  # Wait for batch processing to complete
                        futures.clear()  # Clear the list for the next batch
        concurrent.futures.wait(futures)

def distort_images(image: torch.Tensor, distort_functions: list = None, distort_values: list = None, num_levels: int = 5) -> torch.Tensor:
    """
    Distorts an image using the distortion composition obtained with the image degradation model proposed in the paper
    https://arxiv.org/abs/2310.14918.

    Args:
        image (Tensor): image to distort
        distort_functions (list): list of the distortion functions to apply to the image. If None, the functions are randomly chosen.
        distort_values (list): list of the values of the distortion functions to apply to the image. If None, the values are randomly chosen.
        max_distortions (int): maximum number of distortions to apply to the image
        num_levels (int): number of levels of distortion that can be applied to the image

    Returns:
        image (Tensor): distorted image
        distort_functions (list): list of the distortion functions applied to the image
        distort_values (list): list of the values of the distortion functions applied to the image
    """
    if distort_functions is None or distort_values is None:
        distort_functions, distort_values = get_distortions_composition(num_levels)

    skin_mask = skin_segmentation(image)
    background_proportion = 1 - skin_mask.mean().item()
    if background_proportion < 0.1:
        for idx, func in enumerate(distort_functions):
            if func.__name__ == "color_block":
                distort_values[idx] = 0.0 

    for distortion, value in zip(distort_functions, distort_values):
        image = distortion(image, value)
        image = image.to(torch.float32)
        image = torch.clip(image, 0, 1)

    return image, distort_functions, distort_values

def get_distortions_composition(num_levels: int = 5):
    """
    Image Degradation model proposed in the paper https://arxiv.org/abs/2310.14918. Returns a randomly assembled ordered
    sequence of distortion functions and their values.

    Args:
        max_distortions (int): maximum number of distortions to apply to the image
        num_levels (int): number of levels of distortion that can be applied to the image

    Returns:
        distort_functions (list): list of the distortion functions to apply to the image
        distort_values (list): list of the values of the distortion functions to apply to the image
    """
    MEAN = 0
    STD = 2.5

    distortions = [random.choice(distortion_groups[group]) for group in list(distortion_groups.keys())]
    distort_functions = [distortion_functions[dist] for dist in distortions]
    distort_values = [np.random.choice(distortion_range[dist][:num_levels]) for dist in distortions]

    return distort_functions, distort_values

def map_distortion_values(distort_functions, distort_values):
    """
    Map distortion values to a normalized scale from 0 to 1 based on their defined ranges.

    Args:
        distort_functions (list): List of distortion functions applied to the images.
        distort_values (list): List of actual distortion values corresponding to each function.

    Returns:
        list: Normalized distortion values where 0 represents no distortion and 1 represents the maximum distortion.
    """
    distort_functions = [f.__name__ for f in distort_functions]
    mapped_values = []

    for func, val in zip(distort_functions, distort_values):
        range_vals = distortion_range.get(func)
        if range_vals:
            min_val = min(range_vals)
            max_val = max(range_vals)
            mapped_value = (val - min_val) / (max_val - min_val)
            mapped_values.append(mapped_value)
        else:
            mapped_values.append(val)
    return torch.tensor(mapped_values, dtype=torch.float32)

def map_predictions_to_intervals(predictions):
    """
    Map continuous predictions into discrete intervals.
    Args:
        predictions (np.array): Array of predicted values.
    
    Returns:
        np.array: Array of mapped values.
    """
    bins = np.array([-0.125, 0.125, 0.375, 0.625, 0.875, 1.125])
    interval_indices = np.digitize(predictions, bins, right=True)
    mapped_predictions = bins[interval_indices - 1] + 0.125
    return mapped_predictions

def discretization(scores):
    """Convert continuous scores to categorical by defined thresholds."""
    thresholds = [-0.125, 0.125, 0.375, 0.625, 0.875, 1.125]
    categories = np.digitize(scores, thresholds, right=True) - 1
    categories = np.clip(categories, 0, len(thresholds)-2)
    return categories

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

def get_features_scores(model: torch.nn.Module,
                        dataloader: DataLoader,
                        device: torch.device,
                       ) -> Tuple[np.ndarray, np.ndarray]:        
    feats = np.zeros((0, model.encoder.feat_dim * 2))   # Double the features because of the original and downsampled image (0, 4096)
    scores = np.zeros((0, 7))
    with tqdm(total=len(dataloader), desc="Extracting features", leave=False) as progress_bar:
        for _, batch in enumerate(dataloader):
            img_orig = batch["img"].to(device)
            img_ds = batch["img_ds"].to(device)
            label = batch["label"]

            with torch.no_grad():
                _, f = model(img_orig, img_ds, return_embedding=True)
    
            feats = np.concatenate((feats, f.cpu().numpy()), 0)
            scores = np.concatenate((scores, label.numpy()), 0)
            progress_bar.update(1)
    
    return feats, scores