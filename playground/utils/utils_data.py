import concurrent.futures
import gc
import os
from random import randrange
from typing import Callable, List, Union

import torchvision.transforms.functional as TF
from PIL.Image import Image as PILImage
from torchvision import transforms

from utils.distortions import *

distortion_groups = {
    "orientation": ["perspective_top", "perspective_bottom", "perspective_left", "perspective_right"],
    "focus": ["gaublur", "lensblur", "motionblur"],
    #"resolution": [""],
    "lighting": ["brighten", "darken"],
    #"background": [""],
    "color_calibration": ["colorsat1", "colorsat2"],
}

distortion_groups_mapping = {
    "gaublur": "focus",
    "lensblur": "focus",
    "motionblur": "focus",
    "colorsat1": "color_calibration",
    "colorsat2": "color_calibration",
    "brighten": "lighting",
    "darken": "lighting",
    "perspective_top": "orientation",
    "perspective_bottom": "orientation",
    "perspective_left": "orientation",
    "perspective_right": "orientation",
}

distortion_range = {
    "gaublur": [0, 1, 2, 3, 5],
    "lensblur": [0, 2, 4, 6, 8],
    "motionblur": [0, 2, 4, 6, 8],
    "colorsat1": [0, 0.2, 0.4, 0.6, 0.8],
    "colorsat2": [0, 1, 2, 3, 4],
    "brighten": [0.0, 0.2, 0.4, 0.7, 1.1],
    "darken": [0.0, 0.2, 0.4, 0.6, 0.8],
    "perspective_top": [0.0, 0.2, 0.4, 0.6, 0.8],
    "perspective_bottom": [0.0, 0.2, 0.4, 0.6, 0.8],
    "perspective_left": [0.0, 0.2, 0.4, 0.6, 0.8],
    "perspective_right": [0.0, 0.2, 0.4, 0.6, 0.8],
}

distortion_functions = {
    "gaublur": gaussian_blur,
    "lensblur": lens_blur,
    "motionblur": motion_blur,
    "colorsat1": color_saturation1,
    "colorsat2": color_saturation2,
    "brighten": brighten,
    "darken": darken,
    "perspective_top": perspective_top,
    "perspective_bottom": perspective_bottom,
    "perspective_left": perspective_left,
    "perspective_right": perspective_right,
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
        # Process any remaining futures
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

    probabilities = [1 / (STD * np.sqrt(2 * np.pi)) * np.exp(-((i - MEAN) ** 2) / (2 * STD ** 2))
                     for i in range(num_levels)]  # probabilities according to a gaussian distribution
    normalized_probabilities = [prob / sum(probabilities)
                                for prob in probabilities]  # normalize probabilities
    distort_values = [np.random.choice(distortion_range[dist][:num_levels], p=normalized_probabilities) for dist
                      in distortions]

    return distort_functions, distort_values

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