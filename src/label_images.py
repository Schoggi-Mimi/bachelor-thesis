"""This script does not work currently!!!"""

import argparse
import json
import os

import matplotlib.pyplot as plt
from PIL import Image


def label_images(start_index, end_index, image_folder, output_file):
    distortion_criteria = ["lighting", "focus", "orientation", "color_calibration", "background", "resolution", "field_of_view"]
    scores = {}
    if os.path.exists(output_file):
        with open(output_file, 'r') as file:
            scores = json.load(file)

    all_images = [f for f in os.listdir(image_folder) if f.endswith('.png')]
    selected_images = all_images[start_index:end_index]

    for filename in selected_images:
        image_path = os.path.join(image_folder, filename)
        image = Image.open(image_path)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(image)
        plt.axis('off')
        plt.show()

        image_scores = {}
        for criterion in distortion_criteria:
            valid_score = False
            while not valid_score:
                try:
                    score = float(input(f"Enter score (0-1) for {criterion}: "))
                    if 0 <= score <= 1:
                        image_scores[criterion] = score
                        valid_score = True
                    else:
                        print("Score must be between 0 and 1. Please try again.")
                except ValueError:
                    print("Invalid input. Please enter a numerical value.")
        
        scores[filename] = image_scores

    with open(output_file, "w") as json_file:
        json.dump(scores, json_file)

    print("Scores saved successfully.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label images with quality scores")
    parser.add_argument("start_index", type=int, help="Starting index of images to label")
    parser.add_argument("end_index", type=int, help="Ending index of images to label")
    parser.add_argument("image_folder", type=str, help="Folder containing images to label")
    parser.add_argument("output_file", type=str, help="JSON file to save the scores")

    args = parser.parse_args()

    label_images(args.start_index, args.end_index, args.image_folder, args.output_file)
