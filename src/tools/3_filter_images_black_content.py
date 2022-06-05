#######################################################################################################################
# CS5199 - Individual Masters Project: Breast Cancer Detection in Low Resolution Images.
# Author - Nathan Poole (170004680)
#
# Filter Black Content In Images:
#   Given a directory of images, apply black content filtering to the images and save the results.
#
# Usage:
#   'python3 3_filter_images_black_content.py -i <input_dir> -o <output_dir>'
#   , where:
#       <input_dir> is the directory containing the low-resolution PNG images to perform filtering on,
#       <output_dir> is the directory to save the filtered images to.
#######################################################################################################################

# System Imports:
from datetime import timedelta
import argparse
import os
import sys
import time

# Package Imports:
from PIL import Image


#######################################################################################################################
# Program Arguments.
#######################################################################################################################

# Adding expected program arguments: '-i' (input directory), '-o' (output directory).
parser = argparse.ArgumentParser(description="Generate Tissue Detection Comparison Images:")
parser.add_argument("-i", "--input_dir", dest="input_dir", type=str, required=True,
                    help="Path to input directory containing all images to apply black content filtering to.")
parser.add_argument("-o", "--output_dir", dest="output_dir", type=str, required=True,
                    help="Path to output directory to save filtered images to.")

# Parse program arguments.
args = parser.parse_args()

# Check input directory exists.
if not os.path.isdir(args.input_dir):
    parser.print_help(sys.stdout)
    print("Error: Given input directory does not exist.")
    sys.exit()

# Check output directory exists and create if not.
if not os.path.isdir(args.output_dir):
    print("Output directory ('", args.output_dir, "') does not exist - creating directory.")
    os.makedirs(args.output_dir)


#######################################################################################################################
# Black Content Filtering Process:
#######################################################################################################################

def remove_black(img):
    """
    Removal of black crosses from image via simple thresholding: RGB values < 50 made white.
    Threshold selected based on inspection of RGB values at black cross pixels in images.
    NOTE: Can be improved (optimised, different methodology, etc.)

    :param img: PIL images to apply black content removal to.
    :return: PIL image representing the input PIL image but with black content filtering applied.
    """

    R, G, B = img.split()  # Split image into channels since calculating the per channel mean and std.
    r = R.load()
    g = G.load()
    b = B.load()
    width, height = img.size  # Get image dimensions.

    thresh = 45  # Empirically chosen threshold value to use for filtering.

    # Convert black pixels (i.e., R & G & B < threshold) to white.
    for i in range(width):
        for j in range(height):
            if (r[i, j] < thresh) and (g[i, j] < thresh) and (b[i, j] < thresh):
                r[i, j] = 255
                g[i, j] = 255
                b[i, j] = 255

    # Merge the altered channels into a single RGB image.
    img_alt = Image.merge("RGB", (R, G, B))
    return img_alt


#######################################################################################################################
# Generate Black Content Filtered Images:
#######################################################################################################################

# Get image files in the input directory.
all_data = [loc for loc in os.listdir(args.input_dir) if '.png' in loc.lower()]

start_time = time.time()  # Used to calculate average image conversion time.

img_count = 0
for curr_image_name in all_data:  # Iterate over all images and apply black content filtering.

    curr_image = Image.open(os.path.join(args.input_dir, curr_image_name)).convert("RGB")  # Get current RGB image.

    tissue_img = remove_black(curr_image)  # Apply filtering method to RGB image.

    tissue_img.save(os.path.join(args.output_dir, curr_image_name), "PNG")  # Save image to output folder.
    img_count += 1
    print(f"Saved '{curr_image_name}' to '{args.output_dir}' (Total: {img_count}).")


elapsed_time = time.time() - start_time  # Total time to convert all images in the directory.
avg_conversion_time = elapsed_time / len(all_data)  # Average time to convert a single image.
print(f"Average Conversion Time (h:m:s) = {str(timedelta(seconds=avg_conversion_time))}")  # Output avg time.
