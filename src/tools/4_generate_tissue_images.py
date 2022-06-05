#######################################################################################################################
# CS5199 - Individual Masters Project: Breast Cancer Detection in Low Resolution Images.
# Author - Nathan Poole (170004680)
#
# Generate Tissue Detection Images:
#   Generate the images resulting from tissue detection for the data set (and save with the original images for
#   comparison to verify the success of the tissue detection method).
#
# Usage:
#   'python3 4_generate_tissue_images.py -i <input_dir> -o <output_dir> [-c]'
#   , where:
#       <input_dir> is the directory containing the low-resolution (black filtered) images to apply tissue detection.
#       <output_dir> is the directory to save the generated tissue detected images.
#       [-c] indicates for produced images to show comparisons, so original and tissue images together.
#######################################################################################################################

# System Imports:
from datetime import timedelta
import argparse
import os
import numpy
import sys
import time

# Package Imports:
from PIL import Image
from skimage.filters import threshold_otsu, median
from skimage.morphology import binary_dilation
from torchvision import transforms
from torchvision.utils import make_grid, save_image


#######################################################################################################################
# Program Arguments.
#######################################################################################################################

# Adding expected program arguments: '-i' (input directory), '-o' (output directory).
parser = argparse.ArgumentParser(description="Generate Tissue Detection Comparison Images:")
parser.add_argument("-i", "--input_dir", dest="input_dir", type=str, required=True,
                    help="Path to input directory containing all images to apply tissue detection to.")
parser.add_argument("-o", "--output_dir", dest="output_dir", type=str, required=True,
                    help="Path to output directory to save tissue detection images to.")
parser.add_argument("-c", "--compare", action="store_true", required=False,
                    help="Flag indicating for produced images to show comparison between original and tissue images.")

# Parse program arguments.
args = parser.parse_args()

# Check input directory exists.
if not os.path.isdir(args.input_dir):
    parser.print_help(sys.stdout)
    print("Error: Given input directory does not exist.")
    sys.exit()

# Check output directory exists.
if not os.path.isdir(args.output_dir):
    print("Output directory ('", args.output_dir, "') does not exist - creating directory.")
    os.makedirs(args.output_dir)


#######################################################################################################################
# Tissue Detection Process:
#######################################################################################################################

class DetectTissue(object):
    """
    Custom transform class: Apply Otsu thresholding to a given PIL image for tissue region detection.
    """

    def __call__(self, pil_img):
        """
        Given a PIL image, apply Otsu thresholding and median filtering for tissue detection, and return the cropped
        tissue region.

        :param pil_img: PIL image to apply the tissue detection methodology to.
        :return: PIL image containing the extracted tissue detected area from the input PIL image.
        """

        pil_img_hsv = pil_img.convert("HSV")  # Converting to HSV colour space.
        img_numpy = numpy.array(pil_img_hsv)  # Convert PIL image to numpy array.

        # Apply Otsu thresholding on Hue and Saturation channels (from Camelyon-16 submissions).
        thresh_H = threshold_otsu(img_numpy[:, :, 0])
        thresh_S = threshold_otsu(img_numpy[:, :, 1])

        # Generate binary images from Otsu thresholding.
        binary_img_H = img_numpy[:, :, 0] <= thresh_H
        binary_img_S = img_numpy[:, :, 1] <= thresh_S

        binary_img = binary_img_H + binary_img_S  # Combine H,S channels to create mask (from Camelyon-16 submissions).
        binary_img = (binary_img + 255).astype(numpy.uint8)  # Convert from boolean to integer.

        binary_img = median(binary_img, numpy.ones((7, 7)))  # Applying median filtering to remove spurious regions.
        binary_img = binary_dilation(binary_img, numpy.ones((5, 5)))  # Dilate to add slight tissue buffer.

        # Get (min_x, max_x, min_y, and max_y) where there is tissue in the tissue mask from the img.
        rows = numpy.any(binary_img, axis=1)
        cols = numpy.any(binary_img, axis=0)
        min_y, max_y = numpy.where(rows)[0][[0, -1]]
        min_x, max_x = numpy.where(cols)[0][[0, -1]]

        # Crop out the tissue region and return.
        tissue_det_img = pil_img.crop((min_x, min_y, max_x, max_y))
        return tissue_det_img

    def __repr__(self):
        """
        :return: Provide information about the class.
        """
        return self.__class__.__name__ + '()'


#######################################################################################################################
# Generate Tissue Detection Images From Training Data Set.
#######################################################################################################################

# Get image files in the input directory.
all_data = [loc for loc in os.listdir(args.input_dir) if '.png' in loc.lower()]

start_time = time.time()  # Used to calculate the average image conversion time.

img_count = 0
for curr_image_name in all_data:  # Iterate over all images and apply tissue detection to see/compare result.

    curr_image = Image.open(os.path.join(args.input_dir, curr_image_name)).convert("RGB")  # Get current RGB image.

    tissue_img = DetectTissue()(curr_image)  # Apply tissue detection method to RGB image.

    if args.compare:  # Program arguments specify to create comparison images for manual verification of results.
        curr_image_name = curr_image_name[:-len(".png")] + "_tissue_compare.png"  # Save comparison image.
        curr_image = transforms.Resize(size=(299, 299))(curr_image)
        tissue_img = transforms.Resize(size=(299, 299))(tissue_img)
        # Create side-by-side comparison image.
        comp_img = make_grid([transforms.ToTensor()(curr_image), transforms.ToTensor()(tissue_img)], nrows=2)
        save_image(comp_img, os.path.join(args.output_dir, curr_image_name), "PNG")

    else:  # Save tissue detected image.
        curr_image_name = curr_image_name[:-len(".png")] + "_tissue.png"  # Save tissue image under altered name.
        tissue_img.save(os.path.join(args.output_dir, curr_image_name), "PNG")  # Save image to output folder.

    img_count += 1
    print(f"Saved '{curr_image_name}' to '{args.output_dir}' (Total: {img_count}).")

elapsed_time = time.time() - start_time  # Total time to convert all images in the directory.
avg_conversion_time = elapsed_time / len(all_data)  # Average time to convert a single image.
print(f"Average Conversion Time (h:m:s) = {str(timedelta(seconds=avg_conversion_time))}")  # Output avg time.
