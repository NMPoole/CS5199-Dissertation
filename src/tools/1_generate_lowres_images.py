#######################################################################################################################
# CS5199 - Individual Masters Project: Breast Cancer Detection in Low Resolution Images.
# Author - Nathan Poole (170004680)
#
# Generate Low-Resolution Images:
#   Given input dimensions, or a specified WSI resolution level, convert the WSI TIF images within a given directory
#   to low-resolution PNG images of chosen size (resolution level or thumbnail), and save them to a chosen directory.
#   Note: Thumbnails treat width and height as maximum dimensions, so they will not have the given size definitely as
#         aspect ratio is preserved.
#
# Usage:
#   'python3 1_generate_lowres_images.py <width|-l> <height|res_level> <input_dir> <output_dir>'
#   , where:
#       <width> <height> for a given size (in pixels) or <-l> <res_level> for a chosen resolution (i.e., magnification),
#       <input_dir> is the directory containing the high-resolution TIF images,
#       <output_dir> is the directory to save the generated low-resolution images.
#######################################################################################################################

# System Imports:
from datetime import timedelta
import sys
import os
import time

# Package Imports:
from openslide import open_slide

#######################################################################################################################
# Program Arguments:
#######################################################################################################################

usage_msg = "Usage: 'python3 generate_lowres_images <width|-l> <height|res_level> <input_dir> <output_dir>'."

# Check expected number of arguments (4) given.
if len(sys.argv) != 5:
    print("Invalid number of arguments given.")
    print(usage_msg)
    sys.exit()

# Get program arguments.
first_arg = sys.argv[1]  # Width or '-l'.
second_arg = sys.argv[2]  # Height or res_level.
input_dir = sys.argv[3]  # Input directory containing TIF images.
output_dir = sys.argv[4]  # Output directory to save low-res images to.

# Check if width and height given, or resolution level requested.
res_level_requested = (first_arg == "-l")

if res_level_requested:

    # Check resolution level requested is valid.
    if not second_arg.lstrip("-").isnumeric():
        print("Error: Resolution level must be an integer.")
        sys.exit()

    res_level = int(second_arg)

    # Resolution levels defined for WSIs between 0 and 9.
    if res_level < 0 or res_level > 9:
        print("Error: Invalid resolution level requested - must be between 0 and 9 inclusive.")
        sys.exit()

else:  # not res_level_requested.

    # Check given width and height are valid.
    if (not first_arg.lstrip("-").isnumeric()) or (not second_arg.lstrip("-").isnumeric()):
        print("Error: Width and height must be an integer.")
        sys.exit()

    width = int(first_arg)
    height = int(second_arg)

    if width <= 0 or height <= 0:
        print("Error: Invalid width/height provided - must be greater then 0.")
        sys.exit()

# Check if input and output directories exist (create output directory if required).
if not os.path.isdir(input_dir):
    print("Error: Given input directory does not exist.")
    sys.exit()

if not os.path.isdir(output_dir):
    print("Output directory ('", output_dir, "') does not exist - creating directory.")
    os.makedirs(output_dir)

#######################################################################################################################
# Low-Res Image Generation:
#######################################################################################################################

print("Generating Low-Resolution Images:")
str_barrier = "-" * 100  # For nice output separation.
print(str_barrier)

if res_level_requested:
    print("Images Generated At Resolution Level: ", res_level)
else:
    print("Images Generated As Thumbnails With Dimensions (Width, Height): ", width, ", ", height)

print(str_barrier)

start_time = time.time()  # Record start time so average conversion time per image can be calculated.

num_images = 0
# For every TIF image in input_dir, resize as needed and save as PNG.
for curr_tif_image in os.listdir(input_dir):
    if curr_tif_image.endswith(".tif"):

        num_images += 1

        slide = open_slide(input_dir + curr_tif_image)  # Get TIF image using OpenSlide library.

        print(f"Working On Slide: '{curr_tif_image}'.")

        if res_level_requested:  # If res_level given, down-sample to that size.

            if res_level > (len(slide.level_dimensions) - 1):  # Deal with slides not having requested res_level.
                res_level = len(slide.level_dimensions) - 1

            resized_slide = slide.read_region((0, 0), res_level, slide.level_dimensions[res_level])

        else:  # If width/height given, resize image by obtaining thumbnail (width and height are treated as maximums).
            resized_slide = slide.get_thumbnail(size=(width, height))

        # Save to output_dir as PNG, preserving the file name.
        resized_slide.convert("RGB")
        curr_tif_image = curr_tif_image.replace(".tif", ".png")  # Replace file extension.
        resized_slide.save(os.path.join(output_dir, curr_tif_image))  # Save image.

        # Output status message.
        print(f"Saved '{curr_tif_image}' to '{output_dir}' "
              f"(Width: {resized_slide.width}, Height: {resized_slide.height}).")

elapsed_time = time.time() - start_time  # Total time to convert all images in the directory.
avg_conversion_time = elapsed_time / num_images  # Average time to convert a single image.
print(f"Average Conversion Time (h:m:s) = {str(timedelta(seconds=avg_conversion_time))}")  # Output avg time.
