#######################################################################################################################
# CS5199 - Individual Masters Project: Breast Cancer Detection in Low Resolution Images.
# Author - Nathan Poole (170004680)
#
# Generate Image Labels:
#   Given a directory of images and a corresponding CSV file containing labels for the images, rename all images to
#   include their associated label for convenience. This is useful for the Camelyon-16 test images, but is required for
#   the Camelyon-17 images (training and testing).
#
# Usage:
#   'python3 2_generate_image_labels.py -i <input_dir> -c <csv_file>'
#   , where:
#       <input_dir> is the path to the directory containing PNG images to rename with associated labels.
#       <csv_file> is the path to a CSV file containing the input_dir file names and associated labels.
#######################################################################################################################

# System Imports:
import argparse
import csv
import os
import sys

#######################################################################################################################
# Program Arguments:
#######################################################################################################################

parser = argparse.ArgumentParser(description="Rename Images With Labels:")
parser.add_argument("-i", "--input_dir", dest="input_dir", type=str, required=True,
                    help="Path to the directory containing PNG images to rename with associated labels.")
parser.add_argument("-c", "--csv_file", dest="csv_file", type=str, required=True,
                    help="Path to a CSV file containing the input_dir/ file names and associated labels")


# Parse program arguments.
args = parser.parse_args()

# Check input directory exists.
if not os.path.isdir(args.input_dir):
    parser.print_help(sys.stdout)
    print("Error: Given input directory does not exist.")
    sys.exit()

# Check CSV file is valid location.
if not os.path.isfile(args.csv_file):
    parser.print_help(sys.stdout)
    print("Error: Invalid CSV file given.")
    sys.exit()


#######################################################################################################################
# Renaming Directory Images To Include Associated Output Labels:
#######################################################################################################################

print("Renaming Images To Include Labels:")
STR_BARRIER = "-" * 100  # For nice output separation.
print(STR_BARRIER)

# Get dictionary of image file names in the input directory and corresponding file extensions.
input_dir_images = {}
for curr_file_name in os.listdir(args.input_dir):
    if ".png" in curr_file_name:
        file_name_ext_pair = curr_file_name.rsplit('.', 1)
        input_dir_images[file_name_ext_pair[0]] = file_name_ext_pair[1]  # Use file name as key, and ext as value.
print(f"Input Directory ('{args.input_dir}') Num PNG Images: {len(input_dir_images)}.")
print(STR_BARRIER)

# For every row in the CSV file, check if file exists in input_dir/ and rename if required.
files_renamed = 0
with open(args.csv_file) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')

    print(f"Iterating CSV File: {args.csv_file}.")

    for row in csv_reader:

        file_name = row[0]  # Name of the file to rename including label.
        if "." in file_name:
            file_name = file_name.rsplit(".", 1)[0]  # Only consider file names in CSV, not extensions as well.

        label = row[-1].lower()  # Label to add to file name (assumes in last column, true via manual file inspection).

        if file_name in input_dir_images:  # CSV file name found in input directory, so should be renamed with label.
            print(f"\tFile Found In Input Dir: '{file_name}'. Renaming With Label: '{label}'")

            ext = "." + input_dir_images[file_name]  # Get extension from dictionary.
            original_file = os.path.join(args.input_dir, file_name + ext)  # Original file name.
            new_file = os.path.join(args.input_dir, file_name + f"_{label}" + ext)  # New file name with label.

            os.rename(original_file, new_file)  # Rename file to include label.
            print(f"\t\tRenamed '{original_file}' To '{new_file}'.")
            files_renamed += 1

print(STR_BARRIER)
print(f"Finished Renaming: {files_renamed} Files Renamed (i.e., Labelled).")
print(STR_BARRIER)
