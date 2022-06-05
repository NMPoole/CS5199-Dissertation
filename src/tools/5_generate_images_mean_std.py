#######################################################################################################################
# CS5199 - Individual Masters Project: Breast Cancer Detection in Low Resolution Images.
# Author - Nathan Poole (170004680)
#
# Calculate Images Mean & Standard Deviation:
#   Given a directory containing PNG images, calculate the mean and standard deviation for each channel across
#   all images in the directory.
#
# Usage:
#   'python3 5_generate_images_mean_std.py -i <input_dir> -width <image_width> -height <image_height>'
#   , where:
#       <input_dir> is the path to the directory containing PNG images to calculate the mean and std for.
#       <image_width> is the desired width to convert images to and calculate stats.
#       <image_height> is the desired height to convert images to and calculate stats.
#######################################################################################################################

# System Imports:
import argparse
import os
import sys

# Package Imports:
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


#######################################################################################################################
# Program Arguments:
#######################################################################################################################

parser = argparse.ArgumentParser(description="Calculate Images Mean & Std:")
parser.add_argument("-i", "--input_dir", dest="input_dir", type=str, required=True,
                    help="Path to directory containing (.PNG) images to calculate mean and std for.")
parser.add_argument("-width", dest="width", type=int, required=True,
                    help="Image width to use for calculating statistics.")
parser.add_argument("-height", dest="height", type=int, required=True,
                    help="Image height to use for calculating statistics.")

# Parse program arguments.
args = parser.parse_args()

# Check input directory exists.
if not os.path.isdir(args.input_dir):
    parser.print_help(sys.stdout)
    print("Error: Given input directory does not exist.")
    sys.exit()

# Check width and height are valid.
if not (args.width > 0 and args.height > 0):
    parser.print_help(sys.stdout)
    print("Error: Given width/height must be a positive integer.")
    sys.exit()


#######################################################################################################################
# Simple Image Data Set:
#######################################################################################################################

class ImageDataset(Dataset):
    """
    Custom image data set that simply returns the images within a given input directory.
    """

    def __init__(self, input_dir, width, height):
        """
        Initialise all data simply as file names of all images in the input directory.

        :param input_dir: Directory containing train, validate, and test image subsets.
        :param width: Target image width for mean and std calculation.
        :param height: Target image height for mean and std calculation.
        """
        self.input_dir = input_dir
        self.width = width
        self.height = height
        self.all_data = [loc for loc in os.listdir(self.input_dir) if '.png' in loc.lower()]

    def __len__(self):
        """
        :return: Length of data set is the number of images in the input directory.
        """
        return len(self.all_data)

    def __getitem__(self, index):
        """
        Get the image corresponding to the index, resize and convert to tensor, and return result.

        :param index: Index of image in the data set to retrieve.
        :return: PIL image at the given index in the data set.
        """
        file_name = self.all_data[index]  # Index according to list of all data from initialisation.
        index_img = Image.open(os.path.join(self.input_dir, file_name)).convert("RGB")  # Load image at index.

        index_img = transforms.Resize(size=(self.width, self.height))(index_img)
        index_img = transforms.ToTensor()(index_img)
        index_img = transforms.Normalize(mean=[0, 0, 0], std=[1, 1, 1])(index_img)

        return index_img


#######################################################################################################################
# Calculating Images Mean & Standard Deviation:
#######################################################################################################################

def calc_mean_std(data_loader):
    """
    Given a data loader for images, calculate mean and std for each channel.
    From: https://kozodoi.me/python/deep%20learning/pytorch/tutorial/2021/03/08/image-mean-std.html

    :param data_loader: Data loader to retrieve data set images to calculate mean and std for.
    :return: Mean and Std tensors (for each channel) amongst the images given by the data loader.
    """

    # Placeholders.
    psum = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])
    width = height = 0

    for batch_imgs in data_loader:  # Iterate over images in input directory.

        # Shape = BatchSize x Channels x Width x Height
        width = batch_imgs.shape[2]
        height = batch_imgs.shape[3]

        psum += batch_imgs.sum(axis=[0, 2, 3])
        psum_sq += (batch_imgs ** 2).sum(axis=[0, 2, 3])

    # Pixel Count = Num Images x Image Width x Image Height (...valid since all images are resized to the same size).
    pixel_count = len(data_loader.dataset) * width * height

    # Mean and Std:
    total_mean = psum / pixel_count  # Mean of pixel values across the data set.
    total_var = (psum_sq / pixel_count) - (total_mean ** 2)  # Variance of pixel values across the data set.
    total_std = torch.sqrt(total_var)

    return total_mean, total_std  # Return results.


print("Whole-Slide Classification - Computing Mean And Std:")
STR_BARRIER = "-" * 100  # For nice output separation.

# Get data loader for all images in the input directory.
dataset = ImageDataset(args.input_dir, args.width, args.height)
loader = DataLoader(dataset, batch_size=1, shuffle=False)
print(f"Input Directory Num Images: {len(dataset)} (from '{dataset.input_dir}').")
print(STR_BARRIER)

print(f"Calculating Mean and Std...")
print(STR_BARRIER)

# Calculate mean and standard deviation per channel for all images in the data set.
mean, std = calc_mean_std(loader)

# Output:
print(f"Input Directory Image Stats (at {args.width} x {args.height} pixels):")
print(f"Channel Mean = {str(mean)}")
print(f"Channel Std = {str(std)}")
print(STR_BARRIER)
