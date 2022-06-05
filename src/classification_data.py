#######################################################################################################################
# CS5199 - Individual Masters Project: Breast Cancer Detection in Low Resolution Images.
# Author - Nathan Poole (170004680)
#
# Whole-Slide Classification - Data:
#   Defines a custom dataset class for the whole-slide classification task. Custom dataset can be used for training,
#   evaluating, and testing. The dataset handles any data augmentation to be applied on the data samples.
#######################################################################################################################

# System Imports:
import os
import random

# Package Imports:
from PIL import Image
import torch
from torch.backends.cudnn import deterministic
from torch.utils.data import Dataset
from torchvision import transforms

# Ensure reproducibility via seeding.
SEED = 10
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)


#######################################################################################################################
# Dataset:
#######################################################################################################################

class ClassificationDataset(Dataset):
    """
    Custom dataset for whole-slide classification image data.

    Given a root directory, assume the directory contains folders for training, evaluating, and testing data. The
    dataset is executed in a given mode (i.e., train, eval, test), which determines the type of data being retrieves.
    The dataset size is determined by the number of files within the mode directory.
    """

    def __init__(self, root_dir, mode='train', augment=0, width=299, height=299):
        """
        Initialise the data set with the data root directory and mode.
        Transforms are determined by the mode and augmentation level specified.

        :param root_dir: Directory containing the train, validate, and test sub sets (i.e., subdirectories).
        :param mode: Data set mode of operation: "train"|"eval"|"test" (i.e., which sub set to retrieve from root_dir).
        :param augment: Data augmentation level to apply (i.e., model version).
        :param width: Width of input images to use. Default is 299.
        :param height: Height of input images to use. Default is 299.
        """

        self.root_dir = root_dir  # Directory containing input (i.e., train, eval, or test) data directories.

        assert mode in ['train', 'eval', 'test']  # Only three types of data can be specified.
        self.mode = mode
        self.mode_dir = os.path.join(root_dir, mode)  # Create path to relevant data depending on mode.

        self.all_data = [loc for loc in os.listdir(self.mode_dir) if '.png' in loc.lower()]  # Names of all input files.

        # Determine transform to apply to the input images.
        self.transform = self.determine_transform(mode, augment, width, height)

    def __len__(self):
        """
        :return: Data set length is the number of images files in the corresponding mode directory.
        """
        return len(self.all_data)  # Data set length is the number of image files in mode directory.

    def __getitem__(self, index):
        """
        For a given index, return the image, label, and file name associated with the index.

        :param index: Index of image in the data set to retrieve.
        :return: PIL image at the given index in the data set.
        """

        file_name = self.all_data[index].lower()  # Index according to list of all data from initialisation.
        img = Image.open(os.path.join(self.mode_dir, file_name)).convert("RGB")  # Load image at index.

        if self.transform is not None:  # Apply image transform, if specified during initialisation.
            img = self.transform(img)

        # Get corresponding label - even test set labels should be retrieved by file name.
        if any(label in file_name for label in ["normal", "none", "negative"]):
            label = torch.tensor(0)  # Map "normal" images as negative class, with label '0'.
        elif any(label in file_name for label in ["tumor", "itc", "micro", "macro"]):
            label = torch.tensor(1)  # Map "tumor" images as positive class, with label '1'.
        else:
            print(f"Unexpected label when getting data set item. File: '{file_name}'.")
            label = torch.tensor(-1)  # Should not occur but provide error label to indicate issue.

        return img, label, file_name

    @staticmethod
    def determine_transform(mode, augment, width, height):
        """
        Create transform required for each data set mode.

        :param mode: Mode of data set operation/which sub set is being used (i.e., "train", "eval", or "test").
        :param augment: Augmentation level (i.e., model version) being applied to the data set.
        :param width: Width of input images to use.
        :param height: Height of input images to use.
        :return: Composed transform representing the required transformations to apply to all data in the data sub set.
        """

        # Hard coded dictionary is sufficient - does not need to be made dynamic for investigations. Having the code
        # just calculate what it needs here is better but this will save time in the long run for myself.
        # Means and Stds calculated for the training data prior using: tools/5_generate_images_mean_std.py.
        mean_std_dict = {

            299: {
                # Mean and Std for the original (black filtered) PNG images (i.e., Camelyon 16 & 17 training images).
                "orig": {"mean": [0.9159, 0.8941, 0.9208], "std": [0.1193, 0.1636, 0.1091]},
                # Mean and Std for the tissue detected PNG images at 299 x 299 pixels.
                "tissue": {"mean": [0.8572, 0.8084, 0.8592], "std": [0.1545, 0.2144, 0.1475]}
            },

            # No original mean and std needed as version 3 is used for all investigations of higher input resolutions.

            512: {
                "tissue": {"mean": [0.8938, 0.8598, 0.8968], "std": [0.1371, 0.1917, 0.1306]}
            },

            1024: {
                "tissue": {"mean": [0.8938, 0.8598, 0.8968], "std": [0.1387, 0.1942, 0.1323]}
            },

            2048: {
                "tissue": {"mean": [0.8938, 0.8598, 0.8968], "std": [0.1407, 0.1967, 0.1343]}
            }

        }

        # Apply correct normalisation depending on augmentation level.
        # NOTE: Training mean and std applied when training, evaluating, and testing as model only has train knowledge.
        if augment == 2 or augment == 3:  # Augment levels 2 and 3 indicate tissue detection is used.
            mean, std = mean_std_dict[width]["tissue"]["mean"], mean_std_dict[width]["tissue"]["std"]
        else:
            mean, std = mean_std_dict[width]["orig"]["mean"], mean_std_dict[width]["orig"]["std"]

        # Preparation Transform - resizes to target resolution, converts to tensor, and applies normalisation.
        preparation_transform = transforms.Compose([
            transforms.Resize(size=(width, height)),  # Chosen Resolution.
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        # Augmentation Transform - apply outcome-invariant data augmentation techniques.
        # NOTE: Does not increase the size of the training dataset! However, applying these transformations for
        #       many more epochs is functionally equivalent to adding these augmentations as well as originals.
        augmentation_transform = transforms.Compose([
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            Rotate90Intervals(),
            transforms.ColorJitter(brightness=.25, saturation=.25, hue=.1, contrast=.25),
        ])

        if (augment == 0 or augment == 2) or mode != "train":  # No augmentation, or eval/ or test/ set.
            return preparation_transform
        elif (augment == 1 or augment == 3) and mode == "train":  # Augmentation, but applied only to train/ set.
            return transforms.Compose([augmentation_transform, preparation_transform])


#######################################################################################################################
# Custom Image Rotation Transformation:
#######################################################################################################################

class Rotate90Intervals(object):
    """
    Custom transformation class: Apply random rotation to an image in intervals of 90 degrees.
    """

    def __call__(self, pil_img):
        """
        Given a PIL image, apply a random rotation (from 90 degree intervals) and return the rotated image.

        :param pil_img: PIL image to apply rotation to.
        :return: PIL image possibly rotated by some 90 degree interval.
        """

        angles = [0, 1, 2, 3]  # Corresponds to 0, 90, 180, 270 degree rotations counter-clockwise.
        angle = random.choice(angles)  # Choose a rotation at random.

        tensor_img = transforms.ToTensor()(pil_img)
        rot_tensor_img = torch.rot90(tensor_img, k=angle, dims=[2, 1])  # Rotate maintaining expected data shape.
        rot_img = transforms.ToPILImage()(rot_tensor_img)  # Convert back to PIL image.

        return rot_img

    def __repr__(self):
        """
        :return: Provide information about the class.
        """
        return self.__class__.__name__ + '()'
