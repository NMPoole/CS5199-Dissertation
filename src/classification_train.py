#######################################################################################################################
# CS5199 - Individual Masters Project: Breast Cancer Detection in Low Resolution Images.
# Author - Nathan Poole (170004680)
#
# Whole-Slide Classification - TRAINING:
#   Train a CNN model for the classification of metastases in (low-res) WSIs, as given by the Camelyon challenge(s).
#
# Usage:
#   'python3 classification_train.py <-i INPUT_DIR_PATH> <-o OUTPUT_FILE_PATH> <-a AUGMENT_LEVEL> [-hpo]'
#   , where:
#       <-i INPUT_DIR_PATH> is a path to directory containing all input data.
#       <-o OUTPUT_FILE_PATH> is a path to an output file name to save the trained model to.
#       <-a AUGMENT_LEVEL> is an integer specifying the data augmentation level.
#           - 0 = No Data Augmentation w/ Original Low-Res Images.
#           - 1 = Data Augmentation (i.e., Flip/Rot/Jitter) w/ Original Low-Res Images.
#           - NOTE: Using -a 0|1 implies using original images, but these should still be black content filtered.
#           - 2 = No Data Augmentation w/ Tissue Detected Low-Res Images.
#           - 3 = Data Augmentation w/ Tissue Detected Low-Res Images.
#           - NOTE: Using -a 2|3 implies using tissue detection, so input directory must be for tissue detected images.
#       [-hpo] is a flag indicating to use hyper-parameter optimisation.
#       [-r <INPUT_RES>] is an optional argument to specify the model input resolution to use (Default is 299 x 299).
#######################################################################################################################

# System Imports:
import argparse
from functools import partial
import os
import sys
import math
from multiprocessing import cpu_count
import numpy as np
import time

# Package Imports:
from ray import tune
import torch
from torch import nn
from torch import optim
from torch.backends.cudnn import deterministic
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import inception_v3

# Local Imports:
from classification_data import ClassificationDataset

# Temp Import: PyTorch workaround for downloading pretrained models when there are SSL certificate issues with website.
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


#######################################################################################################################
# Program Arguments:
#######################################################################################################################

# Adding expected program arguments: '-i' (input directory), '-o' (output filename), '-a' (augment input level),
#                                    '-hpo' (hyper-parameter optimisation), '-r' (input resolution)
parser = argparse.ArgumentParser(description="Whole-Slide Classification - TRAINING:")
parser.add_argument("-i", "--input_dir", dest="input_dir", type=str, required=True,
                    help="Path to directory containing all input data.")
parser.add_argument("-o", "--output_filename", dest="output_filename", type=str, required=True,
                    help="Path to output file name to save trained model to.")
parser.add_argument("-a", "--augment", dest="augment", type=int, required=True,
                    help="Argument specifying level of augmentation to use.")
parser.add_argument("-hpo", "--hyperparamoptim", action="store_true", required=False,
                    help="Flag indicating to use hyper-parameter optimisation.")
parser.add_argument("-r", "--res", nargs='?', const=299, type=int, default=299,
                    help="Input image resolution to use [299, 512, 1024, 2048] (Default: 299, i.e. 299 x 299 pixels).")

# Parse program arguments.
args = parser.parse_args()

# Check input directory exists.
if not os.path.isdir(args.input_dir):
    parser.print_help(sys.stdout)
    print("Error: Given input directory does not exist.")
    sys.exit()

# Check output file is valid location.
output_filename_dir = os.path.dirname(args.output_filename)
if output_filename_dir != "" and not os.path.isdir(output_filename_dir):
    parser.print_help(sys.stdout)
    print("Error: Invalid location of output file specified.")
    sys.exit()
if ".pth" not in args.output_filename:  # Check PyTorch model saving convention being followed.
    args.output_filename = args.output_filename + ".pth"

# Augmentation leve specified must be valid.
if not (0 <= args.augment <= 3):
    parser.print_help(sys.stdout)
    print("Error: Augmentation level must be 0, 1, 2, or 3.")
    sys.exit()

# Check input resolution, if given.
if args.res not in [299, 512, 1024, 2048]:
    parser.print_help(sys.stdout)
    print("Error: Specified input resolution unexpected. Should be one of [299, 512, 1024, 2048].")
    sys.exit(-1)


#######################################################################################################################
# Training Functions:
#######################################################################################################################

def model_setup(config):
    """
    Model Set-Up: Create model architecture, loss function, and optimiser.

    :param config: Configuration of model (i.e., dictionary of hyper-parameter values) used for training.
    :return: Model, loss function, and optimisation algorithm to use for training.
    """
    # Model:
    model = inception_v3(pretrained=True)  # Use pretrained Inception_v3 model.
    model.fc = nn.Linear(model.fc.in_features, 1)  # Output layer from 1000 classes to single binary output.
    model.to(DEVICE)  # Move model to GPU, if available.

    # Loss Function:
    ratio = 382 / 236  # Camelyon-16&17: num_negative_training_samples / num_positives_training_samples = ~1.62.
    pos_weight = torch.tensor([ratio], device=DEVICE)  # Class Balancing: Ratio of negative to positive samples.
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)  # Using binary cross-entropy loss.

    # Optimiser: Using Adaptive Moment Estimation (Adam).
    optimiser = optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    return model, criterion, optimiser


def load_data(root_dir, augment_lvl, batch_size, verbose=True, input_img_size=299):
    """
    Given an input directory containing the sub-directories for the training and validation inout images, create a
    training and evaluation/validation data set and loader. The specified augmentation level determines the form of
    augmentation applied to the training data set images.

    :param root_dir: Directory containing train, validate, and test data sub sets (i.e., subdirectories).
    :param augment_lvl: Augmentation level (i.e., model version) being applied.
    :param batch_size: Size of batches to use when loading data.
    :param verbose: Boolean flag indicating whether to print robust output to console.
    :param input_img_size: Dimension to use (width and height) for the model input images.
    :return: DataLoader for the requested training data set.
    """

    # Create training dataset and loader.
    train_dataset = ClassificationDataset(root_dir=root_dir, mode='train', augment=augment_lvl,
                                          width=input_img_size, height=input_img_size)
    train_dataset_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    train_dataset_size = len(train_dataset)
    if verbose:
        print(f"Training Dataset Size: {train_dataset_size} (from '{train_dataset.mode_dir}').")
        print(f"Training Dataset Augmentation Level: {augment_lvl}")

    # Create evaluation (i.e., validation) dataset and loader.
    eval_dataset = ClassificationDataset(root_dir=root_dir, mode='eval', augment=augment_lvl,
                                         width=input_img_size, height=input_img_size)
    eval_dataset_loader = DataLoader(eval_dataset, shuffle=True, batch_size=batch_size)
    eval_dataset_size = len(eval_dataset)
    if verbose:
        print(f"Evaluation Dataset Size: {eval_dataset_size} from '{eval_dataset.mode_dir}').")
        print(STR_BARRIER)

    return train_dataset_loader, train_dataset_size, eval_dataset_loader, eval_dataset_size


def batch_train(batch_train_model, batch_train_dataset_loader, batch_train_optimiser, batch_train_criterion, verbose,
                macro_batch_size=None):
    """
    Batch Training:

    This method will apply the usual batch training loop applied in PyTorch machine learning projects but is also
    equipped to apply accumulated gradients when required. Accumulated gradients will allow for images to be loaded in
    individually (micro batches) but the optimiser is only stepped every macro-batch amount of images, which is
    equivalent to simply having the batch size as the macro batch size but has used memory management to do so. This
    allows for larger model input resolutions to be investigated without having to change the batch size hyper-param.

    :param batch_train_model: Model to use for training.
    :param batch_train_dataset_loader: Loader for the training data set.
    :param batch_train_optimiser: Optimiser to use for training.
    :param batch_train_criterion: Loss function to use for training.
    :param verbose: Boolean flag indicating whether to print robust output to console.
    :param macro_batch_size: Batch size to use when performing accumulated gradients.
    :return: Total average training loss across all batches in the training data.
    """
    batch_train_model.train()  # Set model to training mode.
    running_train_loss = 0.0

    for curr_batch_index, batch_samples in enumerate(batch_train_dataset_loader, 0):  # Iterate training batches.

        if verbose:
            print(f'\tTraining Batch {curr_batch_index + 1} of {len(batch_train_dataset_loader)}: '
                  f'Batch Size = {len(batch_samples[0])}')

        # Get the batch training samples: imgs with corresponding labels.
        imgs, labels, file_names = batch_samples
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)  # Send to GPU, if available.

        # Forward Pass:
        predicted = batch_train_model(imgs)
        # Loss: Calculate loss on binary classes only (not the 1000-classes too).
        loss = batch_train_criterion(predicted[0], labels.unsqueeze(1).float())
        # Back-Propagation:
        loss.backward()

        # Step optimiser if not using accumulated gradients.
        # If using accumulated gradients, wait until the macro batch size is met (or at end of data) before updating.
        if (macro_batch_size is None) or ((curr_batch_index + 1) % macro_batch_size == 0) or \
                (curr_batch_index + 1 == len(batch_train_dataset_loader)):

            if (macro_batch_size is not None) and verbose:
                print(f"\tMacro Batch Size Of {macro_batch_size} Met - Applying Accumulated Gradients.")

            # Step optimiser and zero the gradients.
            batch_train_optimiser.step()
            batch_train_optimiser.zero_grad()

            # Update training loss.
            running_train_loss += loss.item() * imgs.size(0)  # Batch loss is (mean loss for batch * batch size).

    return running_train_loss  # Return overall running loss (i.e., total avg loss for all training images).


def batch_eval(batch_eval_model, batch_eval_dataset_loader, batch_eval_criterion, verbose):
    """
    Batch Evaluation (i.e., Validation):

    :param batch_eval_model: Model to use for validation.
    :param batch_eval_dataset_loader: Loader for the validation data set.
    :param batch_eval_criterion: Loss function to use for validation.
    :param verbose: Boolean flag indicating whether to print robust output to console.
    :return: Total average validation loss across all batches in the validation data.
    """
    batch_eval_model.eval()  # Set model to evaluation mode.
    running_eval_loss = 0.0

    for curr_batch_index, batch_samples in enumerate(batch_eval_dataset_loader, 0):  # Iterate evaluation batches.

        if verbose:
            print(f'\tEvaluating Batch {curr_batch_index + 1} of {len(batch_eval_dataset_loader)}: '
                  f'Batch Size = {len(batch_samples[0])}')

        with torch.no_grad():  # Don't need gradients - reduces memory usage and computation time.

            # Get the batch evaluation samples: imgs with corresponding labels.
            imgs, labels, file_names = batch_samples
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)  # Send to GPU.

            # Get model predictions and calculate loss.
            predicted = batch_eval_model(imgs)
            # Loss:
            loss = batch_eval_criterion(predicted, labels.unsqueeze(1).float())

            # Update evaluation loss.
            running_eval_loss += loss.item() * imgs.size(0)  # Batch loss is (mean loss for batch * batch size).

    return running_eval_loss  # Return overall running loss (i.e., total avg loss for all validation images).


def report_epoch_loss(loss_type, epoch_loss, curr_epoch, augment_lvl, verbose):
    """
    Report the epoch loss to console and TensorBoard.

    :param loss_type: Whether reporting training or validation loss.
    :param epoch_loss: Loss for the epoch.
    :param curr_epoch: Current epoch loss is reported for.
    :param augment_lvl: Augmentation (i.e., model version) being used.
    :param verbose: Boolean flag indicating whether to print robust output to console.
    :return: N/A - Output to console.
    """
    if verbose:
        print(f'\tEpoch {loss_type} Loss: {epoch_loss}')  # Print loss update from the epoch.

    # TensorBoard - Visualise epoch loss vs epoch.
    TB_WRITER.add_scalar(f"Whole-Slide Classification - {loss_type} Loss vs. Epoch (Augment Level: {augment_lvl}).",
                         epoch_loss, curr_epoch)
    TB_WRITER.flush()


def train(config, input_dir, output_file, augment_lvl, verbose=True, input_img_size=299):
    """
    Train, validate, and save the model.

    :param config: Configuration of model (i.e., dictionary of hyper-parameter values) used for training.
    :param input_dir: Directory containing train, validate, and test data sub sets (i.e., subdirectories).
    :param output_file: Output file to save the final trained model to.
    :param augment_lvl: Augmentation (i.e., model version) to use for training.
    :param verbose: Boolean flag indicating whether to print robust output to console.
    :param input_img_size: Dimension to use (width and height) for the model input images.
    :return: N/A - Output to console.
    """

    # Early stopping: 10% of max epochs without improvement - prevent over-fitting, give patience to avoid local minima.
    early_stopping_epochs = math.ceil(CONFIG["max_epochs"] / 10)

    #########
    # Set-Up:
    #########

    # Get model, criterion, and optimiser for training.
    model, criterion, optimiser = model_setup(config)

    # When input resolution is greater than 512, we run into out-of-memory errors at the GPU.
    # Therefore, use accumulated gradients to maintain batch size and manage memory.
    macro_batch_size = None
    if input_img_size > 512:
        if verbose:
            print(f"Using Accumulated Gradients Approach To Handle Input Resolution Size.")
        macro_batch_size = config["batch_size"]
        config["batch_size"] = 1  # Read images in one at a time, then update loss after original batch size amount.

    # Get data set loaders (and data set sizes) for the training and validation sets.
    train_dataset_loader, train_dataset_size, eval_dataset_loader, eval_dataset_size = \
        load_data(input_dir, augment_lvl, config["batch_size"], verbose=verbose, input_img_size=input_img_size)

    ###########
    # Training:
    ###########

    min_eval_loss = np.inf  # Keep minimum validation loss to determine when to save the model.
    epochs_without_improv = 0  # Keep track of the number of epochs without improvement for early stopping.

    train_start_time = time.time()  # Used for calculating elapsed training time.

    for curr_epoch in range(config["max_epochs"]):  # Iterate over the dataset multiple times (epochs).

        if verbose:
            print('Epoch {} of {}:'.format(curr_epoch + 1, config["max_epochs"]))

        # Train model on all batches in training data set.
        total_avg_train_loss = batch_train(model, train_dataset_loader, optimiser, criterion, verbose=verbose,
                                           macro_batch_size=macro_batch_size)
        epoch_train_loss = total_avg_train_loss / train_dataset_size  # Epoch Loss = TotalLoss / DatasetSize.

        # Evaluate (i.e., validate) model on all batches in eval data set.
        total_avg_eval_loss = batch_eval(model, eval_dataset_loader, criterion, verbose=verbose)
        epoch_eval_loss = total_avg_eval_loss / eval_dataset_size  # Epoch Loss = TotalLoss / DatasetSize.

        # Output/report the epoch losses depending on whether using manual of hyper-parameter training.
        report_epoch_loss("Training", epoch_train_loss, curr_epoch, augment_lvl, verbose=True)
        report_epoch_loss("Evaluation", epoch_eval_loss, curr_epoch, augment_lvl, verbose=True)

        ################################
        # Early Stopping/Model Snapshot:
        ################################

        # If validation loss has decreased, then save the current model (i.e., snapshot).
        if epoch_eval_loss < (min_eval_loss - MIN_IMPROV):
            if verbose:
                print(f'\tValidation Loss Decreased ({min_eval_loss:.6f} --> {epoch_eval_loss:.6f}).')

            torch.save(model.state_dict(), output_file)  # Saving Model (State Dict).

            if verbose:
                print("\tSaved Trained Model (State Dict) To: '", output_file, "'.")

            min_eval_loss = epoch_eval_loss
            epochs_without_improv = 0  # Reset early stopping count.

        else:
            epochs_without_improv += 1  # No loss improvement; advance towards early stopping condition.

        # Check if early stopping condition has been met.
        if epochs_without_improv == early_stopping_epochs:
            if verbose:
                print(f'Stopping Training: Early stopping condition met - '
                      f'{epochs_without_improv} epochs without criterion improvement.')
            break  # Leave epoch loop.

    # Show the time elapsed for training.
    time_elapsed = time.time() - train_start_time
    if verbose:
        print('Training Complete In {:.0f}m {:.0f}s.'.format(time_elapsed // 60, time_elapsed % 60))


def train_hpo(config, input_dir, augment_lvl, input_img_size, checkpoint_dir=None):
    """
    Train, validate, and save the model using hyper-parameter optimisation.

    NOTE: Could have been implemented in the regular train() method as well, but kept separate for clarity.

    NOTE: Hyper-parameter optimisation not implemented with accumulated gradients as this was not required for the
    project investigation (redundant work). Thus, should not run -r 1024|2048 with -hpo.

    :param config: Configuration of model (i.e., dictionary of hyper-parameter values) used for training.
    :param input_dir: Directory containing train, validate, and test data sub sets (i.e., subdirectories).
    :param augment_lvl: Augmentation (i.e., model version) to use for training.
    :param checkpoint_dir: Required by HPO function API - where to save model checkpoints to.
    :param input_img_size: Dimension to use (width and height) for the model input images.
    :return: N/A - Output to console and ray_results/ folder.
    """

    os.chdir("../../../")  # Move into src/ for working with relative paths (HPO moves to a ray tune folder by default).

    # Early stopping: 10% of max epochs without improvement - prevent over-fitting, give patience to avoid local minima.
    early_stopping_epochs = math.ceil(config["max_epochs"] / 10)

    #########
    # Set-Up:
    #########

    # Get model, criterion, and optimiser for training.
    model, criterion, optimiser = model_setup(config)

    # Hyper-Parameter Optimisation Checkpointing.
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimiser.load_state_dict(optimizer_state)

    # Get data set loaders (and data set sizes) for the training and validation sets.
    train_dataset_loader, train_dataset_size, eval_dataset_loader, eval_dataset_size = \
        load_data(input_dir, augment_lvl, config["batch_size"], verbose=False, input_img_size=input_img_size)

    ###########
    # Training:
    ###########

    min_eval_loss = np.inf  # Keep minimum validation loss to determine when to save the model.
    epochs_without_improv = 0  # Keep track of the number of epochs without improvement for early stopping.

    for curr_epoch in range(config["max_epochs"]):  # Iterate over the dataset multiple times (epochs).

        # Train model on all batches in training data set.
        batch_train(model, train_dataset_loader, optimiser, criterion, verbose=False)

        # Evaluate (i.e., validate) model on all batches in eval data set.
        total_avg_eval_loss = batch_eval(model, eval_dataset_loader, criterion, verbose=False)
        epoch_eval_loss = total_avg_eval_loss / eval_dataset_size  # Epoch Loss = TotalLoss / DatasetSize.

        #####################################
        # Early Stopping/Model Checkpointing:
        #####################################

        # If validation loss has decreased, then save the current model (i.e., snapshot).
        if epoch_eval_loss < (min_eval_loss - MIN_IMPROV):

            # Report to RayTune the epoch validation loss when the validation loss has improved.
            tune.report(loss=epoch_eval_loss)

            # HPO Checkpointing - save model when there is a reduction in validation loss.
            with tune.checkpoint_dir(step=curr_epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((model.state_dict(), optimiser.state_dict()), path)

            min_eval_loss = epoch_eval_loss
            epochs_without_improv = 0  # Reset early stopping count.

        else:
            epochs_without_improv += 1  # No loss improvement; advance towards early stopping condition.

        # Check if early stopping condition has been met.
        if epochs_without_improv == early_stopping_epochs:
            break  # Stop training.


#######################################################################################################################
# Training:
#######################################################################################################################

print("Whole-Slide Classification - TRAINING:")
STR_BARRIER = "-" * 100  # For nice output separation.
print(STR_BARRIER)

# Seed torch and numpy for reproducibility.
SEED = 10
torch.backends.cudnn.deterministic = True
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

# Get device for training: utilise GPU when available.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training Device: ", DEVICE)

# Configuration: Either default values advised from previous HPO, or a HPO config to search.
CONFIG = {
    "batch_size": 16 if not args.hyperparamoptim else tune.choice([8, 16, 32]),  # >32 is too big for GPU memory!
    "learning_rate": 0.01 if not args.hyperparamoptim else tune.choice([1e-4, 1e-3, 1e-2, 1e-1]),
    "weight_decay": 0,  # if not args.hyperparamoptim else tune.qloguniform(0, 0.5, 0.05),  # L2 regularisation.
    "max_epochs": 500 if not args.hyperparamoptim else tune.choice([100, 300, 500, 700, 1000])
}

MIN_IMPROV = 0.001  # Minimum loss improvement (to prevent early stopping).

# Perform training - either regular training, or training with automated hyper-parameter optimisation:
if args.hyperparamoptim:  # Hyper-parameter training.

    reporter = tune.CLIReporter()  # Type of reporter to use for console output.

    tune_name = f"classification_train_a{args.augment}_{time.time()}"
    analysis = tune.run(
        # Run train_hpo each trial.
        partial(train_hpo, input_dir=args.input_dir, augment_lvl=args.augment, input_img_size=args.res),
        resources_per_trial={"cpu": cpu_count(), "gpu": 1},  # Use GPU for HPO when available.
        num_samples=30,  # Number of trials to execute.
        config=CONFIG,  # Search Space.
        progress_reporter=reporter,  # How to output progress to console.
        local_dir="ray_results/",  # Where to save results.
        name=tune_name,  # Name of this RayTune execution.
        fail_fast=True  # If there is a failure in a trial, terminate trial.
    )

    # Get best trial and trial configuration. Trial with the lowest (val) loss reported is best.
    best_trial = analysis.get_best_trial(metric="loss", mode="min", scope="all")
    print(f"Best Trial Config: {best_trial.config}")
    print(f"Best Trial Validation Loss: {best_trial.last_result['loss']}")  # Last reported loss is lowest.

    # Save model of best trial to output file.
    best_model_checkpoint = analysis.get_last_checkpoint(best_trial)  # Last checkpoint (i.e., lowest validation loss).
    model_state_to_save, optimiser_state = torch.load(os.path.join(best_model_checkpoint, "checkpoint"))
    torch.save(model_state_to_save, args.output_filename)  # Save model state dict to desired location.
    print(f"Best Model From '{best_model_checkpoint}' Saved To '{args.output_filename}'.")

else:  # Regular manual training.

    # Create TensorBoard SummaryWriter for visualisations.
    TB_WRITER = SummaryWriter('tensorboard/')

    train(CONFIG, args.input_dir, args.output_filename, args.augment, input_img_size=args.res)
