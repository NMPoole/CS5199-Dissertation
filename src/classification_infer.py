#######################################################################################################################
# CS5199 - Individual Masters Project: Breast Cancer Detection in Low Resolution Images.
# Author - Nathan Poole (170004680)
#
# Whole-Slide Classification - INFER:
#   Given a data set (root directory of images) and a model file, use the model to predict the test set outputs. The
#   inferences are saved in a (specified) output CSV file, formatted as described with the Camelyon challenges.
#
# Usage:
# 'python3 classification_infer.py <-i INPUT_DIR> <-m MODEL_FILE> <-o OUTPUT_FILE> <-a AUGMENT_LEVEL> [-v]'
#   , where:
#       <-i INPUT_DIR> is a path to a root directory containing all input data,
#       <-m MODEL_FILE> is a path to a '.pth' file where the model is saved,
#       <-o OUTPUT_FILE> is a path to an output CSV file to save the inferred results to.
#       <-a AUGMENT_LEVEL> augmentation level used to train the model.
#           - 0 = No Data Augmentation w/ Original Low-Res Images.
#           - 1 = Data Augmentation (i.e., Flip/Rot/Jitter) w/ Original Low-Res Images.
#           - NOTE: Using -a 0|1 implies using original images, but these should still be black content filtered.
#           - 2 = No Data Augmentation w/ Tissue Detected Low-Res Images.
#           - 3 = Data Augmentation w/ Tissue Detected Low-Res Images.
#           - NOTE: Using -a 2|3 implies using tissue detection, so input directory must be for tissue detected images.
#       [-v] optional argument instructing to infer the validation set (used for debugging mainly).
#       [-r <INPUT_RES>] is an optional argument to specify the model input resolution expected (Default is 299 x 299).
#######################################################################################################################

# System Imports:
import argparse
import csv
from datetime import timedelta
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from statistics import mean
import time

# Package Imports:
from sklearn import metrics
import torch
from torch import nn
from torch.backends.cudnn import deterministic
from torch.utils.data import DataLoader
from torchvision.models import inception_v3

# Local Imports:
from classification_data import ClassificationDataset


#######################################################################################################################
# Program Arguments:
#######################################################################################################################

# Adding expected program arguments: '-i' (input directory), '-m' (model file), '-o' (output file).
parser = argparse.ArgumentParser(description="Whole-Slide Classification - INFER:")
parser.add_argument("-i", "--input_dir", dest="input_dir", type=str, required=True,
                    help="Path to root directory containing all input data.")
parser.add_argument("-m", "--model_file", dest="model_file", type=str, required=True,
                    help="Path to a '.pth' file where the model is saved.")
parser.add_argument("-o", "--output_file", dest="output_file", type=str, required=True,
                    help="Path to an output CSV file to save the inferred results to.")
parser.add_argument("-a", "--augment", dest="augment", type=int, required=True,
                    help="Augmentation level used to train the model.")
parser.add_argument("-v", "--validation_infer", action="store_true", required=False,
                    help="Flag indicating to infer the validation set instead of the test set (default).")
parser.add_argument("-r", "--res", nargs='?', const=299, type=int, default=299,
                    help="Input image resolution expected [299, 512, 1024, 2048] "
                         "(Default: 299 - i.e., 299 x 299 pixels).")

# Parse program arguments.
args = parser.parse_args()

# Check input directory exists.
if not os.path.isdir(args.input_dir):
    parser.print_help(sys.stdout)
    print("Error: Given input directory does not exist.")
    sys.exit()

# Check model file exists.
if not os.path.exists(args.model_file):
    parser.print_help(sys.stdout)
    print("Error: Given model file does not exist.")
    sys.exit()

# Check output file is valid location.
output_file_dir = os.path.dirname(args.output_file)
if output_file_dir != "" and not os.path.isdir(output_file_dir):
    parser.print_help(sys.stdout)
    print("Error: Invalid location of output file specified.")
    sys.exit()

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
# Inference/Metric Functions:
#######################################################################################################################

def load_data(input_dir, mode, augment, input_img_size=299):
    """
    Get specified data set (validation or test) as data set loader.

    :param input_dir: Directory containing train, validate, and test data sub sets (i.e., subdirectories).
    :param mode: Data set mode (i.e., which sub set to use of "train", "eval", or "test").
    :param augment: Augmentation level (i.e., model version) being applied.
    :param input_img_size: Dimension to use (width and height) for the model input images.
    :return: DataLoader for the requested data set.
    """

    batch_size = 16  # Process in batches due to GPU memory limits.
    dataset = ClassificationDataset(root_dir=input_dir, mode=mode, augment=augment, width=input_img_size,
                                    height=input_img_size)
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size)

    print(f"Data Set Size: {len(dataset)} (from '{dataset.mode_dir}').")

    return loader


def model_setup(model_file):
    """
    Set up model for making inferences.

    :param model_file: File containing trained model to load to perform inferences with.
    :return: Model loaded from model_file.
    """
    infer_model = inception_v3()
    num_outputs = 1  # Cancer classification (i.e., cancer or no-cancer/normal).
    infer_model.fc = nn.Linear(infer_model.fc.in_features, num_outputs)

    # Load model. Allow use of available device (i.e., can infer with CPU even when model trained with GPU).
    infer_model.load_state_dict(torch.load(model_file, map_location=DEVICE))
    infer_model.to(DEVICE)  # To GPU, if available.
    infer_model.eval()  # Evaluation mode for testing.

    print(f"Inception v3 Model Created - State Dict Loaded From: '{model_file}'.")

    return infer_model


def infer(data_set_loader, infer_model):
    """
    Infer data set using model.

    :param data_set_loader: Loader of the data set to perform inferences on.
    :param infer_model: Model used to make inferences.
    :return: List of all file names in the data set, along with corresponding predictions, expected outputs, and average
             batch inference times.
    """
    # Store file names and corresponding prediction probabilities for each image file in the data set.
    file_names = []
    predictions = []
    labels = []
    avg_infer_times = []

    # Infer data set:
    for curr_batch_index, batch_samples in enumerate(data_set_loader, 0):  # Iterate testing batches.

        print(f"Inferring Batch {curr_batch_index + 1} of {len(data_set_loader)}. "
              f"Batch Size = {len(batch_samples[0])}.")

        with torch.no_grad():  # Don't need gradients - reduces memory usage and computation time.

            # Get the batch samples (i.e., imgs to infer).
            imgs, batch_labels, batch_file_names = batch_samples
            imgs, batch_labels = imgs.to(DEVICE), batch_labels.to(DEVICE)  # Send to GPU, if available.

            test_start_time = time.time()  # Used for calculating elapsed inference time.

            # Predict on batch of images.
            batch_predictions = nn.Sigmoid()(infer_model(imgs))  # Manually apply Sigmoid() to get probabilities.

            time_elapsed = time.time() - test_start_time  # Time elapsed for inference.
            avg_infer_times.append(time_elapsed / len(batch_samples))

            # Add current file names with corresponding predictions and labels to overall list to return.
            file_names.extend(batch_file_names)
            predictions.extend(curr_pred.item() for curr_pred in batch_predictions)
            labels.extend(curr_label.item() for curr_label in batch_labels)

    return file_names, predictions, labels, avg_infer_times


def write_predictions_to_file(output_file, file_names, predictions):
    """
    Write test image file names and corresponding cancer prediction probabilities to CSV file.

    :param output_file: File to write predictions to.
    :param file_names: List of file names to write as first column in CSV format.
    :param predictions: List of corresponding prediction for each file name to write as second column in CSV format.
    :return: N/A - Method writes to file (NOTE: could be made to return status code).
    """
    with open(output_file, 'w') as output_csv_file:

        csv_writer = csv.writer(output_csv_file, delimiter=',')

        for file_name, prediction in list(zip(file_names, predictions)):
            file_name = file_name[:-len(".png")]  # Remove file extension when writing to CSV file.
            csv_writer.writerow([file_name, prediction])  # Write image name and corresponding model prediction to file.

        output_csv_file.flush()

    print(f"Saved Predictions To: '{output_file}'.")


def get_best_thresh(predictions, expected):
    """
    Get the threshold that maximises some desired metric: F1 Score.

    :param predictions: Set of model predictions.
    :param expected: Set of corresponding expected model outputs.
    :return: Threshold that maximises the metric score when calculated using the predictions and expected outputs.
    """
    thresh = -1
    best_metric_score = -1

    for curr_thresh in range(0, 21):
        curr_thresh = curr_thresh / 20  # Iterate over thresholds in range [0, 1] with 0.05 step.

        curr_thresh_preds = np.where(np.array(predictions) >= curr_thresh, 1, 0)  # Apply threshold to predictions.

        # Threshold that maximises F1 score is best: maximise recall and precision.
        curr_metric_score = metrics.f1_score(expected, curr_thresh_preds)

        if curr_metric_score > best_metric_score:  # Update best threshold when required.
            thresh = curr_thresh
            best_metric_score = curr_metric_score

    return thresh


def plot_pred_probs(predictions, labels, aug_lvl):
    """
    Create a figure plotting predicted probabilities coloured by actual class.

    :param predictions: Model predictions.
    :param labels: Corresponding expected model outputs.
    :param aug_lvl: Augmentation level (i.e., model version) used.
    :return: Figure plotting the predicted probabilities coloured by their actual class label to view class separation.
    """
    figure = plt.figure(figsize=(15, 7))
    num_bins = 50

    # Predicted probabilities in negative (i.e., normal) class.
    negatives = [predictions[i] for i, value in enumerate(predictions, 0) if labels[i] == 0]
    plt.hist(negatives, bins=num_bins, label='Negatives', range=(0, 1), alpha=1, color='b')

    # Predicted probabilities in positive class.
    positives = [predictions[i] for i, value in enumerate(predictions, 0) if labels[i] == 1]
    plt.hist(positives, bins=num_bins, label='Positives', range=(0, 1), alpha=0.7, color='r')

    # Add relevant plot titles and adjust figure for interpretability.
    plt.title(f'Plot Of Predicted Probabilities Coloured By Actual Class (Augment Level: {aug_lvl})', fontsize=20)
    plt.xlabel('Predicted Probability Of Being Positive (Cancer) Class', fontsize=20)
    plt.ylabel(f'Number Of Probabilities In Each Bucket', fontsize=20)
    plt.legend(fontsize=15)
    plt.tick_params(axis='both', labelsize=20, pad=5)

    return figure


def incorrect_preds_dist(predictions, labels, file_names):
    """
    Given a set of class predictions (0 or 1) and their expected labels (0 or 1), return a dictionary for the false
    negative and false positives such that each dictionary gives the distribution of string output labels.
    For example, if a prediction is 0 and the expected class is 1, then this is a false negative, so the string label
    associated with the false negative (tumor, itc, micro, macro) is retrieved and its distribution incremented in the
    dictionary.

    :param predictions: Model predictions.
    :param labels: Corresponding expected model outputs (i.e., 0 or 1).
    :param file_names: Corresponding file names, which contain string versions of the labels.
    :return: Dictionaries for the distribution of incorrect predictions according to string label (e.g., "micro").
    """
    fn_breakdown = {}
    fp_breakdown = {}

    for curr_pred, curr_label, curr_file_name in zip(predictions, labels, file_names):

        curr_file_name = curr_file_name.rsplit(".", 1)[0]  # Remove suffix if need be.
        output_class_left = curr_file_name.split("_", 1)[0]  # Get binary label part of file name, if exists.
        output_class_right = curr_file_name.rsplit("_", 1)[1]  # Get finer label part of file name, if exists.

        # Check if finer label given for file, otherwise just use regular binary label.
        if any(label in curr_file_name for label in ["none", "negative", "itc", "micro", "macro"]):
            output_class = output_class_right
        else:
            output_class = output_class_left

        if curr_label == 1 and curr_label != curr_pred:  # Actual is positive, but predicted negative (FN).

            if output_class in fn_breakdown:  # If label already in dictionary, increment occurrences.
                fn_breakdown[output_class] = fn_breakdown[output_class] + 1
            else:  # Otherwise, add label to the dictionary.
                fn_breakdown[output_class] = 1

        elif curr_label == 0 and curr_label != curr_pred:  # Actual is negative, but predicted positive (FP).

            if output_class in fp_breakdown:  # If label already in dictionary, increment occurrences.
                fp_breakdown[output_class] = fp_breakdown[output_class] + 1
            else:  # Otherwise, add label to the dictionary.
                fp_breakdown[output_class] = 1

    return fn_breakdown, fp_breakdown


def calc_conf_interval(labels, predictions):
    """
    Given a set of labels and predictions, calculate the 95% confidence interval via the percentile bootstrapping method
    (using sampling with replacement) to estimate the AUC score range. This aims to account for the fact that the models
    are trained using a seed for reproducible results, but such a seed may happen to give poor performance where other
    choices of seed would not. The 95% confidence interval is also given with Camelyon-16 submissions.
    Adapted From: https://stackoverflow.com/questions/19124239/scikit-learn-roc-curve-with-confidence-intervals

    The labels and predictions are lists such that their indices are aligned. Simply, labels[i] and predictions[i]
    should reference the actual output label and predicted label for the same input image.

    :param labels: Set of actual labels corresponding to the inferred data set.
    :param predictions: Set of predictions corresponding to the inferred data set.
    :return: Lower and upper bounds for the 95% confidence interval.
    """
    # Convert to numpy arrays instead of lists for ease of implementation.
    labels = np.array(labels)
    predictions = np.array(predictions)

    n_bootstraps = 1000  # Lots of bootstraps for stable confidence intervals.
    bootstrapped_scores = []  # List to contain all bootstrapped AUC scores to calculate confidence interval from.

    rng = np.random.RandomState(SEED)  # Control reproducibility.

    # Generating bootstrap scores for confidence interval calculation.
    for i in range(n_bootstraps):

        # Bootstrap by sampling with replacement on the prediction indices.
        indices = rng.randint(0, len(predictions), len(predictions))

        # Need at least one positive and one negative sample for ROC AUC to be defined: reject the sample if not true.
        labels_at_indices = labels[indices]
        if len(np.unique(labels_at_indices)) < 2:  # Two unique values: 0 (normal) and 1 (cancer).
            continue

        score = metrics.roc_auc_score(labels[indices], predictions[indices])  # Calculate bootstrapped AUC score.
        bootstrapped_scores.append(score)

    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()  # Sort bootstrapped AUC scores so confidence interval can be generated.

    # Computing the lower and upper bound of the 95% confidence interval.
    confidence_lower = sorted_scores[int(0.025 * len(sorted_scores))]
    confidence_upper = sorted_scores[int(0.975 * len(sorted_scores))]

    return confidence_lower, confidence_upper


#######################################################################################################################
# Inference:
#######################################################################################################################

print("Whole-Slide Classification - INFER:")
str_barrier = "-" * 100  # For nice output separation.
print(str_barrier)

# Seed to ensure reproducibility.
SEED = 10
torch.backends.cudnn.deterministic = True
torch.manual_seed(SEED)  # Seed from train script.
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)

# Get device for inference: utilise GPU when available.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Inference Device: ", DEVICE)

# Get data set loader.
mode_str = "test" if not args.validation_infer else "eval"
dataset_loader = load_data(args.input_dir, mode=mode_str, augment=args.augment, input_img_size=args.res)

# Get model for inferences.
model = model_setup(args.model_file)
print(str_barrier)

# Get file names with corresponding predictions, labels and inference times for all test data.
all_file_names, all_predictions, all_labels, all_avg_infer_times = infer(dataset_loader, model)
print(str_barrier)

# Write predictions of model in CSV format to specified output file.
write_predictions_to_file(args.output_file, all_file_names, all_predictions)
print(str_barrier)

#######################################################################################################################
# Metrics:
#######################################################################################################################
print("Metrics:")

# Plot predicted probabilities coloured by actual class.
fig_pred_probs_plot = plot_pred_probs(all_predictions, all_labels, args.augment)
fig_pred_probs_plot.show()

# ROC AUC Score:
auc_score = metrics.roc_auc_score(all_labels, all_predictions)
l_interval, h_interval = calc_conf_interval(all_labels, all_predictions)  # Bootstrapping to report confidence interval.
print(f"ROC AUC Score = {auc_score} (95% Confidence Interval: [{l_interval} - {h_interval}])")

# Average Inference Time:
avg_infer_time = mean(all_avg_infer_times)
print(f"Average Inference Time (h:m:s) = {str(timedelta(seconds=avg_infer_time))}")

# Threshold Moving:
best_thresh = get_best_thresh(all_predictions, all_labels)  # Find best threshold.
thresh_predictions = np.where(np.array(all_predictions) >= best_thresh, 1, 0)  # Apply best threshold to predictions.

# Confusion Matrix:
confusion_matrix = metrics.confusion_matrix(all_labels, thresh_predictions)  # Get corresponding confusion matrix.
print(f"Confusion Matrix (Threshold = {best_thresh}):")
print(confusion_matrix)  # Rows are actual, columns are predicted.

# False Predictions Breakdown From The Confusion Matrix:
fn_dist, fp_dist = incorrect_preds_dist(thresh_predictions, all_labels, all_file_names)
print(f"Confusion Matrix FN Distribution Breakdown = {fn_dist}")  # Shows which tumor types are being missed!
print(f"Confusion Matrix FP Distribution Breakdown = {fp_dist}")  # Less informative, just "normal" or "none".

# Accuracy, Precision, Recall, F1-Score:
accuracy = metrics.accuracy_score(all_labels, thresh_predictions)
precision = metrics.precision_score(all_labels, thresh_predictions)
recall = metrics.recall_score(all_labels, thresh_predictions)
f1_score = metrics.f1_score(all_labels, thresh_predictions)
print(f"Confusion Matrix Metrics:\n"
      f"Accuracy = {accuracy}\n"
      f"Precision = {precision}\n"
      f"Recall = {recall}\n"
      f"F1 Score = {f1_score}")

print(str_barrier)
