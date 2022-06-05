# CS5199-Dissertation

**Breast Cancer Detection in Low Resolution Images:**

Machine learning systems exist for the automatic detection of breast cancer in histopathology whole-slide images with high confidence. 
Such systems can potentially automate large portions of conventional diagnostic procedures used to identify breast cancer, improving support for 
diagnoses via digital second opinion or reducing cognitive load by shifting work away from medical personnel.

However, these current systems are complex as they often fully utilise high-resolution whole-slide images with dimensions that are hundreds of 
thousands of pixels in width and height. Such images represent pathology slides at considerably high magnification. Due to the high resolution of the 
images, these systems are typically resource intensive, requiring either significant time or compute power, which hinders their clinical viability.

This project investigates automated breast cancer detection via deep learning techniques using lower resolution images (i.e., digital histopathology 
slides at a lower magnification). The investigation intends to reveal whether machine learning models can be developed that provide high confidence 
results with some fractional amount of resources by using low- versus high-resolution whole-slide images.


**Information On The Contents Of The Project Directory:**

_CS5199_Report.pdf_
- The final report for the project in PDF format.

_src/_
- Contains the project source code. Primarily, this includes the model training and inference scripts. Also included
  is the _tools/_ subdirectory containing the various data preparation scripts described in the report for making the
  input data set suitable for use. This directory also includes a _ray_results/_ folder, where the hyper-parameter
  optimisation results are stored (not provided given size), and a _tensorboard/_ directory where GUI outputs for training
  are stored (visible via the TensorBoard tool which is a requirement for the program environment). Full user
  instructions for the project source code are found within the appendices of the report.

_models/_
- OMITTED (files too large): Contains the models created for this project. The model names included their expect input resolution (e.g., 299 for
  299 x 299 pixels) as well as the model version used (a0 is model version 0, etc.).

_data/_
- OMITTED (files too large): Contains the low-resolution data sets used in the project for training models at 299 x 299 pixels. This primarily
  includes the full Camelyon data set used in the project. Within the data set folder(s) are the train/, eval/, and
  test/ sub-directories required by the implementation. Note these can be used to replicate the project work carried
  out for models using 299 x 299 pixel inputs, but are not suitable for all higher input resolutions. For higher
  resolutions, the Camelyon data set will have to be downloaded and the data preparation process described in the user
  instructions of the report will have to be followed.

_testdata/_
- Contains a small sub-set of the Camelyon data set used for debugging the source code. Serves no utility purpose now.

_env/_
- Contains the Dockerfile used to create the Docker container (i.e., program environment) at the remote GPU machine.
  This is provided for completeness. Also contains the requirements.txt file which lists the pip packages that are
  required to execute the project code. The only unlisted package required is python3-openslide, whose installation is
  shown in the Dockerfile.
