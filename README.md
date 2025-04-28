# Data_Science_Project
Potato Leaf Disease Classification Using CNN, VGG16, and EfficientNetB0
Overview
This project focuses on detecting potato plant diseases from leaf images using deep learning models.
Three models were developed and compared:

A Custom Convolutional Neural Network (CNN)

A Transfer Learning model using VGG16

A Transfer Learning model using EfficientNetB0

The objective was to build high-accuracy classification models and understand the impact of using pre-trained architectures versus a custom-built CNN.

Project Workflow
Dataset Loading: Images of healthy and diseased potato leaves were loaded from a structured directory.

Exploratory Data Analysis (EDA): Class distribution and pixel intensity distributions were visualized.

Data Preprocessing:

Resizing images to 256x256

Normalizing pixel values to the [0, 1] range

Data augmentation (random flips and rotations)

Model Building:

Custom CNN: Built from scratch with multiple convolution and pooling layers.

VGG16: Pre-trained on ImageNet; custom dense layers added.

EfficientNetB0: Pre-trained model with added dense layers.

Model Training and Evaluation:

Training over 10 epochs

Performance evaluation using accuracy, loss curves, confusion matrices, and precision/recall/F1-scores.

Dataset Structure
The dataset contains three classes:

Potato___Early_blight

Potato___Late_blight

Potato___healthy

Each class has images organized into respective subfolders.

Example structure:

bash
Copy
Edit
/dataset_path
    /Potato___Early_blight
    /Potato___Late_blight
    /Potato___healthy
Model Results

Model	Validation Accuracy	Precision	Recall	F1 Score
Custom CNN	~95%	High	High	High
VGG16 (Transfer Learning)	~98%	0.98	0.98	0.98
EfficientNetB0 (Transfer Learning)	~99%	0.99	0.99	0.99
✅ EfficientNetB0 slightly outperformed other models with the highest accuracy and F1-score.

Technologies Used
Python

TensorFlow / Keras

NumPy

Matplotlib

Seaborn

Scikit-learn

How to Run
Install the necessary libraries:

bash
Copy
Edit
pip install tensorflow matplotlib pandas numpy seaborn scikit-learn
Place your dataset in the correct folder structure.

Update the dataset_path variable in the code.

Run all cells in the Jupyter Notebook.

Files in this Repository
todaycode_09_04.ipynb: Main notebook containing all code (EDA, model building, evaluation)

Saved models (CNN, VGG16, EfficientNetB0) are stored separately (not pushed to GitHub due to file size limits).

Note on Large Files
Trained model files (.keras) are not uploaded here due to GitHub's 100MB file limit.

Models are saved in Google Drive for separate access if needed.

GitHub Link
👉 [Insert your GitHub Repository Link Here]

