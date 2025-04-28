
# Potato Leaf Disease Classification Using CNN, VGG16, and EfficientNetB0

##  Overview
This project focuses on detecting potato plant diseases from leaf images using deep learning models. Three models were developed and compared:
- **Custom Convolutional Neural Network (CNN)** (built from scratch)
- **Transfer Learning with VGG16** (pre-trained on ImageNet)
- **Transfer Learning with EfficientNetB0** (pre-trained model)

The objective was to build high-accuracy classification models and understand the impact of using pre-trained architectures versus a custom-built CNN.

---

##  Project Workflow
- **Dataset Loading**: Images of healthy and diseased potato leaves were loaded from a structured directory.
- **Exploratory Data Analysis (EDA)**:
  - Visualized class distribution
  - Analyzed pixel intensity distributions
- **Data Preprocessing**:
  - Resizing images to **256x256**
  - Normalizing pixel values to **[0, 1]** range
  - Data augmentation (random flips and rotations)
- **Model Building**:
  - Custom CNN: Multiple convolutional and pooling layers
  - VGG16: Transfer learning with added custom dense layers
  - EfficientNetB0: Transfer learning with added dense layers
- **Model Training and Evaluation**:
  - Trained over **10 epochs**
  - Evaluated using accuracy, loss curves, confusion matrices, precision, recall, and F1-score.

---

## Dataset Structure
The dataset contains three classes:
- **Potato___Early_blight**
- **Potato___Late_blight**
- **Potato___healthy**

Example structure:
```bash
/dataset_path/
    /Potato___Early_blight/
    /Potato___Late_blight/
    /Potato___healthy/
```

---

## Model Results

| Model                          | Validation Accuracy | Precision | Recall | F1 Score |
|---------------------------------|----------------------|-----------|--------|----------|
| Custom CNN                     | ~95%                 | High      | High   | High     |
| VGG16 (Transfer Learning)       | ~98%                 | 0.98      | 0.98   | 0.98     |
| EfficientNetB0 (Transfer Learning) | ~99%             | 0.99      | 0.99   | 0.99     |

✅ **EfficientNetB0** slightly outperformed other models with the highest accuracy and F1-score.

---

## Technologies Used
- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn

---

##  How to Run
1. Install the necessary libraries:
   ```bash
   pip install tensorflow matplotlib pandas numpy seaborn scikit-learn
   ```
2. Place your dataset in the correct folder structure as shown above.
3. Update the `dataset_path` variable in the code.
4. Run all cells in the provided Jupyter Notebook(s).

---

## Files in This Repository
- **todaycode_09_04.ipynb**: Main notebook containing EDA, model building, and evaluation.
- **models/**:
  - **CNN_Model_Training.ipynb**
  - **VGG16_Model_Training.ipynb**
  - **EfficientNetB0_Model_Training.ipynb**

>  **Note**: Saved model files (.keras) are **NOT uploaded** due to GitHub's 100MB file limit.  
> Trained models are stored separately in Google Drive for access if needed.

---

## 🔗 GitHub Repository Link
[Visit Project Repository](https://github.com/Sushma897sree/Data_Science_Project)

---
