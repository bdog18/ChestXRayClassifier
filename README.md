# Chest X-Ray Classifier (COVID-19 vs. Pneumonia vs. Normal)

This project uses machine learning models to classify chest X-ray images into one of three categories: **COVID-19**, **Pneumonia**, or **Normal**. 
It compares raw pixel-based features with texture-based features (GLCM), and evaluates multiple classifiers with hyperparameter tuning and model selection.

## Features

- Image cleaning via Roboflow
- GLCM texture feature extraction
- Model comparisons: Logistic Regression, Random Forest, SVC, k-NN
- Evaluation using Accuracy, F1, ROC-AUC, Confusion Matrix
- Feature importance via permutation method
- Model saving (`joblib`) and reproducibility via `.env`


## Setup Instructions

### 1. Create and activate a virtual environment

**Windows:**
bash
python -m venv venv
.\venv\Scripts\activate

**macOS/Linux**
python3 -m venv venv
source venv/bin/activate

### 2. Install Requirements
pip install -r requirements.txt

### 3. Run the project
Run main.ipynb in Jupyter Notebook or VSCode.

### 4. Optional
Set your Roboflow API key in a .env file:
    ROBOFLOW_API_KEY=your_key_here


## Folder Structure
dataroot/: Raw and cleaned X-ray images
models/: Saved model files (pickle format)

## Acknowledgments
Dataset source: Mendeley Chest X-Ray Repository
Roboflow API for image cleaning