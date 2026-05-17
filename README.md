# Deep Learning Medical Diagnosis Predictor

## Objective
Respiratory diseases are one of the leading drivers of ER visits, accounting for nearly one in three patients. Despite this, medical professionals still rely on manual interpretation of x-ray images, which is both time consuming and a bottleneck in high-volume emergency settings. This project examines the feasibility of utilizing machine learning to streamline interpretation and diagnosis of lung diseases by comparing multiple custom and pretrained models, while exploring what a potential deep learning medical tool could look like.  

### Models Compared
- Custom Multi-Layer Perceptron (MLP)
- Custom Convolutional Neural Network (CNN)
- ResNet18 (pretrained)
- DenseNet121 (pretrained)

### Dataset
- **Source**: [National Institute of Health (NIH) via Kaggle (2017)](https://www.kaggle.com/datasets/nih-chest-xrays/data/data?select=Data_Entry_2017.csv)
- Size: 112,120 Chest x-ray images
- Patients: 30,805 unique patients
- Multi-label classification across 14 disease classes

## Project Phases
1. Data Extraction
2. Data Cleaning
3. Exploratory Data Analysis (EDA)
4. Data Preprocessing
5. Model Training & Evaluation
6. Clinical Web Application