# Aircraft Detection Using Deep Learning

## Overview

This project focuses on detecting and classifying military aircraft using deep learning techniques. We employ convolutional neural networks (CNNs) and transfer learning to build models capable of recognizing different types of military aircraft from images with high accuracy.

We explore two advanced deep learning architectures:

- **EfficientNetB3**
- **ResNet50**

Both models are trained and evaluated to compare their performance on this classification task.

## Dataset

The dataset used in this project is sourced from Kaggle:

[Military Aircraft Detection Dataset](https://www.kaggle.com/datasets/a2015003713/militaryaircraftdetectiondataset)

This dataset contains images of various military aircraft types, organized into class-specific directories.

## Description

The code in this project performs the following steps:

1. **Data Loading and Preprocessing**: The script reads image files and their corresponding labels from the dataset directory. It preprocesses the images and splits the data into training, validation, and test sets.

2. **Model Building**: Two deep learning models are constructed using the EfficientNetB3 and ResNet50 architectures. Transfer learning is employed by initializing the models with pre-trained weights from ImageNet and fine-tuning them for our specific task.

3. **Training**: The models are trained using the prepared data. Custom callback functions are implemented to adjust learning rates dynamically and to perform early stopping if the models stop improving.

4. **Evaluation**: After training, the models are evaluated on the test set. Performance metrics such as accuracy, precision, recall, and F1-score are calculated. Confusion matrices are generated to visualize the performance across different classes.

5. **Results Visualization**: The script displays sample predictions alongside the actual labels to provide a visual sense of how well the models are performing.

6. **Model Saving**: The trained models and their class indices are saved for future use or deployment.


 ## Project Structure
 
 ```
 ├── data
 │   ├── raw
 │   │   └── crop
 │   ├── processed
 │   └── results
 ├── src
 │   └── main.py
 ├── models
 │   └── efficientnet_model.h5
 │   └── resnet50_model.h5
 ├── README.md
 └── requirements.txt
 ```
 
 ## Requirements
 
 To install all the dependencies required for this project, run:
 
 ```bash
 pip install -r requirements.txt
 ```
 
 ### Required Libraries
 
 - TensorFlow
 - OpenCV
 - Numpy
 - Pandas
 - Matplotlib
 - Seaborn
 - Scikit-learn

