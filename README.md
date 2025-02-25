Lung Cancer Detection Using CNN (TensorFlow)

Project Overview
This project focuses on using Convolutional Neural Networks (CNN) for detecting lung cancer from histopathological images. The goal is to classify lung tissue samples into normal and cancerous categories. The project uses TensorFlow and Keras to implement the model and achieves an F1 score of over 90% across all classes.

Dataset
The dataset used for this project is sourced from Kaggle and contains images of lung tissue samples, divided into the following categories:

Normal lung tissue
Lung Adenocarcinoma
Lung Squamous Cell Carcinoma
Dataset Size:
Total images: 5,000 images (augmented from original dataset)
The dataset is provided in a zip file and has been split into training and validation sets for model evaluation.

Model Architecture
The model used for this classification task is a Convolutional Neural Network (CNN). Here’s a brief overview of the architecture:

Convolutional Layers: Extract features from images.
Max Pooling Layers: Downsample the feature maps.
Fully Connected Layers: Dense layers for classification with dropout and batch normalization.
Output Layer: Softmax layer for multi-class classification (normal, adenocarcinoma, squamous cell carcinoma).
Training and Results
The model was trained with the following parameters:

Epochs: 10
Batch Size: 64
Validation Split: 20%
The model achieved:

Training Accuracy: ~85%
Validation Accuracy: ~82%
F1 Score: Above 90% for all classes
The model shows strong performance in detecting cancerous lung tissue and classifying it into appropriate categories.

![image](https://github.com/user-attachments/assets/8e187083-d95e-46e3-81b1-ebfcfa3549b0)
![image](https://github.com/user-attachments/assets/db1ec118-adef-454c-ab26-4d4468c018f8)

