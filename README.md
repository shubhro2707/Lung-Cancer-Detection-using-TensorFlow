# **Lung Cancer Detection Using CNN (TensorFlow)**  

## **Project Overview**  
This project utilizes **Convolutional Neural Networks (CNN)** to detect lung cancer from histopathological images. The goal is to classify lung tissue samples into **normal** and **cancerous** categories. Using **TensorFlow & Keras**, the model achieves an **F1 score of over 90%** across all classes.  

---  

## **Dataset**  
**Source:** Kaggle  
**Categories:**  
- **Normal** lung tissue  
- **Lung Adenocarcinoma**  
- **Lung Squamous Cell Carcinoma**  

**Dataset Size:**  
- **Total Images:** 5,000 (including augmented samples)  

[🔗 Dataset Link](https://drive.google.com/file/d/1-md2LF__FOMctSNxa6af0lBLkJ6EsUUW/view?usp=sharing)  

---  

## **Model Architecture**  
- **Convolutional Layers** – Extract features from images  
- **Max Pooling Layers** – Downsample feature maps  
- **Fully Connected Layers** – Classification with dropout & batch normalization  
- **Output Layer** – Softmax for multi-class classification  

---  

## **Training & Results**  
**Training Parameters:**  
- **Epochs:** 10  
- **Batch Size:** 64  
- **Validation Split:** 20%  

**Performance:**  
- **Training Accuracy:** ~85%  
- **Validation Accuracy:** ~82%  
- **F1 Score:** **Above 90%** for all classes  

The model demonstrates strong performance in detecting cancerous lung tissue and classifying it into appropriate categories.  

Dataset: https://drive.google.com/file/d/1-md2LF__FOMctSNxa6af0lBLkJ6EsUUW/view?usp=sharing



