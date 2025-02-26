# **Lung Cancer Detection Using CNN (TensorFlow)**  

## **Project Overview**  
In this project, I developed a **Convolutional Neural Network (CNN)** to detect **lung cancer** from histopathological images. The objective was to classify lung tissue samples into **normal** and **cancerous** categories. Using **TensorFlow & Keras**, I trained the model, achieving an **F1 score of over 90%** across all classes.  

---  

## **Dataset**  
- **Source:** Kaggle  
- **Categories:**  
  - **Normal** lung tissue  
  - **Lung Adenocarcinoma**  
  - **Lung Squamous Cell Carcinoma**  

- **Dataset Size:**  
  - **Total Images:** 5,000 (including augmented samples)  

[🔗 Dataset Link](https://drive.google.com/file/d/1-md2LF__FOMctSNxa6af0lBLkJ6EsUUW/view?usp=sharing)  

---  

## **Model Architecture**  
I designed a CNN architecture consisting of:  
- **Convolutional Layers** – Extracting key features from images  
- **Max Pooling Layers** – Downsampling feature maps to reduce dimensions  
- **Fully Connected Layers** – Dense layers for classification, with dropout & batch normalization for regularization  
- **Output Layer** – A **softmax** layer for multi-class classification  

---  

## **Training & Results**  

### **Training Parameters:**  
- **Epochs:** 10  
- **Batch Size:** 64  
- **Validation Split:** 20%  

### **Performance:**  
- **Training Accuracy:** ~85%  
- **Validation Accuracy:** ~82%  
- **F1 Score:** **Above 90%** for all classes  

The model performed well in **classifying lung cancer types** and exhibited **strong generalization** on validation data, making it a reliable approach for lung cancer detection.

Dataset: https://drive.google.com/file/d/1-md2LF__FOMctSNxa6af0lBLkJ6EsUUW/view?usp=sharing



