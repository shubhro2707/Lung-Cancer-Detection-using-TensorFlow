import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
import cv2
import os
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import warnings

warnings.filterwarnings('ignore')

# Extract the dataset
dataset_zip_path = 'lung-and-colon-cancer-histopathological-images.zip'
with ZipFile(dataset_zip_path, 'r') as zip_ref:
    zip_ref.extractall()
    print('Dataset successfully extracted.')

# Visualize sample images from each class
image_directory = 'lung_colon_image_set/lung_image_sets'
categories = os.listdir(image_directory)
print("Categories found:", categories)

for category in categories:
    category_path = os.path.join(image_directory, category)
    category_images = os.listdir(category_path)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f'Random images for {category} class', fontsize=20)

    for i in range(3):
        random_index = np.random.randint(0, len(category_images))
        img_path = os.path.join(category_path, category_images[random_index])
        img = np.array(Image.open(img_path))
        axes[i].imshow(img)
        axes[i].axis('off')
    
    plt.show()

# Prepare the data for training
image_size = 256
validation_split = 0.2
epochs = 10
batch_size = 64

X_data = []
y_data = []

for index, category in enumerate(categories):
    category_images = glob(f'{image_directory}/{category}/*.jpeg')
    for image_path in category_images:
        img = cv2.imread(image_path)
        resized_img = cv2.resize(img, (image_size, image_size))
        X_data.append(resized_img)
        y_data.append(index)

X_data = np.asarray(X_data)
y_data_one_hot = pd.get_dummies(y_data).values

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_data, y_data_one_hot, test_size=validation_split, random_state=2022)
print(f"Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}")

# Define the CNN model architecture
cnn_model = keras.models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=(image_size, image_size, 3), padding='same'),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(2, 2),
    
    layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D(2, 2),
    
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.BatchNormalization(),
    layers.Dense(3, activation='softmax')
])

cnn_model.summary()

# Compile the model
cnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define the callback for early stopping and learning rate reduction
class CustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('val_accuracy') > 0.90:
            print('\nValidation accuracy has reached 90%, stopping training.')
            self.model.stop_training = True

early_stop = EarlyStopping(patience=3, monitor='val_accuracy', restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5, verbose=1)

# Train the model
training_history = cnn_model.fit(X_train, y_train,
                                 validation_data=(X_val, y_val),
                                 batch_size=batch_size,
                                 epochs=epochs,
                                 verbose=1,
                                 callbacks=[early_stop, reduce_lr, CustomCallback()])

# Visualize training history
history_df = pd.DataFrame(training_history.history)
history_df[['loss', 'val_loss']].plot()
history_df[['accuracy', 'val_accuracy']].plot()
plt.show()
