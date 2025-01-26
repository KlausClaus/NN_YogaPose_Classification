import os
import keras
import numpy as np
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from scipy import ndimage
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from keras.preprocessing.image import ImageDataGenerator
import plotly.express as px
import matplotlib.pyplot as plt
from keras import Sequential, Model, Input
from keras.layers import (
    Dense, Dropout, Conv2D, MaxPooling2D,
    UpSampling2D, GlobalAveragePooling2D,
    Flatten, Reshape
)
from keras.callbacks import EarlyStopping, ModelCheckpoint


# Data Generator setup
data_gen = ImageDataGenerator(
    rotation_range=20,
    validation_split=0.2,
    rescale=1. / 255
)

# Load data
root_path = r'C:\Users\Anderson\Documents\GitHub\COMP9444_project\Dataset\Yoga-82\yoga_dataset_links_skeleton'
train_ds = data_gen.flow_from_directory(
    root_path,
    target_size=(256, 256),
    class_mode='binary',
    subset='training'
)
valid_ds = data_gen.flow_from_directory(
    root_path,
    target_size=(256, 256),
    class_mode='binary',
    subset='validation'
)


# Define Autoencoder
def build_autoencoder(input_shape=(256, 256, 3)):
    # Encoder
    input_img = Input(shape=input_shape)

    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = UpSampling2D((2, 2))(x)
    decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)

    # Create models
    autoencoder = Model(input_img, decoded)
    encoder = Model(input_img, encoded)

    return autoencoder, encoder


# Build and compile autoencoder
autoencoder, encoder = build_autoencoder()
autoencoder.compile(optimizer='adam', loss='mse')


# Train autoencoder
def prepare_data(dataset, num_samples):
    images = []
    labels = []
    for i in range(num_samples):
        img, label = next(dataset)
        images.append(img[0])
        labels.append(label[0])
    return np.array(images), np.array(labels)


# Prepare training data
n_samples = len(train_ds.filenames)
X_train, y_train = prepare_data(train_ds, n_samples)
X_valid, y_valid = prepare_data(valid_ds, len(valid_ds.filenames))

# Train autoencoder
autoencoder.fit(
    X_train, X_train,
    epochs=50,
    batch_size=32,
    shuffle=True,
    validation_data=(X_valid, X_valid),
    callbacks=[EarlyStopping(patience=3, restore_best_weights=True)]
)

# Extract features using encoder
train_features = encoder.predict(X_train)
valid_features = encoder.predict(X_valid)

# Flatten features for Random Forest
train_features_flat = train_features.reshape(train_features.shape[0], -1)
valid_features_flat = valid_features.reshape(valid_features.shape[0], -1)

# Train Random Forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(train_features_flat, y_train)

# Evaluate
y_pred = rf_classifier.predict(valid_features_flat)
print("\nClassification Report:")
print(classification_report(y_valid, y_pred))


# Visualization function for results
def plot_results(original, reconstructed, n=5):
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # Original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(original[i])
        plt.title("Original")
        plt.axis("off")

        # Reconstructed
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(reconstructed[i])
        plt.title("Reconstructed")
        plt.axis("off")
    plt.show()


# Show some results
reconstructed_images = autoencoder.predict(X_valid[:5])
plot_results(X_valid[:5], reconstructed_images)