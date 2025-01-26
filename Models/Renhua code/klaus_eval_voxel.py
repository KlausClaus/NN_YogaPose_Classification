# Import Data Science Libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import itertools
import random

# Import visualization libraries
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
from PIL import Image, ImageFile

# Tensorflow Libraries
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,models
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import Callback, EarlyStopping,ModelCheckpoint,TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import Model
# from tensorflow.keras.layers import Rescaling, Normalization, RandomFlip, RandomRotation
from tensorflow.keras.layers.experimental import preprocessing

# 检查 GPU 是否可用
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if tf.config.list_physical_devices('GPU'):
    print("Using GPU:", tf.config.list_physical_devices('GPU')[0])
else:
    print("No GPU detected, using CPU instead.")


# System libraries
from pathlib import Path
import os.path
import requests
import sys
import os

# Metrics
from sklearn.metrics import classification_report, confusion_matrix























import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader

class YogaConv3d(nn.Module):
    def __init__(self, num_classes):
        super(YogaConv3d, self).__init__()
        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, padding=1)  # Assuming 1 channel for grayscale voxel grid
        self.pool = nn.MaxPool3d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.dropout2 = nn.Dropout(0.25)
        
        self.flatten = nn.Flatten()
        # Update the linear layer's input size based on the new 64x64x64 grid
        self.fc1 = nn.Linear(256 * 4 * 4 * 4, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
        self.dropout3 = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout1(x)
        
        x = torch.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout2(x)

        x = torch.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout2(x)

        x = self.flatten(x)
        x = torch.relu(self.fc1(x))
        x = self.dropout3(x)
        x = self.fc2(x)
        return x
    
# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the saved model
model = YogaConv3d(num_classes=82)  
model.load_state_dict(torch.load("model_checkpoint.pth"))
model = model.to(device)
model.eval()  # Set model to evaluation mode

# Define the Grad-CAM function
def make_gradcam_heatmap(input_tensor, model, target_layer):
    gradients = []
    activations = []
    
    # Hook for gradients and activations
    def save_gradient(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    target_layer.register_forward_hook(lambda m, inp, out: activations.append(out))
    target_layer.register_backward_hook(save_gradient)

    output = model(input_tensor)
    pred_class = output.argmax(dim=1).item()

    # Calculate gradients for the predicted class
    model.zero_grad()
    output[0, pred_class].backward(retain_graph=True)
    
    gradient = gradients[0].cpu().detach()
    activation = activations[0].cpu().detach()
    
    # Global Average Pooling
    pooled_gradients = torch.mean(gradient, dim=[0, 2, 3, 4])
    for i in range(activation.shape[1]):
        activation[:, i, :, :, :] *= pooled_gradients[i]
    
    heatmap = torch.mean(activation, dim=1).squeeze()
    heatmap = torch.relu(heatmap)
    heatmap /= torch.max(heatmap)
    
    return heatmap.numpy()

# Display function for overlaying the heatmap on the input image
def display_gradcam(img_tensor, heatmap, alpha=0.4):
    heatmap = np.uint8(255 * heatmap)
    plt.imshow(img_tensor.squeeze().permute(1, 2, 0).cpu().numpy(), cmap='gray')
    plt.imshow(heatmap, cmap='jet', alpha=alpha)
    plt.colorbar()
    plt.show()

# Load test data
# Assume `test_loader` is defined similarly as in your original code
# Test loader should load data in (1, 32, 32, 32) shape

# Evaluate on random samples
for idx in random.sample(range(len(test_loader.dataset)), 5):
    input_image, label = test_loader.dataset[idx]
    input_image = input_image.unsqueeze(0).to(device)  # Add batch dimension

    # Generate Grad-CAM heatmap
    heatmap = make_gradcam_heatmap(input_image, model, model.conv3)

    # Display the heatmap overlayed on the input image
    display_gradcam(input_image, heatmap)
