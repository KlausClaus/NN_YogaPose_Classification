# Common
import os
import keras
import numpy as np
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from scipy import ndimage

# Data
from keras.preprocessing.image import ImageDataGenerator

# Data VIz
import plotly.express as px
import matplotlib.pyplot as plt

# Model
from keras import Sequential
from keras.layers import GlobalAveragePooling2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import load_model

# Transfer Learning
from tensorflow.keras.applications import Xception

# Callbacks
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

# Initialize data generator
data_gen = ImageDataGenerator(
    rotation_range=20,
    validation_split=0.2,
    rescale=1./255
)

# This is the main path to our data set
root_path = r'C:\Users\Anderson\Documents\GitHub\COMP9444_project\Dataset\Yoga-82\yoga_dataset_links_skeleton'

# The folder names are our Classes
class_names = sorted(os.listdir(root_path))
n_classes = len(class_names)

print(f"Total Number of Classes : {n_classes}")
print(f"Classes : \n{class_names}")

# Load Data
train_ds = data_gen.flow_from_directory(root_path, target_size=(256,256), class_mode='binary', subset='training')
valid_ds = data_gen.flow_from_directory(root_path, target_size=(256,256), class_mode='binary', subset='validation')

def show_image(image, title=None):
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')

# Loading Xception for Transfer-Learning
base_model = Xception(
    include_top=False,
    weights='imagenet',
    input_shape=(256,256,3)
)
base_model.trainable=False

# Model Architecture
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(n_classes, activation='softmax')
])

# Compiling Model
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# Callbacks
# [修改1] 添加ModelCheckpoint来保存最佳模型
checkpoint = ModelCheckpoint(
    'final/yoga_pose_model_best.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# [修改2] 在callbacks列表中添加checkpoint
callbacks = [
    EarlyStopping(
        patience=7,              # 1. 允许5轮没有提升才停止
        monitor='val_accuracy',  # 2. 监控验证集准确率
        mode='max',             # 3. 期望准确率越大越好
        verbose=1,              # 4. 打印停止相关的信息
        restore_best_weights=True  # 5. 恢复最佳模型权重
    ),
    ModelCheckpoint(
        'final/yoga_pose_model_best.h5', # 6. 保存的模型文件名
        monitor='val_accuracy',     # 7. 监控验证集准确率
        save_best_only=True,       # 8. 只保存最好的模型
        mode='max',                # 9. 期望准确率越大越好
        verbose=1                  # 10. 打印保存信息
    )
]

# [修改3] 保存训练历史
history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=50,
    callbacks=callbacks
)

# [修改4] 训练完成后保存最终模型
model.save('yoga_pose_model_final.h5')

# Model Evaluation
model.evaluate(valid_ds)