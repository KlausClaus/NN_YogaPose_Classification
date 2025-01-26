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
from keras.layers import GlobalAveragePooling2D, BatchNormalization
from keras.layers import Dense
from keras.layers import Dropout
from keras.models import load_model

# Transfer Learning
from tensorflow.keras.applications import Xception

# Callbacks
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam

# 初始化数据生成器
data_gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,  # 添加水平移动
    height_shift_range=0.2,  # 添加垂直移动
    zoom_range=0.2,  # 添加缩放
    horizontal_flip=True,  # 添加水平翻转
    validation_split=0.2,
    rescale=1. / 255,
    fill_mode='nearest'  # 设置填充模式
)

# 设置数据路径
root_path = r'C:\Users\Anderson\Documents\GitHub\COMP9444_project\Dataset\Yoga-82\yoga_dataset_links_skeleton'

# 获取类别名称
class_names = sorted(os.listdir(root_path))
n_classes = len(class_names)

print(f"Total Number of Classes : {n_classes}")
print(f"Classes : \n{class_names}")

# 加载数据
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


def show_image(image, title=None):
    plt.imshow(image)
    plt.title(title)
    plt.axis('off')


# 加载基础模型
base_model = Xception(
    include_top=False,
    weights='imagenet',
    input_shape=(256, 256, 3)
)
base_model.trainable = False

# 构建优化后的模型结构
model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    BatchNormalization(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(n_classes, activation='softmax')
])

# 编译模型
model.compile(
    optimizer=Adam(learning_rate=1e-4),  # 使用更小的学习率
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 设置回调函数
callbacks = [
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,  # 每次将学习率降低一半
        patience=3,
        verbose=1,
        min_lr=1e-6
    ),
    EarlyStopping(
        patience=8,  # 增加patience
        monitor='val_accuracy',
        mode='max',
        verbose=1,
        restore_best_weights=True
    ),
    ModelCheckpoint(
        'final/yoga_pose_model_best.h5',
        monitor='val_accuracy',
        save_best_only=True,
        mode='max',
        verbose=1
    )
]

# 训练模型
history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=50,
    callbacks=callbacks
)

# 保存最终模型
model.save('yoga_pose_model_final.h5')

# 评估模型
evaluation = model.evaluate(valid_ds)
print(f"Final Loss: {evaluation[0]:.4f}")
print(f"Final Accuracy: {evaluation[1]:.4f}")


# 可视化训练历史
def plot_training_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # 绘制准确率曲线
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()

    # 绘制损失曲线
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()

    plt.show()


# 绘制训练历史
plot_training_history(history)
