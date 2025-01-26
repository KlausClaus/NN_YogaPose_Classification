import os
import keras
import numpy as np
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator
import plotly.express as px
import matplotlib.pyplot as plt

# Model imports
from keras import Sequential
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.models import load_model
from tensorflow.keras.applications import Xception
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, LearningRateScheduler

# Data Generator setup - with augmentation
data_gen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    validation_split=0.2,
    rescale=1. / 255
)

# Load data
root_path = r'C:\Users\Anderson\Documents\GitHub\COMP9444_project\Dataset\Yoga-82\yoga_dataset_links_skeleton'  # Update with your path
train_ds = data_gen.flow_from_directory(
    root_path,
    target_size=(256, 256),
    batch_size=32,
    class_mode='sparse',
    subset='training'
)

valid_ds = data_gen.flow_from_directory(
    root_path,
    target_size=(256, 256),
    batch_size=32,
    class_mode='sparse',
    subset='validation'
)

# Get number of classes
n_classes = len(train_ds.class_indices)


def lr_schedule(epoch):
    """适用于长期训练的学习率调度策略"""
    initial_lr = 0.001

    if epoch < 5:
        return initial_lr
    elif epoch < 15:
        return initial_lr * 0.5
    elif epoch < 30:
        return initial_lr * 0.1
    elif epoch < 50:
        return initial_lr * 0.05
    elif epoch < 70:
        return initial_lr * 0.01
    else:
        return initial_lr * 0.005


def build_model(trainable=False, learning_rate=0.001):
    """构建模型"""
    base_model = Xception(
        include_top=False,
        weights='imagenet',
        input_shape=(256, 256, 3)
    )

    # 设置基础模型是否可训练
    base_model.trainable = trainable

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(n_classes, activation='softmax')
    ])

    optimizer = Adam(learning_rate=learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_model():
    """两阶段训练模型，支持更长时间的训练"""
    # 第一阶段：训练顶层分类器
    print("Stage 1: Training top layers...")
    model = build_model(trainable=False, learning_rate=0.001)

    callbacks_stage1 = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            'yoga_pose_model_stage1.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]

    history1 = model.fit(
        train_ds,
        epochs=30,
        validation_data=valid_ds,
        callbacks=callbacks_stage1
    )

    # 第二阶段：微调整个模型
    print("\nStage 2: Fine-tuning the entire model...")
    model = build_model(trainable=True, learning_rate=0.0001)

    # 解冻后几层进行微调
    for layer in model.layers[0].layers[:-20]:
        layer.trainable = False

    callbacks_stage2 = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True
        ),
        ModelCheckpoint(
            'yoga_pose_model_stage2.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=7,
            min_lr=1e-7,
            verbose=1
        ),
        LearningRateScheduler(lr_schedule, verbose=1)
    ]

    history2 = model.fit(
        train_ds,
        epochs=70,
        validation_data=valid_ds,
        callbacks=callbacks_stage2
    )

    return model, history1, history2


def plot_training_history(history1, history2):
    """改进的训练历史可视化"""
    # 合并两个阶段的历史
    acc = history1.history['accuracy'] + history2.history['accuracy']
    val_acc = history1.history['val_accuracy'] + history2.history['val_accuracy']
    loss = history1.history['loss'] + history2.history['loss']
    val_loss = history1.history['val_loss'] + history2.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(15, 5))

    # 准确率曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'b', label='Training Accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation Accuracy')
    plt.axvline(x=len(history1.history['accuracy']), color='g', linestyle='--', label='Stage Change')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    # 损失曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.axvline(x=len(history1.history['loss']), color='g', linestyle='--', label='Stage Change')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def evaluate_model(model, test_ds):
    """评估模型性能"""
    print("\nEvaluating model...")
    test_loss, test_accuracy = model.evaluate(test_ds)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

    return test_loss, test_accuracy


if __name__ == "__main__":
    print("Starting training process...")
    print(f"Number of classes: {n_classes}")

    # 训练模型
    model, history1, history2 = train_model()

    # 绘制训练历史
    plot_training_history(history1, history2)

    # 评估最终模型
    final_loss, final_accuracy = evaluate_model(model, valid_ds)

    # 保存最终模型
    model_save_path = 'final/yoga_pose_model_final.h5'
    model.save(model_save_path)
    print(f"\nModel saved to {model_save_path}")

    # 打印训练总结
    print("\nTraining Summary:")
    print(f"Total epochs trained: {len(history1.history['accuracy']) + len(history2.history['accuracy'])}")
    print(
        f"Best validation accuracy: {max(history1.history['val_accuracy'] + history2.history['val_accuracy']) * 100:.2f}%")
    print(f"Final validation accuracy: {final_accuracy * 100:.2f}%")