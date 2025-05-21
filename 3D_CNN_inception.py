#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 14:56:40 2025

@author: smoradi
"""

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import AdamW
from sklearn.metrics import confusion_matrix
import numpy as np
from glob import glob
from os.path import basename, join
import SimpleITK as sitk
from natsort import natsorted
import random

# Set seeds for reproducibility
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)
random.seed(seed)

np.set_printoptions(threshold=np.inf)
# 97.65078, 92.80124 
def load_files(pet_paths, seg_paths, max_value=97.65078):
    images, labels = [], []
    for pet_file, seg_file in zip(pet_paths, seg_paths):
        img_arr = sitk.GetArrayFromImage(sitk.ReadImage(pet_file))
        seg_arr = sitk.GetArrayFromImage(sitk.ReadImage(seg_file))
        masked_arr = np.where(seg_arr == 1, img_arr, 0) / max_value
        label = int(basename(pet_file).split(".")[0][-1])
        images.append(masked_arr)
        labels.append(label)
    return np.asarray(images), np.asarray(labels)

def nifti_loader(path_to_execution):
    train_files_pet = natsorted(glob(join(path_to_execution, "Train", "*", "*PET*nii.gz")))
    train_files_seg = natsorted(glob(join(path_to_execution, "Train", "*", "*mask*nii.gz")))
    test_files_pet = natsorted(glob(join(path_to_execution, "Test", "*", "*PET*nii.gz")))
    test_files_seg = natsorted(glob(join(path_to_execution, "Test", "*", "*mask*nii.gz")))
    train_images, train_labels = load_files(train_files_pet, train_files_seg)
    test_images, test_labels = load_files(test_files_pet, test_files_seg)
    return {"train_images": train_images, "train_labels": train_labels, "test_images": test_images, "test_labels": test_labels}

def inception_module(input_layer):
    conv1x1 = layers.Conv3D(32, (1, 1, 1), activation='relu', padding='same')(input_layer)
    conv3x3 = layers.Conv3D(32, (3, 3, 3), activation='relu', padding='same')(input_layer)
    conv5x5 = layers.Conv3D(32, (5, 5, 5), activation='relu', padding='same')(input_layer)
    pool = layers.MaxPooling3D((3, 3, 3), strides=(1, 1, 1), padding='same')(input_layer)
    concat_layer = layers.concatenate([conv1x1, conv3x3, conv5x5, pool], axis=-1)
    return concat_layer

def create_inception_cnn_model(learning_rate, dropout_rate):
    inputs = layers.Input(shape=(32, 32, 32, 1))
    x = inception_module(inputs)
    x = layers.MaxPooling3D((2, 2, 2))(x)
    x = inception_module(x)
    x = layers.MaxPooling3D((2, 2, 2))(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer=AdamW(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    return model

def train_cnn_and_evaluate(n_reps, initial_lr, dropout_rate):
    test_accuracies, conf_mats = [], []
    for _ in range(n_reps):
        path_to_execution = "/home/smoradi/anaconda3/envs/myenv/Test/3layer/Executions_done/EXECUTIONS/E1"
        data = nifti_loader(path_to_execution)
        x_train, y_train = data["train_images"], data["train_labels"]
        x_test, y_test = data["test_images"], data["test_labels"]
        model = create_inception_cnn_model(initial_lr, dropout_rate)
        model.fit(x_train, y_train, epochs=n_epochs, verbose=0, validation_data=(x_test, y_test))
        test_predictions = np.round(model.predict(x_test)).astype(int)
        test_conf_matrix = confusion_matrix(y_test, test_predictions)
        test_accuracies.append(model.evaluate(x_test, y_test, verbose=0)[1])
        conf_mats.append(test_conf_matrix)
    return test_predictions, conf_mats

# Hyperparameters
n_epochs = 100
n_reps = 1
initial_lr = 0.001
dropout_rate = 0.5

# Training and evaluation
test_predictions, conf_mats = train_cnn_and_evaluate(n_reps, initial_lr, dropout_rate)
print('Confusion Matrices:', conf_mats)
print('Test Prediction:', test_predictions)