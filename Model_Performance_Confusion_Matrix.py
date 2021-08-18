#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Tue May 11 13:31:29 2021

@author: xingshuli
"""
import os
import csv
import numpy as np
from keras import backend as K
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from SE_DANet import Weakly_DenseNet
#from SE_weakly_densenet import Weakly_DenseNet
#from Ensemble_model import E_Net
#from Progressive_Network import P_Net
 
from sklearn.metrics import classification_report, confusion_matrix
from sklearn_plot_confusion_matrix import plot_confusion_matrix
import matplotlib.pyplot as plt

#from keras.models import load_model

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

#model hyper-parameters
batch_size = 16
num_classes = 14
img_height, img_width = 224, 224

if K.image_dim_ordering() == 'th':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

#load test images
validation_data_dir = os.path.join(os.getcwd(), 'S_PD_images/validation')
nb_validation_samples = 858

#load data
test_datagen = ImageDataGenerator(rescale = 1. / 255)
validation_generator = test_datagen.flow_from_directory(
        validation_data_dir, 
        target_size = (img_width, img_height), 
        batch_size = batch_size, 
        class_mode = 'categorical', 
        shuffle = False)

#model reconstruct
model = Weakly_DenseNet(input_shape = input_shape, classes = num_classes)

#load model weights
#load_dir = os.path.join(os.getcwd(), 'SKNet_Plus_Plus_CPD_train/Join_training/E_Net')
#model_weights_name = 'keras_trained_weights_E_Net.h5'
load_dir = '/home/xingshuli/Desktop/model_save_SE_DANet'
#model_name = 'model.424-0.8884.h5'
#model_path = os.path.join(load_dir, model_name)
model_weights_name = 'weights.007-0.8064.h5'
load_path = os.path.join(load_dir, model_weights_name)
model.load_weights(load_path, by_name = True)
#model = load_model(model_path)

#model compile
optimizer = SGD(lr = 1e-3, momentum = 0.9, nesterov = True)
model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy']) 

#model test
print("-- Evaluate --")
steps = validation_generator.n // validation_generator.batch_size
scores = model.evaluate_generator(validation_generator, steps)
eval_acc = scores[1] * 100
print(f"{model.metrics_names[0]}:{scores[0]}\n{model.metrics_names[1]}:{eval_acc:.4f}")

# Confution Matrix and Classification Report
labels = ['c_l', 't', 'a', 'sl', 't_s_s_m', 'w_f', 
          'an', 'g_m', 'l_b', 'l_sc', 'l_s', 'p_m', 
          'h_f', 'h_l']
#labels = [str(i) for i in range(45)]

validation_generator.reset()
Y_pred = model.predict_generator(validation_generator, steps+1)
classes = validation_generator.classes[validation_generator.index_array]
y_pred = np.argmax(Y_pred, axis=-1)
cnt = 0
for i in range(len(y_pred)):
    if y_pred[i] == classes[i]:
        cnt += 1
print(f"Prediction acc: {(cnt/nb_validation_samples)*100:.4f}\n\n")

print('-- Confusion Matrix --')
conf_mat = confusion_matrix(validation_generator.classes[validation_generator.index_array], y_pred)
print(conf_mat)
with open(f"report/confusion-matrix_SE_DANet_dp.csv", "w") as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)
    # write.writerow(labels)
    write.writerows(conf_mat)
cls_report = classification_report(validation_generator.classes[validation_generator.index_array], 
                                   y_pred, 
                                   target_names=labels, 
                                   digits = 4)
print(cls_report)
file = open(f"report/classification_report_SE_DANet_dp.txt", "w")
file.write("Evaluate\n")
file.write(f"{model.metrics_names[1]}: {eval_acc:.4f}\n\n")
file.write("Classification report\n\n")
file.write(cls_report)
file.close()

plt.figure()
plot_confusion_matrix(conf_mat, 
                      labels, 
                      normalize=False, 
                      title='Confusion matrix', 
                      cmap=plt.cm.Blues)

#plt.savefig('confusion_matrix_SE_DANet.png', dpi = 1800)

