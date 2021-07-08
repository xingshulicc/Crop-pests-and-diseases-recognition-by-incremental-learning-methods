# -*- coding: utf-8 -*-
from __future__ import print_function
"""
Created on Fri Jun 12 10:20:58 2020

@author: Admin
"""

import os
#from keras.models import Model
#from keras.layers import Input
#from keras.layers import Dense
#from keras.layers import Dropout
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
#from keras.callbacks import ModelCheckpoint

#from Bridge_VGG19 import Bridge_VGG
#from VGG_model import VGG_19
#from VGG16_SE import VGG_16
#from VGG19_SE import VGG_19
#from ResNet_50_test import ResNet50
#from keras.applications.vgg19 import VGG19
#from keras.applications.vgg16 import VGG16
#from Weakly_Dense_Net import Weakly_DenseNet
#from SE_weakly_densenet import Weakly_DenseNet
#from Progressive_Network import P_Net
#from Ensemble_model import E_Net
#from Grouped_Weakly_DenseNet import Weakly_DenseNet
#from Narrow_Weakly_DenseNet import Weakly_DenseNet
#from SK_Net_Plus import Weakly_DenseNet
#from SK_GC_NN import Weakly_DenseNet
#from IGC_Split_attention_share_layer import Weakly_DenseNet
#from Split_attention_SK_NN import Weakly_DenseNet
from Split_attention_share_layer_NN import Weakly_DenseNet
#from New_model_comparison_Inception import Weakly_DenseNet
#from SE_DANet import Weakly_DenseNet
from learning_rate import choose


#pre-parameters
os.environ['CUDA_VISIBLE_DEVICES'] = '1' # '1' or '0' GPU

img_height, img_width = 224, 224

if K.image_dim_ordering() == 'th':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

#input_tensor = Input(shape = input_shape)

batch_size = 16
epochs = 500

train_data_dir = os.path.join(os.getcwd(), 'Apple_2020/train')
validation_data_dir = os.path.join(os.getcwd(), 'Apple_2020/validation')

num_classes = 4
nb_train_samples = 1365
nb_validation_samples = 456

model = Weakly_DenseNet(input_shape = input_shape, classes = num_classes)
#base_model = VGG19(include_top = False, 
#                   weights = 'imagenet', 
#                   input_tensor = input_tensor, 
#                   pooling = 'avg')
#
#x = base_model.output
#x = Dense(512, activation = 'relu')(x)
#x = Dropout(rate = 0.5)(x)
#x = Dense(num_classes, activation = 'softmax')(x)
#
#model = Model(base_model.input, outputs= x, name = 'VGG19_Modified')

#load model weights
load_dir = os.path.join(os.getcwd(), 'SKNet_Plus_Plus_CPD_train/without_group_shuffle_num_4_0.003')
model_weights_name = 'keras_trained_weights.h5'
load_path = os.path.join(load_dir, model_weights_name)
model.load_weights(load_path, by_name = True)

for layer in model.layers:
    layer.trainable = True

#for layer in model.layers[:-3]:
#    layer.trainable = False
#for layer in model.layers[-3:]:
#    layer.trainable = True
#list_list = []
#Freeze_Layer_File = 'Normal_test_NonLocal.txt'
#The_file_path = os.path.join(os.getcwd(), Freeze_Layer_File)
#with open(The_file_path, 'r') as fr:
#    for line in fr:
#        list_list.append(int(line.rstrip()))
#
#for index in list_list:
#    model.layers[index].trainable = False

#model.layers[1].trainable = False
#model.layers[5].trainable = False
#model.layers[8].trainable = False
#model.layers[11].trainable = False
#model.layers[20].trainable = False
#model.layers[23].trainable = False            
#model.layers[26].trainable = False            
#model.layers[37].trainable = False            
#model.layers[40].trainable = False            
#model.layers[43].trainable = False            
#model.layers[52].trainable = False            
#model.layers[55].trainable = False            
#model.layers[58].trainable = False 
#model.layers[69].trainable = False            
#model.layers[72].trainable = False            
#model.layers[75].trainable = False 

optimizer = SGD(lr = 0.0001, momentum = 0.9, nesterov = True) 
model.compile(loss = 'categorical_crossentropy', optimizer = optimizer, metrics = ['accuracy'])
model.summary()


train_datagen = ImageDataGenerator(rescale = 1. / 255, 
                                   rotation_range = 30, 
                                   width_shift_range = 0.2, 
                                   height_shift_range = 0.2, 
                                   horizontal_flip = True, 
                                   zoom_range = 0.1, 
                                   fill_mode = 'nearest')

test_datagen = ImageDataGenerator(rescale = 1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size = (img_width, img_height),
    batch_size = batch_size,
    class_mode = 'categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size = (img_width, img_height),
    batch_size = batch_size,
    class_mode = 'categorical')

#set learning rate schedule
lr_monitorable = True
lr_reduce = choose(lr_monitorable = lr_monitorable)

##save the best performance model weights
check_file_path = '/home/xingshuli/Desktop/Journal_paper_exper'
#model_weights_name = 'weights.{epoch:03d}-{val_acc:.4f}.h5'
#save_model_weights_path = os.path.join(check_file_path, model_weights_name)
#checkpoint = ModelCheckpoint(filepath = save_model_weights_path, 
#                             monitor = 'val_acc', 
#                             verbose = 1, 
#                             save_best_only = False, 
#                             save_weights_only = True)

#set callbacks for model fit
callbacks = [lr_reduce]

#model fit
hist = model.fit_generator(
    train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = epochs,
    validation_data = validation_generator,
    validation_steps = nb_validation_samples // batch_size, 
    callbacks = callbacks)

# save whole model
model.save_weights(os.path.join(check_file_path, "DANet_Apple_2020_weights.h5"))

#print acc and stored into acc.txt
f = open('/home/xingshuli/Desktop/acc.txt','w')
f.write(str(hist.history['acc']))
f.close()
#print val_acc and stored into val_acc.txt
f = open('/home/xingshuli/Desktop/val_acc.txt','w')
f.write(str(hist.history['val_acc']))
f.close()
#print val_loss and stored into val_loss.txt   
f = open('/home/xingshuli/Desktop/val_loss.txt', 'w')
f.write(str(hist.history['val_loss']))
f.close()
