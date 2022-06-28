# -*- coding: utf-8 -*-
"""
Created on Tue Jun 28 18:15:16 2022

@author: John
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization, Lambda
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D, Input
from tensorflow.keras.preprocessing import image
from tensorflow.keras import regularizers
import tensorflow.keras.backend as K

def make_3_vggs(in_shape, tune, classes):
    
    base_model_early = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=in_shape, pooling=None, classes=classes)

    for layer in base_model_early.layers:
        layer._name = layer._name + str('_E')  
    
    layer_dict = dict([(layer.name, layer) for layer in base_model_early.layers])

    for layer in base_model_early.layers:
        layer.trainable = False
    for layer in base_model_early.layers[28:]:
        layer.trainable = True
        
    early3 = layer_dict['block3_pool_E'].output   
    #early3 = Conv2D(128, (3, 3), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early3)
    early3 = tf.keras.layers.BatchNormalization()(early3)
    early3 = tf.keras.layers.Dropout(0.5)(early3)
    early3= tf.keras.layers.GlobalAveragePooling2D()(early3)    
        
    early4 = layer_dict['block4_pool_E'].output   
    #early4 = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early4)
    early4 = tf.keras.layers.BatchNormalization()(early4)
    early4 = tf.keras.layers.Dropout(0.5)(early4)
    early4= tf.keras.layers.GlobalAveragePooling2D()(early4)     
    
    x1 = layer_dict['block5_conv3_E'].output 
    x1= tf.keras.layers.GlobalAveragePooling2D()(x1)
    x = tf.keras.layers.concatenate([x1, early4, early3], axis=-1)  
    

    

    base_model_late = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=in_shape, pooling=None, classes=classes)

    for layer in base_model_late.layers:
        layer._name = layer._name + str('_L')  
    layer_dict_late = dict([(layer.name, layer) for layer in base_model_late.layers])
    for layer in base_model_late.layers:
        layer.trainable = False
    for layer in base_model_late.layers[28:]:
        layer.trainable = True
        
    early3_late = layer_dict_late['block3_pool_L'].output   
    #early3 = Conv2D(128, (3, 3), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early3)
    early3_late = tf.keras.layers.BatchNormalization()(early3_late)
    early3_late = tf.keras.layers.Dropout(0.5)(early3_late)
    early3_late= tf.keras.layers.GlobalAveragePooling2D()(early3_late)    
        
    early4_late = layer_dict_late['block4_pool_L'].output   
    #early4 = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early4)
    early4_late = tf.keras.layers.BatchNormalization()(early4_late)
    early4_late = tf.keras.layers.Dropout(0.5)(early4_late)
    early4_late = tf.keras.layers.GlobalAveragePooling2D()(early4_late)     
    
    y1 = layer_dict_late['block5_conv3_L'].output 
    y1= tf.keras.layers.GlobalAveragePooling2D()(y1)
    y = tf.keras.layers.concatenate([y1, early4_late, early3_late], axis=-1)  
    
 

    base_model_sub = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=in_shape, pooling=None, classes=classes)
    
    for layer in base_model_sub.layers:
        layer._name = layer._name + str('_S')  
    
    layer_dict_sub = dict([(layer.name, layer) for layer in base_model_sub.layers])
    for layer in base_model_sub.layers:
        layer.trainable = False
    for layer in base_model_sub.layers[28:]:
        layer.trainable = True
        
    early3_sub = layer_dict_sub['block3_pool_S'].output   
    #early3 = Conv2D(128, (3, 3), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early3)
    early3_sub = tf.keras.layers.BatchNormalization()(early3_sub)
    early3_sub = tf.keras.layers.Dropout(0.5)(early3_sub)
    early3_sub = tf.keras.layers.GlobalAveragePooling2D()(early3_sub)    
        
    early4_sub = layer_dict_sub['block4_pool_S'].output   
    #early4 = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early4)
    early4_sub = tf.keras.layers.BatchNormalization()(early4_sub)
    early4_sub = tf.keras.layers.Dropout(0.5)(early4_sub)
    early4_sub = tf.keras.layers.GlobalAveragePooling2D()(early4_sub)     
    
    z1 = layer_dict_sub['block5_conv3_S'].output 
    z1= tf.keras.layers.GlobalAveragePooling2D()(z1)
    z = tf.keras.layers.concatenate([z1, early4_sub, early3_sub], axis=-1) 

    
    exodus = tf.keras.layers.concatenate([x, y, z], axis=-1) 
    
    exodus = tf.keras.layers.Dropout(0.5)(exodus)
    exodus = tf.keras.layers.Dense(750, activation="relu")(exodus)
    exodus = tf.keras.layers.BatchNormalization()(exodus)
    exodus = tf.keras.layers.Dense(256, activation="relu")(exodus)
    outputs = tf.keras.layers.Dense(classes, activation="softmax")(exodus)    
    
    model = tf.keras.Model(inputs=[base_model_early.input,base_model_late.input,base_model_sub.input], outputs = outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    #model.summary()
    return model