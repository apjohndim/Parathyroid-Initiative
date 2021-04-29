# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 21:52:08 2021

@author: John
"""


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.preprocessing import image
from keras import regularizers

#in_shape = (300, 300, 3)
#classes = 2

def make_vgg (in_shape, tune, classes): #tune = 0 is off the self, tune = 1 is scratch, tune 
    
#import pydot
    
    base_model = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=in_shape, pooling=None, classes=classes)
    #base_model = keras.applications.resnet.ResNet50(include_top=False, weights='imagenet', input_shape=(32, 32, 3), pooling='max')
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    #base_model.summary()
    
    if tune == 1:
        for layer in base_model.layers:
            layer.trainable = True
    
    if tune == 0:
        for layer in base_model.layers:
            layer.trainable = False
        
    if tune == 20:   
        for layer in base_model.layers:
            layer.trainable = False
        for layer in base_model.layers[20:]:
            layer.trainable = True
    #base_model.summary()
  
    
    x1 = layer_dict['block5_conv3'].output 
    x1= tf.keras.layers.GlobalAveragePooling2D()(x1)
    #x1 = Flatten()(x1) 
    
    #x = Flatten()(x)
    #x = Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(2500, activation='relu')(x1)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)

    #model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #plot_model(model, to_file='vggmod19.png')
    print("[INFO] Model Compiled!")
    return model

def make_lvgg (in_shape, tune, classes):
    
#import pydot
    
    base_model = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet', input_tensor=None, input_shape=in_shape, pooling=None, classes=classes)
    #base_model = keras.applications.resnet.ResNet50(include_top=False, weights='imagenet', input_shape=(32, 32, 3), pooling='max')
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    #base_model.summary()
    for layer in base_model.layers:
        layer.trainable = False
    for layer in base_model.layers[18:]:
        layer.trainable = True
    #base_model.summary()
    
    # early2 = layer_dict['block2_pool'].output 
    # #early2 = Conv2D(64, (3, 3), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early2)
    # early2 = tf.keras.layers.BatchNormalization()(early2)
    # early2 = tf.keras.layers.Dropout(0.5)(early2)
    # early2= tf.keras.layers.GlobalAveragePooling2D()(early2)
        
    early3 = layer_dict['block3_pool'].output   
    #early3 = Conv2D(128, (3, 3), padding='valid', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early3)
    early3 = tf.keras.layers.BatchNormalization()(early3)
    early3 = tf.keras.layers.Dropout(0.5)(early3)
    early3= tf.keras.layers.GlobalAveragePooling2D()(early3)    
        
    early4 = layer_dict['block4_pool'].output   
    #early4 = Conv2D(256, (3, 3), padding='same', activation='relu', kernel_regularizer=regularizers.l2(0.001))(early4)
    early4 = tf.keras.layers.BatchNormalization()(early4)
    early4 = tf.keras.layers.Dropout(0.5)(early4)
    early4= tf.keras.layers.GlobalAveragePooling2D()(early4)     
    
    x1 = layer_dict['block5_conv3'].output 
    x1= tf.keras.layers.GlobalAveragePooling2D()(x1)
    #x1 = Flatten()(x1)
    x = tf.keras.layers.concatenate([x1, early4, early3], axis=-1)  
    
    #x = Flatten()(x)
    #x = Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(1000, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    #model.summary()
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    #plot_model(model, to_file='vggmod19.png')
    print("[INFO] Model Compiled!")
    return model

def make_xception (in_shape, tune, classes):
    
    base_model = tf.keras.applications.Xception(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=in_shape,
    pooling=None,
    classes=classes)
    
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    
    if tune == 1:
        for layer in base_model.layers:
            layer.trainable = True
    
    if tune == 0:
        for layer in base_model.layers:
            layer.trainable = False
        
    if tune is not 0:   
        for layer in base_model.layers:
            layer.trainable = False
        for layer in base_model.layers[tune:]:
            layer.trainable = True
    
    x1 = layer_dict['block14_sepconv2'].output 
    x1= tf.keras.layers.GlobalAveragePooling2D()(x1)
    #x1 = Flatten()(x1) 
    
    #x = Flatten()(x1)
    #x = Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(2500, activation='relu')(x1)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)

    #model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #plot_model(model, to_file='vggmod19.png')
    print("[INFO] Model Compiled!")
    return model 

def make_inception (in_shape, tune, classes):
    
    base_model = tf.keras.applications.InceptionV3(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=in_shape,
    pooling=None,
    classes=classes)
    
    #for layer in base_model.layers:
        #print(layer.name)
    
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    
    if tune == 1:
        for layer in base_model.layers:
            layer.trainable = True
    
    if tune == 0:
        for layer in base_model.layers:
            layer.trainable = False
        
    if tune != 0:   
        for layer in base_model.layers:
            layer.trainable = False
        for layer in base_model.layers[tune:]:
            layer.trainable = True
    
    x1 = layer_dict['mixed10'].output 
    x1= tf.keras.layers.GlobalAveragePooling2D()(x1)
    #x1 = Flatten()(x1) 
    
    #x = Flatten()(x1)
    #x = Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(2500, activation='relu')(x1)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)

    #model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #plot_model(model, to_file='vggmod19.png')
    #print("[INFO] Model Compiled!")
    return model 

def make_resnet (in_shape, tune, classes):
    
    base_model = tf.keras.applications.ResNet152V2(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=in_shape,
    pooling=None,
    classes=classes)
    
    for layer in base_model.layers:
        print(layer.name)
    
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    
    if tune == 1:
        for layer in base_model.layers:
            layer.trainable = True
    
    if tune == 0:
        for layer in base_model.layers:
            layer.trainable = False
        
    if tune != 0:   
        for layer in base_model.layers:
            layer.trainable = False
        for layer in base_model.layers[tune:]:
            layer.trainable = True
    
    x1 = layer_dict['post_relu'].output 
    x1= tf.keras.layers.GlobalAveragePooling2D()(x1)
    #x1 = Flatten()(x1) 
    
    #x = Flatten()(x1)
    #x = Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(2500, activation='relu')(x1)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)

    #model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #plot_model(model, to_file='vggmod19.png')
    print("[INFO] Model Compiled!")
    return model 


def make_mobile (in_shape, tune, classes):
    
    base_model = tf.keras.applications.MobileNetV2(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=in_shape,
    pooling=None,
    classes=classes)
    
    for layer in base_model.layers:
        print(layer.name)
    
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    
    if tune == 1:
        for layer in base_model.layers:
            layer.trainable = True
    
    if tune == 0:
        for layer in base_model.layers:
            layer.trainable = False
        
    if tune != 0:   
        for layer in base_model.layers:
            layer.trainable = False
        for layer in base_model.layers[tune:]:
            layer.trainable = True
    
    x1 = layer_dict['out_relu'].output 
    x1= tf.keras.layers.GlobalAveragePooling2D()(x1)
    #x1 = Flatten()(x1) 
    
    #x = Flatten()(x1)
    #x = Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(2500, activation='relu')(x1)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)

    #model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #plot_model(model, to_file='vggmod19.png')
    print("[INFO] Model Compiled!")
    return model 


def make_dense (in_shape, tune, classes):
    
    base_model = tf.keras.applications.DenseNet201(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=in_shape,
    pooling=None,
    classes=classes)
    
    for layer in base_model.layers:
        print(layer.name)
    
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    
    if tune == 1:
        for layer in base_model.layers:
            layer.trainable = True
    
    if tune == 0:
        for layer in base_model.layers:
            layer.trainable = False
        
    if tune != 0:   
        for layer in base_model.layers:
            layer.trainable = False
        for layer in base_model.layers[tune:]:
            layer.trainable = True
    
    x1 = layer_dict['relu'].output 
    x1= tf.keras.layers.GlobalAveragePooling2D()(x1)
    #x1 = Flatten()(x1) 
    
    #x = Flatten()(x1)
    #x = Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(2500, activation='relu')(x1)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)

    #model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #plot_model(model, to_file='vggmod19.png')
    print("[INFO] Model Compiled!")
    return model 

def make_eff (in_shape, tune, classes):
    
    base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_tensor=None,
    input_shape=in_shape,
    pooling=None,
    classes=classes)
    
    for layer in base_model.layers:
        print(layer.name)
    
    layer_dict = dict([(layer.name, layer) for layer in base_model.layers])
    
    if tune == 1:
        for layer in base_model.layers:
            layer.trainable = True
    
    if tune == 0:
        for layer in base_model.layers:
            layer.trainable = False
        
    if tune != 0:   
        for layer in base_model.layers:
            layer.trainable = False
        for layer in base_model.layers[tune:]:
            layer.trainable = True
    
    x1 = layer_dict['relu'].output 
    x1= tf.keras.layers.GlobalAveragePooling2D()(x1)
    #x1 = Flatten()(x1) 
    
    #x = Flatten()(x1)
    #x = Dense(256, activation='relu')(x)
    x = tf.keras.layers.Dense(2500, activation='relu')(x1)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(classes, activation='softmax')(x)
    model = tf.keras.Model(inputs=base_model.input, outputs=x)

    #model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #plot_model(model, to_file='vggmod19.png')
    print("[INFO] Model Compiled!")
    return model

def make_self (in_shape, tune, classes):
    
    inputs = tf.keras.Input(shape=in_shape)
    x = tf.keras.layers.BatchNormalization()(inputs)
    x = tf.keras.layers.Conv2D(32, 3, strides=1, padding="valid")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D(2, strides=1, padding="valid")(x)
    x = tf.keras.layers.Conv2D(64, 3, strides=1, padding="valid")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D(2, strides=1, padding="valid")(x)
    x = tf.keras.layers.Conv2D(128, 3, strides=1, padding="valid")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D(2, strides=1, padding="valid")(x)
    x = tf.keras.layers.Conv2D(256, 3, strides=1, padding="valid")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    x = tf.keras.layers.Dense(500, activation="relu")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    outputs = tf.keras.layers.Dense(classes, activation="softmax")(x)
    
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    
    
    
    #x = layers.SeparableConv2D(size, 3, padding="same")(x)
    
    model.summary()
    return model

def make_multi_self (in_shape, tune, classes):
    
    inputs_a = tf.keras.Input(shape=in_shape)
    x = tf.keras.layers.BatchNormalization()(inputs_a)
    x = tf.keras.layers.Conv2D(32, 3, strides=1, padding="valid")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid")(x)
    x = tf.keras.layers.Conv2D(64, 3, strides=1, padding="valid")(x)
    x = tf.keras.layers.Conv2D(64, 3, strides=1, padding="valid")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid")(x)
    x = tf.keras.layers.Conv2D(128, 3, strides=1, padding="valid")(x)
    x = tf.keras.layers.Conv2D(128, 3, strides=1, padding="valid")(x)
    x = tf.keras.layers.Conv2D(128, 3, strides=1, padding="valid")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid")(x)
    x = tf.keras.layers.GlobalMaxPooling2D()(x)
    
    inputs_b = tf.keras.Input(shape=in_shape)
    y = tf.keras.layers.BatchNormalization()(inputs_b)
    y = tf.keras.layers.Conv2D(32, 3, strides=1, padding="valid")(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Activation("relu")(y)
    y = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid")(y)
    y = tf.keras.layers.Conv2D(64, 3, strides=1, padding="valid")(y)
    y = tf.keras.layers.Conv2D(64, 3, strides=1, padding="valid")(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Activation("relu")(y)
    y = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid")(y)
    y = tf.keras.layers.Conv2D(128, 3, strides=1, padding="valid")(y)
    y = tf.keras.layers.Conv2D(128, 3, strides=1, padding="valid")(y)
    y = tf.keras.layers.Conv2D(128, 3, strides=1, padding="valid")(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.Activation("relu")(y)
    y = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid")(y)
    y = tf.keras.layers.GlobalMaxPooling2D()(y)

    inputs_c = tf.keras.Input(shape=in_shape)
    z = tf.keras.layers.BatchNormalization()(inputs_c)
    z = tf.keras.layers.Conv2D(32, 3, strides=1, padding="valid")(z)
    z = tf.keras.layers.BatchNormalization()(z)
    z = tf.keras.layers.Activation("relu")(z)
    z = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid")(z)
    z = tf.keras.layers.Conv2D(64, 3, strides=1, padding="valid")(z)
    z = tf.keras.layers.Conv2D(64, 3, strides=1, padding="valid")(z)
    z = tf.keras.layers.BatchNormalization()(z)
    z = tf.keras.layers.Activation("relu")(z)
    z = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid")(z)
    z = tf.keras.layers.Conv2D(128, 3, strides=1, padding="valid")(z)
    z = tf.keras.layers.Conv2D(128, 3, strides=1, padding="valid")(z)
    z = tf.keras.layers.Conv2D(128, 3, strides=1, padding="valid")(z)
    z = tf.keras.layers.BatchNormalization()(z)
    z = tf.keras.layers.Activation("relu")(z)
    z = tf.keras.layers.MaxPooling2D(2, strides=2, padding="valid")(z)
    z = tf.keras.layers.GlobalMaxPooling2D()(z)    
    
    
    output = tf.keras.layers.concatenate([x, y, z], axis=1)
    
    
    
    
    exodus = tf.keras.layers.Dropout(0.5)(output)
    exodus = tf.keras.layers.Dense(1200, activation="relu")(exodus)
    exodus = tf.keras.layers.BatchNormalization()(exodus)
    exodus = tf.keras.layers.Dense(512, activation="relu")(exodus)
    outputs = tf.keras.layers.Dense(classes, activation="softmax")(exodus)
    
    model = tf.keras.Model(inputs=[inputs_a,inputs_b,inputs_c], outputs = outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    
    
    
    #x = layers.SeparableConv2D(size, 3, padding="same")(x)
    
    model.summary()
    return model


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
    
    model.summary()
    return model
    