# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 21:50:16 2021

@author: John
"""

'''CODES FOR PARATHYROID'''

import matplotlib.pyplot as plt
plt.style.use('ggplot')
from imutils import paths
import numpy as np
import cv2
import os
import sys
from PIL import Image 
from sklearn.model_selection import KFold
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from imutils import paths
import numpy as np
import os
import numpy
import tensorflow as tf
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

sys.path.insert(1, 'C:\\Users\\User\\ZZZ. Parathyroid 1')

# sys.path.insert(1, 'C:\\Users\\japostol\\Downloads\\Para')

from data_loader import load_parathyroid,load_parathyroid_perioxes, load_parathyroid_dual, load_parathyroid_multi
from main_functions import train,model_save_load,feature_maps,train_multi



#%% PARAMETER ASSIGNEMENT

##############################################################################################################################################################################################################################################################################################################################
#
#
#
#
#   in_shape: must be tuple in channels-last format. E.G. (200,200,3)
#   tune: a paramter to control trainable layers. Tune=1 means training from scratch. Tune = 0 means feature extraction via transfer learning. Any other number 
#   classes: the number of classes
#   epochs: the number of training epochs
#   batch_size: the desired batch size
#   n_split: integer, number of k where K= k-fold cross validation
#   augmentation: True or False. Choose True to perform predefined augmentations
#   verbose: True/False. True prints every step
#   class_names: A list containing all the class names
#   model: the model you want to use. Accepts: 'vgg','lvgg','xception','inception','dense','resnet','mobile','efficient'
#
#
#
#
#
#
###############################################################################################################################################################
###############################################################################################################################################################

#excel_path = 'C:\\Users\\japostol\\Downloads\\Para\\PARATHYROID-for Giannis.xls'



'''PARAMETERS'''


''' SIMPLE PHASE (DUAL+SUB) NO PERIOXES'''
# path = "C:\\Users\\User\\ZZZ. Parathyroid 1\\PreExp\\Test\\CROPPED\\"
# #path = "C:\\Users\\User\\ZZZ. Parathyroid 1\\PreExp\\Test\\CROPPED_NO_FACE_ALL3\\"
# excel_path = "C:\\Users\\User\\ZZZ. Parathyroid 1\\PreExp\\Test\\labels.xls" #needs xls not xlsx
#excel_path = "C:\\Users\\User\\ZZZ. Parathyroid 1\\PreExp\\Test\\PARATHYROID-for Giannis.xls"
#path = "C:\\Users\\User\\ZZZ. Parathyroid 1\\PreExp\\Test\\SPECIAL\\CROPPED_PERIOXES_NOFACE_ALL3\\"





'''PERIOXES'''
# path = 'C:\\Users\\User\\ZZZ. Parathyroid 1\\PreExp\\Test\\SPECIAL\\CROPPED_PERIOXES_NOFACE_ALL3\\'
# excel_path = 'C:\\Users\\User\\ZZZ. Parathyroid 1\\PreExp\\Test\\labels_perioxes.xls'


'''DUAL PHASE ONLY'''
path = "C:\\Users\\User\\ZZZ. Parathyroid 1\\PreExp\\Test\\CROPPED_NOSUBS\\"
excel_path = "C:\\Users\\User\\ZZZ. Parathyroid 1\\PreExp\\Test\\labels_dual.xls" #needs xls not xlsx
# excel_path = "C:\\Users\\User\\ZZZ. Parathyroid 1\\PreExp\\Test\\labels_surgery.xls" #needs xls not xlsx



'''SUB ONLY'''
# path = "C:\\Users\\User\\ZZZ. Parathyroid 1\\PreExp\\Test\\CROPPED_ONLY_SUB\\"
# excel_path = "C:\\Users\\User\\ZZZ. Parathyroid 1\\PreExp\\Test\\labels_sub.xls" #needs xls not xlsx


'''MULTI'''
# path_1 = 'C:\\Users\\User\\ZZZ. Parathyroid 1\\PreExp\\Test\\CROPPED_ONLY_EARLY\\'
# path_2 = 'C:\\Users\\User\\ZZZ. Parathyroid 1\\PreExp\\Test\\CROPPED_ONLY_LATE\\'
# path_3 = 'C:\\Users\\User\\ZZZ. Parathyroid 1\\PreExp\\Test\\CROPPED_ONLY_SUB\\'
# excel_path = "C:\\Users\\User\\ZZZ. Parathyroid 1\\PreExp\\Test\\labels.xls"







#in_shape = (100,279,3) #width,heigh,channels NO_PERIOXES

'''PERIOXES'''
#in_shape = (175,489,3) #PERIOXES
#in_shape = (200,558,3) #for attention, image diminsions must be >139


'''DUAL'''
# in_shape  = (350,652,3) # dual
in_shape  = (175,326,3) # dual


'''SUB ONLY'''
# in_shape  = (326,350,3) # SUBONLY


'''MULTI'''
# in_shape  = (326,350,3) # MULTI
# in_shape  = (163,175,3) # MULTI


tune = 1 # SET: 1 FOR TRAINING SCRATCH, 0 FOR OFF THE SHELF, INTEGER FOR TRAINABLE LAYERS (FROM TUNE AND DOWN, THE LAYERS WILL BE TRAINABLE)
#tune = 249 # for inceptionV3
classes = 2
epochs = 40
batch_size = 8
n_split = 10
augmentation=False
verbose=True
class_names = ["Healhty", "Parathyroid"]
model = 'lvgg'
# model = '3vggs'





#%%
##%% IMAGE LOAD




  # LOAD IMAGES WITH DIFERRENT FUNCTIONS
  
  
'''SIMPLE'''
#data, labels, labeltemp, image,image_max = load_parathyroid (path,excel_path,in_shape,verbose=False)


'''PERIOXES'''
#data, labels, labeltemp, image,image_max = load_parathyroid_perioxes (path,excel_path,in_shape,verbose=False)


'''DUAL_ONLY'''
data, labels, labeltemp, image,image_max = load_parathyroid_dual (path,excel_path,in_shape,verbose=False)

'''SUB_ONLY'''
#data, labels, labeltemp, image,image_max = load_parathyroid_dual (path,excel_path,in_shape,verbose=False)


'''MULTI'''
# data_early, data_late, data_sub, labels, labeltemp, image, image_max, info = load_parathyroid_multi (path_1, path_2, path_3, excel_path, in_shape, verbose=False)


  # SHOW A SAMPLE
from matplotlib import pyplot as plt
plt.imshow(image, interpolation='nearest')
plt.show()
img = data[4,:,:,:]
# img = img*10
plt.imshow(img, interpolation='nearest')
plt.show()







#%% FIT THE MODEL TO THE DATA (FOR PHASE 2)


''' TRAIN - EVALUATE MODEL - GET METRICS '''

in_shape_model = (in_shape[1],in_shape[0],in_shape[2])


##### NON MULTI#####

model3, group_results, fold_scores, pd_metrics,predictions_all,predictions_all_num,test_labels,labels,conf_final,cnf_matrix,conf_img,history = train(data,labels,epochs=epochs,batch_size=batch_size, model=model, in_shape=in_shape_model, tune=tune, classes=classes,n_split=n_split,augmentation=augmentation,verbose=verbose,logs=True,plot_results=True,class_names=class_names,save_variables=True)


##### MULTI#####

# model3, group_results, fold_scores, pd_metrics,predictions_all,predictions_all_num,test_labels,labels,conf_final,cnf_matrix,conf_img,history = train_multi(data_early,data_late,data_sub,labels=labels,epochs=epochs,batch_size=batch_size, model=model, in_shape=in_shape_model, tune=tune, classes=classes,n_split=n_split,augmentation=augmentation,verbose=verbose,logs=True,plot_results=True,class_names=class_names,save_variables=True)





#%%


''' GET TRAINED MODEL '''


loaded_trained_model = model_save_load(data,labels,epochs=epochs,batch_size=batch_size, model=model, in_shape=in_shape, tune=tune, classes=classes,n_split=n_split,augmentation=augmentation,verbose=verbose)






#%%


''' SAVE FEATURE MAPS '''

save_path = 'C:\\Users\\User\\'
returned_img_path = feature_maps(path,save_path,loaded_trained_model,in_shape)







