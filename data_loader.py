# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 21:51:50 2021

@author: John
"""


''' LOAD BASIC LIBRARIRES'''

import matplotlib as plt
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from imutils import paths
import numpy as np
import random
import cv2
import os
from PIL import Image 
import numpy
import keras
import pandas as pd
SEED = 124   # set random seed


#imgplot = plt.imshow(img)


def load_parathyroid (path,excel_path,in_shape,verbose):
    
    #LOADS 3IN1 IMAGES (DUAL + SUBSTRACTION). NO PERIOXES
    
    WS = pd.read_excel(excel_path)
    excel = np.array(WS)
    excel[:,0] = excel[:,0].astype(int)
    
    
    width = in_shape[0]
    height = in_shape[1]
    if verbose:
        print("[INFO] loading images")
    data = [] # Here, data will be stored in numpy array
    labels = [] # Here, the lables of each image are stored
    imagePaths = sorted(list(paths.list_images(path)))  # data folder with 2 categorical folders
    random.seed(SEED) 
    random.shuffle(imagePaths) # Shuffle the image data
    # loop over the input images
    for l,imagePath in enumerate(imagePaths): #load, resize, normalize, etc
        if verbose:
            print("Preparing Image: {}".format(imagePath))
        image = cv2.imread(imagePath)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        
        '''CONSTRAST'''
    
        
        alpha = 1.5 # Contrast control (1.0-3.0)
        beta = 0 # Brightness control (0-100)
        
        adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        image = cv2.resize(adjusted, (width, height))/image.max()
        
        
        if verbose:
            if l<2:
                cv2.imshow('image',image)
                cv2.imshow('adjusted',adjusted)
                cv2.waitKey(0)
        data.append(image)
        # extract the class label from the image path and update the labels list
        img_num = imagePath.split(os.path.sep)[-1]
        img_num = img_num[:3]
        img_num = int(img_num)
        if verbose:
            print(img_num)
        for j in range(len(excel)):
            if int(excel[j,0]) == img_num:
                if verbose:
                    print("found_match")
                label = int(excel[j,2])
                if label == 1:
                    label2 = 'Parathyroid'
                    if verbose:
                        print ("IMG_NUM: {}, EXCEL LABEL: {}, Translated: {}".format(img_num,label,label2))
                else:
                    label2 = 'ND'
                    if verbose:
                        print ("IMG_NUM: {}, EXCEL LABEL: {}, Translated: {}".format(img_num,label,label2))
        
        
        labels.append(label2)
    data = np.array(data, dtype="float")
    labeltemp=labels
    labels = np.array(labels)
    #data = data.reshape(data.shape[0], 32, 32, 1)  
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels) 
    labels = keras.utils.to_categorical(labels, num_classes=2, dtype='float32')
    print("Data and labels loaded and returned")
    return data, labels,labeltemp, image, image.max()

def load_parathyroid_dual (path,excel_path,in_shape,verbose):
    
    #LOADS 3IN1 IMAGES (DUAL + SUBSTRACTION). NO PERIOXES
    
    WS = pd.read_excel(excel_path)
    excel = np.array(WS)
    excel[:,0] = excel[:,0].astype(int)
    
    
    width = in_shape[0]
    height = in_shape[1]
    if verbose:
        print("[INFO] loading images")
    data = [] # Here, data will be stored in numpy array
    labels = [] # Here, the lables of each image are stored
    imagePaths = sorted(list(paths.list_images(path)))  # data folder with 2 categorical folders
    random.seed(SEED) 
    random.shuffle(imagePaths) # Shuffle the image data
    # loop over the input images
    for l,imagePath in enumerate(imagePaths): #load, resize, normalize, etc
        if verbose:
            print("Preparing Image: {}".format(imagePath))
        image = cv2.imread(imagePath)
        
        '''CONSTRAST'''
    
        
        # alpha = 1.5 # Contrast control (1.0-3.0)
        # beta = 0 # Brightness control (0-100)
        
        # adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
        #adjusted = np.clip(image,30,255)
        image = cv2.resize(image, (width, height))/image.max()        

        
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (width, height))/image.max()
        if verbose:
            if l<0:
                cv2.imshow('image',image)
                #cv2.imshow('ajusted',adjusted)
                
                cv2.waitKey(0)
        
        # extract the class label from the image path and update the labels list
        img_num = imagePath.split(os.path.sep)[-1]
        img_num = img_num[:3]
        img_num = int(img_num)
        if verbose:
            print(img_num)
        for j in range(len(excel)):
            if int(excel[j,0]) == img_num:
                if verbose:
                    print("found_match")
                try:
                    label = int(excel[j,2])
                except Exception as e:
                    print(e)
                    label = 'else'
                if label == 1:
                    label2 = 'Parathyroid'
                    if verbose:
                        print ("IMG_NUM: {}, EXCEL LABEL: {}, Translated: {}".format(img_num,label,label2))
                elif label == 0:
                    label2 = 'ND'
                    if verbose:
                        print ("IMG_NUM: {}, EXCEL LABEL: {}, Translated: {}".format(img_num,label,label2))
        
        if label != 'else':
            data.append(image)
            labels.append(label2)
    data = np.array(data, dtype="float")
    labeltemp=labels
    labels = np.array(labels)
    #data = data.reshape(data.shape[0], 32, 32, 1)  
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels) 
    labels = keras.utils.to_categorical(labels, num_classes=2, dtype='float32')
    print("Data and labels loaded and returned")
    return data, labels,labeltemp, image, image.max()



def load_parathyroid_multi (path_1, path_2, path_3, excel_path, in_shape, verbose):
    
    #LOADS 3IN1 IMAGES (DUAL + SUBSTRACTION). NO PERIOXES
    
    WS = pd.read_excel(excel_path)
    excel = np.array(WS)
    excel[:,0] = excel[:,0].astype(int)
    
    info = np.empty([0,4])
    
    
    width = in_shape[0]
    height = in_shape[1]
    if verbose:
        print("[INFO] loading images")
        
        
    data_early = [] # Here, data will be stored in numpy array
    data_late = []
    data_sub = []
    labels = [] # Here, the lables of each image are stored
    imagePaths_early = sorted(list(paths.list_images(path_1)))  # data folder with 2 categorical folders
    random.seed(SEED) 
    random.shuffle(imagePaths_early) # Shuffle the image data
    imagePaths_late = sorted(list(paths.list_images(path_2)))
    imagePaths_sub = sorted(list(paths.list_images(path_3)))
    
    # loop over the input images
    for l,imagePath_early in enumerate(imagePaths_early): #load, resize, normalize, etc
        if verbose:
            print("Preparing Image: {}".format(imagePath_early))
            
        image = cv2.imread(imagePath_early)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (width, height))/image.max()
        if verbose:
            if l<0:
                cv2.imshow('image',image)
                cv2.waitKey(0)
        
        # extract the class label from the image path and update the labels list
        img_num = imagePath_early.split(os.path.sep)[-1]
        img_num = img_num[:3]
        img_num = int(img_num)
        if verbose:
            print(img_num)
        found_label = False
        found_late = False
        found_sub = False
        for j in range(len(excel)):
            if int(excel[j,0]) == img_num:
                found_label = True
                subject_no = j
                if verbose:
                    print("found_match")
                try:
                    label = int(excel[j,2])
                except Exception as e:
                    print(e)
                    label = 'else'
                if label == 1:
                    label2 = 'Parathyroid'
                    if verbose:
                        print ("IMG_NUM: {}, EXCEL LABEL: {}, Translated: {}".format(img_num,label,label2))
                elif label == 0:
                    label2 = 'ND'
                    if verbose:
                        print ("IMG_NUM: {}, EXCEL LABEL: {}, Translated: {}".format(img_num,label,label2))
                
                for i,imagePath_late in enumerate(imagePaths_late):
                    img_num_late = imagePath_late.split(os.path.sep)[-1]
                    img_num_late = img_num_late[:3]
                    img_num_late = int(img_num_late)
                    
                    if img_num_late == img_num:
                        print("Found late")
                        found_late = True
                        image_late = cv2.imread(imagePath_late)
                        image_late = cv2.resize(image_late, (width, height))/image_late.max()
                        break
                
                if not found_late:
                    if verbose:
                        print ("Did not find late match")
                        
                
                for i,imagePath_sub in enumerate(imagePaths_sub):
                    img_num_sub = imagePath_sub.split(os.path.sep)[-1]
                    img_num_sub = img_num_sub[:3]
                    img_num_sub = int(img_num_sub)
                    
                    if img_num_sub == img_num:
                        print("Found late")
                        found_sub = True
                        image_sub = cv2.imread(imagePath_sub)
                        image_sub = cv2.resize(image_sub, (width, height))/image_sub.max()
                        break
                
                if not found_late:
                    if verbose:
                        print ("Did not find late match")

                if not found_sub:
                    if verbose:
                        print ("Did not find sub match")
                break
            
            
        if not found_label:
            if verbose:
                print ("Did not find label")
            label = 'else'
        
        if (label != 'else') and (found_late) and (found_sub):
            info = np.concatenate([info,excel[subject_no,:].reshape(1,4)])
            data_early.append(image)
            data_late.append(image_late)
            data_sub.append(image_sub)
            labels.append(label2)
    data_early = np.array(data_early, dtype="float")
    data_late = np.array(data_late, dtype="float")
    data_sub =  np.array(data_sub, dtype="float")
    labeltemp=labels
    labels = np.array(labels)
    #data = data.reshape(data.shape[0], 32, 32, 1)  
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels) 
    labels = keras.utils.to_categorical(labels, num_classes=2, dtype='float32')
    print("Data and labels loaded and returned")
    return data_early, data_late, data_sub, labels, labeltemp, image, image.max(), info





def labels_maker_perioxes_from_original_excel_file (excel_path):
    
    # AN INTELLIGENT WAY TO PRODUCE THE LABELS FOR THE PERIOXES EXCEL
    
    columns = ["img_no", "name", "ΔΕ ΑΝΩ", "ΑΡ ΑΝΩ", "ΔΕ ΚΑΤΩ", "ΑΡ ΚΑΤΩ"]
    new_file = pd.DataFrame(columns=columns)
    instance = np.zeros([1,6],dtype=object)
    #instance = pd.DataFrame(columns=columns)
    
    WS = pd.read_excel(excel_path)
    excel = np.array(WS)
    
    excel = excel[3:,:]
    excel[:,0] = excel[:,0].astype(float)
    
    for i in range (len(excel)):
        
        if str(excel[i,0]) == 'nan':
            continue
        
        
        if str(excel[i,11]) == 'nan':
            print('No label found for: {}: {}'.format(i,excel[i,1]))
            continue

        if str(excel[i,10]) == 'nan':
            print('No label found for: {}: {}'.format(i,excel[i,1]))
            continue
        
        
        instance[0,0] = str(excel[i,0])
        instance[0,1] = str(excel[i,1])   
        
        if int(excel[i,11]) == 0:
            instance[0,2] = instance[0,3] = instance[0,4] = instance[0,5] = str(0)
            new_file = new_file.append(pd.DataFrame(instance, columns=columns), ignore_index=True)
            continue
        
        #afou eftase edw, tote sto keli 11 yparxei assos (ara positive)
        
        acceptable = ["ΑΡ ΑΝΩ", "ΑΡ ΚΑΤΩ", "ΑΡ", "ΔΕ ΚΑΤΩ", "ΔΕ ΑΝΩ", "ΔΕ"]
        
        # ara pigaine twra stin idia eutheia kai des ti leei to keli 7
        if int(excel[i,10]) == 0: #an stin idia eutheia me ton arxiko asso iparxei 0 tote kati den paei kala
            print ('Problem in {}: {}'.format(excel[i,0],excel[i,1]))        
        
        for k in range(5):
            
            if i == len(excel)-1:
                break
            
            if k!=0:
                if (str(excel[i+k,0]) != 'nan') : #an to apokatw kali anoikei se allo astheni, break the for
                    break
    
                if str(excel[i+k,10]) == 'nan': #an to apokatw kali anoikei se allo astheni, break the for
                    break
            else:
                if (i+k)!=(len(excel)-1):
                    if (str(excel[i+k+1,0]) != 'nan') : #an to apokatw kali anoikei se allo astheni, break the for
                        break
        
                    if str(excel[i+k+1,10]) == 'nan': #an to apokatw kali anoikei se allo astheni, break the for
                        break               
            if str(excel[i+k,10]) == 'nan':
                break
            
            if int(excel[i+k,10]) == 1:
                
                #tsekare se pio keli anaferetai perioxi thetiki
                if str(excel[i+k,7]) in acceptable :
                    place = str(excel[i+k,7])
                elif str(excel[i+k,3]) in acceptable :
                     place = str(excel[i+k,3])
                else: 
                    place = 'all'
                

                if place == 'ΔΕ':
                    instance[0,2] = 1
                    instance[0,4] = 1
                elif place == 'ΑΡ':
                    instance[0,3] = 1
                    instance[0,5] = 1
                elif place=='all':
                    instance[0,2] = 1
                    instance[0,4] = 1
                    instance[0,3] = 1
                    instance[0,5] = 1
                else:
                    instance = pd.DataFrame(instance, columns=columns)
                    instance[place] = 1
                    instance = np.array(instance)
            
             
            
                            
        instance = np.array(instance)
        new_file = new_file.append(pd.DataFrame(instance, columns=columns), ignore_index=True)
        instance = np.zeros([1,6],dtype=object)
        
        
        
        places = ["ΑΡ ΑΝΩ", "ΑΡ ΚΑΤΩ", "ΔΕ ΚΑΤΩ", "ΔΕ ΑΝΩ", ]
        for place in places:
            new_file[place] = new_file[place].astype(float)
            
        positives = new_file.sum(axis=1)
        postivives = positives.sum()
        balance_ratio = postivives / (len(new_file)*4)
        new_file["img_no"] = new_file["img_no"].astype(float)   
            
        
            

def load_parathyroid_perioxes (path,excel_path,in_shape,verbose):
    
    #LOADS FROM FOLDER WHERE IMAGES CONTAIN 3IN1 BUT PERIOXES. NEEDS EXCEL
    
    perioxes_dict = {1:2,
                     2:3,
                     3:4,
                     4:5
        }
    
    WS = pd.read_excel(excel_path)
    excel = np.array(WS)
    excel[:,0] = excel[:,0].astype(int)
    
    
    width = in_shape[0]
    height = in_shape[1]
    if verbose:
        print("[INFO] loading images")
    data = [] # Here, data will be stored in numpy array
    labels = [] # Here, the lables of each image are stored
    imagePaths = sorted(list(paths.list_images(path)))  # data folder with 2 categorical folders
    random.seed(SEED) 
    random.shuffle(imagePaths) # Shuffle the image data
    # loop over the input images
    for l,imagePath in enumerate(imagePaths): #load, resize, normalize, etc
        if verbose:
            print("Preparing Image: {}".format(imagePath))
        image = cv2.imread(imagePath)
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (width, height))/image.max()
        if verbose:
            if l<2:
                cv2.imshow('image',image)
                cv2.waitKey(0)
        data.append(image)
        # extract the class label from the image path and update the labels list
        img_num = imagePath.split(os.path.sep)[-1]
        img_num = img_num[:3]
        img_num = int(img_num)
        if verbose:
            print(img_num)
        for j in range(len(excel)):
            if int(excel[j,0]) == img_num:
                if verbose:
                    print("found_match")
    
                perioxi = int(imagePath.split(os.path.sep)[-1][4:5])
                label_perioxis = perioxes_dict[perioxi]
                label = excel[j,label_perioxis]
                
                if label == 1:
                    label2 = 'Parathyroid'
                    if verbose:
                        print ("IMG_NUM: {}, EXCEL LABEL: {}, Translated: {}".format(img_num,label,label2))
                else:
                    label2 = 'ND'
                    if verbose:
                        print ("IMG_NUM: {}, EXCEL LABEL: {}, Translated: {}".format(img_num,label,label2))
        
        
        labels.append(label2)
    data = np.array(data, dtype="float")
    labeltemp=labels
    labels = np.array(labels)
    #data = data.reshape(data.shape[0], 32, 32, 1)  
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels) 
    labels = keras.utils.to_categorical(labels, num_classes=2, dtype='float32')
    print("Data and labels loaded and returned")
    return data, labels,labeltemp, image, image.max()