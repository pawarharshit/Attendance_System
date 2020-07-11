import tensorflow as tf

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def loadmodel(path):
    return tf.keras.models.load_model(path)

def generate_from_array(model,image_array):
#     generate the embedding for multiple faces images
#     return embedding dataframe
    X = list()
    imgCount=0
    for image in image_array:
        imgCount += 1
        image = image.astype('float32')
        mean,std = image.mean(),image.std()
        image = (image - mean)/std
        image = image.reshape((1,160,160,3))
        em = model.predict(image)
        X.append(em)
    X = np.asarray(X)
    X = X.reshape((imgCount,128))
    data = pd.DataFrame(X)
    return data


def GenerateEmbedding(base_path,model,save_csv=False,name=None):
    print(os.listdir(base_path))
    X,y = list(),list()
    imgCount=0
    for subdir in os.listdir(base_path):
        class_path = os.path.join(base_path,subdir)
        for img_name in os.listdir(class_path):
            imgCount += 1
            img_path = os.path.join(class_path,img_name)
            img = plt.imread(img_path)
            img = img.astype('float32')
            mean,std = img.mean(),img.std()
            img = (img - mean)/std
            img = img.reshape((1,160,160,3))
            em = model.predict(img)
            X.append(em)
            y.append(subdir)
    
    X = np.asarray(X)
    X = X.reshape((imgCount,128))
    print('X.shape:' ,X.shape)
    y = np.asarray(y)
    print('y.shape' ,y.shape)
    data = pd.DataFrame(X)
    data['labels'] = y
    print('data.shape',data.shape)
    if(save_csv):
        if(save_path):
            data.to_csv(name+'.csv')
        else:
            print('please provide valid path for saveing embedding in csv format')
    return data







