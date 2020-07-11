import Crop_face_functions as cff
import Generate_Embeddings as ge

import tensorflow as tf

import mtcnn
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
import numpy as np
import pandas as pd



classifier = tf.keras.models.load_model('tri_classifier.h5')


def Classify_image(image_path,classifier):
    image = cff.read_img(image_path)
    image_face = cff.find_faces(image)
    cut_face = cff.crop_face(image,image_face)
    embeddingModel = ge.loadmodel('facenet_keras.h5')
    embedding = ge.generate_from_array(embeddingModel,cut_face)
    X = embedding.values
    y = classifier.predict(X)
    personidentified = []
    for i in y:
        if i[np.argmax(i)] < 0.50:
            personidentified.append(None)
        else:
            personidentified.append(classes[np.argmax(i)])
    result = []
    for x,y in zip(image_face,personidentified):
        result.append([x,y])
    resultingImage = cff.showResult(image,result)
    return  resultingImage