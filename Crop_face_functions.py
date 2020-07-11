import mtcnn
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
from PIL import Image


def read_img(path):
    img = cv2.imread(path)
#     if img == None:
#         print('img is null ')
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img


def find_faces(image):
    detector = mtcnn.MTCNN()
    img_copy = image.copy()
    faces = detector.detect_faces(img_copy)
    return faces

def crop_face(image,face_d):
    all_faces = []
    for point in face_d:
        if point['confidence'] > 0.1:
            x,y,w,h = point['box']
#             face = image[y-h//5:y+h+h//5,x-w//5:x+w+w//5]
            if x < 0 : x = 0
            if y < 0 : y = 0
            face = image[y:y+h,x:x+w]
            face = Image.fromarray(face)
            face = face.resize((160,160))
            face = np.asarray(face)
            all_faces.append(face)
    
    return all_faces

def draw_rectangle(image,face_d):
    image_copy = image.copy()
    thickness = 7
    for point in face_d:
        x,y,w,h = point['box']
        if x < 0 : x = 0
        if y < 0 : y = 0
        image_copy = cv2.rectangle(image_copy,(x,y), (x+w,y+h),(255, 0, 0) ,7)
    return image_copy

def showResult(image,result):
    image_copy = image.copy()
    thickness = 7
    for point,label in result:
        if label:
            x,y,w,h = point['box']
            if x < 0 : x = 0
            if y < 0 : y = 0
            image_copy = cv2.rectangle(image_copy,(x,y), (x+w,y+h),(0, 0, 255) ,7)
            font = cv2.FONT_HERSHEY_SIMPLEX 
            fontScale = 3
            color = (255, 0, 0) 
            image_copy = cv2.putText(image_copy,label,(x+w,y+h),font,fontScale,color,thickness=10)
    
    return image_copy
        
def Extract_faces(base_path,crop_path):
    for subdir in os.listdir(base_path):
        if subdir not in os.listdir(crop_path):
            os.mkdir(os.path.join(crop_path,subdir))
    count= 0
    for subdir in os.listdir(base_path):
        subdir_path = os.path.join(base_path,subdir)
        crop_subdir_path = os.path.join(crop_path,subdir)
        for img_name in os.listdir(subdir_path):
            count+=1
            img_path = os.path.join(subdir_path,img_name)
            image = read_img(img_path)
            face_d = find_faces(image)
            face = crop_face(image,face_d)
            save_path = os.path.join(crop_subdir_path,img_name)
#             face0 = Image.fromarray(face[0])
#             face0 = face0.resize((160,160))
#             face0 = np.asarray(face0)
            plt.imsave(save_path,face[0])
        print("_______________________________________________________________________")
