import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import np_utils
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

class image2data:
    def __init__(self, image_path, output_path, percent_split, img_h, img_w):
        
        self.image_path = image_path
        self.output_path = output_path
        self.percent_split = percent_split
        self.image_h = img_h
        self.image_w = img_w 
        self.create_data()
        
    def create_data(self):
        x = []
        y = []

        for root, folder, files in os.walk(self.image_path):
            for f in files:
                if f.endswith(".jpg"):
                    file_path = os.path.join(root, f)
                    image = cv2.imread(file_path, 1)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = cv2.resize(image, (self.image_h, self.image_w))
                    x.append(image)

                    folder = file_path.split('\\')[0]
                    folder = folder.split("/")[1]
                    y.append(folder)   
                    
        x = np.array(x)
        y = np.array(y)

        y = self.one_hot_encode(y)
        x, y = self.shuffle_data(x, y)
        self.split_data(x, y)

    def one_hot_encode(self, y):
        le = LabelEncoder()
        le.fit(y)
        le.classes_
        y_enconded = le.transform(y)
        y_enconded = np_utils.to_categorical(y_enconded, 2)

        return y_enconded

    def shuffle_data(self, x_data, y_data):
        x, y = shuffle(x_data, y_data, random_state=0)
        
        return x, y
    
    def split_data(self, x, y):
        
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= self.percent_split, random_state=0)
        
        np.save(os.path.join(self.output_path, "x_train.npy"), x_train)
        np.save(os.path.join(self.output_path, "y_train.npy"), y_train)
        np.save(os.path.join(self.output_path, "x_test.npy"), x_test)
        np.save(os.path.join(self.output_path, "y_test.npy"), y_test)
        
        print("Data Created.")

if __name__ == "__main__":
    
    OUTPUT = 'data/'
    IMAGES = 'images/'
    H,W,C = 32, 32, 3
    SPLIT_PERCENT = 0.15

    image2data(IMAGES, OUTPUT, SPLIT_PERCENT, H, W)