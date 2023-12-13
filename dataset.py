import numpy as np
import cv2
import os
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import img_to_array


def Dataset(path,size):
    color_img = []
    gray_img = []
    files_colour = os.listdir(path+'/color/')
    for i in tqdm(files_colour):
            img = cv2.imread(path + '/color/'+i , 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (size, size))
            img = img.astype('float32') / 255.0
            color_img.append(img_to_array(img))
            img = cv2.imread(path + '/gray/'+i, 1)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (size, size))
            img = img.astype('float32') / 255.0
            gray_img.append(img_to_array(img))

    color_dataset=tf.data.Dataset.from_tensor_slices(np.array(color_img[:1800])).batch(64)
    gray_dataset=tf.data.Dataset.from_tensor_slices(np.array(gray_img[:1800])).batch(64)

    color_dataset_test=tf.data.Dataset.from_tensor_slices(np.array(color_img[1800:])).batch(8)
    gray_dataset_test=tf.data.Dataset.from_tensor_slices(np.array(gray_img[1800:])).batch(8)

    return (color_dataset,gray_dataset),(color_dataset_test,gray_dataset_test)
