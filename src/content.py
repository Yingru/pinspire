import os
import sys
import scipy.io
import scipy.misc

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

from PIL import Image
import numpy as np
import tensorflow as tf

#from keras.applications import vgg19, inception_v3, resnet50
from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.imagenet_utils import decode_predictions
from collections import defaultdict


HEIGHT, WIDTH = 224, 224
#HEIGHT, WIDTH = 300, 300



def predict_content(imagePath, weights='imagenet'):
    #base_model = vgg19.VGG19(weights=weights)
    #base_model = ResNet50(weights='imagenet',
    #                        input_shape=(HEIGHT, WIDTH, 3))
    base_model = ResNet50(weights='imagenet')

    files = os.listdir(imagePath)
    imagePath = [os.path.join(imagePath, f) for f in files if f.endswith('.jpg')]
    length = len(imagePath)
    totalProb = {}

    for _ in imagePath:
        img = load_img(_, target_size = (HEIGHT, WIDTH))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        probs = base_model.predict(x)
        idx = probs[0].argmax()
        _label = decode_predictions(probs)
        _probability = probs[0][idx]
        #print('predicting: {}',format(_), _label)
        for i in _label[0]:
            split = i[1].split('_')
            for j in split:
                if j not in totalProb:
                    totalProb[j] = i[2]
                else:
                    totalProb[j] += i[2]


    for i in totalProb.keys():
        totalProb[i] /= length

    return totalProb


if __name__ == '__main__':
    image_path = '/home/ubuntu/Pinterest_final/webApp/board/interior'
    result = predict_content(image_path)
    result = sorted(result.items(), key=lambda item: item[1], reverse=True)
    print(result)
