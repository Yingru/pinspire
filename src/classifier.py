from keras.applications.resnet50 import ResNet50, preprocess_input                                                             
#from keras.applications.vgg19 import VGG19, preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.applications.imagenet_utils import decode_predictions
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.models import Sequential, Model
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
from keras import metrics
import numpy as np
import os, sys
from collections import defaultdict

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import argparse



class FeatureClassifier(object):
    def __init__(self, base='resnet50', weights='imagenet', HEIGHT=224, WIDTH=224, class_list = None):
        '''
        create a classifier to make prediction 
        Input -- base ['resnet50', 'vgg19']
              -- weights ['imagenet']
              -- HEIGHT, WIDTH [int]
              -- class_list [list[string]]
        '''
        if base == 'resnet50':
            self.model = ResNet50(weights=weights)
        elif base == 'vgg19':
            self.model = VGG19(weights=weights)
        
        self.HEIGHT = HEIGHT
        self.WIDTH = WIDTH
        self.class_list = class_list
    

    def _predict(self, imagePath, totalProb={}):
        '''
        make a prediction of the image with respect to content
        '''

        img = load_img(imagePath, target_size = (self.HEIGHT, self.WIDTH))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        probs = self.model.predict(x)
        _label = decode_predictions(probs)
        for i in _label[0]:
            split = i[1].split('_')
            for j in split:
                j = j.lower()
                if j not in totalProb:
                    totalProb[j] = i[2]
                else:
                    totalProb[j] += i[2]
            
        return totalProb


class FeatureClassifierTransfered(object):
    def __init__(self, base='resnet50', HEIGHT=300, WIDTH=300, FC_LAYERS = [1024,1024], dropout=0.5, 
                    class_list = None, weights=None):
        '''
        create a classifier from a model that comes from the transfered learning
        Input -- base: ['resnet50', 'vgg19', 'inceptionv3']
              -- (HEIGHT, WIDTH): input layer shape
              -- FC_LAYERS: additional fully connected layer in addition to the base mode 
              -- dropout: Dropout layer for preventing overfitting
              -- class_list: list of class you'd like to train 
              -- weights: (weight table for the NN)
        '''
        if base == 'resnet50':
            baseModel = ResNet50(weights='imagenet', 
                                include_top=False,
                                input_shape = (HEIGHT, WIDTH, 3))

        for layer in baseModel.layers:
            layer.trainable = False

        x = baseModel.output
        x = Flatten()(x)
        
        for fc in FC_LAYERS:
            x = Dense(fc, activation='relu')(x)
            x = Dropout(dropout)(x)

        predictions = Dense(len(class_list), activation='softmax')(x)
        self.model = Model(inputs=baseModel.input, output=predictions)
        self.HEIGHT = HEIGHT
        self.WIDTH = WIDTH
        self.class_list = class_list
        self.weights = weights
        
        # load weights if it exist
        if self.weights:
            self.model.load_weights(self.weights)
        

    def _train(self, TrainDir, ValidDir=None, batch_size=8, num_epochs=5, checkpoint=None):
        '''
        train the model based on (training_x, training_y)
        Input -- TrainDir: training dataset. must stored organized in order to use the ImageDataGenerator
              -- ValidDir: valid dataset. 
        '''
        # 1. prepare model, training_dataset using ImageDataGenerator (very efficient!)
        train_datagen = ImageDataGenerator(preprocessing_function = preprocess_input,
                                           rotation_range=90,
                                           horizontal_flip = True,
                                           vertical_flip = True)

        valid_dategen = ImageDataGenerator(preprocessing_function = preprocess_input)

        train_generator = train_datagen.flow_from_directory(TrainDir,
                                                    target_size = (HEIGHT, WIDTH),
                                                    batch_size = batch_size)

        if valid_datagen:
            valid_generator = valid_datagen.flow_from_directory(ValidDir,
                                                    target_size = (HEIGHT, WIDTH),
                                                    batch_size = batch_size)
        else:
            valid_generator = valid_datagen.flow_from_directory(TrainDir,
                                                    target_size = (HEIGHT, WIDTH),
                                                    batch_size=batch_size)

        num_train_images = 66544
        num_valid_images = 7376

        # 2. create the optimizer and compiler
        adam = Adam(lr=0.0001)
        self.model.compile(adam, loss='categorical_crossentropy', 
                            metrics=['accuracy', metrics.top_k_categorical_accuracy])

        checkpoint = ModeCheckPoint(checkpoint, monitor=['acc'], verbose=1, mode='max')
        callback_list = [checkpoint]

        # 3. start training
        history = self.model.fit_generator(train_generator, 
                                            epochs=num_epochs,
                                            workers=4,
                                            steps_per_epoch = num_train_images//batch_size,
                                            shuffle=True,
                                            call_backs=callback_list)
        predIdx = self.model.predict_generator(valid_generator,
                                               step_per_epoch = num_valid_images//batch_size)

        predLabels = np.argmax(predIdx, axis=1)
        print('[INFO] evaluating network ...', predIdx)

    def _predict(self, imagePath, totalProb={}):
        '''
        make a prediction of the image with respect to content 
        '''
        # 1. preprocessing the images
        img = load_img(imagePath, target_size=(self.HEIGHT, self.WIDTH))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # 2. make prediction
        probs = self.model.predict(x)[0]

        # 3. mapping the prediction to actual label
        probs_mapping = dict(zip(self.class_list, probs))
        probs_mapping = sorted(probs_mapping.items(), key=lambda x: x[1], reverse=True)

        for x in probs_mapping[:3]:
            if x[0] not in totalProb:
                totalProb[x[0]] = x[1]
            else:
                totalProb[x[0]] += x[1]

        totalProb = sorted(totalProb.items(), key=lambda x: x[1], reverse=True)
        return totalProb




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='description')
    parser.add_argument('--classifier', default='content', type=str)  # class= ['content', 'pattern', 'artStyle']
    args = parser.parse_args()
    
    if args.classifier == 'content':
        classifier = FeatureClassifier()

    elif args.classifier == 'pattern':
        class_list = ['checkered','dotted','floral', 'solid', 'striped','zig zag']
        weights = '/home/ubuntu/Pinterest_final/webApp/weights/ResNet50_model_weights.h5'
        weights = '/home/yingru/Documents/Project/Insight/Pinterest/Pinterest_final/webApp/weights/ResNet50_model_weights.h5'
        classifier = FeatureClassifierTransfered(class_list = class_list, weights=weights)
   
    elif args.classifier == 'artStyle':
        #class_list = ['abstract', 'abstract_expression', 'art_informel', 'art_nouveau', 'baroque', \
        #       'color_field_painting', 'cubism', 'early_renaissance', 'expressionism', 'high_renaissance', \
        #       'impressionism', 'magic_realism', 'nannerism', 'minimalism', 'naive_art', 'neoclassicim', \
        #       'northern_renaissance', 'pop_art', 'post_impressionism', 'realism', \
        #       'rococo', 'romanticism', 'surrealism', 'symbolism', 'Ukiyo-e']
        class_list = ['abstract', 'abstract', 'informal', 'nouveau', 'baroque', \
                'color field', 'cubism', 'renaissance', 'expressionism', 'renaissance', \
                'impressionism', 'realism', 'nannerism', 'minimalism', 'naive', 'neoclassicim', \
                'renaissance', 'pop art', 'impressionism', 'realism', \
                'rococo', 'romanticism', 'surrealism', 'symbolism', 'ukiyoe']
        weights = '/home/yingru/Documents/Project/Insight/Pinterest/Pinterest_final/webApp/weights/ResNet50_art_style_weights_valid.h5'
        classifier = FeatureClassifierTransfered(HEIGHT=224, WIDTH=224, class_list = class_list, weights=weights)

    #imagePath = '../board/interior'
    imagePath = '../board/art'
    files = os.listdir(imagePath)
    imagePath = [os.path.join(imagePath, f) for f in files if f.endswith('.jpg?b=t')]
    for i in imagePath:
        print(i, classifier._predict(i, {}))

