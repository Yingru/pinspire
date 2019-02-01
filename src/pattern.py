from keras.applications.resnet50 import ResNet50, preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.models import Sequential, Model
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint
import numpy as np
import os
from collections import defaultdict



HEIGHT, WIDTH = 300, 300
class_list = ['checkered','dotted','floral', 'solid', 'striped','zig zag']
FC_LAYERS = [1024, 1024]
dropout = 0.5


def build_finetune_model(dropout=dropout, fc_layers=FC_LAYERS, num_classes=len(class_list)):
    base_model = ResNet50(weights='imagenet',
                            include_top=False,
                            input_shape=(HEIGHT, WIDTH, 3))
 
    for layer in base_model.layers:
        layer.trainable = False

    x = base_model.output
    x = Flatten()(x)

    for fc in fc_layers:
        x = Dense(fc, activation='relu')(x)
        x = Dropout(dropout)(x)

    predictions = Dense(num_classes, activation = 'softmax')(x)
    finetune_model = Model(inputs = base_model.input, output=predictions)
    return finetune_model


def train_pattern(TrainDir, batch_size=8, num_epochs=5, weight=None, checkpoint=None):
    ## 1. prepare base_model, training_dataset
    train_datagen = ImageDataGenerator(preprocessing_function = preprocess_input,
                                        rotation_range = 90,
                                        horizontal_flip = True,
                                        vertical_flip = True)
    train_generator = train_datagen.flow_from_directory(TrainDir,
                                                        target_size = (HEIGHT, WIDTH),
                                                        batch_size = batch_size)
    num_train_images = 400

    finetune_model = build_finetune_model()

    if weight:
        finetune_model.load_weights(weights)

    adam = Adam(lr=0.0001)
    finetune_model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])
    
    checkpoint = ModelCheckpoint(checkpoint, monitor=['acc'], verbose=1, mode='max')
    callback_list = [checkpoint]

    history = finetune_model.fit_generator(train_generator, epochs=num_epochs, 
                                            workers=4,
                                            steps_per_epoch = num_train_images//batch_size,
                                            shuffle=True,
                                            callbacks = callback_list)
    


def predict_pattern(imagePath, weights):
    finetune_model = build_finetune_model()
    finetune_model.load_weights(weights)

   
    files = os.listdir(imagePath)
    imagePath = [os.path.join(imagePath, f) for f in files if f.endswith('.jpg')]
    length = len(imagePath)

    totalProb = {}
    for _ in imagePath:
        img = image.load_img(_, target_size=(HEIGHT, WIDTH))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
    
        probs = finetune_model.predict(x)[0]
        idx = probs.argmax()
        _label = class_list[idx]
        _probability = probs[idx]

        if _label not in totalProb:
            totalProb[_label] = _probability
        else:
            totalProb[_label] += _probability


        #print('Predicting {}: '.format(_), class_list[idx], ', probability: ', probs[idx])

    for i in totalProb.keys():
        totalProb[i] /= length


    return totalProb

if __name__ == '__main__':
    image_path = '../Boards/interior/'
    weights = '../checkoutpoints/ResNet50_model_weights.h5'
    result = predict_pattern(image_path, weights)
    result = sorted(result.items(), key=lambda item: item[1])
    print(result)
