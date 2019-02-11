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



HEIGHT, WIDTH = 224, 224
#class_list = ['checkered','dotted','floral', 'solid', 'striped','zig zag']
class_list = ['abstract', 'abstract_expression', 'art_informel', 'art_nouveau', 'baroque', \
                'color_field_painting', 'cubism', 'early_renaissance', 'expressionism', 'high_renaissance', \
                'impressionism', 'magic_realism', 'nannerism', 'minimalism', 'naive_art', 'neoclassicim', \
                'northern_renaissance', 'pop_art', 'post_impressionism', 'realism', \
                'rococo', 'romanticism', 'surrealism', 'symbolism', 'Ukiyo-e']

print(len(class_list))
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
    num_train_images = 66544

    finetune_model = build_finetune_model()

    if weight:
        finetune_model.load_weights(weight)

    adam = Adam(lr=0.0001)
    finetune_model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])
    
    checkpoint = ModelCheckpoint(checkpoint, monitor=['acc'], verbose=1, mode='max')
    callback_list = [checkpoint]

    history = finetune_model.fit_generator(train_generator, epochs=num_epochs, 
                                            workers=4,
                                            steps_per_epoch = num_train_images//batch_size,
                                            shuffle=True,
                                            callbacks = callback_list)
    
def train_valid_pattern(TrainDir, ValidDir, batch_size=8, num_epochs=5, weight=None, checkpoint=None):
    ## 1. prepare base_model, training_dataset
    train_datagen = ImageDataGenerator(preprocessing_function = preprocess_input,
                                        rotation_range = 90,
                                        horizontal_flip = True,
                                        vertical_flip = True)

    valid_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)

    train_generator = train_datagen.flow_from_directory(TrainDir,
                                                        target_size = (HEIGHT, WIDTH),
                                                        batch_size = batch_size)
    valid_generator = valid_datagen.flow_from_directory(ValidDir, 
                                                        target_size=(HEIGHT, WIDTH),
                                                        batch_size = batch_size)

    num_train_images = 66544
    num_valid_images = 7376

    finetune_model = build_finetune_model()

    if weight:
        finetune_model.load_weights(weight)

    adam = Adam(lr=0.0001)
    finetune_model.compile(adam, loss='categorical_crossentropy', metrics=['accuracy'])
    
    checkpoint = ModelCheckpoint(checkpoint, monitor=['acc'], verbose=1, mode='max')
    callback_list = [checkpoint]

    history = finetune_model.fit_generator(train_generator, epochs=num_epochs, 
                                            workers=4,
                                            steps_per_epoch = num_train_images//batch_size,
                                            shuffle=True,
                                            validation_data = valid_generator, 
                                            validation_steps = num_valid_images//batch_size,
                                            callbacks = callback_list)

def train_valid_test_pattern(TrainDir, ValidDir, batch_size=8, num_epochs=5, weight=None, checkpoint=None):
    ## 1. prepare base_model, training_dataset
    train_datagen = ImageDataGenerator(preprocessing_function = preprocess_input,
                                        rotation_range = 90,
                                        horizontal_flip = True,
                                        vertical_flip = True)

    valid_datagen = ImageDataGenerator(preprocessing_function = preprocess_input)

    train_generator = train_datagen.flow_from_directory(TrainDir,
                                                        target_size = (HEIGHT, WIDTH),
                                                        batch_size = batch_size)
    valid_generator = valid_datagen.flow_from_directory(ValidDir, 
                                                        target_size=(HEIGHT, WIDTH),
                                                        batch_size = batch_size)

    num_train_images = 66544
    num_valid_images = 7376

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

    predIdx = finetune_model.predict_generator(valid_generator, 
                                               steps_per_epoch = num_valid_images//batch_size)

    predLabels = np.argmax(predIdx, axis=1)
    print('[INFO] evaluating network ...', predIdx)


def predict_artStyle(imagePath, weights):
    finetune_model = build_finetune_model()
    finetune_model.load_weights(weights)

   
    files = os.listdir(imagePath)
    print(os.getcwd())
    imagePath = [os.path.join(imagePath, f) for f in files if f.endswith('.jpg')]
    length = len(imagePath)

    totalProb = {}
    totalProbs2 = {}
    for _ in imagePath:
        img = image.load_img(_, target_size=(HEIGHT, WIDTH))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
    
        probs = finetune_model.predict(x)[0]
        probs_mapping = dict(zip(class_list, probs))
        probs_mapping = sorted(probs_mapping.items(), key=lambda x: x[1], reverse=True)

        idx = probs.argmax()
        _label = class_list[idx]
        _probability = probs[idx]

        if _label not in totalProb:
            totalProb[_label] = _probability
        else:
            totalProb[_label] += _probability


        print('Predicting {}: '.format(_), class_list[idx], ', probability: ', probs[idx])
        for x in probs_mapping[:3]:
            if x[0] not in totalProbs2:
                totalProbs2[x[0]] = x[1]
            else:
                totalProbs2[x[0]] += x[1]

        print(probs_mapping[0], probs_mapping[1], probs_mapping[2])

    totalProbs2 = sorted(totalProbs2.items(), key=lambda x: x[1], reverse=True)
    print(totalProbs2)

    for i in totalProb.keys():
        totalProb[i] /= length


    return totalProb, totalProbs2



if __name__ == '__main__':
    # first part , training the model
    #TrainDir  = './wikipaintings_train'
    #weights = 'checkpoints/ResNet50_art_style_weights.h5'
    #train_pattern(TrainDir, batch_size=16, num_epochs=1, weight=weights, checkpoint=weights)

    '''
    TrainDir  = './wikipaintings_train'
    ValidDir = './wikipaintings_val'
    weights = 'checkpoints/ResNet50_art_style_weights_valid.h5'
    train_valid_pattern(TrainDir, ValidDir, batch_size=16, num_epochs=10, weight=weights, checkpoint=weights)
    '''

    
    
    image_path = '../../../rasta/data/wikipaintings_small/wikipaintings_test/Rococo/'
    weights = '../weights/ResNet50_art_style_weights_valid.h5'
    result, _ = predict_artStyle(image_path, weights)
    result = sorted(result.items(), key=lambda item: item[1])
    print(result)
