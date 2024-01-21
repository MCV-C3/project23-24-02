import os, sys
os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch"!
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils_main import *
import keras
import tensorflow as tf
from keras.layers import Dense
from keras.utils import plot_model
from keras.optimizers import SGD, Adagrad, RMSprop, Adam

from keras.applications import ResNet50
from keras.applications.resnet import preprocess_input

from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D

import seaborn as sns
import pandas as pd

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
from sklearn.feature_extraction import image
from PIL import Image
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from itertools import cycle
import itertools
from scipy import interp


def lr_plots(learning_rates, results, accuracy_training, accuracy_validation, number_of_epochs):
    plt.figure(figsize=(12, 6))
    for lr, result in zip(learning_rates, results):
        plt.bar(str(lr), result[1], label=str(lr))
    plt.xlabel('Learning Rate')
    plt.ylabel('Test Accuracy')
    plt.title('Comparison of test accuracy for different momentum values')

    # Show the plot
    plt.savefig('./test_accuracy_momentum_nesterov.jpg')

    training_data = np.array(accuracy_training)
    validation_data = np.array(accuracy_validation)
    difference_data = validation_data - training_data

    # Set up the matplotlib figure
    plt.figure(figsize=(12, 6))

    # Use matplotlib's plot to create the plot
    for i, exp in enumerate(learning_rates):
        plt.plot(list(range(1, (number_of_epochs+1))), difference_data[i], label=f'{exp} - Difference', marker='o')

    # Add labels and title
    plt.xlabel('Epoch')
    plt.ylabel('Difference (Validation - Training Accuracy)')
    plt.title('Overfitting Analysis: Difference Between Validation and Training Accuracy')

    # Show the legend
    plt.legend()
    plt.savefig('./difference_accuracy_momentum_nesterov.jpg')

def define_data_generator(config, data):
    if data == 'training':
        # Define the data generator for data augmentation and preprocessing
        data_generator = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            #rotation_range=20,
            #width_shift_range=0.2,
            #height_shift_range=0.2,
            #shear_range=10,
            #zoom_range=0.2,
            #horizontal_flip=True,
            #vertical_flip=False
        )
    elif data == 'validation' or data == 'test':
        # Load and preprocess the validation dataset
        data_generator = ImageDataGenerator(
            preprocessing_function=preprocess_input
        )
    else:
        raise ValueError('data must be either training, validation or test')
    return data_generator


def load_and_preprocess(config, data_generator, data):

    if data == 'training':
        # Load and preprocess the training dataset
        dataset = data_generator.flow_from_directory(
            directory=config['TRAINING_DATASET_DIR'],
            target_size=(config['IMG_WIDTH'], config['IMG_HEIGHT']),
            batch_size=config['BATCH_SIZE'],
            class_mode='categorical',
            shuffle=True
        )
    elif data == 'test':
        # Load and preprocess the test or validation dataset
        dataset = data_generator.flow_from_directory(
            directory=config['TEST_DATASET_DIR'],
            target_size=(config['IMG_WIDTH'], config['IMG_HEIGHT']),
            batch_size=config['BATCH_SIZE'],
            class_mode='categorical',
            shuffle=False
        )
    elif data == 'validation':
        # Load and preprocess the test or validation dataset
        dataset = data_generator.flow_from_directory(
            directory=config['VALIDATION_DATASET_DIR'],
            target_size=(config['IMG_WIDTH'], config['IMG_HEIGHT']),
            batch_size=config['BATCH_SIZE'],
            class_mode='categorical',
            shuffle=True
        )
    else:
        raise ValueError('data must be either training, validation or test')
    return dataset

def create_trainable_model(config):
    # create the base pre-trained model
    base_model = ResNet50(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    # Reduce number of parameters by selecting until which block you want
    # x = base_model.get_layer('conv4_block6_out').output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(8, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional VGG16 layers
    for layer in base_model.layers:
        layer.trainable = False
        
    # compile the model (should be done *after* setting layers to non-trainable)
    if config['optimizer'] == 'SGD':
        model.compile(optimizer=SGD(learning_rate=config['learning_rate'], momentum=config['momentum'], nesterov=True), loss='categorical_crossentropy', metrics=['accuracy'])
    elif config['optimizer'] == 'Adagrad':
        model.compile(optimizer=Adagrad(learning_rate=config['learning_rate']), loss='categorical_crossentropy', metrics=['accuracy'])
    elif config['optimizer'] == 'RMSprop':
        model.compile(optimizer=RMSprop(learning_rate=config['learning_rate'], momentum=config['momentum']), loss='categorical_crossentropy', metrics=['accuracy'])
    elif config['optimizer'] == 'Adam':
        model.compile(optimizer=Adam(learning_rate=config['learning_rate']), loss='categorical_crossentropy', metrics=['accuracy'])
    else:
        raise ValueError('optimizer must be SGD, Adagrad, RMSprop or Adam')
    

    return model

def train_topN_layers(model, N):
    for layer in model.layers[:N]:
        layer.trainable = False
    for layer in model.layers[N:]:
        layer.trainable = True
    model.compile(optimizer=SGD(lr=0.0001, momentum=0.9), loss='categorical_crossentropy', metrics=['accuracy'])

def train_val(config):
    training_data_generator = define_data_generator(config, 'training')
    train_dataset = load_and_preprocess(config, training_data_generator, 'training')

    validation_data_generator = define_data_generator(config, 'validation')
    validation_dataset = load_and_preprocess(config, validation_data_generator, 'validation')

    test_data_generator = define_data_generator(config, 'test')
    test_dataset = load_and_preprocess(config, test_data_generator, 'test')


    model = create_trainable_model(config)

    # train the model on the new data for a few epochs
    history = model.fit(train_dataset,
                        epochs=config['NUMBER_OF_EPOCHS'],
                        validation_data=validation_dataset,
                        verbose=2)
    
    
    if config['layers_to_train'] != 0:
        train_topN_layers(model, config['layers_to_train'])

        # we train our model again (this time fine-tuning the top 2 inception blocks
        # alongside the top Dense layers
        history = model.fit(train_dataset,
                            epochs=config['NUMBER_OF_EPOCHS'],
                            validation_data=validation_dataset,
                            verbose=0)

    result = model.evaluate(test_dataset)
    
    
    # VISUALIZATION
    test_labels = test_dataset.classes
    predictions = model.predict(test_dataset, verbose=1)
    pred_labels = np.argmax(predictions, axis=1)
    class_names = ['Opencountry', 'coast', 'forest', 'highway', 'inside_city', 'mountain', 'street', 'tallbuilding']
    
    # ROC curve
    compute_roc(test_labels, predictions, class_names, 'untilconv5_roc.png', 'ROC curve') 
    
    # Confusion matrix
    conf_mat = confusion_matrix(test_labels, pred_labels)
    conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
    
    save_confusion_matrix(conf_mat, class_names, 'untilconv5_confusion_matrix.png')

    return history, result, model



config = {'IMG_WIDTH': 224, 
      'IMG_HEIGHT': 224, 
      'BATCH_SIZE': 50,
      'NUMBER_OF_EPOCHS': 20,
      'TRAINING_DATASET_DIR': '/ghome/mcv/datasets/C3/MIT_small_train_1/train/',
      'TEST_DATASET_DIR': '/ghome/mcv/datasets/C3/MIT_split/test/',
      'VALIDATION_DATASET_DIR': '/ghome/mcv/datasets/C3/MIT_small_train_1/test/',
      'layers_to_train': 0, #starting from the last layer
      'learning_rate': 0.1,
      'momentum': 0.1,
      'optimizer': 'SGD'}

accuracy_training = []
accuracy_validation = []
results = []

history, result, model = train_val(config)
accuracy_training.append(history.history['accuracy'])
accuracy_validation.append(history.history['val_accuracy'])
results.append(result)
model.save_weights('model_untilconv5.weights.h5')

print(accuracy_training)
print(accuracy_validation)
print(results)

with open('untilconv5.txt', 'w') as file:
    file.write('ACCTRAINING,'+str(accuracy_training)+ '\n\n')
    file.write('ACCVALIDATION,'+str(accuracy_validation)+ '\n\n')
    file.write('RESULTS,'+str(results))
    
 
if True:
    import matplotlib
    #matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('untilconv5_accuracy.jpg')
    plt.close()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig('untilconv5_loss.jpg')
