import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch"!
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import matplotlib.pyplot as plt
import keras
import tensorflow as tf
from keras.layers import Dense
from keras.utils import plot_model
from keras.optimizers import SGD, Adagrad, RMSprop, Adam
from keras.applications import ResNet50
from keras.applications.resnet import preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.initializers import GlorotNormal, RandomNormal
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import roc_curve, auc, confusion_matrix
from itertools import cycle
import itertools
from scipy import interp
import pickle
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation, Input, SpatialDropout2D, Dropout
from contextlib import redirect_stdout



gpus = tf.config.list_physical_devices('GPU')
print(f'GPUs available: {gpus}')

def define_data_generator(config, data):
    if data == 'training':
        # Define the data generator for data augmentation and preprocessing
        data_generator = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            #rotation_range=20,
            #width_shift_range=0.2,
            #height_shift_range=0.2,
            #shear_range=0.2,
            #zoom_range=0.2,
            #horizontal_flip=True,
            vertical_flip=False
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

def create_model(config):

    input = Input(shape=(config['IMG_WIDTH'], config['IMG_HEIGHT'], 3), batch_size=config['BATCH_SIZE'])
    x = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', kernel_initializer=GlorotNormal())(input)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', kernel_initializer=GlorotNormal())(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Conv2D(filters=32, kernel_size=3, padding='same', activation='relu', kernel_initializer=GlorotNormal())(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = BatchNormalization()(x)
    #x = Dropout(0.2)(x)
    #x = SpatialDropout2D(0.2)(x)
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    #x = Dense(16, activation='relu', kernel_initializer=GlorotNormal())(x)
    predictions = Dense(8, activation='softmax', kernel_initializer=GlorotNormal())(x)
    model = Model(inputs=input, outputs=predictions)
        
    # compile the model (should be done *after* setting layers to non-trainable)
        
    if config['scheduler'] == 'cosine':
        lr = keras.optimizers.schedules.CosineDecay(initial_learning_rate=config['learning_rate'], decay_steps=20) # 4 * (20/4)
    elif config['scheduler'] == 'exponential':
        lr = keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=config['learning_rate'], decay_steps=20, decay_rate=0.96) # 4 * (20/4)
    else:
        lr = config['learning_rate']

    if config['optimizer'] == 'SGD':
        optimizer=SGD(learning_rate=lr, momentum=config['momentum'], nesterov=config['nesterov'])
    elif config['optimizer'] == 'Adagrad':
        optimizer=Adagrad(learning_rate=lr)
    elif config['optimizer'] == 'RMSprop':
        optimizer=RMSprop(learning_rate=lr, momentum=config['momentum'])
    elif config['optimizer'] == 'Adam':
        optimizer=Adam(learning_rate=lr)
    else:
        raise ValueError('optimizer must be SGD, Adagrad, RMSprop or Adam')
    
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def train(config, model, train_dataset, validation_dataset, test_dataset):
    # train the model on the new data for a few epochs
    history = model.fit(train_dataset,
                        epochs=config['NUMBER_OF_EPOCHS'],
                        validation_data=validation_dataset,
                        verbose=2)

    model.save_weights('./output/'+config['MODEL_FNAME']+'.weights.h5')

    result = model.evaluate(test_dataset)

    accuracy_training = history.history['accuracy']
    accuracy_validation = history.history['val_accuracy']

    with open('./output/'+config['MODEL_FNAME']+'.txt', 'w') as file:
        with redirect_stdout(file):
            print(model.summary())
        file.write('ACCTRAINING,'+str(accuracy_training)+ '\n\n')
        file.write('ACCVALIDATION,'+str(accuracy_validation)+ '\n\n')
        file.write('RESULTS,'+str(result))

    epochs = list(range(1, (config['NUMBER_OF_EPOCHS']+1)))
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, accuracy_training, label='Training Accuracy')
    plt.plot(epochs, accuracy_validation, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Evolution of Training and Validation Accuracy through epochs')
    plt.legend()
    # Show the plot
    plt.savefig('./plot_prueba.jpg')

def val(config, model, test_dataset):
    model.load_weights(config['MODEL_FNAME']+'.weights.h5')

    class_names = ['Opencountry', 'coast', 'forest', 'highway', 'inside_city', 'mountain', 'street', 'tallbuilding']
    Y_pred = model.predict(test_dataset)
    y_pred = np.argmax(Y_pred, axis=1)
    test_labels = test_dataset.classes
    conf_mat = confusion_matrix(test_labels, y_pred)
    conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]


def train_val(config):
    training_data_generator = define_data_generator(config, 'training')
    train_dataset = load_and_preprocess(config, training_data_generator, 'training')

    validation_data_generator = define_data_generator(config, 'validation')
    validation_dataset = load_and_preprocess(config, validation_data_generator, 'validation')

    test_data_generator = define_data_generator(config, 'test')
    test_dataset = load_and_preprocess(config, test_data_generator, 'test')

    model = create_model(config)

    if config['MODE']=='train':

        train(config, model, train_dataset, validation_dataset, test_dataset)

    elif config['MODE']=='val':
    
        val(config, model, test_dataset)
    
    elif config['MODE']=='train_val':
        
        train(config, model, train_dataset, validation_dataset, test_dataset)
        val(config, model, test_dataset)

    else:
        raise ValueError('MODE must be train or val or train_val')



config = {  'IMG_WIDTH': 32, 
            'IMG_HEIGHT': 32, 
            'BATCH_SIZE': 50,
            'NUMBER_OF_EPOCHS': 50,
            'TRAINING_DATASET_DIR': './data/MIT_small_train_1/train/',
            'TEST_DATASET_DIR': './data/MIT_split/test/',
            'VALIDATION_DATASET_DIR': './data/MIT_small_train_1/test/',
            'layers_to_train': 177, # list containing the modules to train 
            'learning_rate': 0.001,
            'momentum': 0.0,
            'nesterov': False,
            'optimizer': 'Adam', 
            'scheduler': None, 
            'MODEL_FNAME': '14', 
            'MODE': 'train'
            }

train_val(config)


