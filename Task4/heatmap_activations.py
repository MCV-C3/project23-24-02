
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import cv2
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
from IPython.display import Image, display
import matplotlib as mpl

# function extracted from: https://keras.io/examples/vision/grad_cam/
def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.utils.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.utils.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array

# function extracted from: https://keras.io/examples/vision/grad_cam/
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = keras.models.Model(
        model.inputs, [last_conv_layer_name.output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# function extracted from: https://keras.io/examples/vision/grad_cam/
def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    # Load the original image
    img = keras.utils.load_img(img_path)
    img = keras.utils.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = mpl.colormaps["jet"]

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # Display Grad CAM
    # display(Image(cam_path))

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
    x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    predictions = Dense(8, activation='softmax', kernel_initializer=GlorotNormal())(x)
    model = Model(inputs=input, outputs=predictions)
        
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

config = {  'IMG_WIDTH': 32, 
            'IMG_HEIGHT': 32, 
            'BATCH_SIZE': 50,
            'NUMBER_OF_EPOCHS': 60,
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

model = create_model(config)

model.load_weights('./output/'+config['MODEL_FNAME']+'.weights.h5')

for i, l in enumerate(model.layers):
    print(i, l.name)


layer_index = 9 
selected_layer = model.layers[layer_index]

img_path = './data/MIT_small_train_1/train/forest/bost100.jpg'

img_size = (32, 32)
img = image.load_img(img_path, target_size=(32,32))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array /= 255.0  # Normalize pixel values

# Get activations
activations = model.predict(img_array)

# Generate class activation heatmap
heatmap = make_gradcam_heatmap(img_array, model, selected_layer)

save_and_display_gradcam(img_path, heatmap)