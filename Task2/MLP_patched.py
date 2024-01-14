import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch"!

from utils import load_model_data, plot_evaluation, generate_image_patches_db, softmax
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Reshape, Input
from keras.utils import plot_model
import numpy as np
from PIL import Image
from sklearn.feature_extraction import image
from keras import initializers

## User defined variables
PATCH_SIZE  = 64
BATCH_SIZE  = 16

DATASET_DIR = '/ghome/mcv/datasets/C3/MIT_split'

EXPERIMENT_FNAME = "mlp_patched_model"+str(PATCH_SIZE)
PATCHES_DIR = '/ghome/group02/lab2/data/MIT_split_patches'+str(PATCH_SIZE)
WEIGHTS_FNAME = '/ghome/group02/lab2/model/'+EXPERIMENT_FNAME+'.weights.h5'
RESULTS = '/ghome/group02/lab2/results/MLP_PATCHED/'

if not os.path.exists(DATASET_DIR):
    print('ERROR: dataset directory '+DATASET_DIR+' does not exist!\n')
    quit()

if not os.path.exists(PATCHES_DIR):
  print('WARNING: patches dataset directory '+PATCHES_DIR+' does not exist!\n')
  print('Creating image patches dataset into '+PATCHES_DIR+'\n')
  generate_image_patches_db(DATASET_DIR, PATCHES_DIR, patch_size=PATCH_SIZE)
  print('patxes generated!\n')

print('EXPERIMENT NAME: '+ EXPERIMENT_FNAME + '...\n')

print('Setting up data ...\n')
train_dataset, test_dataset = load_model_data(PATCHES_DIR, BATCH_SIZE, PATCH_SIZE)

print('Building MLP model...\n')
#Build the Multi Layer Perceptron model
model = Sequential()
input = Input(shape=(PATCH_SIZE, PATCH_SIZE, 3,), name='input')
model.add(input) # Input tensor
model.add(Reshape((PATCH_SIZE*PATCH_SIZE*3,), name='reshape'))
model.add(Dense(units=1024, activation='relu', kernel_initializer=initializers.GlorotUniform(), name='first'))
model.add(Dense(units=2048, activation='relu', kernel_initializer=initializers.GlorotUniform(), name='second'))
model.add(Dense(units=8, activation='softmax', kernel_initializer=initializers.GlorotUniform(), name='classification'))
model.compile(loss='categorical_crossentropy',
              optimizer="sgd",
              metrics=['accuracy'])

print('Start training...\n')
history = model.fit(
        train_dataset,
        epochs=25,
        validation_data=test_dataset,
        verbose=0)

print('Saving the model into '+WEIGHTS_FNAME+' \n')
plot_model(model, to_file=RESULTS+EXPERIMENT_FNAME+'.png', show_shapes=True, show_layer_names=True)
model.save_weights(WEIGHTS_FNAME)  # always save your weights after training or during training

# summarize history for accuracy
plot_evaluation(history, RESULTS+EXPERIMENT_FNAME)

print('Done!')