import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch"!

from utils import *
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Reshape, Input
from keras.utils import plot_model
from keras import initializers
from sklearn import svm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

def create_model(IMG_SIZE):
  #Build the Multi Layer Perceptron model
  model = Sequential()
  input = Input(shape=(IMG_SIZE, IMG_SIZE, 3,),name='input')
  model.add(input) # Input tensor
  model.add(Reshape((IMG_SIZE*IMG_SIZE*3,),name='reshape'))
  model.add(Dense(units=32, activation='relu', kernel_initializer=initializers.GlorotUniform(), name='first'))
  model.add(Dense(units=64, activation='relu', kernel_initializer=initializers.GlorotUniform(), name='second'))
  model.add(Dense(units=32, activation='relu', kernel_initializer=initializers.GlorotUniform(), name='last'))
  model.add(Dense(units=8, activation='softmax',kernel_initializer=initializers.GlorotUniform(), name='classification'))
  sgd = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, weight_decay=1e-06, nesterov=False)
  model.compile(loss='categorical_crossentropy',
                optimizer=sgd,
                metrics=['accuracy'])
  return input, model

#user defined variables
IMG_SIZE    = 32
BATCH_SIZE  = 16
DATASET_DIR = '/ghome/mcv/datasets/C3/MIT_split'
EXPERIMENT_FNAME = "3_layers_escalera_pequeña_neurons" # experiment name
MODEL_FNAME = '/ghome/group02/lab2/model/'+EXPERIMENT_FNAME+'.weights.h5'
RESULTS = '/ghome/group02/lab2/results/MLP_SIMPLE/'


if not os.path.exists(DATASET_DIR):
  print('ERROR: dataset directory '+DATASET_DIR+' does not exist!\n')
  quit()

mode = 'val'

input, model = create_model(IMG_SIZE)
print(model.summary())
plot_model(model, to_file=RESULTS+EXPERIMENT_FNAME+'modelMLP.png', show_shapes=True, show_layer_names=True)

if mode=='train':
  train_dataset, validation_dataset = load_model_data(DATASET_DIR, BATCH_SIZE, IMG_SIZE)
  print('Start training...\n')
  history = model.fit(train_dataset, epochs=50, validation_data=validation_dataset, verbose=0)
  plot_evaluation(history, RESULTS+EXPERIMENT_FNAME)

  print('Saving the model into '+MODEL_FNAME+' \n')
  model.save_weights(MODEL_FNAME)  # always save your weights after training or during training


elif mode=='val':
  print('Loading the model from '+MODEL_FNAME+' \n')
  model.load_weights(MODEL_FNAME)

  train_images_filenames, train_labels, test_images_filenames, test_labels = load_train_test_data(DATASET_DIR)

  LAYER_NAME = 'last'
  model_layer = keras.Model(inputs=input, outputs=model.get_layer(LAYER_NAME).output)

  # Get the features of the train and test data for the given layer
  train_features = get_features(train_images_filenames, model_layer, IMG_SIZE)
  test_features = get_features(test_images_filenames, model_layer, IMG_SIZE)

  # Create the SVM classifier
  KERNEL_NAME = 'rbf'
  classifier = svm.SVC(kernel=KERNEL_NAME)
  classifier.fit(train_features,train_labels)

  # Evaluate
  tr_predictions = classifier.predict(train_features)
  tr_accuracy, tr_precision, tr_recall, tr_f1 = evaluate_model_performance(train_labels, tr_predictions)

  te_predictions = classifier.predict(test_features)
  te_accuracy, te_precision, te_recall, te_f1 = evaluate_model_performance(test_labels, te_predictions)

  save_on_file(RESULTS, EXPERIMENT_FNAME, [tr_accuracy, tr_precision, tr_recall, tr_f1], [te_accuracy, te_precision, te_recall, te_f1])

  create_roc_curve(train_features, test_features, train_labels,test_labels, classifier,RESULTS+EXPERIMENT_FNAME+'_roc_curve.png')
  create_confusion_matrix(test_labels, te_predictions, RESULTS+EXPERIMENT_FNAME+'_confusion_matrix.png')

print('Done!')