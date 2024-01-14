import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch"!

from utils import *
import keras
from keras import initializers
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Reshape, Input
from keras.utils import plot_model
import numpy as np
from PIL import Image
from itertools import product
from sklearn.feature_extraction import image
from sklearn import svm
from sklearn.cluster import MiniBatchKMeans
import collections
from tqdm import tqdm 

## Define functions
# Based on solution by @Ivan in https://stackoverflow.com/questions/5953373/how-to-split-image-into-multiple-pieces-in-python
def tile(img, PATCH_SIZE):
    w, h = img.size
    grid = product(range(0, h-h%PATCH_SIZE, PATCH_SIZE), range(0, w-w%PATCH_SIZE, PATCH_SIZE))
    patches = []
    for i, j in grid:
        box = (j, i, j+PATCH_SIZE, i+PATCH_SIZE)
        patches.append(img.crop(box))
    return np.array(patches)

def compute_descriptors(images_filenames, PATCH_SIZE, embedding_size=512):
    # Get the dimensions of the images and compute the number of patches per image
    w, h = Image.open(images_filenames[0]).size
    n_patches = (w // PATCH_SIZE)**2

    descriptors = np.empty((len(images_filenames), n_patches, embedding_size))
    for i,filename in tqdm(enumerate(images_filenames), desc="Creating descriptors...", total=len(images_filenames)):
        img = Image.open(filename)
        patches = tile(img, PATCH_SIZE)
        descriptors[i, :, :] = model_layer.predict(patches/255., verbose=False)
    return descriptors

def compute_visual_words(features, codebook, k):
  visual_words=np.zeros((len(features), k), dtype=np.float32)
  for i, desc in enumerate(features):
      prediction = codebook.predict(desc)
      visual_words[i,:] = np.bincount(prediction, minlength=k)
  return visual_words

## User defined variables
PATCH_SIZE  = 64
BATCH_SIZE  = 16
DATASET_DIR = '/ghome/mcv/datasets/C3/MIT_split'

EXPERIMENT_FNAME = "first_layer_mlp_patched_svm_"+str(PATCH_SIZE)
WEIGHTS_FNAME = f"/ghome/group02/lab2/model/{'mlp_patched_model64'}.weights.h5"
RESULTS = '/ghome/group02/lab2/results/MLP_PATCHED/'

if not os.path.exists(DATASET_DIR):
  print('ERROR: dataset directory '+DATASET_DIR+' do not exists!\n')
  quit()

## Load MLP model
print('Building MLP model for testing...\n')
model = Sequential()
input = Input(shape=(PATCH_SIZE, PATCH_SIZE, 3,), name='input')
model.add(input) # Input tensor
model.add(Reshape((PATCH_SIZE*PATCH_SIZE*3,), name='reshape'))
model.add(Dense(units=1024, activation='relu', name='first'))
model.add(Dense(units=2048, activation='relu', name='second'))
model.add(Dense(units=8, activation='softmax', name='classification'))

print(model.summary())
print('Loading weights from '+WEIGHTS_FNAME+' ...\n')
model.load_weights(WEIGHTS_FNAME)
    
LAYER_NAME = 'first'
model_layer = keras.Model(inputs=input, outputs=model.get_layer(LAYER_NAME).output)
size_layer = 1024


## Load data files
print("Creating features...")
train_images_filenames, train_labels, test_images_filenames, test_labels = load_train_test_data(DATASET_DIR)
# Divide image into patches, compute descriptors from 
train_features = compute_descriptors(train_images_filenames, PATCH_SIZE, size_layer)
print("Training features shape:", train_features.shape)
test_features = compute_descriptors(test_images_filenames, PATCH_SIZE, size_layer)
print("Test features shape:", test_features.shape)

## Grid search to train SVM
best_accuracy, best_k = -1, None
results_grid = {}
for k_codebook in [32, 64, 128, 256, 512, 1024, 2048]:
  print(f"[GRID SEARCH] TRAINING MODEL WITH K={k_codebook}")
  # Codebook
  print("Creating the bag of words...")
  codebook = MiniBatchKMeans(n_clusters=k_codebook, verbose=False, batch_size=k_codebook * 20, compute_labels=False, reassignment_ratio=10**-4, random_state=1)
  codebook_data = np.vstack(train_features) # For training the codebook, all the patches are used independently
  codebook.fit(codebook_data)

  visual_words_train = compute_visual_words(train_features, codebook, k_codebook)
  visual_words_test = compute_visual_words(test_features, codebook, k_codebook)

  print("Training the SVM Classifier")
  # Create the SVM classifier
  KERNEL_NAME = 'rbf'
  classifier = svm.SVC(kernel=KERNEL_NAME)
  classifier.fit(visual_words_train, train_labels)

  print('Start evaluation ...')
  tr_predictions = classifier.predict(visual_words_train)
  tr_accuracy, tr_precision, tr_recall, tr_f1 = evaluate_model_performance(train_labels, tr_predictions)

  te_predictions = classifier.predict(visual_words_test)
  te_accuracy, te_precision, te_recall, te_f1 = evaluate_model_performance(test_labels, te_predictions)

  print(f"K={k_codebook}, Test Acc. = {str(te_accuracy)} \n")
  results_grid[k_codebook] = {"acc": te_accuracy, "pr": te_precision, "r": te_recall, "f1": te_f1}
  if te_accuracy > best_accuracy:
     best_accuracy = te_accuracy
     best_k = k_codebook

  with open(f"{RESULTS+EXPERIMENT_FNAME}_k{k_codebook}_eval_metrics.txt", 'w') as file:
      file.write(f'Train Accuracy: {tr_accuracy}\n')
      file.write(f'Train Precision: {tr_precision}\n')
      file.write(f'Train Recall: {tr_recall}\n')
      file.write(f'Train F1: {tr_f1}\n')
      file.write(f'Test Accuracy: {te_accuracy}\n')
      file.write(f'Test Precision: {te_precision}\n')
      file.write(f'Test Recall: {te_recall}\n')
      file.write(f'Test F1: {te_f1}\n')

  create_roc_curve(visual_words_train, visual_words_test, train_labels, test_labels, classifier, f"{RESULTS+EXPERIMENT_FNAME}_k{k_codebook}_roc_curve.png")
  create_confusion_matrix(test_labels, te_predictions, f"{RESULTS+EXPERIMENT_FNAME}_k{k_codebook}_confusion_matrix.png")

print("\n\n[GRID SEARCH] FINAL RESULTS")
print(f"Best k={best_k} with acc={best_accuracy} \nDictionary of results:")
print(results_grid)