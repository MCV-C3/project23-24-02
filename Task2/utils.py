#from __future__ import print_function
import os,sys
import numpy as np
from sklearn.feature_extraction import image
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from itertools import cycle
import itertools
import keras
import tensorflow as tf

def load_model_data(DATASET_DIR, BATCH_SIZE, IMG_SIZE):
  # Load and preprocess the training dataset

  train_dataset = keras.utils.image_dataset_from_directory(
    directory=DATASET_DIR+'/train/',
    labels='inferred',
    label_mode='categorical',
    batch_size=BATCH_SIZE,
    class_names=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
    image_size=(IMG_SIZE, IMG_SIZE),
    shuffle=True,
    validation_split=None,
    subset=None
    )

  # Load and preprocess the validation dataset
  validation_dataset = keras.utils.image_dataset_from_directory(
    directory=DATASET_DIR+'/test/',
    labels='inferred',
    label_mode='categorical',
    batch_size=BATCH_SIZE,
    class_names=['coast', 'forest', 'highway', 'inside_city', 'mountain', 'Opencountry', 'street', 'tallbuilding'],
    image_size=(IMG_SIZE, IMG_SIZE),
    shuffle=True,
    seed=123,
    validation_split=None,
    subset=None
  )

  # Data augmentation and preprocessing
  preprocessing_train = keras.Sequential([
    keras.layers.Rescaling(1./255),
    keras.layers.RandomFlip("horizontal")
  ])

  preprocessing_validation = keras.Sequential([
    keras.layers.Rescaling(1./255)
  ])

  train_dataset = train_dataset.map(lambda x, y: (preprocessing_train(x, training=True), y))
  validation_dataset = validation_dataset.map(lambda x, y: (preprocessing_validation(x, training=False), y))

  train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
  validation_dataset = validation_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

  return train_dataset, validation_dataset

def load_train_test_data(DATASET_DIR):
   # Upload train and test data
  train_directory = DATASET_DIR+'/train'
  train_images_filenames = []
  train_labels = []

  for class_dir in os.listdir(train_directory):
      for imname in os.listdir(os.path.join(train_directory,class_dir)):
          train_images_filenames.append(train_directory+'/'+class_dir+'/'+imname)
          train_labels.append(class_dir)

  test_directory = DATASET_DIR+'/test'
  test_images_filenames = []
  test_labels = []

  for class_dir in os.listdir(test_directory):
      for imname in os.listdir(os.path.join(test_directory,class_dir)):
          test_images_filenames.append(test_directory+'/'+class_dir+'/'+imname)
          test_labels.append(class_dir)

  return train_images_filenames, train_labels, test_images_filenames, test_labels

def get_features(images_filenames, model_layer, IMG_SIZE):
  features = []

  for i in range(len(images_filenames)):
      filename = images_filenames[i]
      x = np.asarray(Image.open(filename))
      x = np.expand_dims(np.resize(x, (IMG_SIZE, IMG_SIZE, 3)), axis=0)
      x_features = model_layer.predict(x/255.0, verbose=0)
      features.append(np.asarray(x_features).reshape(-1))

  return np.asarray(features)

def save_on_file(RESULTS, EXPERIMENT_FNAME, train_metrics, test_metrics):
  with open(RESULTS+EXPERIMENT_FNAME+'_eval_metrics.txt', 'w') as file:
    file.write(f'Train Accuracy: {train_metrics[0]}\n')
    file.write(f'Train Precision: {train_metrics[1]}\n')
    file.write(f'Train Recall: {train_metrics[2]}\n')
    file.write(f'Train F1-Score: {train_metrics[3]}\n')
    file.write(f'Test Accuracy: {test_metrics[0]}\n')
    file.write(f'Test Precision: {test_metrics[1]}\n')
    file.write(f'Test Recall: {test_metrics[2]}\n')
    file.write(f'Test F1-score: {test_metrics[3]}\n')


def plot_evaluation(history, file):
  # summarize history for accuracy
  plt.plot(history.history['accuracy'])
  plt.plot(history.history['val_accuracy'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='upper left')
  plt.savefig(file+'_accuracy.jpg')
  plt.close()
  # summarize history for loss
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'validation'], loc='upper left')
  plt.savefig(file+'_loss.jpg')

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def generate_image_patches_db(in_directory,out_directory, patch_size=64):
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)
  
    total = 2688
    count = 0  
    for split_dir in os.listdir(in_directory):
      if not os.path.exists(os.path.join(out_directory,split_dir)):
        os.makedirs(os.path.join(out_directory,split_dir))
    
      for class_dir in os.listdir(os.path.join(in_directory,split_dir)):
        count = 0
        if not os.path.exists(os.path.join(out_directory,split_dir,class_dir)):
          os.makedirs(os.path.join(out_directory,split_dir,class_dir))
    
        for imname in os.listdir(os.path.join(in_directory,split_dir,class_dir)):
          count += 1
          im = Image.open(os.path.join(in_directory,split_dir,class_dir,imname))
          print(im.size, patch_size)
          print('Processed images: '+str(count)+' / '+str(total), end='\r')
          patches = image.extract_patches_2d(np.array(im), (patch_size, patch_size),  max_patches=(im.size[0]//patch_size)**2)
          for i,patch in enumerate(patches):
            patch = Image.fromarray(patch)
            patch.save(os.path.join(out_directory,split_dir,class_dir,imname.split('.')[0]+'_'+str(i)+'.jpg'))
    print('\n')

def evaluate_model_performance(groundtruth, pred):
    # Calculate accuracy
    accuracy = accuracy_score(groundtruth, pred)

    # Calculate precision
    precision = precision_score(groundtruth, pred, average='weighted')

    # Calculate recall
    recall = recall_score(groundtruth, pred, average='weighted')

    # Calculate F1 score
    f1 = f1_score(groundtruth, pred, average='weighted')

    return accuracy, precision, recall, f1

def create_roc_curve(train_features, test_features, train_labels, test_labels, classifier, results_path):
    # Binarize the labels
    n_classes = len(np.unique(train_labels))
    y_train_bin = label_binarize(train_labels, classes=np.unique(train_labels))
    y_test_bin = label_binarize(test_labels, classes=np.unique(train_labels))

    plt.figure(figsize=(8, 8))
    lw = 2
    all_fpr = np.linspace(0, 1, 100)

    mean_tpr = 0
    mean_auc = 0

    for i in range(n_classes):
        # Train the classifier for each class
        classifier.fit(train_features, y_train_bin[:, i])

        # Get the decision function on the test set
        y_score = classifier.decision_function(test_features)

        # Compute ROC curve and ROC area for each class
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score)
        roc_auc = auc(fpr, tpr)
        mean_auc += roc_auc

        # Interpolate ROC curve to get the mean TPR at the same FPR points
        mean_tpr += np.interp(all_fpr, fpr, tpr)

        # Plot ROC curves for each class
        plt.plot(fpr, tpr, lw=lw,
                label='Class {0} (AUC = {1:0.3f})'
                      ''.format(i, roc_auc))
        
    mean_tpr /= n_classes
    mean_auc /= n_classes

    # Plot average ROC curve
    plt.plot(all_fpr, mean_tpr, color='navy', linestyle='--', lw=lw,
             label='Macro-average ROC curve (AUC = {0:0.3f})'.format(mean_auc))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Each Class')
    plt.legend(loc="lower right")

    plt.savefig(results_path)
    plt.close()

def create_confusion_matrix(groundtruth, pred, file):
    # Compute confusion matrix
    classes = list(set(groundtruth))
    cm = confusion_matrix(groundtruth, pred, labels=classes)
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    # Plot confusion matrix
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    # Add text annotations
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], '.2f'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    # Set labels and layout
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()

    # Save and close the plot
    plt.savefig(file)
    plt.close()