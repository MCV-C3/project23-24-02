import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch"!
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import matplotlib.pyplot as plt
#from utils_example import *
import keras
import tensorflow as tf
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
from tensorflow.keras.optimizers import SGD, Adam, RMSprop
from keras.applications.resnet import preprocess_input
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout, AveragePooling2D, SpatialDropout2D, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.initializers import GlorotNormal
from keras.regularizers import L1, L2, L1L2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import wandb

# Global variables
DATASET_DIR = 'data/MIT_split'
RESULTS_FOLDER = "results/2layers"
IMG_WIDTH = 32
IMG_HEIGHT = 32
BATCH_SIZE = 8
NUMBER_OF_EPOCHS = 50
N_LAST_LAYERS_TO_UNFREEZE = 0

gpus = tf.config.list_physical_devices('GPU')
print(f'GPUs available: {gpus}')

def save_history(history, result, result_name, model, notes=None):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(f"{RESULTS_FOLDER}/{result_name}/accuracy.jpg")
    plt.close()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('Loss (CELoss)')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(f"{RESULTS_FOLDER}/{result_name}/loss.jpg")

    with open(f"{RESULTS_FOLDER}/{result_name}/eval_metrics.txt", 'w') as file:
        file.write(f'Accuracy: {result[1]}\n')
        file.write(f'Loss: {result[0]}\n')
        file.write(f'Number of parameters: {model.count_params()}\n')
        model.summary(print_fn=lambda x: file.write(x + '\n'))
        if notes:
            file.write(f"Notes: {notes}\n")

def create_dataset(directory, preprocess_args={}, shuffle=False):
    data_generator = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        **preprocess_args
    )
    dataset = data_generator.flow_from_directory(
        directory=directory,
        target_size=(IMG_WIDTH, IMG_HEIGHT),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=shuffle
    )
    return dataset

# Create folders (if necessary)
os.makedirs(f"results/2layers/", exist_ok=True)
os.makedirs(f"model_files/2layers/", exist_ok=True)

# Define the data generator for data augmentation and preprocessing
train_preprocess_args = {
    "rotation_range":1,
    "width_shift_range":0.0,
    "height_shift_range":0.0,
    "shear_range":1.4,
    "zoom_range":0.0,
    "horizontal_flip":True,
    "vertical_flip":False
}
train_dataset = create_dataset("data/MIT_small_train_1/train/", shuffle=True)#, preprocess_args=train_preprocess_args, shuffle=True)
val_dataset = create_dataset('data/MIT_small_train_1/test/')
test_dataset = create_dataset('data/MIT_split/test/')

# Create model
input = Input((IMG_WIDTH, IMG_HEIGHT, 3), batch_size=BATCH_SIZE)
regularizer = L1L2(l1=0.0011729749774142836, l2=0.0017419871174629649)
x = Conv2D(8, kernel_size=3, kernel_regularizer=regularizer, padding="same", kernel_initializer=GlorotNormal())(input)
x = MaxPool2D((2, 2))(x)
x = Dropout(0.25)(x)
x = Conv2D(8, kernel_size=3, kernel_regularizer=regularizer, padding="same", kernel_initializer=GlorotNormal())(x)
x = MaxPool2D((2, 2))(x)
x = Dropout(0.25)(x)
#x = SpatialDropout2D(0.5)(x)

# Classification part
x = GlobalAveragePooling2D()(x)
predictions = Dense(8, activation='softmax', kernel_regularizer=regularizer, kernel_initializer=GlorotNormal())(x)
model = Model(inputs=input, outputs=predictions)

# compile the model (should be done *after* setting layers to non-trainable)
model.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=0.0027645685878352807), metrics=['accuracy'])
print(model.summary())
print(f"Number of parameters: {model.count_params()}")

# Train model
history = model.fit(train_dataset,
                    epochs=NUMBER_OF_EPOCHS,
                    validation_data=val_dataset,
                    verbose=2)

# Evaluate model
result = model.evaluate(test_dataset)

model_name = "model23"
# Save model
model.save(f'{RESULTS_FOLDER}/{model_name}.keras')

os.makedirs(f"{RESULTS_FOLDER}/{model_name}", exist_ok=True)

# Save stats, plots
save_history(history, result, model_name, model, notes="")