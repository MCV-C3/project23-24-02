import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch"!
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import matplotlib.pyplot as plt
#from utils_example import *
import keras
import tensorflow as tf
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
from tensorflow.keras.optimizers import SGD, Adam, Adagrad, RMSprop
from keras.applications.resnet import preprocess_input
from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout, AveragePooling2D, SpatialDropout2D, Flatten, Conv2D, MaxPool2D, BatchNormalization
from keras.initializers import GlorotNormal
from keras.regularizers import L1, L2, L1L2
from tensorflow.keras.optimizers.schedules import CosineDecay, ExponentialDecay
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import wandb
import optuna

# Global variables
DATASET_DIR = 'data/MIT_split'
IMG_WIDTH = 32
IMG_HEIGHT = 32
BATCH_SIZE = 8
NUMBER_OF_EPOCHS = 50

gpus = tf.config.list_physical_devices('GPU')
print(f'GPUs available: {gpus}')

def save_history(history, result, model):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(f"results/best_accuracy.jpg")
    plt.close()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('Loss (CELoss)')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(f"results/best_loss.jpg")

    with open(f"results/best_eval_metrics.txt", 'w') as file:
        file.write(f'Accuracy: {result[1]}\n')
        file.write(f'Loss: {result[0]}\n')
        file.write(f'Number of parameters: {model.count_params()}\n')
        model.summary(print_fn=lambda x: file.write(x + '\n'))

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

# Define the objective function (classification accuracy)
def objective(trial):
    lr_value = trial.suggest_float('lr_value', 0.0001, 0.1, log=True)
    momentum = trial.suggest_float('momentum', 0.0001, 1, log=True)
    optimizer = trial.suggest_categorical("optimizer", [Adam, SGD, Adagrad, RMSprop])
    l1_value = trial.suggest_float("l1_value", 0.00001, 0.1, log=True)
    l2_value = trial.suggest_float("l2_value", 0.00001, 0.1, log=True)

    scheduler = trial.suggest_categorical("scheduler", [False, CosineDecay, ExponentialDecay])
    if scheduler == CosineDecay:
        decay_steps = int(NUMBER_OF_EPOCHS/10) # 10% of the total steps
        lr = scheduler(initial_learning_rate=lr_value, decay_steps=decay_steps)
    elif scheduler == ExponentialDecay:
        decay_steps = int(NUMBER_OF_EPOCHS/10) # 10% of the total steps
        lr = scheduler(initial_learning_rate=lr_value, decay_steps=decay_steps, decay_rate=decay_steps/4)
    else:
        lr = lr_value

    # Define the data generator for data augmentation and preprocessing
    train_preprocess_args = {
        "rotation_range": trial.suggest_int("rotation_range", 0, 55, step=1),
        "width_shift_range": trial.suggest_float("width_shift_range", 0, 0.5, step=0.1),
        "height_shift_range": trial.suggest_float("width_shift_range", 0, 0.5, step=0.1),
        "shear_range": trial.suggest_float("shear_range", 0, 2, step=0.2),
        "zoom_range": trial.suggest_float("zoom_range", 0, 0.5, step=0.1),
        "horizontal_flip":trial.suggest_categorical("horizontal_flip", [True, False]),
        "vertical_flip":False
    }
    train_dataset = create_dataset("data/MIT_small_train_1/train/", preprocess_args=train_preprocess_args, shuffle=True)
    val_dataset = create_dataset('data/MIT_small_train_1/test/')
    test_dataset = create_dataset('data/MIT_split/test/')

    # Create model
    input = Input((IMG_WIDTH, IMG_HEIGHT, 3), batch_size=BATCH_SIZE)
    regularizer = L1L2(l1=l1_value, l2=l2_value)
    x = Conv2D(8, kernel_size=3, kernel_regularizer=regularizer, padding="same", kernel_initializer=GlorotNormal())(input)
    x = MaxPool2D((2, 2))(x)
    x = Dropout(0.25)(x)
    x = Conv2D(8, kernel_size=3, kernel_regularizer=regularizer, padding="same", kernel_initializer=GlorotNormal())(x)
    x = MaxPool2D((2, 2))(x)
    x = Dropout(0.25)(x)
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(8, activation='softmax', kernel_regularizer=regularizer, kernel_initializer=GlorotNormal())(x)
    model = Model(inputs=input, outputs=predictions)

    # Compile the model
    if optimizer == SGD:
        model.compile(
            loss='categorical_crossentropy', 
            optimizer=optimizer(learning_rate=lr, momentum=momentum), 
            metrics=['accuracy'])
    else:
        model.compile(
            loss='categorical_crossentropy', 
            optimizer=optimizer(learning_rate=lr), 
            metrics=['accuracy'])

    # Train model
    history = model.fit(train_dataset,
                        epochs=NUMBER_OF_EPOCHS,
                        validation_data=val_dataset,
                        verbose=2)

    # Evaluate model
    result = model.evaluate(test_dataset)
    accuracy = result[1]
    
    # Log hyperparameters and metrics to Weights & Biases
    wandb.log({'accuracy': accuracy, **trial.params})
    
    return accuracy

# Initialize Weights & Biases
wandb.init(project='c3_week4', name='hyperparam_search')

# Create an Optuna study
study = optuna.create_study(direction='maximize')
# Optimize the objective function
study.optimize(objective, n_trials=150, n_jobs=1)


# Access the best trial and its parameters
best_trial = study.best_trial

# Print the results
print('Best trial:')
print('Accuracy: ', best_trial.value)
print('Params: ', best_trial.params)

with open("results/best_params.json", "w") as outfile: 
    for k, v in best_trial.params.items():
        outfile.writelines(f"{k}: {v}\n")

wandb.finish()
