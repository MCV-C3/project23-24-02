import os
os.environ["KERAS_BACKEND"] = "tensorflow"  # Or "jax" or "torch"!
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import matplotlib.pyplot as plt
from utils_example import *
import keras
import tensorflow as tf
from keras.layers import Dense
from keras.utils.vis_utils import plot_model
from tensorflow.keras.optimizers import SGD, Adam
from keras.applications.resnet import ResNet50
from keras.applications.resnet import preprocess_input
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, BatchNormalization
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import wandb

# Global variables
DATASET_DIR = 'data/MIT_split'
IMG_WIDTH = 224
IMG_HEIGHT= 224
BATCH_SIZE=4
NUMBER_OF_EPOCHS=20
N_LAST_LAYERS_TO_UNFREEZE=0

gpus = tf.config.list_physical_devices('GPU')
print(f'GPUs available: {gpus}')

def save_history(history, result, result_name):
    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(f"results/{result_name}_accuracy.jpg")
    plt.close()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('Loss (CELoss)')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.savefig(f"results/{result_name}_loss.jpg")

    with open(f"results/{result_name}_eval_metrics.txt", 'w') as file:
        file.write(f'Accuracy: {result[1]}\n')
        file.write(f'Loss: {result[0]}\n')

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

# Define the data generator for data augmentation and preprocessing
train_preprocess_args = {
    "rotation_range":20,
    "width_shift_range":0.2,
    "height_shift_range":0.2,
    "shear_range":0.2,
    "zoom_range":.2,
    "horizontal_flip":True,
    "vertical_flip":False
}
train_dataset = create_dataset("data/MIT_small_train_1/train/", preprocess_args=train_preprocess_args, shuffle=True)
val_dataset = create_dataset('data/MIT_small_train_1/test/')
test_dataset = create_dataset('data/MIT_split/test/')


for do_batchnorm in [False, True]:   
    for layer_size in [1024, 512, 2048]:
        wandb.init(project='c3_week3', name=f"batchnorm_{do_batchnorm}_layersize_{layer_size}")
        for n_fc_layers in range(1, 7):
            # Load pretrained model
            base_model = ResNet50(weights='imagenet', include_top=False)
            x = base_model.output
            if do_batchnorm:
                x = BatchNormalization()(x)
            x = GlobalAveragePooling2D()(x)
            for _ in range(n_fc_layers):
                x = Dense(layer_size, activation='relu')(x)
                if do_batchnorm:
                    x = BatchNormalization()(x)

            predictions = Dense(8, activation='softmax')(x)

            model = Model(inputs=base_model.input, outputs=predictions)

            # Freeze model (Only ResNet50 layers)
            for i, layer in enumerate(base_model.layers):
                layer.trainable = False

            # compile the model (should be done *after* setting layers to non-trainable)
            model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.01),  metrics=['accuracy'])

            # train the model on the new data for a few epochs
            history = model.fit(train_dataset,
                                epochs=NUMBER_OF_EPOCHS,
                                validation_data=val_dataset,
                                verbose=2)

            result = model.evaluate(test_dataset)
            save_history(history, result, f"batchnorm/do_batchnorm_{do_batchnorm}_nlayers_{n_fc_layers}")
            wandb.log({'Num_layers': n_fc_layers, 'accuracy': result[1], 'loss': result[0]})
        wandb.finish()