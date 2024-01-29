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
from tensorflow.keras.datasets import cifar100



gpus = tf.config.list_physical_devices('GPU')
print(f'GPUs available: {gpus}')

def save_activation_map_from_path(img_path, model, layer, config):

    img = image.load_img(img_path, target_size=(224,224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    model_features = Model(inputs=model.input, outputs=model.get_layer(layer).output)
    features = model_features.predict(x)

    # Flatten the features
    flat_features = features.flatten()

    # Normalize the flattened features to the range [0, 1]
    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(flat_features.reshape(-1, 1)).reshape(features.shape)

    # Save the scaled features as an image
    plt.imshow(scaled_features[0, :, :, 0], cmap='bwr')  # Assuming a 3D feature map, adjust if needed
    plt.savefig('./activation_maps'+config['MODEL_FNAME']+'.png')

def compute_roc(test_labels, y_score, classes, results_path, title_roc):
    # First, binarize the labels
    y_test = LabelBinarizer().fit_transform(test_labels)
    n_classes = y_test.shape[1]

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    plt.figure(figsize=(8, 8))

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

        # Plot each ROC curve without specifying color
        plt.plot(fpr[i], tpr[i], lw=1,
                 label='Class {0} (AUC = {1:0.3f})'.format(classes[i], roc_auc[i]))

    # Plot the average ROC curve without specifying color
    mean_tpr = np.mean([interp(np.linspace(0, 1, 100), fpr[i], tpr[i]) for i in range(n_classes)], axis=0)
    mean_auc = auc(np.linspace(0, 1, 100), mean_tpr)
    plt.plot(np.linspace(0, 1, 100), mean_tpr, color='navy', linestyle='--', lw=2,
             label='Average (AUC = {0:0.3f})'.format(mean_auc))

    # Customize plot details
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Each Class')

    # Adjust legend to avoid overlapping
    plt.legend(loc="lower right")

    # Save the plot
    plt.savefig(results_path, bbox_inches='tight')
    plt.close()

def save_confusion_matrix(cm, classes, output_file):
    plt.figure(figsize=(8, 8))
    
    ## Plot confusion matrix
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

    # Customize plot details
    plt.title('Confusion matrix')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    
    # Save the plot without specifying a specific color
    plt.savefig(output_file, bbox_inches='tight')
    plt.close()

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
        dataset = data_generator.flow(
            config['x_train'],
            config['y_train'],
            batch_size=config['BATCH_SIZE'],
            shuffle=True
        )
    elif data == 'test':
        # Load and preprocess the test or validation dataset
        dataset = data_generator.flow(
            config['x_test'],
            config['y_test'],
            batch_size=config['BATCH_SIZE'],
            shuffle=False
        )
    else:
        raise ValueError('data must be either training or test')
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

    # choose the number of layers to train
    """if config['layers_to_train'] == 0:
        for layer in base_model.layers:
            layer.trainable = False
    else:
        for layer in model.layers[:config['layers_to_train']]:
            layer.trainable = False
        for layer in model.layers[-(config['layers_to_train']+3):]: # +3 due to the 3 layers added
            layer.trainable = True
    """
    #plot_model(model, to_file='modelResNet50.png', show_shapes=True, show_layer_names=True)
        
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

def train(config, model, train_dataset, test_dataset):
    # train the model on the new data for a few epochs
    history = model.fit(train_dataset,
                        epochs=config['NUMBER_OF_EPOCHS'],
                        validation_data=test_dataset,
                        verbose=2)

    model.save_weights('./output/'+config['MODEL_FNAME']+'.weights.h5')

    #result = model.evaluate(test_dataset)

    accuracy_training = history.history['accuracy']
    accuracy_validation = history.history['val_accuracy']

    with open('./output/'+config['MODEL_FNAME']+'.txt', 'w') as file:
        with redirect_stdout(file):
            print(model.summary())
        file.write('ACCTRAINING,'+str(accuracy_training)+ '\n\n')
        file.write('ACCVALIDATION,'+str(accuracy_validation)+ '\n\n')
        #file.write('RESULTS,'+str(result))

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
    compute_roc(test_labels, Y_pred, class_names, config['MODEL_FNAME']+'_ROC.png', 'ROC curve') 
    conf_mat = confusion_matrix(test_labels, y_pred)
    conf_mat = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis]
    save_confusion_matrix(conf_mat, class_names, config['MODEL_FNAME']+'_confusionmatrix.png')
    
    layers = ['conv5_block3_out', 'conv4_block6_out', 'conv3_block4_out', 'conv2_block3_out']
    for layer in layers:
        save_activation_map_from_path('./data/MIT_small_train_1/train/coast/bea10.jpg', model, layer, config)



def train_val(config):
    training_data_generator = define_data_generator(config, 'training')
    train_dataset = load_and_preprocess(config, training_data_generator, 'training')

    test_data_generator = define_data_generator(config, 'test')
    test_dataset = load_and_preprocess(config, test_data_generator, 'test')

    model = create_model(config)

    if config['MODE']=='train':

        train(config, model, train_dataset, test_dataset)

    elif config['MODE']=='val':
    
        val(config, model, test_dataset)
    
    elif config['MODE']=='train_val':
        
        train(config, model, train_dataset, test_dataset)
        val(config, model, test_dataset)

    else:
        raise ValueError('MODE must be train or val or train_val')

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

(x_train, y_train), (x_test, y_test) = cifar100.load_data()


config = {  'IMG_WIDTH': 32, 
            'IMG_HEIGHT': 32, 
            'BATCH_SIZE': 50,
            'NUMBER_OF_EPOCHS': 50,
            'x_train': x_train,
            'y_train': y_train,
            'x_test': x_test,
            'y_test': y_test,
            'layers_to_train': 177, # list containing the modules to train 
            'learning_rate': 0.0001,
            'momentum': 0.4,
            'nesterov': True,
            'optimizer': 'Adam', 
            'scheduler': None, 
            'MODEL_FNAME': '14', 
            'MODE': 'train'
            }

train_val(config)


