# Project
State shortly what you did during each week. Just a table with the main results is enough. Remind to upload a brief presentation (pptx?) at virtual campus. Do not modify the previous weeks code. If you want to reuse it, just copy it on the corespondig week folder.

## Task1 work

This week we have solved an image classification task using the idea of a Bag of Visual Words to create the image features.
We have explored a basic machine learning pipeline, and we have tried to improve creating more complex features (Dense SIFT, Spatial Pyramid, Fisher Vectors),
introducing some preprocessing techniques (Normalization, scaling and dimensionality reduction) and finally trying different models (SVMs with different kernels, KNN and Logistic Regression).
Also, we have performed many hyperparameter searches following both naive and optimal approaches (Optuna).

From all the testing, we have obtained the following model configuration:

| Hyperparameter             	    | Value 	    | Hyperparameter                         	| Value                   |
|-------------------------------- |------------	|---------------------------------------- |------------------------	|
| **Descriptor:**              	  | Dense SIFT 	| **Dimensionality reduction algorithm:** | No                     	|
| **Descriptor parameter:** 	    | 771        	| **Number of components:**               | None                   	|
| **Scale factor (Dense SIFT):**  | 8          	| **Model type:**                      	  | SVM                    	|
| **Step size (Dense SIFT):**     | 10         	| **Regularization parameter (C):** 	    | 0.13146                	|
| **K (Codebook):**            	  | 348        	| **Kernel:**                          	  | Histogram Intersection 	|
| **Normalization:**           	  | No         	| **Alpha parameter (HI):**               | 0.7                    	|
| **Scaling:**                   	| No         	| **Pyramid level:**                   	  | 2                      	|

And we have obtained the following results on the test set:

| F1-Score    | Accuracy 	  | Precision  	| Recall     |
|-------------|------------	|-------------|----------- |
| 0.9318      | 0.8724    	| 0.9412     	|  0.9227    |


## Task2 work

This week we have explored the same classification problem as the previous week, but now using a MLP in different ways: A classifier and a feature extractor.
#### Approach 1: MLP training and classification
The MLP network trained is used as a classification algorithm.

#### Approach 2: MLP training and SVM classification
The MLP network trained is used as a feature extractor, and the embeddings of the last hidden layer are used as the image descriptors that will be used to train a SVM classifier. 

#### Approach 3: MLP embeddings + BoVW + SVM
First we have trained a MLP model to classify random patches of the original MIT_split dataset -> 0.4857 accuracy score on test set
Then, divide dataset images into patches, computed the embeddings using the trained network, created a BoVW, and then trained an SVM model:

| MLP                    | K (Codebook)  | Acc (Training)  	| Acc (Test)   |
|------------------------|---------------|------------------|--------------|
| 3 layers (64, 128, 64)  | -            | 0.6871          	| 0.4803       |
| 4 layers (512, 512, 512, 512)  | -            | 0.4583          	| 0.4002       |
| 2 layers (1024, 2048)  | 64            | 0.7602          	| 0.6245       |

## Task3 work

## Task4 work

