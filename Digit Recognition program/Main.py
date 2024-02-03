# -*- coding: utf-8 -*-
"""
# Handwritten Digit Recognition using Neural Network

Handwritten digit recognition using MNIST dataset is a major project made with the help of Neural Network. It basically detects the scanned images of handwritten digits.

I have taken this a step further where our handwritten digit recognition system not only detects scanned images of handwritten digits but also allows writing digits on the screen with the help of an integrated GUI for recognition.

Extract the data from mnist-original.mat file

### Import Libraries
"""

from scipy.io import loadmat
import numpy as np
from Model import neural_network
from RandInitialize import initialise
from Prediction import predict
from scipy.optimize import minimize

"""### Loading The Mat File"""

# from google.colab import files # when using colab
# uploaded = files.upload() # when using colab 

# import scipy.io # when using colab
import pandas as pd
import matplotlib.pyplot as plt
# Load the .mat file # when using colab
# mat = scipy.io.loadmat('mnist-original.mat') # when using colab

mat = loadmat('Data/mnist-original.mat')
# The data is typically stored in the 'data' key
data = mat['data']

# The data is usually flattened, so we need to reshape it to view it as an image
# MNIST images are 28x28 pixels
image_data = np.reshape(data[:, 0], (28, 28))

# Display the first image
plt.imshow(image_data, cmap='gray')
plt.show()

# Now let's display the second image
image_data = np.reshape(data[:, 1], (28, 28))
plt.imshow(image_data, cmap='gray')
plt.show()

"""### Extracting Features from mat file"""

X = mat['data']
X =X.transpose()

# Normalizing the data
X = X /255

# Extracting Lbles from Mat file
y = mat['label']
y = y.flatten()

"""### Spliting data into Traning set with 60, 000"""

X_train = X[:60000, :]
y_train = y[:60000]

"""### Spliting data into testing set 10,000"""

X_test = X[60000: , :]
y_test = y[60000:]

m = X.shape[0]

input_layer_size = 784 # Images are of 28 x 28 px so there will be 784 features
hidden_layer_size = 100
num_labels = 10 # there are 10 classes [0 , 9]

# Randomly Initializing Thetas
initial_Theta1 = initialise(hidden_layer_size, input_layer_size)
initial_Theta2 = initialise(num_labels, hidden_layer_size)

"""# Unrolling prameters into a single column vector"""

initial_nn_params = np.concatenate((initial_Theta1.flatten(), initial_Theta2.flatten()))
maxiter = 100
lambda_reg = 0.1 # To avoid overfitting
myargs = (input_layer_size, hidden_layer_size, num_labels, X_train, y_train, lambda_reg)

# Calling minimize function to minimize cost function and to traib weights
results = minimize(neural_network, x0=initial_nn_params, args = myargs, options = {'disp': True, 'maxiter':maxiter},
                   method = "L-BFGS-B", jac =True)

nn_params = results["x"] # Trained Theta is extracted

# weights are split back to Theta1, Theta2
Theta1 = np.reshape(nn_params[:hidden_layer_size * (input_layer_size + 1)], (
    hidden_layer_size, input_layer_size + 1 # Shape = (100, 785)
))
Theta2 = np.reshape(nn_params[hidden_layer_size * (input_layer_size + 1):], 
                      (num_labels, hidden_layer_size + 1))  # shape = (10, 101)

# Checking test set accuracy of our model
pred = predict(Theta1, Theta2, X_test)
print("Test set Accuracy : {:f}".format((np.mean(pred == y_test) * 100)))

# Cheking train set accuracy of our model
pred = predict(Theta1, Theta2, X_train)
print('Training Set Accuracy: {:f}'.format((np.mean(pred == y_train) * 100)))
 
# Evaluating precision of our model
true_positive = 0
for i in range(len(pred)):
    if pred[i] == y_train[i]:
        true_positive += 1
false_positive = len(y_train) - true_positive
print('Precision =', true_positive/(true_positive + false_positive))
 
# Saving Thetas in .txt file
np.savetxt('Theta1.txt', Theta1, delimiter=' ')
np.savetxt('Theta2.txt', Theta2, delimiter=' ')