# This imports Python's built-in operating system library,
#  which lets you interact with the computer’s file system 
# (e.g., reading/writing files, navigating directories).
import os

# This imports OpenCV-python
# a powerful library for computer vision tasks (like image processing). 
import cv2

# NumPy is a library for numerical computing in Python.
# It provides support for arrays (like lists but more powerful) 
# and many mathematical functions that operate on these arrays. 
# Here, it’s often used to handle image data and matrices
import numpy as np

# Matplotlib is a plotting library that can create graphs and visualizations. 
# The alias plt is typically used for its pyplot module 
# (usually imported as import matplotlib.pyplot as plt)
import matplotlib.pyplot as plt

# TensorFlow is a popular deep learning library developed by Google. 
# It provides tools for building and training neural networks. The code uses
# TensorFlow’s high-level API, Keras, to build the model.
import tensorflow as tf

# # TensorFlow comes with several pre-loaded datasets. Here, we’re using the MNIST 
# # dataset, which contains 70,000 images of handwritten digits (0-9). 
# # Each image is 28x28 pixels.
# mnist = tf.keras.datasets.mnist

# # mnist.load_data()
# # This function downloads and loads the dataset.
# # It returns two tuples: 
# # (x_train, y_train) the training data
# # (x_test, y_test) the testing data
# # x_train and x_test contain the images.
# # y_train and y_test contain the corresponding digit labels (for example, 7 or 3).
# (x_train, y_train), (x_test, y_test) = mnist.load_data()


# # The pixel values in the MNIST images originally range from 0 to 255.
# # Normalization scales these values to a smaller range (typically between 0 and 1)
# # which helps the neural network learn more efficiently.
# x_train = tf.keras.utils.normalize(x_train, axis=1)
# x_test = tf.keras.utils.normalize(x_test, axis=1)

# # This creates a Sequential model, which is a linear stack of layers. 
# # Think of it as a series of building blocks (layers) where the output 
# # of one layer is the input of the next.
# model = tf.keras.models.Sequential()

# # The MNIST images are 2D (28x28 pixels). The Flatten layer converts each 
# # 2D image into a 1D array (vector) of 784 pixels (since 28 × 28 = 784).
# model.add(tf.keras.layers.Flatten(input_shape=(28,28)))

# # A Dense (or fully connected) layer means every input neuron 
# # (pixel value) is connected to every neuron in this layer.
# # The ReLU (Rectified Linear Unit) activation function is applied to each neuron’s output. 
# # ReLU is defined as max(0, x) which means it outputs the input directly if it’s
# # positive; otherwise, it outputs 0. This introduces non-linearity into the model,
# # allowing it to learn more complex patterns.
# model.add(tf.keras.layers.Dense(128, activation='relu'))
# model.add(tf.keras.layers.Dense(128, activation='relu'))

# # The output layer has 10 neurons, one for each digit (0-9).
# # The softmax activation function converts the output values into probabilities
# # that add up to 1. The digit corresponding to the neuron with the highest 
# # probability is the model’s prediction for that image.
# model.add(tf.keras.layers.Dense(10, activation='softmax'))


# # Before training a neural network, you need to configure it for the learning process.
# # This step is called "compiling" the model.
# # 1) The optimizer is an algorithm that adjusts the weights of the network to minimize the error
# # 2) The loss function measures how far the model's predictions are from the true values.
# # The goal is to minimize this loss during training.
# # sparse_categorical_crossentropy is used here because we're dealing with 
# # a classification problem (recognizing digits 0-9) and the labels are 
# # provided as integers (like 0, 1, 2, ... 9)
# model.compile(optimizer ='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# # This line tells the model to start learning from the training data.
# model.fit(x_train, y_train, epochs=3)

# model.save('handwritten.keras')

model = tf.keras.models.load_model('handwritten.keras')

image_number = 1
while os.path.isfile(f"digits/digit{image_number}.png"):
    try :

        # Uses OpenCV to read the image from the file.
        # [:,:,0]: After reading the image, this slice selects only the first color channel 
        # because the model expects a grayscale img
        img = cv2.imread(f"digits/digit{image_number}.png")[:,:,0]

        # Keras expects inputs in the form (batch_size, height, width). So we wrap the img in an array and 
        # convert it to a NumPy array 
        img = np.invert(np.array([img]))

        # The model returns an array of probabilities for each digit (0 through 9).
        prediction = model.predict(img)

        # np.argmax(prediction): Finds the index of the highest value in the prediction array.
        print(f"This digit is probably a {np.argmax(prediction)}")

        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()

    except:
        print("Error!")
    finally:
        image_number += 1