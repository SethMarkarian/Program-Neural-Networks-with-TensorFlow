import tensorflow as tf
import numpy as np
from tensorflow import keras

''' Simplest possible neural network
    Has one layer, that one layer has one neuron and the input shape to it has one value '''
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

''' This compiles the neural network
    The loss function measures the guessed answers against the known correct answers and measures how well or not it did
    The optimizer function makes a guess based on how well the loss function returns '''
model.compile(optimizer='sgd', loss='mean_squared_error')

''' Data to be fed to model '''
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

''' This function helps train the neural network
    It learns the relationship between the X's and Y's 
    It will go through the loop before making a guess, measuring how good or bad it its
    If not, it will use the optimizer to make another guess
    It will print out the loss for each epoch'''
model.fit(xs, ys, epochs=500)

''' At this point, model has been trained to learn the relationship between X and Y
    This method will figure out the Y for the previously unknown X
    For example, if X is 10, what will Y be?'''
print(model.predict([10.0]))