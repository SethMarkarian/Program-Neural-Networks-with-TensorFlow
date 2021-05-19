import tensorflow as tf
import matplotlib.pyplot as plt

''' The Fashion MNIST data that will be used from the tf.keras.datasets API '''
mnist = tf.keras.datasets.fashion_mnist

''' load_data returns a tuple with two lists:
        training_images- graphics of clothing
        training_labels- labeling each graphic of clothing '''
(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

''' Printing a trainer image and label that corresponds to the image'''
plt.imshow(training_images[0])
# print(training_labels[0])
# print(training_images[0])

''' When training a neural network, easier to treat all values between 0 and 1
    This process is called normalization'''
training_images = training_images / 255.0
test_images = test_images / 255.0

''' To design the model, needs 3 layers
    Sequential defines a sequence of layers in the neural network
    Flatten takes a square and turns it into a one dimensional vector
    Dense adds a layer of neurons
    Exercise changed it to 1024
    Activation tells each layer of neurons wha to do
    Relu effectively means if X is greater than 0, return X, else return 0
    Softmax takes a set of values and picks the biggest one'''
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(), 
                                    tf.keras.layers.Dense(1024, activation=tf.nn.relu), 
                                    tf.keras.layers.Dense(10, activation=tf.nn.softmax)])

''' Build the model using optimizer and loss functions
    Goal is to have the model figure out the relationship between the training data and labels
    Using the metrics parameter: reports the accuracy of the training'''
model.compile(optimizer = tf.keras.optimizers.Adam(),
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)

''' Helps the model perform on data it hasn't seen yets'''
model.evaluate(test_images, test_labels)


#====================Exercise 1====================
print("\nEXERCISE 1 \n")

''' Creates a set of classifications for each of the test images
    Prints the first entry in classifications
    Output is a list of numbers'''
classifications = model.predict(test_images)
print("First element in classifications: " + str(classifications[0]))
print("First element in test_labels: " + str(test_labels[0]))