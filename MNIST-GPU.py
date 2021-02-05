import keras
from keras import models, layers
from keras.datasets import mnist
from IPython.display import Image
import matplotlib.pyplot as plt
from keras.utils import to_categorical

Image(filename='data/05-Chollet-MNIST-sample.jpg')
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

Image(filename='data/05-MNIST.png')
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

train_images1 = train_images.reshape((60000, 28 * 28))
train_images1 = train_images1.astype('float32') / 255

test_images1 = test_images.reshape((10000, 28 * 28))
test_images1 = test_images1.astype('float32') / 255

train_labels_categ = to_categorical(train_labels)
test_labels_categ = to_categorical(test_labels)

history = network.fit(train_images1, train_labels_categ, epochs= 5, batch_size=128, 
                      validation_data=(test_images1, test_labels_categ))
