import keras
from keras import models, layers
from keras.datasets import mnist
from IPython.display import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.utils import to_categorical


# 讀入mnist資料集中的訓練資料及測試資料
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 資料預處理及訓練,train資料大小60000,test資料大小10000,像素28*28
train_images1 = train_images.reshape((60000, 28 * 28))
train_images1 = train_images1.astype('float32') / 255
test_images1 = test_images.reshape((10000, 28 * 28))
test_images1 = test_images1.astype('float32') / 255
train_labels_categ = to_categorical(train_labels)
test_labels_categ = to_categorical(test_labels)

# 編譯模型
network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

#訓練神經網路
history = network.fit(train_images1, train_labels_categ, epochs= 10, batch_size=100, 
                      validation_data=(test_images1, test_labels_categ))

# Performance
prediction = network.predict_classes(test_images1)
pd.crosstab(test_labels, prediction,
            rownames=['label'], colnames=['predict'])

for i in range(len(test_labels)):
    if (test_labels[i] == 1) & (test_labels[i] != prediction[i]) :
        print('when ', i, ': test_labels is ', test_labels[i], ', but predict ', prediction[i])
        plt.imshow(test_images[i,:,:], cmap = plt.cm.gray) 
        plt.show()

print(network.predict_classes(test_images1))
print(network.predict(test_images1))