# 手寫數字數據集-機器學習實作
import keras
import cv2
from keras import models, layers, regularizers
from keras.datasets import mnist
from IPython.display import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score


# 各項函數
layerkernelregularizer1 = regularizers.l1(0.001) # layer1 Regularization function
layersDense1= 512 # layer1 hidden units
activation1= 'relu' # layer1 Classifier 
outputnums= 10 # Output layer units
activation2= 'softmax' # Output layer Classifier
compileoptimizer= 'rmsprop' # 可改用 keras.optimizers.Adam(learning_rate=0.01)
compileloss= 'categorical_crossentropy' # 可改用 'mse'
compilemetrics= 'accuracy' # 可改用 'mae'
epochsnum= 5
batchsize= 300    
lablenumber= 1 # 看哪個標籤: 0~9
Dropoutrate= 0.6
Layernum = 0

# 讀入mnist資料集中的訓練資料及測試資料
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 資料預處理及訓練,train資料大小60000筆,test資料大小10000筆,
# Reshape: 將圖片(三維)轉化為28*28(二維)矩陣
train_images1 = train_images.reshape((60000, 28 * 28))
train_images1 = train_images1.astype('float32') / 255
test_images1 = test_images.reshape((10000, 28 * 28))
test_images1 = test_images1.astype('float32') / 255
train_labels_categ = to_categorical(train_labels)
test_labels_categ = to_categorical(test_labels)

# 計算起始時間
e1 = cv2.getTickCount()

# 編譯模型
# Sequential Model: 單一輸入,單一輸出,按順序一層一層的執行
# Kernel_initializer: 權重的初始值, l1: Loss Regularization; l2: Ridge Regularation
# Loss function: 針對要解決的問題,決定最小化什麼目標函數(如:categorical_crossentropy、mse、hinge)
# Activation function: 過濾門檻,轉化方式(如: softmax, sigmoid, relu, tanh)
## softmax：值介於 [0,1] 之間，且機率總和等於 1，適合多分類使用。
## sigmoid：值介於 [0,1] 之間，且分布兩極化，大部分不是 0，就是 1，適合二分法。
## relu (Rectified Linear Units)：忽略負值，介於 [0,∞] 之間。
## tanh：與sigmoid類似，但值介於[-1,1]之間，即傳導有負值。
network = models.Sequential()
network.add(layers.Dense(units=512, activation= activation1, input_shape=(28*28,)))
for j in range(Layernum):
    network.add(layers.Dense(units=64, activation= 'relu'))
network.add(layers.Dropout(Dropoutrate)) #dropout layer
network.add(layers.Dense(outputnums, activation= activation2))
network.compile(optimizer= compileoptimizer,
                loss= compileloss,
                metrics= [compilemetrics])

# 訓練神經網路
history = network.fit(train_images1, train_labels_categ, epochs= epochsnum, batch_size= batchsize, 
                      validation_data=(test_images1, test_labels_categ))

# 訓練中 accuracy 及 val_accurary的變動
print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

# 訓練成果
scores = network.evaluate(train_images1, train_labels_categ)
print('Scores:', scores)

# 進行預測
prediction = network.predict_classes(test_images1)

# 預測cross table成果
crosstablex = pd.crosstab(test_labels, prediction,
            rownames=['label'], colnames=['predict'])

accuracyscore = accuracy_score(test_labels, prediction)

for i in range(len(test_labels)):
    if (test_labels[i] == lablenumber) & (test_labels[i] != prediction[i]):
        print('when ', i, ': test_labels is ', test_labels[i], ', but predict ', prediction[i])
        plt.imshow(test_images[i,:,:], cmap = plt.cm.gray) 
        plt.show()

# 終止時間
e2 = cv2.getTickCount()
t = (e2-e1) / cv2.getTickFrequency()
print('Time duration:%5.2f'%t)
        
print(network.predict_classes(test_images1))
print(network.predict(test_images1))
print(prediction)
print(crosstablex)
print(accuracyscore)
