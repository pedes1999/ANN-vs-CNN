import tensorflow as tf
from tensorflow.keras import datasets,layers,models
import matplotlib.pyplot as plt
import numpy as np

#Loading Sets
(X_train , y_train) ,(X_test ,y_test) = datasets.cifar10.load_data()

#reshaping y_train
y_train= y_train.reshape(-1,)

#setting a classes variable
classes = ["airplane" , "automobile" ,"bird","cat","deer","dog","frog","horse","ship","truck"]

#Creating funt to show img
def plot_sample(X,y,index):
    plt.figure(figsize = (15,2))
    plt.imshow(X[index])
    plt.xlabel(classes[y[index]])

#Scaling the sets
X_train = X_train / 255
X_test = X_test / 255

#Building ANN model
ann = models.Sequential([
    layers.Flatten(input_shape=(32,32,3)),
    layers.Dense(3000,activation = 'relu'),
    layers.Dense(3000,activation = 'relu'),
    layers.Dense(10,activation = 'sigmoid')
])

ann.compile(optimizer="SGD",
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
ann.fit(X_train,y_train,epochs=5)

#Evaluating ANN model
ann.evaluate(X_test ,y_test)

#Classification Report for ANN
from sklearn.metrics import confusion_matrix , classification_report
import numpy as np
y_pred = ann.predict(X_test)
y_pred_classes = [np.argmax(element) for element in y_pred]

print("Classification Report: \n", classification_report(y_test, y_pred_classes))

#Building CNN model
cnn = models.Sequential([
    # cnn
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    # dense
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
cnn.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])
cnn.fit(X_train,y_train,epochs=10)

#Evaluating CNN model
cnn.evaluate(X_test,y_test)
y_pred = cnn.predict(X_test)

#Classification Report for CNN
print("Classification Report: \n",classification_report(y_test,y_classes))