import os
import tensorflow as tf
import keras as keras
import numpy as np
import random as rand
import matplotlib.pyplot as plt

#set random seeds to not be random
os.environ['PYTHONHASHSEED'] = '2022'
np.random.seed(2022)
tf.random.set_seed(2022)
rand.seed(2022)

print(tf.keras.__version__)
print(tf.__version__)
print(np.__version__)

train_data = np.loadtxt('train_data_2022.csv', delimiter =',', skiprows = 1)
test_data = np.loadtxt('test_data_2022.csv', delimiter =',', skiprows = 1)
validation_data = np.loadtxt('val_data_2022.csv', delimiter =',', skiprows = 1)

# train_data = pd.read_csv('train_data_2022.csv')
# test_data = pd.read_csv('test_data_2022.csv')
# validation_data = pd.read_csv("val_data_2022.csv")

x_train = train_data[:,:60]
y_train = train_data[:,60:61]
x_val = validation_data[:,:60]
y_val = validation_data[:,60:61]
x_test = test_data[:,:60]
y_test = test_data[:,60:61]

# x_train = train_data.iloc[:,:60].values
# y_train = train_data.iloc[:,60:61].values
# x_val = validation_data.iloc[:,:60].values
# y_val = validation_data.iloc[:,60:61].values
# x_test = test_data.iloc[:,:60].values
# y_test = test_data.iloc[:,60:61].values

# sc = StandardScaler()
# x_train = sc.fit_transform(x_train)
# x_val = sc.fit_transform(x_val)
# x_test = sc.fit_transform(x_test)

# ohe = OneHotEncoder()
# y_train = ohe.fit_transform(y_train).toarray()
# y_val = ohe.fit_transform(y_val).toarray()
# y_test = ohe.fit_transform(y_test).toarray()

neurons = [4, 16, 32, 64]
models = {}
train_accuracy = []
val_accuracy = []
for x in neurons:
    #create sequential model so each layer will serve as input to the next
    model = keras.models.Sequential()
    #creates first layer of ANN with relu activation function
    model.add(keras.layers.Dense(x, activation = 'relu'))
    #creates second layer of ANN with sigmoid activation function
    model.add(keras.layers.Dense(1, activation = 'sigmoid'))
    #specifies that we want to judge our performance based on accuracy and optimize with adams function
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    #trains the model with the training data and validates with the validation data using
    #5 epochs per model and a batch size of 10
    result = model.fit(x_train, y_train, validation_data = (x_val, y_val), epochs = 5, batch_size = 10)
    #adds training accuracy, validation accuracy, and model to a list in order to keep track of the
    #most optimal model for testing later
    train_accuracy.append(result.history['accuracy'][4])
    val_accuracy.append(result.history['val_accuracy'][4])
    models[x] = model

_, accuracy = models[16].evaluate(x_test, y_test)
print("16 neuron accuracy: ", accuracy * 100)
# `for x in neurons:
#     m = models[x]
#     _, accuracy = m.evaluate(x_test, y_test)
#     print(x, "neuron accuracy: ", accuracy * 100)`
# model = models[16]
# y_pred = model.predict(x_test)
# pred = list()
# for i in range(len(y_pred)):
#     pred.append(np.argmax(y_pred[i]))
# test = list()
# for i in range(len(y_test)):
#     test.append(np.argmax(y_test[i]))
# test_accuracy = accuracy_score(pred, test)
# print("16 Neuron Accuracy: ", test_accuracy * 100)

# model = models[64]
# y_pred = model.predict(x_test)
# pred = list()
# for i in range(len(y_pred)):
#     pred.append(np.argmax(y_pred[i]))
# test = list()
# for i in range(len(y_test)):
#     test.append(np.argmax(y_test[i]))
# test_accuracy = accuracy_score(pred, test)
# print("64 Neuron Accuracy: ", test_accuracy * 100)

plt.scatter(neurons, train_accuracy, c="blue")
plt.scatter(neurons, val_accuracy, c="red")
plt.title('Number of Neurons vs. Accuracy')
plt.xlabel("Number of Neurons")
plt.ylabel("Accuracy")
plt.legend(['Training Accuracy', 'Validation Accuracy'], loc = 'upper left')
plt.show()
