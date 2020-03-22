"""
Created on Sun Oct 20 19:44:34 2019
Predicting the stock prices of Google using LSTM
@author: aditya
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
# selecting in the column we are interested in using iloc 
# second, creating a numpy array
training_set = dataset_train.iloc[:,1:2].values # data frame created. To make it numpy array(as they are the only mode of input in Keras), we add values()

# when there is a sigmoid function as the activation function in output layer; normalization is preferred.
# normalization and standardization are two imp methods for feature scaling
# feature scaling begins
from sklearn.preprocessing import MinMaxScaler 
sc = MinMaxScaler(feature_range = (0,1))
# it is recommended to keep the original training set
training_set_scaled = sc.fit_transform(training_set)

# now we need to define a data structure specifying what the RNN need to remember when predticting the next stock price
# this is known as the number of time steps
# wrong number of time steps may lead to overfitting or nonsense prediction

# creating a data structure with 60 timesteps and 1 output 
# we mean that the RNN will be going to look at 60 stock prices between 60 days before time t and time t.
# so we explore some past, to lean some trends
# we predict it at time t+1
# we obtained 60 by trial and error
# hence 60 time steps and one output at time t+1

# X_train : input of neural network(the 60 prices before that financial day)
# y_train : value of stock price the day after
X_train = []
y_train = [] 
# populate X_train with 60 previous stock prices 
# populate y_train with the next stock price
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0]) # hence we get the stock prices from day 0 to day 59 (prev 60). Output y_train will have the stock price of day 60, as the RNN will learn to predict. 
    y_train.append(training_set_scaled[i, 0])
    # X_train and y_train are list..so we have to convert them in numpy array 
X_train, y_train = np.array(X_train), np.array(y_train)

# introducing new dimensonality : till now we are using one indicator to predict the stock price. Ehat if we need more?
# for that we will enhance the dimensonality using the reshape function in the numpy array.
# we only need to do this for X_train
# open keras documentation
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1)) # 1 for open google stock price

# Designing a stacked LSTM without dropout regularization
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
regressor = Sequential() # initializes the RNN
# adding layer 1 and dropout regularization 
# LSTM(<number of cells>, <return sequences = True ; as we are building a stacked model>, <input shape = shape of the input in X_train>)
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1],1) )) # we will add an object of LSTM class
regressor.add(Dropout(0.2))
# adding new layers of LSTM with dropout regularization
regressor.add(LSTM(units = 50, return_sequences = True)) # we will add an object of LSTM class
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = True)) # we will add an object of LSTM class
regressor.add(Dropout(0.2))
regressor.add(LSTM(units = 50, return_sequences = False)) # we will add an object of LSTM class
regressor.add(Dropout(0.2))

# adding the output layer
# the output layer is fully connected to the output layer
regressor.add(Dense(units = 1))
# compiling
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error') 
# training
regressor.fit(X_train, y_train, epochs = 100, batch_size = 32 )

# save
from keras.models import model_from_json
model_json = regressor.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
regressor.save_weights("model.h5")
print("Saved model to disk")

# get model
from keras.models import model_from_json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
loaded_model.compile(loss='mean_squared_error', optimizer='adam')
# loading over 

# test set
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:,1:2].values
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
# at each day of financial day, we need the prices of prev 60 days
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60: ].values
inputs = inputs.reshape(-1,1) 
inputs = sc.transform(inputs)
X_test = [] 
for i in range(60,80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = loaded_model.predict(X_test)
# inverse the scaling on the prediction 
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# visualizing the result
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()
