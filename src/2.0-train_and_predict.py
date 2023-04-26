import pandas as pd
from pandas import concat
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


sys.path.append('./pre_processing.py')
from pre_processing import BTCUSD_Features

BTCUSD = BTCUSD_Features()
BTCUSD_df = BTCUSD.get_full_df()

X = []
y = []

BTCUSD_df = BTCUSD_df.drop("DOGEUSD", axis=1).drop("Volume", axis=1)

training_length = int(np.ceil(BTCUSD_df.shape[0] * 0.9))
testing_length = int(BTCUSD_df.shape[0] - training_length)

training_df = BTCUSD_df.head(training_length)
testing_df = BTCUSD_df.tail(testing_length)

training_df = training_df.dropna()
testing_df = testing_df.dropna()

scaler = MinMaxScaler()
training_data = scaler.fit_transform(training_df)

X_train = [] 
Y_train = []
training_data.shape[0]
for i in range(60, training_data.shape[0]):
    X_train.append(training_data[i-60:i])
    Y_train.append(training_data[i,0])

X_train, Y_train = np.array(X_train), np.array(Y_train)

model = Sequential() 
model.add(LSTM(units = 50, activation = 'linear', return_sequences = True, input_shape = (X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units = 60, activation = 'linear', return_sequences = True))
model.add(Dropout(0.3)) 
model.add(LSTM(units = 80, activation = 'linear', return_sequences = True))
model.add(Dropout(0.4)) 
model.add(LSTM(units = 120, activation = 'linear'))
model.add(Dense(units =1))
model.summary()

model.compile(optimizer = 'rmsprop', loss = 'mean_squared_error')
history= model.fit(X_train, Y_train, epochs = 40, batch_size =60, validation_split=0.1)

loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))
plt.figure()
plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title("Training and Validation Loss")
plt.legend()
plt.show()

# Testing Set

# Get 60 candles of data prior to testing set for time series data
prior_60_candles = training_df.tail(60)
df= prior_60_candles.append(testing_df, ignore_index = True)

transform_train = scaler.transform(df)

X_test = []
Y_test = []
for i in range (60, transform_train.shape[0]):
    X_test.append(transform_train[i-60:i]) 
    Y_test.append(transform_train[i, 0])

X_test, Y_test = np.array(X_test), np.array(Y_test) 

X_test.shape, Y_test.shape
Y_pred = model.predict(X_test) 

Y_pred, Y_test

plt.figure(figsize=(14,5))
plt.plot(Y_test, color = 'black', label = 'Real Bitcoin High')
plt.plot(Y_pred, color = 'yellow', label = 'Predicted Bitcoin High')
plt.title('Bitcoin Price Prediction Using LSTM Neural Network')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

mse = mean_squared_error(Y_test, Y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(Y_test, Y_pred)
r2 = r2_score(Y_test, Y_pred)

print("MSE:", mse)
print("RMSE:", rmse)
print("MAE:", mae)
print("R-squared:", r2)