
# Stock Price Prediction using LSTM Model

This project demonstrates how to predict stock prices using a Long Short-Term Memory (LSTM) neural network in Python, implemented in a Google Colab environment. The project involves data preprocessing, building and training an LSTM model, and making predictions on stock price data.

## Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Data](#data)
- [Preprocessing](#preprocessing)
- [Model](#model)
- [Training](#training)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction
This project aims to predict future stock prices using historical stock price data. We use an LSTM model, a type of recurrent neural network (RNN) that is well-suited for time series forecasting.

## Installation
To run this project in Google Colab, you'll need to install the following dependencies:

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `tensorflow`

You can install these dependencies using the following commands in a Google Colab cell:

```python
!pip install numpy pandas matplotlib scikit-learn tensorflow
```

## Data
For this project, we use Amazon historical stock price data. You can upload your dataset to Google Colab using Google Drive.

## Preprocessing
We preprocess the data by normalizing it and creating sequences for the LSTM model. The steps include:

1. Loading the dataset.
2. Normalizing the data.
3. Creating sequences for the LSTM model.

```python
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')
df = pd.read_csv('/AMZN.csv')
df1 = df.reset_index()['Close']

import matplotlib.pyplot as plt
plt.plot(df1)

import numpy as np
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))

# Splitting dataset into train and test split
training_size = int(len(df1) * 0.65)
test_size = len(df1) - training_size
train_data, test_data = df1[0:training_size, :], df1[training_size:len(df1), :1]

# Convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - time_step - 1):
        a = dataset[i:(i + time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

time_step = 200
X_train, y_train = create_dataset(train_data, time_step)
X_test, ytest = create_dataset(test_data, time_step)

# Reshape input to be [samples, time steps, features] which is required for LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
```

## Model
We build an LSTM model using TensorFlow/Keras. The model consists of:

- LSTM layers
- Dense layers

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(200, 1)))
model.add(LSTM(50, return_sequences=True))
model.add(LSTM(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.summary()
```

## Training
We train the model using the preprocessed data.

```python
model.fit(X_train, y_train, validation_data=(X_test, ytest), epochs=100, batch_size=64, verbose=1)
```

## Evaluation
We evaluate the model's performance using metrics such as Mean Squared Error (MSE) and visualize the predicted vs. actual stock prices.

```python
# Prediction
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# Transform back to original form
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)

# Calculate RMSE performance metrics
import math
from sklearn.metrics import mean_squared_error

train_rmse = math.sqrt(mean_squared_error(y_train, train_predict))
test_rmse = math.sqrt(mean_squared_error(ytest, test_predict))

# Plotting
look_back = 200
trainPredictPlot = np.empty_like(df1)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict

testPredictPlot = np.empty_like(df1)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict) + (look_back * 2) + 1:len(df1) - 1, :] = test_predict

plt.plot(scaler.inverse_transform(df1))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
```

## Usage
To use this project:

1. Open the Google Colab notebook.
2. Upload your dataset or use the provided sample dataset.
3. Run the notebook cells step-by-step to preprocess the data, build and train the model, and make predictions.

## Results
The results of the stock price prediction are visualized using Matplotlib. The predicted stock prices are compared with the actual stock prices to evaluate the model's performance.

```python
# Demonstrate prediction for next 30 days
x_input = test_data[241:].reshape(1, -1)
temp_input = list(x_input)
temp_input = temp_input[0].tolist()

from numpy import array

lst_output = []
n_steps = 200
i = 0
while (i < 100):
    if (len(temp_input) > 200):
        x_input = np.array(temp_input[1:])
        x_input = x_input.reshape(1, -1)
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]
        lst_output.extend(yhat.tolist())
        i = i + 1
    else:
        x_input = x_input.reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        lst_output.extend(yhat.tolist())
        i = i + 1

day_new = np.arange(1, 201)
day_pred = np.arange(201, 301)

plt.plot(day_new, scaler.inverse_transform(df1[1059:]))
plt.plot(day_pred, scaler.inverse_transform(lst_output))
df3 = df1.tolist()
df3.extend(lst_output)
plt.plot(df3[1200:])
plt.show()
```

## Contributing
Contributions are welcome! If you have any ideas or suggestions, please open an issue or submit a pull request.
