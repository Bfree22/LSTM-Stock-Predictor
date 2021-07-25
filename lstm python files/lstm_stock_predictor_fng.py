#!/usr/bin/env python
# coding: utf-8

# # LSTM Stock Predictor Using Fear and Greed Index
# 
# In this notebook, you will build and train a custom LSTM RNN that uses a 10 day window of Bitcoin fear and greed index values to predict the 11th day closing price. 
# 
# You will need to:
# 
# 1. Prepare the data for training and testing
# 2. Build and train a custom LSTM RNN
# 3. Evaluate the performance of the model

# ## Data Preparation
# 
# In this section, you will need to prepare the training and testing data for the model. The model will use a rolling 10 day window to predict the 11th day closing price.
# 
# You will need to:
# 1. Use the `window_data` function to generate the X and y values for the model.
# 2. Split the data into 70% training and 30% testing
# 3. Apply the MinMaxScaler to the X and y values
# 4. Reshape the X_train and X_test data for the model. Note: The required input format for the LSTM is:
# 
# ```python
# reshape((X_train.shape[0], X_train.shape[1], 1))
# ```

# In[1]:


import numpy as np
import pandas as pd
import hvplot.pandas


# In[2]:


# Set the random seed for reproducibility
# Note: This is for the homework solution, but it is good practice to comment this out and run multiple experiments to evaluate your model
from numpy.random import seed
seed(1)
from tensorflow import random
random.set_seed(2)


# In[3]:


# Load the fear and greed sentiment data for Bitcoin
df = pd.read_csv('btc_sentiment.csv', index_col="date", infer_datetime_format=True, parse_dates=True)
df = df.drop(columns="fng_classification")
df.head()


# In[4]:


# Load the historical closing prices for Bitcoin
df2 = pd.read_csv('btc_historic.csv', index_col="Date", infer_datetime_format=True, parse_dates=True)['Close']
df2 = df2.sort_index()
df2.tail()


# In[5]:


# Join the data into a single DataFrame
df = df.join(df2, how="inner")
df.tail()


# In[6]:


df.head()


# In[7]:


# This function accepts the column number for the features (X) and the target (y)
# It chunks the data up with a rolling window of Xt-n to predict Xt
# It returns a numpy array of X any y
def window_data(df, window, feature_col_number, target_col_number):
    X = []
    y = []
    for i in range(len(df) - window - 1):
        features = df.iloc[i:(i + window), feature_col_number]
        target = df.iloc[(i + window), target_col_number]
        X.append(features)
        y.append(target)
    return np.array(X), np.array(y).reshape(-1, 1)


# In[8]:


# Predict Closing Prices using a 10 day window of previous fng values
# Then, experiment with window sizes anywhere from 1 to 10 and see how the model performance changes
window_size = 10

# Column index 0 is the 'fng_value' column
# Column index 1 is the `Close` column
feature_column = 0
target_column = 1
X, y = window_data(df, window_size, feature_column, target_column)


# In[9]:


# Use 70% of the data for training and the remaineder for testing
split = int(0.7 * len(X))
X_train = X[: split]
X_test = X[split:]
y_train = y[: split]
y_test = y[split:]


# In[10]:


from sklearn.preprocessing import MinMaxScaler
# Use the MinMaxScaler to scale data between 0 and 1.
scaler = MinMaxScaler()
scaler.fit(X)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
scaler.fit(y)
y_train = scaler.transform(y_train)
y_test = scaler.transform(y_test)


# In[12]:


# Reshape the features for the model
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))


# ---

# ## Build and Train the LSTM RNN
# 
# In this section, you will design a custom LSTM RNN and fit (train) it using the training data.
# 
# You will need to:
# 1. Define the model architecture
# 2. Compile the model
# 3. Fit the model to the training data
# 
# ### Hints:
# You will want to use the same model architecture and random seed for both notebooks. This is necessary to accurately compare the performance of the FNG model vs the closing price model. 

# In[13]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout


# In[14]:


# Build the LSTM model. 
# The return sequences need to be set to True if you are adding additional LSTM layers, but 
# You don't have to do this for the final layer. 
# Note: The dropouts help prevent overfitting
# Note: The input shape is the number of time steps and the number of indicators
# Note: Batching inputs has a different input shape of Samples/TimeSteps/Features

model = Sequential()

number_units = 30 # The number of units in each LSTM layer, is equal to the size of the time window
dropout_fraction = 0.2 # fraction of nodes that will be dropped on each epoch. randomly drop 20% of the units.

# Layer 1
model.add(LSTM(
    units=number_units,
    return_sequences=True,
    input_shape=(X_train.shape[1], 1))
    )

model.add(Dropout(dropout_fraction))
# Layer 2
model.add(LSTM(units=number_units, return_sequences=True))
model.add(Dropout(dropout_fraction))

# Layer 3
model.add(LSTM(units=number_units))
model.add(Dropout(dropout_fraction))

# Output layer
model.add(Dense(1))


# In[15]:


# Compile the model
model.compile(optimizer="adam", loss="mean_squared_error")


# In[16]:


# Summarize the model
model.summary()


# In[17]:


# Train the model
# Use at least 10 epochs
# Do not shuffle the data
# Experiement with the batch size, but a smaller batch size is recommended
model.fit(X_train, y_train, epochs=20, shuffle=False, batch_size=2, verbose=1)


# ---

# ## Model Performance
# 
# In this section, you will evaluate the model using the test data. 
# 
# You will need to:
# 1. Evaluate the model using the `X_test` and `y_test` data.
# 2. Use the X_test data to make predictions
# 3. Create a DataFrame of Real (y_test) vs predicted values. 
# 4. Plot the Real vs predicted values as a line chart
# 
# ### Hints
# Remember to apply the `inverse_transform` function to the predicted and y_test values to recover the actual closing prices.

# In[18]:


# Evaluate the model
model.evaluate(X_test, y_test)


# In[25]:


# Make some predictions
predictions = model.predict(X_test)


# In[26]:


# Recover the original prices instead of the scaled version
predicted_prices = scaler.inverse_transform(predictions)
real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))


# In[27]:


# Create a DataFrame of Real and Predicted values
stocks = pd.DataFrame({
    "Real": real_prices.ravel(),
    "Predicted": predicted_prices.ravel()
}, index = df.index[-len(real_prices): ]) 
stocks.head()


# In[28]:


# Plot the real vs predicted values as a line chart
stocks.hvplot.line(xlabel="Date", ylabel="Price")


# In[ ]:




