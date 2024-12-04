#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 21:42:21 2024

@author: saharqaadan
"""

import os
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
import statsmodels.api as sm
from statsmodels.regression.mixed_linear_model import MixedLM
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.colors as mcolors

# Define the base path to the data directory
base_path = '/Users/saharqaadan/Documents/Research2023/researchStay2024/KDD2024/Lithium-ion battery degradation dataset based on a realistic forklift operation profile'  # Update this with your actual path

# Function to load data from a cell directory
def load_cell_data(cell_path):
    all_data = []
    rounds = [f for f in os.listdir(cell_path) if os.path.isdir(os.path.join(cell_path, f))]
    for round_folder in rounds:
        round_path = os.path.join(cell_path, round_folder)
        for file_name in ['RPT.csv', 'Ageing.csv']:
            file_path = os.path.join(round_path, file_name)
            if os.path.isfile(file_path):
                data = pd.read_csv(file_path)
                data['Cell'] = os.path.basename(cell_path)
                data['Round'] = round_folder
                data['File_Type'] = file_name.split('.')[0]
                all_data.append(data)
    return pd.concat(all_data, ignore_index=True)

# Load data for each cell
cell_paths = [os.path.join(base_path, f'Cell{i}') for i in range(1, 4)]
all_data = pd.concat([load_cell_data(cell_path) for cell_path in cell_paths], ignore_index=True)

# Separate RPT and Aging data
rpt_df = all_data[all_data['File_Type'] == 'RPT']
aging_df = all_data[all_data['File_Type'] == 'Ageing']

# Convert 'Round' to a numeric value for sorting
rpt_df['Round'] = rpt_df['Round'].str.extract('(\d+)').astype(int)
aging_df['Round'] = aging_df['Round'].str.extract('(\d+)').astype(int)

# Sample 0.01 of the data for RPT and Aging data
rpt_sample = rpt_df.sample(frac=0.03, random_state=4)
aging_sample = aging_df.sample(frac=0.003, random_state=4)

import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import GRU, Dense
from keras import backend as K
from scipy.stats import weibull_min
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import tensorflow as tf

# Custom Weibull loss function using TensorFlow operations
def weibull_loss(y_true, y_pred):
    shape = 2.0  # Example Weibull shape parameter
    scale = 500.0  # Example Weibull scale parameter (cycles or time)

    # Weibull CDF using TensorFlow operations
    failure_prob = 1 - tf.exp(-((y_pred / scale) ** shape))  # Weibull CDF: 1 - exp(-(x/scale)^shape)

    # Custom loss: Mean Squared Error combined with Weibull failure probability
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    loss = mse * (1 + failure_prob)
    return loss

# Function to train and evaluate the GRU-DINN model for a given dataset (RPT or Aging)
def train_and_evaluate(data_sample, dataset_name):
    # Preprocess the data (scale the features)
    scaler = StandardScaler()
    features = ['Current', 'Voltage', 'Energy', 'Temperature']  # Use the available features
    X_scaled = scaler.fit_transform(data_sample[features])
    y = data_sample['Voltage'].values  # Convert to numpy array for easier slicing
    
    # Reshape the data for time series (timesteps, features)
    timesteps = 10  # You can adjust the window size
    X = []
    y_final = []
    
    # Adjust the loop to properly slice both X and y
    for i in range(timesteps, len(X_scaled)):
        X.append(X_scaled[i-timesteps:i])
        y_final.append(y[i])  # Match the correct index in y to the time-series slice in X
    
    X = np.array(X)
    y_final = np.array(y_final)
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_final, test_size=0.2, random_state=42)
    
    # Build the GRU-based DINN model
    model = Sequential()
    model.add(GRU(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(GRU(units=32))
    model.add(Dense(1))  # Output layer predicting voltage
    
    # Compile the model with the Weibull-based custom loss function
    model.compile(optimizer='adam', loss=weibull_loss)
    
    # Train the model
    history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)
    
    # Evaluate the model on test data
    y_pred = model.predict(X_test)
    
    # Plot the predicted vs actual values
    plt.figure(figsize=(10, 6))
    plt.plot(y_test, label='True Values')
    plt.plot(y_pred, label='Predicted Values')
    plt.title(f"Predictions for {dataset_name} Data using Weibull-informed DINN")
    plt.legend()
    plt.show()

# Predict on RPT Sample
print("Training on RPT Data Sample...")
train_and_evaluate(rpt_sample, "RPT")


# Predict on Aging Sample
print("Training on Aging Data Sample...")
train_and_evaluate(aging_sample, "Aging")

# Function to sample different fractions of data and train the model on each sample
def train_on_multiple_samples(data, dataset_name, n_samples=3, frac=0.01):
    for i in range(n_samples):
        print(f"Training on {dataset_name} Sample {i+1}...")
        data_sample = data.sample(frac=frac, random_state=i * 42)  # Sample with different random seeds
        train_and_evaluate(data_sample, f"{dataset_name} Sample {i+1}")

# Train on multiple 0.01 samples for RPT and Aging data
train_on_multiple_samples(rpt_df, "RPT", n_samples=3, frac=0.01)
train_on_multiple_samples(aging_df, "Aging", n_samples=3, frac=0.01)
##########################COMPARISON: GRU, GRU_DINN
import numpy as np
import matplotlib.pyplot as plt
from keras.regularizers import l2
from keras.layers import Dropout
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import GRU, Dense

# Weibull loss function
def weibull_loss(y_true, y_pred):
    shape = 2.0
    scale = 500.0
    failure_prob = 1 - tf.exp(-((y_pred / scale) ** shape))
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    loss = mse * (1 + failure_prob)
    return loss

# Add regularization to the GRU-based DINN model
def train_gru_dinn_regularized(data_sample, timesteps=10, epochs=5):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data_sample[['Current', 'Voltage', 'Energy', 'Temperature']])
    y = data_sample['Voltage'].values

    X, y_final = [], []
    for i in range(timesteps, len(X_scaled)):
        X.append(X_scaled[i-timesteps:i])
        y_final.append(y[i])

    X = np.array(X)
    y_final = np.array(y_final)

    X_train, X_test, y_train, y_test = train_test_split(X, y_final, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(GRU(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]), kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))  # Dropout regularization
    model.add(GRU(units=32, kernel_regularizer=l2(0.001)))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss=weibull_loss)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.2, verbose=0)
    
    return history.history['loss'], history.history['val_loss']

# Regularized DINN model
def train_dinn_regularized(data_sample, epochs=5):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data_sample[['Current', 'Voltage', 'Energy', 'Temperature']])
    y = data_sample['Voltage'].values

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.5))  # Adding Dropout regularization
    model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(1))  # Output layer predicting voltage

    model.compile(optimizer='adam', loss=weibull_loss)
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.2, verbose=0)

    return history.history['loss'], history.history['val_loss']

# GRU model (without Weibull or DINN)
def train_gru(data_sample, timesteps=10, epochs=5):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data_sample[['Current', 'Voltage', 'Energy', 'Temperature']])
    y = data_sample['Voltage'].values

    X, y_final = [], []
    for i in range(timesteps, len(X_scaled)):
        X.append(X_scaled[i-timesteps:i])
        y_final.append(y[i])

    X = np.array(X)
    y_final = np.array(y_final)

    X_train, X_test, y_train, y_test = train_test_split(X, y_final, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(GRU(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(GRU(units=32))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_split=0.2, verbose=0)

    return history.history['loss'], history.history['val_loss']

# Function to average results over 10 samples and compute standard deviation
def average_over_samples(data, train_function, epochs=5, n_samples=10):
    train_losses, val_losses = [], []
    
    for i in range(n_samples):
        data_sample = data.sample(frac=0.01, random_state=i*42)
        train_loss, val_loss = train_function(data_sample, epochs=epochs)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
    
    avg_train_loss = np.mean(train_losses, axis=0)
    avg_val_loss = np.mean(val_losses, axis=0)
    
    std_train_loss = np.std(train_losses, axis=0)
    std_val_loss = np.std(val_losses, axis=0)
    
    return avg_train_loss, avg_val_loss, std_train_loss, std_val_loss

# Train on multiple samples for each model and average
gru_loss, gru_val_loss, gru_std_loss, gru_val_std = average_over_samples(rpt_sample, train_gru)
gru_dinn_loss, gru_dinn_val_loss, gru_dinn_std_loss, gru_dinn_val_std = average_over_samples(rpt_sample, train_gru_dinn_regularized)
dinn_loss, dinn_val_loss, dinn_std_loss, dinn_val_std = average_over_samples(rpt_sample, train_dinn_regularized)

# Plot comparison with uncertainty (STD as shaded area)
epochs = list(range(1, 6))  # For 5 epochs
colors = ['#9370DB', '#32CD32', '#1E90FF']

plt.figure(figsize=(10, 6))

# GRU alone with uncertainty
plt.plot(epochs, gru_loss, label='GRU Training Loss', color=colors[0])
plt.fill_between(epochs, gru_loss - gru_std_loss, gru_loss + gru_std_loss, color=colors[0], alpha=0.3)
plt.plot(epochs, gru_val_loss, label='GRU Validation Loss', color=colors[0], linestyle='dashed')
plt.fill_between(epochs, gru_val_loss - gru_val_std, gru_val_loss + gru_val_std, color=colors[0], alpha=0.3)

# GRU-DINN with uncertainty
plt.plot(epochs, gru_dinn_loss, label='GRU-DINN Training Loss', color=colors[1])
plt.fill_between(epochs, gru_dinn_loss - gru_dinn_std_loss, gru_dinn_loss + gru_dinn_std_loss, color=colors[1], alpha=0.3)
plt.plot(epochs, gru_dinn_val_loss, label='GRU-DINN Validation Loss', color=colors[1], linestyle='dashed')
plt.fill_between(epochs, gru_dinn_val_loss - gru_dinn_val_std, gru_dinn_val_loss + gru_dinn_val_std, color=colors[1], alpha=0.3)

# DINN alone with uncertainty
plt.plot(epochs, dinn_loss, label='DINN Training Loss', color=colors[2])
plt.fill_between(epochs, dinn_loss - dinn_std_loss, dinn_loss + dinn_std_loss, color=colors[2], alpha=0.3)
plt.plot(epochs, dinn_val_loss, label='DINN Validation Loss', color=colors[2], linestyle='dashed')
plt.fill_between(epochs, dinn_val_loss - dinn_val_std, dinn_val_loss + dinn_val_std, color=colors[2], alpha=0.3)

# Plot details
plt.title("Comparison of GRU, Regularized GRU-DINN, and Regularized DINN with Uncertainty")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Print the results
print("Regularized GRU-DINN Results (Average Training and Validation Loss over 10 samples):")
print(f"Training Loss: {gru_dinn_loss}")
print(f"Validation Loss: {gru_dinn_val_loss}\n")

print("Regularized DINN Results (Average Training and Validation Loss over 10 samples):")
print(f"Training Loss: {dinn_loss}")
print(f"Validation Loss: {dinn_val_loss}\n")

print("GRU Results (Average Training and Validation Loss over 10 samples):")
print(f"Training Loss: {gru_loss}")
print(f"Validation Loss: {gru_val_loss}\n")
#############################################################
#############################
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout, BatchNormalization, Activation
from keras.regularizers import l2
from keras import backend as K
from scipy.stats import weibull_min
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf

# Define epochs as a global variable for easy adjustment
EPOCHS = 1  # You can change this value to adjust all models

# Weibull loss function using TensorFlow operations
def weibull_loss(y_true, y_pred):
    shape = 2.0  # Weibull shape parameter
    scale = 500.0  # Weibull scale parameter (cycles or time)
    failure_prob = 1 - tf.exp(-((y_pred / scale) ** shape))
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    loss = mse * (1 + failure_prob)
    return loss

# Function to train and evaluate the GRU-DINN model with regularization and batch normalization
def train_gru_dinn_regularized(data_sample, timesteps=10, epochs=EPOCHS):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data_sample[['Current', 'Voltage', 'Energy', 'Temperature']])
    y = data_sample['Voltage'].values

    # Create time-series data
    X, y_final = [], []
    for i in range(timesteps, len(X_scaled)):
        X.append(X_scaled[i-timesteps:i])
        y_final.append(y[i])
    
    X = np.array(X)
    y_final = np.array(y_final)

    # Train-test split (splitting into train, validation, and test sets)
    X_train, X_test, y_train, y_test = train_test_split(X, y_final, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2

    # GRU-DINN model with regularization and batch normalization
    model = Sequential()
    model.add(GRU(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]), kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.6))
    model.add(GRU(units=32, kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.6))
    model.add(Dense(16, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(1))

    # Compile the model with Weibull loss
    model.compile(optimizer='adam', loss=weibull_loss)

    # Train the model, using validation set for monitoring
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_val, y_val), verbose=0)

    # Evaluate the model on the test set
    test_loss = model.evaluate(X_test, y_test, verbose=0)

    return history.history['val_loss'], test_loss


# Fully connected network with regularization and batch normalization
def train_dinn_regularized(data_sample, epochs=EPOCHS):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data_sample[['Current', 'Voltage', 'Energy', 'Temperature']])
    y = data_sample['Voltage'].values

    # Train-test split (splitting into train, validation, and test sets)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)  # 0.25 * 0.8 = 0.2

    # Fully connected network with regularization and batch normalization
    model = Sequential()
    model.add(Dense(32, input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.6))

    model.add(Dense(16, kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.6))

    model.add(Dense(1))

    model.compile(optimizer='adam', loss=weibull_loss)

    # Train the model, using validation set for monitoring
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_val, y_val), verbose=0)

    # Evaluate the model on the test set
    test_loss = model.evaluate(X_test, y_test, verbose=0)

    return history.history['val_loss'], test_loss


# Regularized GRU model (with dropout and L2 regularization added)
def train_gru_regularized(data_sample, timesteps=1, epochs=EPOCHS):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data_sample[['Current', 'Voltage', 'Energy', 'Temperature']])
    y = data_sample['Voltage'].values

    # Create time-series data
    X, y_final = [], []
    for i in range(timesteps, len(X_scaled)):
        X.append(X_scaled[i-timesteps:i])
        y_final.append(y[i])
    
    X = np.array(X)
    y_final = np.array(y_final)

    # Train-test split (splitting into train, validation, and test sets)
    X_train, X_test, y_train, y_test = train_test_split(X, y_final, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    # GRU model with regularization (Dropout and L2 regularization)
    model = Sequential()
    model.add(GRU(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]), kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.6))
    model.add(GRU(units=32, kernel_regularizer=l2(0.001)))
    model.add(Dropout(0.6))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')

    # Train the model, using validation set for monitoring
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_val, y_val), verbose=0)

    # Evaluate the model on the test set
    test_loss = model.evaluate(X_test, y_test, verbose=0)

    return history.history['val_loss'], test_loss


# Function to average results over n samples and compute standard deviation
def average_over_samples(data, train_function, epochs=EPOCHS, n_samples=2):
    val_losses, test_losses = [], []
    
    for i in range(n_samples):
        data_sample = data.sample(frac=1.0, random_state=i*42)
        val_loss, test_loss = train_function(data_sample, epochs=epochs)
        val_losses.append(val_loss)
        test_losses.append(test_loss)
    
    avg_val_loss = np.mean(val_losses, axis=0)
    avg_test_loss = np.mean(test_losses, axis=0)
    
    std_val_loss = np.std(val_losses, axis=0)
    std_test_loss = np.std(test_losses, axis=0)
    
    return avg_val_loss, avg_test_loss, std_val_loss, std_test_loss


# Train on 10 samples for each model and average
gru_val_loss, gru_test_loss, gru_val_std, gru_test_std = average_over_samples(rpt_sample, train_gru)
gru_dinn_val_loss, gru_dinn_test_loss, gru_dinn_val_std, gru_dinn_test_std = average_over_samples(rpt_sample, train_gru_dinn_regularized)
dinn_val_loss, dinn_test_loss, dinn_val_std, dinn_test_std = average_over_samples(rpt_sample, train_dinn_regularized)

# Plot comparison with uncertainty (STD as shaded area)
epochs = list(range(1, EPOCHS + 1))  # For the number of epochs set
colors = ['#9370DB', '#32CD32', '#1E90FF']

plt.figure(figsize=(10, 6))

# GRU with uncertainty
#plt.plot(epochs, gru_val_loss, label='GRU Validation Loss', color=colors[0])
plt.fill_between(epochs, gru_val_loss - gru_val_std, gru_val_loss + gru_val_std, color=colors[0], alpha=0.3)
plt.plot(epochs, [gru_test_loss] * len(epochs), label='GRU Test Loss', color=colors[0], linestyle='dashed')

# GRU-DINN with uncertainty
#plt.plot(epochs, gru_dinn_val_loss, label='GRU-DINN Validation Loss', color=colors[1])
plt.fill_between(epochs, gru_dinn_val_loss - gru_dinn_val_std, gru_dinn_val_loss + gru_dinn_val_std, color=colors[1], alpha=0.3)
plt.plot(epochs, [gru_dinn_test_loss] * len(epochs), label='GRU-DINN Test Loss', color=colors[1], linestyle='dashed')

# DINN with uncertainty
#plt.plot(epochs, dinn_val_loss, label='DINN Validation Loss', color=colors[2])
plt.fill_between(epochs, dinn_val_loss - dinn_val_std, dinn_val_loss + dinn_val_std, color=colors[2], alpha=0.3)
plt.plot(epochs, [dinn_test_loss] * len(epochs), label='DINN Test Loss', color=colors[2], linestyle='dashed')

plt.yscale('log')
# Plot details
plt.title("Comparison of Validation and Test Loss for GRU, GRU-DINN, and DINN with Uncertainty")
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

# Print the results
print("Regularized GRU-DINN Results:")
print(f"Validation Loss: {gru_dinn_val_loss}")
print(f"Test Loss: {gru_dinn_test_loss}\n")

print("Regularized DINN Results:")
print(f"Validation Loss: {dinn_val_loss}")
print(f"Test Loss: {dinn_test_loss}\n")

print("GRU Results:")
print(f"Validation Loss: {gru_val_loss}")
print(f"Test Loss: {gru_test_loss}\n")

##############
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout, BatchNormalization, Activation, LSTM
from keras.regularizers import l2
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from scipy.stats import weibull_min
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

# Define epochs and samples as global variables for easy adjustment
EPOCHS = 1  # Change this value to adjust the number of epochs for all models
N_SAMPLES = 1  # Number of samples for averaging

# Weibull loss function using TensorFlow operations
def weibull_loss(y_true, y_pred):
    shape = 2.0  # Weibull shape parameter
    scale = 500.0  # Weibull scale parameter (cycles or time)
    failure_prob = 1 - tf.exp(-((y_pred / scale) ** shape))
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    loss = mse * (1 + failure_prob)
    return loss

# GRU-DINN Model with regularization and batch normalization
def train_gru_dinn_regularized(data_sample, timesteps=1, epochs=EPOCHS):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data_sample[['Current', 'Voltage', 'Energy', 'Temperature']])
    y = data_sample['Voltage'].values

    # Create time-series data
    X, y_final = [], []
    for i in range(timesteps, len(X_scaled)):
        X.append(X_scaled[i-timesteps:i])
        y_final.append(y[i])

    X = np.array(X)
    y_final = np.array(y_final)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_final, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    # GRU-DINN model with regularization and batch normalization
    model = Sequential()
    model.add(GRU(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]), kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.6))
    model.add(GRU(units=32, kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.6))
    model.add(Dense(16, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(1))

    # Compile the model with Weibull loss
    model.compile(optimizer='adam', loss=weibull_loss)

    # Train the model, using validation set for monitoring
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_val, y_val), verbose=0)

    # Evaluate the model on the test set
    test_loss = model.evaluate(X_test, y_test, verbose=0)

    return history.history['val_loss'], test_loss

# Fully connected DINN Model with regularization and batch normalization
def train_dinn_regularized(data_sample, epochs=EPOCHS):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data_sample[['Current', 'Voltage', 'Energy', 'Temperature']])
    y = data_sample['Voltage'].values

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    # Fully connected DINN model with regularization and batch normalization
    model = Sequential()
    model.add(Dense(32, input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.6))

    model.add(Dense(16, kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.6))

    model.add(Dense(1))

    model.compile(optimizer='adam', loss=weibull_loss)

    # Train the model, using validation set for monitoring
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_val, y_val), verbose=0)

    # Evaluate the model on the test set
    test_loss = model.evaluate(X_test, y_test, verbose=0)

    return history.history['val_loss'], test_loss

# LSTM model
def train_lstm(data_sample, timesteps=10, epochs=EPOCHS):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data_sample[['Current', 'Voltage', 'Energy', 'Temperature']])
    y = data_sample['Voltage'].values

    X, y_final = [], []
    for i in range(timesteps, len(X_scaled)):
        X.append(X_scaled[i-timesteps:i])
        y_final.append(y[i])

    X = np.array(X)
    y_final = np.array(y_final)

    X_train, X_test, y_train, y_test = train_test_split(X, y_final, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    model = Sequential()
    model.add(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(units=32))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_val, y_val), verbose=0)
    test_loss = model.evaluate(X_test, y_test, verbose=0)

    return history.history['val_loss'], test_loss

# Simple ANN
def train_ann(data_sample, epochs=EPOCHS):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data_sample[['Current', 'Voltage', 'Energy', 'Temperature']])
    y = data_sample['Voltage'].values

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    model = Sequential()
    model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],), kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.6))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_val, y_val), verbose=0)
    test_loss = model.evaluate(X_test, y_test, verbose=0)

    return history.history['val_loss'], test_loss

# Random Forest with 10 estimators
def train_rf(data_sample):
    X = data_sample[['Current', 'Voltage', 'Energy', 'Temperature']]
    y = data_sample['Voltage'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=10, random_state=42)
    rf.fit(X_train, y_train)
    val_loss = np.mean((rf.predict(X_train) - y_train) ** 2)  # RF doesn't support val, so use train loss here
    test_loss = np.mean((rf.predict(X_test) - y_test) ** 2)

    return val_loss, test_loss

# Linear Regression
def train_lr(data_sample):
    X = data_sample[['Current', 'Voltage', 'Energy', 'Temperature']]
    y = data_sample['Voltage'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    val_loss = np.mean((lr.predict(X_train) - y_train) ** 2)
    test_loss = np.mean((lr.predict(X_test) - y_test) ** 2)

    return val_loss, test_loss

# Physics-Informed Neural Network (PINN)
def train_pinn(data_sample, epochs=EPOCHS):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data_sample[['Current', 'Voltage', 'Energy', 'Temperature']])
    y = data_sample['Voltage'].values

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    model = Sequential()
    model.add(Dense(64, input_shape=(X_train.shape[1],), activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))

    # Example custom physics-informed loss function
    def custom_pinn_loss(y_true, y_pred):
        physics_loss = tf.reduce_mean(tf.square(y_true - y_pred))  # Example constraint
        return physics_loss

    model.compile(optimizer='adam', loss=custom_pinn_loss)

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_val, y_val), verbose=0)
    test_loss = model.evaluate(X_test, y_test, verbose=0)

    return history.history['val_loss'], test_loss




# Function to average results over n samples and compute standard deviation
def average_over_samples(data, train_function, epochs=EPOCHS, n_samples=N_SAMPLES):
    val_losses, test_losses = [], []
    
    for i in range(n_samples):
        data_sample = data.sample(frac=1.0, random_state=i*42)
        val_loss, test_loss = train_function(data_sample, epochs=epochs)
        val_losses.append(val_loss)
        test_losses.append(test_loss)
    
    avg_val_loss = np.mean(val_losses, axis=0)
    avg_test_loss = np.mean(test_losses)

    return avg_val_loss, avg_test_loss

gru_dinn_val_loss, gru_dinn_test_loss = average_over_samples(rpt_sample, train_gru_dinn_regularized)
dinn_val_loss, dinn_test_loss = average_over_samples(rpt_sample, train_dinn_regularized)
lstm_val_loss, lstm_test_loss = average_over_samples(rpt_sample, train_lstm)
ann_val_loss, ann_test_loss = average_over_samples(rpt_sample, train_ann)
gru_val_loss, gru_test_loss = average_over_samples(rpt_sample, train_gru_regularized)


# Plot comparison with uncertainty (STD as shaded area)
epochs = list(range(1, EPOCHS + 1))
colors = ['#9370DB', '#32CD32', '#1E90FF', '#FF6347', '#FFD700', '#6A5ACD']

plt.figure(figsize=(12, 8))

# GRU-DINN Results
#plt.plot(epochs, gru_dinn_val_loss, label='GRU-DINN Validation Loss', color=colors[0])
plt.plot(epochs, [gru_dinn_test_loss] * len(epochs), label='GRU-DINN Test Loss', color=colors[0], linestyle='dashed')

# DINN Results
#plt.plot(epochs, dinn_val_loss, label='DINN Validation Loss', color=colors[1])
plt.plot(epochs, [dinn_test_loss] * len(epochs), label='DINN Test Loss', color=colors[1], linestyle='dashed')

# LSTM Results
#plt.plot(epochs, lstm_val_loss, label='LSTM Validation Loss', color=colors[2])
plt.plot(epochs, [lstm_test_loss] * len(epochs), label='LSTM Test Loss', color=colors[2], linestyle='dashed')

# ANN Results
#plt.plot(epochs, ann_val_loss, label='ANN Validation Loss', color=colors[3])
plt.plot(epochs, [ann_test_loss] * len(epochs), label='ANN Test Loss', color=colors[3], linestyle='dashed')

# Random Forest Results
#rf_val_loss, rf_test_loss = average_over_samples(rpt_sample, train_rf)
#plt.plot(epochs, [rf_val_loss] * len(epochs), label='RF Validation Loss', color=colors[4])
#plt.plot(epochs, [rf_test_loss] * len(epochs), label='RF Test Loss', color=colors[4], linestyle='dashed')

# Linear Regression Results
#lr_val_loss, lr_test_loss = average_over_samples(rpt_sample, train_lr)
#plt.plot(epochs, [lr_val_loss] * len(epochs), label='LR Validation Loss', color=colors[5])
#plt.plot(epochs, [lr_test_loss] * len(epochs), label='LR Test Loss', color=colors[5], linestyle='dashed')

# Set the y-axis to log scale
plt.yscale('log')

# Plot details
plt.title("Comparison of Validation and Test Loss for Multiple Models with Uncertainty (Log Scale)")
plt.xlabel('Epochs')
plt.ylabel('Loss (Log Scale)')
plt.legend()
plt.grid(True)
plt.show()

# Print results for key models
print(f"GRU-DINN Validation Loss: {gru_dinn_val_loss}, Test Loss: {gru_dinn_test_loss}")
print(f"GRU Validation Loss: {gru_val_loss}, Test Loss: {gru_test_loss}")
print(f"DINN Validation Loss: {dinn_val_loss}, Test Loss: {dinn_test_loss}")
print(f"LSTM Validation Loss: {lstm_val_loss}, Test Loss: {lstm_test_loss}")
print(f"ANN Validation Loss: {ann_val_loss}, Test Loss: {ann_test_loss}")
#print(f"RF Validation Loss: {rf_val_loss}, Test Loss: {rf_test_loss}")
#print(f"LR Validation Loss: {lr_val_loss}, Test Loss: {lr_test_loss}")


###################
import os
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for seaborn
sns.set(style='whitegrid')

# Define the base path to the data directory
base_path = '/Users/saharqaadan/Documents/Research2023/researchStay2024/KDD2024/Lithium-ion battery degradation dataset based on a realistic forklift operation profile'  # Update this with your actual path

# Function to load data from a cell directory
def load_cell_data(cell_path):
    all_data = []
    rounds = [f for f in os.listdir(cell_path) if os.path.isdir(os.path.join(cell_path, f))]
    for round_folder in rounds:
        round_path = os.path.join(cell_path, round_folder)
        for file_name in ['RPT.csv', 'Ageing.csv']:
            file_path = os.path.join(round_path, file_name)
            if os.path.isfile(file_path):
                data = pd.read_csv(file_path)
                data['Cell'] = os.path.basename(cell_path)
                data['Round'] = round_folder
                data['File_Type'] = file_name.split('.')[0]
                all_data.append(data)
    return pd.concat(all_data, ignore_index=True)

# Load data for each cell
cell_paths = [os.path.join(base_path, f'Cell{i}') for i in range(1, 4)]
all_data = pd.concat([load_cell_data(cell_path) for cell_path in cell_paths], ignore_index=True)

# Separate RPT and Aging data
rpt_df = all_data[all_data['File_Type'] == 'RPT']
aging_df = all_data[all_data['File_Type'] == 'Ageing']

# Convert 'Round' to a numeric value for sorting
rpt_df['Round'] = rpt_df['Round'].str.extract('(\d+)').astype(int)
aging_df['Round'] = aging_df['Round'].str.extract('(\d+)').astype(int)

# Sample 0.1 of the data for RPT and Aging data
rpt_sample = rpt_df.sample(frac=0.1, random_state=4)
aging_sample = aging_df.sample(frac=0.01, random_state=4)

# Function to perform t-SNE and visualize feature contributions
def tsne_visualization(data, title):
    # Selecting features for t-SNE
    features = ['Current', 'Voltage', 'Energy', 'Temperature']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[features])
    
    # Perform t-SNE for dimensionality reduction
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=300)
    tsne_results = tsne.fit_transform(scaled_features)
    
    # Create a DataFrame with t-SNE results
    tsne_df = pd.DataFrame(tsne_results, columns=['TSNE1', 'TSNE2'])
    tsne_df['Cell'] = data['Cell']
    tsne_df['Round'] = data['Round']
    
    # Plotting t-SNE results
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='TSNE1', y='TSNE2', hue='Cell', data=tsne_df, palette='tab10', alpha=0.7)
    plt.title(f't-SNE Visualization for {title}')
    plt.show()

# Function to perform clustering and generate summary tables
def clustering_analysis(data, dataset_name):
    # Selecting features for clustering
    features = ['Current', 'Voltage', 'Energy', 'Temperature']
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[features])
    
    # Apply Gaussian Mixture Model for clustering
    gmm = GaussianMixture(n_components=3, random_state=42)
    clusters = gmm.fit_predict(scaled_features)
    
    # Add cluster information to the DataFrame
    data['Cluster'] = clusters
    
    # Generate summary table for each cluster
    summary = data.groupby('Cluster')[features].mean()
    summary['Count'] = data['Cluster'].value_counts()
    
    # Display the summary table using the print function
    print(f'Clustering Summary for {dataset_name}')
    print(summary)
    
    return summary

# Visualize t-SNE for each cell in RPT and Aging datasets
tsne_visualization(rpt_sample, 'RPT Data')
tsne_visualization(aging_sample, 'Aging Data')

# Perform clustering analysis and create tables for RPT and Aging datasets
rpt_clustering_summary = clustering_analysis(rpt_sample, 'RPT Data')
aging_clustering_summary = clustering_analysis(aging_sample, 'Aging Data')

# Save clustering summaries to CSV files
rpt_clustering_summary.to_csv('RPT_Clustering_Summary.csv', index=False)
aging_clustering_summary.to_csv('Aging_Clustering_Summary.csv', index=False)

print("t-SNE visualization and clustering analysis completed.")
