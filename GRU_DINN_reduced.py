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
rpt_sample = rpt_df.sample(frac=0.1, random_state=4)
aging_sample = aging_df.sample(frac=0.1, random_state=4)
#############################


#############################

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Activation
from keras.regularizers import l2
import tensorflow as tf
import matplotlib.pyplot as plt

# Define the base path to the data directory
base_path = '/Users/saharqaadan/Documents/Research2023/researchStay2024/KDD2024/Lithium-ion battery degradation dataset based on a realistic forklift operation profile'

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

# Sample 10% of the data for RPT and Aging data
rpt_sample = rpt_df.sample(frac=0.001, random_state=4)
aging_sample = aging_df.sample(frac=0.0001, random_state=4)

# DINN model with regularization and batch normalization
def weibull_loss(y_true, y_pred):
    shape = 2.0  # Weibull shape parameter
    scale = 500.0  # Weibull scale parameter (cycles or time)
    failure_prob = 1 - tf.exp(-((y_pred / scale) ** shape))
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    loss = mse * (1 + failure_prob)
    return loss

def train_dinn_regularized(data_sample, epochs=10):
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

    # Train the model
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_val, y_val), verbose=0)

    # Evaluate the model on the test set
    test_loss = model.evaluate(X_test, y_test, verbose=0)

    return history.history['val_loss'], test_loss

# Cross-validation function for DINN
def cross_validate_model(data_sample, model_train_function, k=5, epochs=10):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data_sample[['Current', 'Voltage', 'Energy', 'Temperature']])
    y = data_sample['Voltage'].values

    kf = KFold(n_splits=k)
    val_losses, test_losses = [], []

    for train_index, test_index in kf.split(X_scaled):
        X_train, X_test = X_scaled[train_index], X_scaled[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Train the model
        val_loss, test_loss = model_train_function(data_sample, epochs=epochs)
        val_losses.append(val_loss)
        test_losses.append(test_loss)

    # Compute average validation and test losses
    avg_val_loss = np.mean(val_losses, axis=0)
    avg_test_loss = np.mean(test_losses)
    return avg_val_loss, avg_test_loss

# Run cross-validation for DINN on the RPT sample
val_loss_dinn, test_loss_dinn = cross_validate_model(rpt_sample, train_dinn_regularized)
print(f"DINN Cross-Validation Results (RPT): Validation Loss = {val_loss_dinn}, Test Loss = {test_loss_dinn}")

# Run cross-validation for DINN on the Aging sample
val_loss_dinn_aging, test_loss_dinn_aging = cross_validate_model(aging_sample, train_dinn_regularized)
print(f"DINN Cross-Validation Results (Aging): Validation Loss = {val_loss_dinn_aging}, Test Loss = {test_loss_dinn_aging}")



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
N_SAMPLES = 2  # Number of samples for averaging

# Weibull loss function using TensorFlow operations
def weibull_loss(y_true, y_pred):
    shape = 2.0  # Weibull shape parameter
    scale = 500.0  # Weibull scale parameter (cycles or time)
    failure_prob = 1 - tf.exp(-((y_pred / scale) ** shape))
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    loss = mse * (1 + failure_prob)
    return loss

# GRU Model with regularization and batch normalization
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

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y_final, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    # GRU model with regularization (Dropout and L2 regularization)
    model = Sequential()
    model.add(GRU(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]), kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.6))
    model.add(GRU(units=32, kernel_regularizer=l2(0.001)))
    model.add(BatchNormalization())
    model.add(Dropout(0.6))
    model.add(Dense(16, activation='relu', kernel_regularizer=l2(0.001)))
    model.add(Dense(1))

    # Compile the model with Mean Squared Error loss
    model.compile(optimizer='adam', loss='mse')

    # Train the model, using validation set for monitoring
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_val, y_val), verbose=0)

    # Evaluate the model on the test set
    test_loss = model.evaluate(X_test, y_test, verbose=0)

    return history.history['val_loss'], test_loss


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
        val_losses.append(val_loss[-1])  # Only consider the last validation loss per sample
        test_losses.append(test_loss)
    
    # Compute mean and standard deviation for validation and test losses
    avg_val_loss = np.mean(val_losses)
    std_val_loss = np.std(val_losses)
    avg_test_loss = np.mean(test_losses)
    std_test_loss = np.std(test_losses)

    return avg_val_loss, std_val_loss, avg_test_loss, std_test_loss

# Train and evaluate each model on the RPT dataset
print("Results on RPT Dataset:")
gru_dinn_val_loss_rpt, gru_dinn_val_std_rpt, gru_dinn_test_loss_rpt, gru_dinn_test_std_rpt = average_over_samples(rpt_sample, train_gru_dinn_regularized)
dinn_val_loss_rpt, dinn_val_std_rpt, dinn_test_loss_rpt, dinn_test_std_rpt = average_over_samples(rpt_sample, train_dinn_regularized)
lstm_val_loss_rpt, lstm_val_std_rpt, lstm_test_loss_rpt, lstm_test_std_rpt = average_over_samples(rpt_sample, train_lstm)
ann_val_loss_rpt, ann_val_std_rpt, ann_test_loss_rpt, ann_test_std_rpt = average_over_samples(rpt_sample, train_ann)
gru_val_loss_rpt, gru_val_std_rpt, gru_test_loss_rpt, gru_test_std_rpt = average_over_samples(rpt_sample, train_gru_regularized)

# Print out the RPT dataset results with standard deviation
print(f"GRU-DINN Validation Loss (RPT): {gru_dinn_val_loss_rpt} ± {gru_dinn_val_std_rpt}, Test Loss (RPT): {gru_dinn_test_loss_rpt} ± {gru_dinn_test_std_rpt}")
print(f"DINN Validation Loss (RPT): {dinn_val_loss_rpt} ± {dinn_val_std_rpt}, Test Loss (RPT): {dinn_test_loss_rpt} ± {dinn_test_std_rpt}")
print(f"LSTM Validation Loss (RPT): {lstm_val_loss_rpt} ± {lstm_val_std_rpt}, Test Loss (RPT): {lstm_test_loss_rpt} ± {lstm_test_std_rpt}")
print(f"ANN Validation Loss (RPT): {ann_val_loss_rpt} ± {ann_val_std_rpt}, Test Loss (RPT): {ann_test_loss_rpt} ± {ann_test_std_rpt}")
print(f"GRU Validation Loss (RPT): {gru_val_loss_rpt} ± {gru_val_std_rpt}, Test Loss (RPT): {gru_test_loss_rpt} ± {gru_test_std_rpt}")

# Train and evaluate each model on the Aging dataset
print("\nResults on Aging Dataset:")
gru_dinn_val_loss_aging, gru_dinn_val_std_aging, gru_dinn_test_loss_aging, gru_dinn_test_std_aging = average_over_samples(aging_sample, train_gru_dinn_regularized)
dinn_val_loss_aging, dinn_val_std_aging, dinn_test_loss_aging, dinn_test_std_aging = average_over_samples(aging_sample, train_dinn_regularized)

lstm_val_loss_aging, lstm_val_std_aging, lstm_test_loss_aging, lstm_test_std_aging = average_over_samples(aging_sample, train_lstm)
ann_val_loss_aging, ann_val_std_aging, ann_test_loss_aging, ann_test_std_aging = average_over_samples(aging_sample, train_ann)
gru_val_loss_aging, gru_val_std_aging, gru_test_loss_aging, gru_test_std_aging = average_over_samples(aging_sample, train_gru_regularized)

# Print out the Aging dataset results with standard deviation
print(f"GRU-DINN Validation Loss (Aging): {gru_dinn_val_loss_aging} ± {gru_dinn_val_std_aging}, Test Loss (Aging): {gru_dinn_test_loss_aging} ± {gru_dinn_test_std_aging}")
print(f"DINN Validation Loss (Aging): {dinn_val_loss_aging} ± {dinn_val_std_aging}, Test Loss (Aging): {dinn_test_loss_aging} ± {dinn_test_std_aging}")
print(f"LSTM Validation Loss (Aging): {lstm_val_loss_aging} ± {lstm_val_std_aging}, Test Loss (Aging): {lstm_test_loss_aging} ± {lstm_test_std_aging}")
print(f"ANN Validation Loss (Aging): {ann_val_loss_aging} ± {ann_val_std_aging}, Test Loss (Aging): {ann_test_loss_aging} ± {ann_test_std_aging}")
print(f"GRU Validation Loss (Aging): {gru_val_loss_aging} ± {gru_val_std_aging}, Test Loss (Aging): {gru_test_loss_aging} ± {gru_test_std_aging}")


# Plot comparison with uncertainty (STD as shaded area)
epochs = list(range(1, EPOCHS + 1))
colors = ['#9370DB', '#32CD32', '#1E90FF', '#FF6347', '#FFD700', '#6A5ACD']
###########################################################################################
#plotting:
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# Set custom colors for the plot
colors = ['#9370DB', '#32CD32', '#1E90FF']

# 3D Plot: Voltage, Current, and Temperature with Color-Coded Cells
# Adjust the data to sample only 1%
rpt_sample_reduced = rpt_sample.sample(frac=0.0001, random_state=42)

# 3D Plot: Voltage, Current, and Temperature with reduced data
def plot_3d(data):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Use different colors for each cell
    cell_colors = {'Cell1': colors[0], 'Cell2': colors[1], 'Cell3': colors[2]}
    
    for cell in data['Cell'].unique():
        subset = data[data['Cell'] == cell]
        ax.scatter(subset['Voltage'], subset['Current'], subset['Temperature'],
                   color=cell_colors[cell], label=cell, s=50)

    # Labeling the axes
    ax.set_xlabel('Voltage (V)')
    ax.set_ylabel('Current (A)')
    ax.set_zlabel('Temperature (°C)')
    ax.set_title('3D Scatter Plot: Voltage, Current, and Temperature by Cell (1% Sample)')

    # Legend
    ax.legend(title='Cell', loc='upper right')

    plt.show()

# Apply the 3D plot to the reduced data sample
plot_3d(rpt_sample_reduced)

import matplotlib.pyplot as plt
import seaborn as sns

# Custom Colors
colors = ['#9370DB', '#32CD32', '#1E90FF']

# Function to plot time series for a specific dataset and cell with custom colors
def plot_time_series(data, dataset_name):
    plt.figure(figsize=(14, 10))

    # Plot Voltage vs. Time
    plt.subplot(2, 2, 1)
    sns.lineplot(x='Time', y='Voltage', data=data, hue='Cell', palette=colors)
    plt.title(f'Voltage vs Time for {dataset_name}')
    plt.xlabel('Time')
    plt.ylabel('Voltage (V)')

    # Plot Current vs. Time
    plt.subplot(2, 2, 2)
    sns.lineplot(x='Time', y='Current', data=data, hue='Cell', palette=colors)
    plt.title(f'Current vs Time for {dataset_name}')
    plt.xlabel('Time')
    plt.ylabel('Current (A)')

    # Plot Energy vs. Time
    plt.subplot(2, 2, 3)
    sns.lineplot(x='Time', y='Energy', data=data, hue='Cell', palette=colors)
    plt.title(f'Energy vs Time for {dataset_name}')
    plt.xlabel('Time')
    plt.ylabel('Energy (Wh)')

    # Plot Temperature vs. Time
    plt.subplot(2, 2, 4)
    sns.lineplot(x='Time', y='Temperature', data=data, hue='Cell', palette=colors)
    plt.title(f'Temperature vs Time for {dataset_name}')
    plt.xlabel('Time')
    plt.ylabel('Temperature (°C)')

    plt.tight_layout()
    plt.show()

# Sample 1% of the RPT and Aging data for clearer plots
rpt_sample_reduced = rpt_sample.sample(frac=0.00001, random_state=4)
aging_sample_reduced = aging_sample.sample(frac=0.000009, random_state=4)

# Plot time series for RPT and Aging datasets with custom colors
plot_time_series(rpt_sample_reduced, 'RPT Dataset')
plot_time_series(aging_sample_reduced, 'Aging Dataset')


rpt_sample_reduced = rpt_sample.sample(frac=0.1, random_state=42)
aging_sample_reduced = aging_sample.sample(frac=0.01, random_state=42)

#######
rpt_sample_reduced = rpt_sample.sample(frac=0.0001, random_state=42)
aging_sample_reduced = aging_sample.sample(frac=0.0001, random_state=42)

###################################
# Example: Plotting Energy vs Voltage vs Current for Aging Data (Cell1)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import numpy as np

# Function to create a 3D surface plot for Energy vs Voltage vs Current
def plot_3d_surface_energy(data, feature_x, feature_y, feature_z, title, color):
    # Create grid data for plotting
    grid_x, grid_y = np.mgrid[data[feature_x].min():data[feature_x].max():100j, 
                              data[feature_y].min():data[feature_y].max():100j]
    
    # Interpolate the z values (Energy) for the grid
    grid_z = griddata((data[feature_x], data[feature_y]), data[feature_z], (grid_x, grid_y), method='cubic')
    
    # Create the 3D surface plot
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot the surface
    surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='coolwarm', edgecolor='none', alpha=0.7)
    
    # Add labels and title
    ax.set_xlabel(f'{feature_x} (V)', labelpad=15)
    ax.set_ylabel(f'{feature_y} (A)', labelpad=15)
    ax.set_zlabel(f'{feature_z} (Energy)', labelpad=15)
    ax.set_title(f'{title}')
    
    # Adjust layout to make space for color bar and prevent overlap
    fig.tight_layout()
    
    # Add color bar with space adjustment
    cbar = fig.colorbar(surf, shrink=0.6, aspect=10, pad=0.15)  # Adjust padding here
    cbar.set_label(f'{feature_z} (Energy)')
    
    plt.show()
subset = rpt_sample_reduced[rpt_sample_reduced['Cell'] == 'Cell1']
plot_3d_surface_energy(subset, 'Voltage', 'Current', 'Energy', '3D Surface Plot: Energy vs Voltage vs Current', 'coolwarm')

# Example: Plotting Energy vs Voltage vs Current for Cell1 in Aging Data
aging_subset = aging_sample_reduced[aging_sample_reduced['Cell'] == 'Cell1']
plot_3d_surface_energy(aging_subset, 'Voltage', 'Current', 'Energy', '3D Surface Plot: Energy vs Voltage vs Current (Aging Data)', 'coolwarm')
#################################
# Plot separated time-series plots for each cell in the RPT and Aging samples
rpt_sample_reduced = rpt_sample.sample(frac=0.0001, random_state=50)
aging_sample_reduced = aging_sample.sample(frac=0.0001, random_state=50)

# Updated function to exclude temperature plot and include 3D plot in the 2x2 layout
def plot_time_series_with_3d(data, dataset_name, plot_3d_func):
    fig = plt.figure(figsize=(14, 10))

    # Plot Voltage vs. Time
    plt.subplot(2, 2, 1)
    sns.lineplot(x='Time', y='Voltage', data=data, hue='Cell', palette=colors)
    plt.title(f'Voltage vs Time for {dataset_name}')
    plt.xlabel('Time')
    plt.ylabel('Voltage (V)')

    # Plot Current vs. Time
    plt.subplot(2, 2, 2)
    sns.lineplot(x='Time', y='Current', data=data, hue='Cell', palette=colors)
    plt.title(f'Current vs Time for {dataset_name}')
    plt.xlabel('Time')
    plt.ylabel('Current (A)')

    # Plot Energy vs. Time
    plt.subplot(2, 2, 3)
    sns.lineplot(x='Time', y='Energy', data=data, hue='Cell', palette=colors)
    plt.title(f'Energy vs Time for {dataset_name}')
    plt.xlabel('Time')
    plt.ylabel('Energy (Wh)')

    # Insert the 3D plot in the last quadrant
    ax_3d = fig.add_subplot(2, 2, 4, projection='3d')
    plot_3d_func(ax_3d, data)  # Pass the ax and data to the provided 3D plotting function

    plt.tight_layout()
    plt.show()

# Function to create the 3D plot based on Voltage, Current, and Energy
def plot_3d_energy_voltage_current(ax, data):
    # Create grid data for plotting
    grid_x, grid_y = np.meshgrid(np.linspace(data['Voltage'].min(), data['Voltage'].max(), 50),
                                 np.linspace(data['Current'].min(), data['Current'].max(), 50))

    # Use scipy's griddata to interpolate the Z (Energy) values
    grid_z = griddata((data['Voltage'], data['Current']), data['Energy'], (grid_x, grid_y), method='cubic')

    # Plot surface
    surf = ax.plot_surface(grid_x, grid_y, grid_z, cmap='coolwarm', edgecolor='none', alpha=0.8)
    ax.set_title('3D Surface: Energy vs Voltage vs Current')
    ax.set_xlabel('Voltage (V)')
    ax.set_ylabel('Current (A)')
    ax.set_zlabel('Energy (Wh)')

    # Add colorbar for energy
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)

# Plotting with the new setup for both RPT and Aging datasets
plot_time_series_with_3d(rpt_sample_reduced, 'RPT Dataset', plot_3d_energy_voltage_current)
plot_time_series_with_3d(aging_sample_reduced, 'Aging Dataset', plot_3d_energy_voltage_current)

# Adjusting color palette dynamically based on the number of unique cells in the dataset
############
rpt_sample_reduced = rpt_sample.sample(frac=1, random_state=42)
aging_sample_reduced = aging_sample.sample(frac=0.1, random_state=42)


#####################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import weibull_min
import seaborn as sns
from statsmodels.tsa.stattools import grangercausalitytests
import networkx as nx
from causalgraphicalmodels import CausalGraphicalModel

# Define colors
colors = ['#9370DB', '#32CD32', '#1E90FF']

plt.yscale('log')
# Weibull Trend: Fit and plot Weibull distribution for degradation
def weibull_trend_plot(data, feature, cell_label):
    # Fit Weibull distribution
    shape, loc, scale = weibull_min.fit(data[feature], floc=0)
    weibull_data = weibull_min.rvs(shape, scale=scale, size=len(data[feature]))
    
    # Plot Weibull distribution
    plt.figure(figsize=(8, 6))
    sns.histplot(data[feature], bins=30, color=colors[0], kde=False, label=f'{feature} Histogram')
    plt.plot(np.sort(data[feature]), np.sort(weibull_data), color=colors[1], label=f'Weibull Fit (Shape={shape:.2f}, Scale={scale:.2f})')
    plt.title(f'Weibull Trend for {feature} in {cell_label}')
    plt.xlabel(f'{feature}')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()

# Granger Causality Test
def check_granger_causality(data, variables, max_lag=3):
    print(f"\nRunning Granger Causality Test with max_lag={max_lag}")
    results = {}
    for var in variables:
        test_result = grangercausalitytests(data[['Energy', var]].dropna(), max_lag, verbose=False)
        results[var] = test_result
    return results

# Causal Graph for Battery Degradation
def plot_causal_graph():
    causal_graph = CausalGraphicalModel(
        nodes=["Current", "Voltage", "Energy", "Temperature"],
        edges=[("Current", "Voltage"), ("Voltage", "Energy"), ("Energy", "Temperature")]
    )
    causal_graph.draw()

# Run Weibull trend for RPT and Aging datasets
def analyze_weibull_trends_and_causality(rpt_sample, aging_sample):
    # Plot Weibull trends for Voltage, Current, Energy for one cell (RPT)
    cells = rpt_sample['Cell'].unique()
    for cell in cells:
        subset = rpt_sample[rpt_sample['Cell'] == cell]
        weibull_trend_plot(subset, 'Voltage', f'Cell {cell}')
        weibull_trend_plot(subset, 'Current', f'Cell {cell}')
        weibull_trend_plot(subset, 'Energy', f'Cell {cell}')

    # Repeat for Aging dataset
    cells_aging = aging_sample['Cell'].unique()
    for cell in cells_aging:
        subset_aging = aging_sample[aging_sample['Cell'] == cell]
        weibull_trend_plot(subset_aging, 'Voltage', f'Cell {cell} (Aging)')
        weibull_trend_plot(subset_aging, 'Current', f'Cell {cell} (Aging)')
        weibull_trend_plot(subset_aging, 'Energy', f'Cell {cell} (Aging)')

    # Check Granger causality for RPT and Aging datasets
    variables = ['Voltage', 'Current', 'Temperature']
    print("Granger Causality Test for RPT Dataset:")
    rpt_causality = check_granger_causality(rpt_sample, variables)
    print("Granger Causality Test for Aging Dataset:")
    aging_causality = check_granger_causality(aging_sample, variables)

    # Plot causal graph
    plot_causal_graph()

# Run the analysis
analyze_weibull_trends_and_causality(rpt_sample_reduced, aging_sample_reduced)
###########
rpt_sample_reduced = rpt_sample.sample(frac=0.0001, random_state=42)
aging_sample_reduced = aging_sample.sample(frac=0.0001, random_state=42)

##############
import numpy as np
from scipy.stats import weibull_min
import matplotlib.pyplot as plt
import seaborn as sns

# Extract the voltage data for a specific cell (e.g., Cell 1)
voltage_data_rpt = rpt_sample[rpt_sample['Cell'] == 'Cell1']['Voltage'].dropna().values
voltage_data_aging = aging_sample[aging_sample['Cell'] == 'Cell1']['Voltage'].dropna().values

# Weibull fit for RPT data
shape_rpt, loc_rpt, scale_rpt = weibull_min.fit(voltage_data_rpt, floc=0)

# Weibull fit for Aging data
shape_aging, loc_aging, scale_aging = weibull_min.fit(voltage_data_aging, floc=0)

# Generate Weibull PDF values for plotting
x_rpt = np.linspace(voltage_data_rpt.min(), voltage_data_rpt.max(), 100)
pdf_fitted_rpt = weibull_min.pdf(x_rpt, shape_rpt, loc_rpt, scale_rpt)

x_aging = np.linspace(voltage_data_aging.min(), voltage_data_aging.max(), 100)
pdf_fitted_aging = weibull_min.pdf(x_aging, shape_aging, loc_aging, scale_aging)

# Plotting the Weibull fits and histograms
plt.figure(figsize=(14, 6))

# RPT Data Plot
plt.subplot(1, 2, 1)
sns.histplot(voltage_data_rpt, bins=30, kde=False, color='purple', stat="density", label='Voltage Histogram')
plt.plot(x_rpt, pdf_fitted_rpt, 'g-', label=f'Weibull Fit (Shape={shape_rpt:.2f}, Scale={scale_rpt:.2f})')
plt.title('Weibull Trend for Voltage in Cell1 (RPT)')
plt.xlabel('Voltage')
plt.ylabel('Density')
plt.legend()

# Aging Data Plot
plt.subplot(1, 2, 2)
sns.histplot(voltage_data_aging, bins=30, kde=False, color='purple', stat="density", label='Voltage Histogram')
plt.plot(x_aging, pdf_fitted_aging, 'g-', label=f'Weibull Fit (Shape={shape_aging:.2f}, Scale={scale_aging:.2f})')
plt.title('Weibull Trend for Voltage in Cell1 (Aging)')
plt.xlabel('Voltage')
plt.ylabel('Density')
plt.legend()

plt.tight_layout()
plt.show()
##################
###########
rpt_sample_reduced = rpt_sample.sample(frac=0.0001, random_state=42)
aging_sample_reduced = aging_sample.sample(frac=0.0001, random_state=42)

################
import numpy as np
from scipy.stats import weibull_min
import matplotlib.pyplot as plt
import seaborn as sns

# Define colors
colors = ['#9370DB', '#32CD32', '#1E90FF']

# Custom Weibull parameters for the loss function
shape_loss_func = 2.0
scale_loss_func = 500.0

# Define a function to plot Weibull trends for all cells
def plot_weibull_for_cells(rpt_data, aging_data, cell, color, ax_rpt, ax_aging):
    # RPT Data
    voltage_data_rpt = rpt_data[rpt_data['Cell'] == cell]['Voltage'].dropna().values
    x_rpt = np.linspace(voltage_data_rpt.min(), voltage_data_rpt.max(), 100)
    pdf_fitted_rpt = weibull_min.pdf(x_rpt, shape_loss_func, scale=scale_loss_func)
    
    # Plot RPT
    sns.histplot(voltage_data_rpt, bins=30, kde=False, color=color, stat="density", ax=ax_rpt, label=f'Voltage Histogram for {cell}')
    ax_rpt.plot(x_rpt, pdf_fitted_rpt, 'g-', label=f'Weibull Fit (Shape={shape_loss_func}, Scale={scale_loss_func})')
    ax_rpt.set_title(f'Weibull Trend for Voltage in {cell} (RPT)')
    ax_rpt.set_xlabel('Voltage')
    ax_rpt.set_ylabel('Density')
    ax_rpt.legend()

    # Aging Data
    voltage_data_aging = aging_data[aging_data['Cell'] == cell]['Voltage'].dropna().values
    x_aging = np.linspace(voltage_data_aging.min(), voltage_data_aging.max(), 100)
    pdf_fitted_aging = weibull_min.pdf(x_aging, shape_loss_func, scale=scale_loss_func)
    
    # Plot Aging
    sns.histplot(voltage_data_aging, bins=30, kde=False, color=color, stat="density", ax=ax_aging, label=f'Voltage Histogram for {cell}')
    ax_aging.plot(x_aging, pdf_fitted_aging, 'g-', label=f'Weibull Fit (Shape={shape_loss_func}, Scale={scale_loss_func})')
    ax_aging.set_title(f'Weibull Trend for Voltage in {cell} (Aging)')
    ax_aging.set_xlabel('Voltage')
    ax_aging.set_ylabel('Density')
    ax_aging.legend()

# Plot for all cells in RPT and Aging
fig, axes = plt.subplots(3, 2, figsize=(14, 18))

# Plot Weibull trends for each cell in RPT and Aging data
plot_weibull_for_cells(rpt_sample, aging_sample, 'Cell1', colors[0], axes[0, 0], axes[0, 1])
plot_weibull_for_cells(rpt_sample, aging_sample, 'Cell2', colors[1], axes[1, 0], axes[1, 1])
plot_weibull_for_cells(rpt_sample, aging_sample, 'Cell3', colors[2], axes[2, 0], axes[2, 1])

plt.tight_layout()
plt.show()
#################
import numpy as np
from scipy.stats import weibull_min
import matplotlib.pyplot as plt
import seaborn as sns

# Define colors
colors = ['#9370DB', '#32CD32', '#1E90FF']

# Define a function to plot Weibull trends for all cells and fit Weibull parameters from data
def plot_weibull_for_cells(rpt_data, aging_data, cell, color, ax_rpt, ax_aging):
    # RPT Data
    voltage_data_rpt = rpt_data[rpt_data['Cell'] == cell]['Voltage'].dropna().values
    shape_rpt, loc_rpt, scale_rpt = weibull_min.fit(voltage_data_rpt, floc=0)
    x_rpt = np.linspace(voltage_data_rpt.min(), voltage_data_rpt.max(), 100)
    pdf_fitted_rpt = weibull_min.pdf(x_rpt, shape_rpt, loc_rpt, scale_rpt)
    
    # Plot RPT
    sns.histplot(voltage_data_rpt, bins=30, kde=False, color=color, stat="density", ax=ax_rpt, label=f'Voltage Histogram for {cell}')
    ax_rpt.plot(x_rpt, pdf_fitted_rpt, 'g-', label=f'Weibull Fit (Shape={shape_rpt:.2f}, Scale={scale_rpt:.2f})')
    ax_rpt.set_title(f'Weibull Trend for Voltage in {cell} (RPT)')
    ax_rpt.set_xlabel('Voltage')
    ax_rpt.set_ylabel('Density')
    ax_rpt.legend()

    # Aging Data
    voltage_data_aging = aging_data[aging_data['Cell'] == cell]['Voltage'].dropna().values
    shape_aging, loc_aging, scale_aging = weibull_min.fit(voltage_data_aging, floc=0)
    x_aging = np.linspace(voltage_data_aging.min(), voltage_data_aging.max(), 100)
    pdf_fitted_aging = weibull_min.pdf(x_aging, shape_aging, loc_aging, scale_aging)
    
    # Plot Aging
    sns.histplot(voltage_data_aging, bins=30, kde=False, color=color, stat="density", ax=ax_aging, label=f'Voltage Histogram for {cell}')
    ax_aging.plot(x_aging, pdf_fitted_aging, 'g-', label=f'Weibull Fit (Shape={shape_aging:.2f}, Scale={scale_aging:.2f})')
    ax_aging.set_title(f'Weibull Trend for Voltage in {cell} (Aging)')
    ax_aging.set_xlabel('Voltage')
    ax_aging.set_ylabel('Density')
    ax_aging.legend()

# Plot for all cells in RPT and Aging
fig, axes = plt.subplots(3, 2, figsize=(14, 18))

# Plot Weibull trends for each cell in RPT and Aging data
plot_weibull_for_cells(rpt_sample, aging_sample, 'Cell1', colors[0], axes[0, 0], axes[0, 1])
plot_weibull_for_cells(rpt_sample, aging_sample, 'Cell2', colors[1], axes[1, 0], axes[1, 1])
plot_weibull_for_cells(rpt_sample, aging_sample, 'Cell3', colors[2], axes[2, 0], axes[2, 1])

plt.tight_layout()
plt.show()
####################
###############################################################################
rpt_sample_reduced = rpt_sample.sample(frac=0.1, random_state=42)
aging_sample_reduced = aging_sample.sample(frac=0.001, random_state=42)
############
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from math import pi

# Sample features
features = ['Current', 'Voltage', 'Energy', 'Temperature']
n_clusters = 3  # Adjust based on your clustering

# Assuming you already have a clustered dataframe like 'rpt_sample' with clusters
# Scale the features for t-SNE
scaler = StandardScaler()
features_scaled = scaler.fit_transform(rpt_sample_reduced[features])

# Apply t-SNE to reduce to 2D for visualization
tsne = TSNE(n_components=2, random_state=42)
tsne_data = tsne.fit_transform(features_scaled)

# Apply KMeans for clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
rpt_sample_reduced['Cluster'] = kmeans.fit_predict(tsne_data)

# Colors for clusters - Same as those in the spider plots
colors = ['green', 'blue', 'purple']

# Map clusters to their respective colors
cluster_colors = [colors[label] for label in rpt_sample_reduced['Cluster']]

# Plot t-SNE alone with matching colors
plt.figure(figsize=(8, 6))
scatter = plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=cluster_colors, s=50, alpha=0.8)
plt.title('t-SNE Clustering of RPT Data')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.grid(True)
plt.show()

# Function to plot spider (radar) chart
def plot_real_spider_chart(ax, data, features, cluster_label, color):
    """
    Function to plot a radar (spider) chart for feature contributions.
    
    Parameters:
    - ax: Axes object for the radar plot.
    - data: Pandas DataFrame containing the features.
    - features: List of feature names.
    - cluster_label: The cluster label for which to plot the feature contributions.
    - color: Color for the radar plot.
    """
    cluster_data = data[data['Cluster'] == cluster_label]
    mean_values = cluster_data[features].mean().values  # Mean values of the features for this cluster
    
    # Radar chart setup
    num_vars = len(features)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    # The plot requires the values to be cyclic, so we append the first value to the end
    mean_values = np.concatenate((mean_values, [mean_values[0]]))
    angles += angles[:1]

    # Plot on the provided Axes
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    ax.plot(angles, mean_values, color=color, linewidth=2, linestyle='solid')
    ax.fill(angles, mean_values, color=color, alpha=0.25)
    ax.set_yticklabels([])  # Hide radial ticks
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features)
    ax.set_title(f'Feature Contributions - Cluster {cluster_label}', size=15, color=color, y=1.1)

# Create a figure with 1x3 subplots for feature contributions
fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(polar=True))

# Loop over the clusters and plot their feature contributions as spider plots
for i in range(n_clusters):
    plot_real_spider_chart(axes[i], rpt_sample_reduced, features, i, colors[i])

# Adjust layout for better spacing
plt.tight_layout()
plt.show()

###############################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from math import pi

# Sample features
features = ['Current', 'Voltage', 'Energy', 'Temperature']
n_clusters = 3  # Adjust based on your clustering

# Assuming you already have a clustered dataframe like 'aging_sample_reduced' with clusters
# Scale the features for t-SNE
scaler = StandardScaler()
features_scaled = scaler.fit_transform(aging_sample_reduced[features])

# Apply t-SNE to reduce to 2D for visualization
tsne = TSNE(n_components=2, random_state=42)
tsne_data = tsne.fit_transform(features_scaled)

# Apply KMeans for clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
aging_sample_reduced['Cluster'] = kmeans.fit_predict(tsne_data)

# Colors for clusters - Same as those in the spider plots (using blue instead of cyan)
colors = ['green', 'blue', 'purple']

# Map clusters to their respective colors
cluster_colors = [colors[label] for label in aging_sample_reduced['Cluster']]

# Plot t-SNE alone with matching colors
plt.figure(figsize=(8, 6))
scatter = plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=cluster_colors, s=50, alpha=0.8)
plt.title('t-SNE Clustering of Aging Data')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.grid(True)
plt.colorbar(scatter, label='Cluster')
plt.show()

# Function to plot spider (radar) chart
def plot_real_spider_chart(ax, data, features, cluster_label, color):
    """
    Function to plot a radar (spider) chart for feature contributions.
    
    Parameters:
    - ax: Axes object for the radar plot.
    - data: Pandas DataFrame containing the features.
    - features: List of feature names.
    - cluster_label: The cluster label for which to plot the feature contributions.
    - color: Color for the radar plot.
    """
    cluster_data = data[data['Cluster'] == cluster_label]
    mean_values = cluster_data[features].mean().values  # Mean values of the features for this cluster
    
    # Radar chart setup
    num_vars = len(features)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    
    # The plot requires the values to be cyclic, so we append the first value to the end
    mean_values = np.concatenate((mean_values, [mean_values[0]]))
    angles += angles[:1]

    # Plot on the provided Axes
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    ax.plot(angles, mean_values, color=color, linewidth=2, linestyle='solid')
    ax.fill(angles, mean_values, color=color, alpha=0.25)
    ax.set_yticklabels([])  # Hide radial ticks
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(features)
    ax.set_title(f'Feature Contributions - Cluster {cluster_label}', size=15, color=color, y=1.1)

# Create a figure with 1x3 subplots for feature contributions
fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(polar=True))

# Loop over the clusters and plot their feature contributions as spider plots
for i in range(n_clusters):
    plot_real_spider_chart(axes[i], aging_sample_reduced, features, i, colors[i])

# Adjust layout for better spacing
plt.tight_layout()
plt.show()


###################
# Function to plot t-SNE for each cell
def plot_tsne_clusters_for_cell(cell_data, cell_name, features, n_clusters, colors):
    # Scale the features for t-SNE
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(cell_data[features])

    # Apply t-SNE to reduce to 2D for visualization
    tsne = TSNE(n_components=2, random_state=42)
    tsne_data = tsne.fit_transform(features_scaled)

    # Apply KMeans for clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    cell_data['Cluster'] = kmeans.fit_predict(tsne_data)

    # Map clusters to their respective colors
    cluster_colors = [colors[label] for label in cell_data['Cluster']]

    # Plot t-SNE with matching colors for clusters
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(tsne_data[:, 0], tsne_data[:, 1], c=cluster_colors, s=50, alpha=0.8)
    plt.title(f't-SNE Clustering of {cell_name} Data')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.grid(True)
    plt.show()

# List of unique cells in the RPT dataset
cells = rpt_sample_reduced['Cell'].unique()

# Plot t-SNE clusters for each cell in the RPT dataset
for cell in cells:
    # Filter data for the current cell
    cell_data = rpt_sample_reduced[rpt_sample_reduced['Cell'] == cell]
    
    # Plot t-SNE clusters for the current cell
    plot_tsne_clusters_for_cell(cell_data, f'Cell {cell} - RPT', features, n_clusters, colors)

# Repeat the process for the Aging dataset
cells_aging = aging_sample_reduced['Cell'].unique()

# Plot t-SNE clusters for each cell in the Aging dataset
for cell in cells_aging:
    # Filter data for the current cell
    cell_data_aging = aging_sample_reduced[aging_sample_reduced['Cell'] == cell]
    
    # Plot t-SNE clusters for the current cell
    plot_tsne_clusters_for_cell(cell_data_aging, f'Cell {cell} - Aging', features, n_clusters, colors)

# List of unique cells in the RPT dataset
cells = rpt_sample_reduced['Cell'].unique()

# Create spider plots for each cell in the RPT dataset
for cell in cells:
    # Filter data for the current cell
    cell_data = rpt_sample_reduced[rpt_sample_reduced['Cell'] == cell]
    
    # Create a figure with 1x3 subplots for feature contributions for each cell
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(polar=True))
    
    # Loop over the clusters and plot their feature contributions as spider plots for each cell
    for i in range(n_clusters):
        plot_real_spider_chart(axes[i], cell_data, features, i, colors[i])
    
    # Add a title to indicate which cell is being plotted
    plt.suptitle(f'Spider Plot for Cell {cell} - RPT Data', size=20)
    
    # Adjust layout for better spacing
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Show the plot
    plt.show()

# Repeat the process for the Aging dataset
cells_aging = aging_sample_reduced['Cell'].unique()

# Create spider plots for each cell in the Aging dataset
for cell in cells_aging:
    # Filter data for the current cell
    cell_data_aging = aging_sample_reduced[aging_sample_reduced['Cell'] == cell]
    
    # Create a figure with 1x3 subplots for feature contributions for each cell
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), subplot_kw=dict(polar=True))
    
    # Loop over the clusters and plot their feature contributions as spider plots for each cell
    for i in range(n_clusters):
        plot_real_spider_chart(axes[i], cell_data_aging, features, i, colors[i])
    
    # Add a title to indicate which cell is being plotted
    plt.suptitle(f'Spider Plot for Cell {cell} - Aging Data', size=20)
    
    # Adjust layout for better spacing
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Show the plot
    plt.show()

###############################
# Extract the mean feature contributions for each cluster in the RPT and Aging datasets

import pandas as pd
# Calculate the means of each feature per cluster for RPT
rpt_cluster_means = rpt_sample_reduced.groupby('Cluster')[features].mean()
# Calculate the means of each feature per cluster for Aging
aging_cluster_means = aging_sample_reduced.groupby('Cluster')[features].mean()


# Assuming rpt_cluster_means and aging_cluster_means are DataFrames:
print("RPT Cluster Means:")
print(rpt_cluster_means)

print("\nAging Cluster Means:")
print(aging_cluster_means)

from IPython.display import display

# Display the dataframes in a cleaner format in Jupyter
display(rpt_cluster_means)
display(aging_cluster_means)

rpt_cluster_means.to_csv("rpt_cluster_means.csv")
aging_cluster_means.to_csv("aging_cluster_means.csv")

###############################

#####################################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import weibull_min, expon, lognorm

# Define a function to refine Weibull fitting with parameter control and display alternative distributions
def refined_weibull_fit(data, feature, cell_label, ax, distribution='weibull'):
    # Define different distribution options
    dist_options = {
        'weibull': weibull_min,
        'lognorm': lognorm,
        'expon': expon
    }
    
    dist = dist_options.get(distribution, weibull_min)
    
    # Fit Weibull (or other) distribution
    shape, loc, scale = dist.fit(data[feature], floc=0)
    x = np.linspace(data[feature].min(), data[feature].max(), 100)
    pdf_fitted = dist.pdf(x, shape, loc=loc, scale=scale)

    # Plot histogram and the fitted distribution
    sns.histplot(data[feature], bins=30, color='purple', kde=False, stat="density", ax=ax, label=f'{feature} Histogram')
    ax.plot(x, pdf_fitted, 'r-', label=f'{distribution.capitalize()} Fit (Shape={shape:.2f}, Scale={scale:.2f})')
    ax.set_title(f'{distribution.capitalize()} Trend for {feature} in {cell_label}')
    ax.set_xlabel(f'{feature}')
    ax.set_ylabel('Density')
    ax.legend()

# Refine the fitting process for each feature in the RPT and Aging datasets
def refined_weibull_analysis(rpt_sample_reduced, aging_sample_reduced, distribution='weibull'):
    fig, axes = plt.subplots(3, 2, figsize=(14, 18))
    cells = rpt_sample['Cell'].unique()
    
    # RPT data analysis
    for i, cell in enumerate(cells):
        subset = rpt_sample[rpt_sample['Cell'] == cell]
        refined_weibull_fit(subset, 'Voltage', f'Cell {cell}', axes[i, 0], distribution)
        refined_weibull_fit(subset, 'Current', f'Cell {cell}', axes[i, 1], distribution)
        
    plt.tight_layout()
    plt.show()
    
    # Aging data analysis
    fig, axes = plt.subplots(3, 2, figsize=(14, 18))
    cells_aging = aging_sample_reduced['Cell'].unique()
    
    for i, cell in enumerate(cells_aging):
        subset_aging = aging_sample_reduced[aging_sample_reduced['Cell'] == cell]
        refined_weibull_fit(subset_aging, 'Voltage', f'Cell {cell} (Aging)', axes[i, 0], distribution)
        refined_weibull_fit(subset_aging, 'Current', f'Cell {cell} (Aging)', axes[i, 1], distribution)

    plt.tight_layout()
    plt.show()

# Run the refined Weibull analysis on RPT and Aging datasets
refined_weibull_analysis(rpt_sample_reduced, aging_sample_reduced, distribution='weibull')
###############################################################################
rpt_sample_reduced = rpt_sample.sample(frac=1, random_state=42)
aging_sample_reduced = aging_sample.sample(frac=0.01, random_state=42)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from math import pi

# Sample features
features = ['Current', 'Voltage', 'Energy', 'Temperature']
n_clusters = 3  # Adjust based on your clustering

# Assuming you already have a clustered dataframe like 'rpt_sample' with clusters
# Scale the features for t-SNE
scaler = StandardScaler()
features_scaled_rpt = scaler.fit_transform(rpt_sample[features])

# Apply t-SNE to reduce to 2D for visualization
tsne = TSNE(n_components=2, random_state=42)
tsne_data = tsne.fit_transform(features_scaled_rpt)

# Apply KMeans for clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
rpt_sample['Cluster'] = kmeans.fit_predict(tsne_data)
# Sample features
# Assuming you already have a clustered dataframe like 'aging_sample_reduced' with clusters
# Scale the features for t-SNE
scaler = StandardScaler()
features_scaled_aging = scaler.fit_transform(aging_sample[features])

# Apply t-SNE to reduce to 2D for visualization
tsne = TSNE(n_components=2, random_state=42)
tsne_data = tsne.fit_transform(features_scaled_aging)

# Apply KMeans for clustering
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
aging_sample_reduced['Cluster'] = kmeans.fit_predict(tsne_data)
#########plots
import matplotlib.pyplot as plt

# Extract the t-SNE coordinates and clusters from the dataframe
tsne_data_x = tsne_data[:, 0]
tsne_data_y = tsne_data[:, 1]
clusters = rpt_sample['Cluster']

# Create a scatter plot
plt.figure(figsize=(8, 6))
scatter = plt.scatter(tsne_data_x, tsne_data_y, c=clusters, cmap='viridis', s=50)
plt.colorbar(scatter, label='Cluster')
plt.title('t-SNE Visualization with KMeans Clusters')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.grid(True)
plt.show()

############################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import weibull_min, lognorm, expon

# Function to fit Weibull or alternative distributions with parameter optimization
def refined_weibull_fit(data, feature, cluster_label, ax, distribution='weibull'):
    # Define different distribution options
    dist_options = {
        'weibull': weibull_min,
        'lognorm': lognorm,
        'expon': expon
    }
    
    dist = dist_options.get(distribution, weibull_min)
    
    # Fit the distribution
    params = dist.fit(data[feature])
    x = np.linspace(data[feature].min(), data[feature].max(), 100)
    pdf_fitted = dist.pdf(x, *params)  # Fit PDF based on optimized parameters

    # Plot histogram and fitted distribution
    sns.histplot(data[feature], bins=30, color='cyan', kde=False, stat="density", ax=ax, label=f'{feature} Histogram')
    ax.plot(x, pdf_fitted, 'r-', label=f'{distribution.capitalize()} Fit (Shape={params[0]:.2f}, Scale={params[2]:.2f})')
    ax.set_title(f'{distribution.capitalize()} Trend for {feature} in {cluster_label}')
    ax.set_xlabel(f'{feature}')
    ax.set_ylabel('Density')
    ax.legend()

# Refine Weibull fit and compare with other distributions
def refined_weibull_analysis(sample_data, features, distribution='weibull'):
    clusters = sample_data['Cluster'].unique()

    for cluster in clusters:
        cluster_data = sample_data[sample_data['Cluster'] == cluster]
        fig, axes = plt.subplots(len(features), 1, figsize=(10, 14))

        for i, feature in enumerate(features):
            refined_weibull_fit(cluster_data, feature, f'Cluster {cluster}', axes[i], distribution)
        
        plt.tight_layout()
        plt.show()

# Apply Weibull fitting and alternative fits for RPT and Aging datasets
features = ['Current', 'Voltage', 'Energy', 'Temperature']
refined_weibull_analysis(rpt_sample_reduced, features, distribution='weibull')
refined_weibull_analysis(aging_sample_reduced, features, distribution='weibull')

#refined_weibull_analysis(aging_sample_reduced, features, distribution='lognorm')  # Try log-normal for aging dataset
#extract the info
import numpy as np
import pandas as pd
from scipy.stats import weibull_min
import seaborn as sns
import matplotlib.pyplot as plt

# Function to fit Weibull distribution and return the shape and scale parameters
def get_weibull_params(data, feature):
    shape, loc, scale = weibull_min.fit(data[feature], floc=0)
    return shape, scale

# Function to print the extracted Weibull parameters
def print_weibull_params(cluster_data, features, cluster_label):
    print(f"\nWeibull Parameters for Cluster {cluster_label}:")
    for feature in features:
        shape, scale = get_weibull_params(cluster_data, feature)
        print(f"{feature}:")
        print(f"  Shape: {shape:.2f}, Scale: {scale:.2f}")
    print("-" * 40)

# Example features to analyze
features = ['Current', 'Voltage', 'Energy', 'Temperature']

# Assuming 'rpt_sample' and 'aging_sample' have a 'Cluster' column from previous clustering
clusters = rpt_sample_reduced['Cluster'].unique()

# Loop over each cluster and print Weibull parameters for each feature
for cluster in clusters:
    cluster_data_rpt = rpt_sample_reduced[rpt_sample_reduced['Cluster'] == cluster]
    cluster_data_aging = aging_sample_reduced[aging_sample_reduced['Cluster'] == cluster]
    
    print_weibull_params(cluster_data_rpt, features, f"{cluster} (RPT)")
    print_weibull_params(cluster_data_aging, features, f"{cluster} (Aging)")


########shapescale
import matplotlib.pyplot as plt
import numpy as np

# Data for Weibull Parameters
clusters = ['Cluster 0', 'Cluster 1', 'Cluster 2']
features = ['Current', 'Voltage', 'Energy', 'Temperature']

# Shape (β) and Scale (λ) for each feature in each cluster
shape_data = {
    'Current': [0.07, 4.60, 0.11],
    'Voltage': [26.38, 42.06, 29.17],
    'Energy': [0.05, 0.06, 0.06],
    'Temperature': [14.46, 14.77, 8.22]
}

scale_data = {
    'Current': [0.00, 44.60, 0.00],
    'Voltage': [3.37, 3.29, 3.30],
    'Energy': [0.00, 0.00, 0.00],
    'Temperature': [27.02, 26.71, 27.95]
}

# Colors for each feature with gradients for clusters
shape_colors = ['#d3d3d3', '#a9a9a9', '#808080']   # Grey gradients for shape
scale_colors = ['#87CEFA', '#4682B4', '#1E90FF']   # Blue gradients for scale

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# Plotting
for i, feature in enumerate(features):
    ax = axes[i // 2, i % 2]
    
    # Plot Shape (left bars)
    bars1 = ax.bar(np.arange(len(clusters)) - 0.2, shape_data[feature], 0.4, 
                   label='Shape', color=shape_colors)
    
    # Plot Scale (right bars)
    bars2 = ax.bar(np.arange(len(clusters)) + 0.2, scale_data[feature], 0.4, 
                   label='Scale', color=scale_colors)
    
    # Adding text labels on the bars
    for bar in bars1:
        yval = bar.get_height()
        if yval > 0:  # Avoid showing labels for 0 values
            ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 2), ha='center', va='bottom')  # Shape value
        
    for bar in bars2:
        yval = bar.get_height()
        if yval > 0:  # Avoid showing labels for 0 values
            ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 2), ha='center', va='bottom')  # Scale value
    
    # Adding labels and title
    ax.set_xticks(np.arange(len(clusters)))
    ax.set_xticklabels(clusters)
    ax.set_title(f'{feature} (Shape and Scale)')
    ax.set_ylabel('Value')
    ax.legend()
    
    # Apply log scale where values are too small
    if feature in ['Current', 'Energy']:
        ax.set_yscale('log')
        ax.set_ylim(1e-3, 100)  # Prevent the log scale from crashing on small values

plt.tight_layout()
plt.show()
###########################################
# DINN Evaluation on Clusters with Progress Updates

# Increase the number of epochs and samples for better training
EPOCHS = 100
N_SAMPLES = 2

# Weibull loss function using TensorFlow operations
def weibull_loss(y_true, y_pred):
    shape = 2.0  # Weibull shape parameter
    scale = 500.0  # Weibull scale parameter (cycles or time)
    failure_prob = 1 - tf.exp(-((y_pred / scale) ** shape))
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    loss = mse * (1 + failure_prob)
    return loss

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
    print("Training DINN...")
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=32, validation_data=(X_val, y_val), verbose=0)

    # Evaluate the model on the test set
    test_loss = model.evaluate(X_test, y_test, verbose=0)

    print(f"Final Validation Loss: {history.history['val_loss'][-1]:.5f}")
    print(f"Test Loss: {test_loss:.5f}")

    return history.history['val_loss'], test_loss

# Average loss over multiple samples
def average_over_samples(data, train_function, epochs=EPOCHS, n_samples=N_SAMPLES):
    val_losses, test_losses = [], []
    
    for i in range(n_samples):
        print(f"\nRunning sample {i + 1}/{n_samples}...")
        data_sample = data.sample(frac=1.0, random_state=i*42)
        val_loss, test_loss = train_function(data_sample, epochs=epochs)
        val_losses.append(val_loss[-1])  # Only consider the last validation loss per sample
        test_losses.append(test_loss)
        print(f"Validation Loss for sample {i + 1}: {val_loss[-1]:.5f}")
        print(f"Test Loss for sample {i + 1}: {test_loss:.5f}")

    # Compute mean and standard deviation for validation and test losses
    avg_val_loss = np.mean(val_losses)
    std_val_loss = np.std(val_losses)
    avg_test_loss = np.mean(test_losses)
    std_test_loss = np.std(test_losses)

    return avg_val_loss, std_val_loss, avg_test_loss, std_test_loss

# Train and evaluate the model on each cluster in RPT and Aging datasets
def evaluate_dinn_per_cluster(dataset, train_function):
    clusters = dataset['Cluster'].unique()
    for cluster in clusters:
        print(f"\n--- Evaluating Cluster {cluster} ---")
        cluster_data = dataset[dataset['Cluster'] == cluster]
        avg_val_loss, std_val_loss, avg_test_loss, std_test_loss = average_over_samples(cluster_data, train_function)
        print(f"\nResults for Cluster {cluster}:")
        print(f"Validation Loss: {avg_val_loss:.5f} ± {std_val_loss:.5f}")
        print(f"Test Loss: {avg_test_loss:.5f} ± {std_test_loss:.5f}")

# Run evaluation on RPT and Aging datasets
print("Evaluating DINN on RPT Dataset Clusters:")
evaluate_dinn_per_cluster(rpt_sample, train_dinn_regularized)

print("\nEvaluating DINN on Aging Dataset Clusters:")
evaluate_dinn_per_cluster(aging_sample_reduced, train_dinn_regularized)
