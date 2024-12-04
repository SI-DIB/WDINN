#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Battery Degradation Prediction: Parameterized Model Implementation
===================================================================
This repository implements parameterized machine learning and deep learning models 
for lithium-ion battery degradation prediction. Hyperparameters are passed as arguments, 
allowing seamless integration with a hyperparameter optimization script.

Author: Sahar Qaadan
Date: September 13, 2024
"""

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout, BatchNormalization, Activation, LSTM
from keras.regularizers import l2
import tensorflow as tf
from statsmodels.regression.mixed_linear_model import MixedLM

# Define Weibull Loss Function
def weibull_loss(y_true, y_pred, shape=2.0, scale=500.0):
    """
    Custom loss function combining MSE with Weibull-based failure probabilities.
    """
    failure_prob = 1 - tf.exp(-((y_pred / scale) ** shape))
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    return mse * (1 + failure_prob)

# Mixed Linear Effects (LME) Model
def fit_lme(data, formula="Voltage ~ Current + Energy + Temperature", group_col="Cluster"):
    """
    Fit a mixed linear effects model with configurable formula and group column.
    """
    model = MixedLM.from_formula(formula, data=data, groups=data[group_col])
    result = model.fit()
    print(result.summary())
    return result

# GRU Model
def train_gru(data, params, timesteps=10):
    """
    Train a GRU model with configurable parameters.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data[['Current', 'Energy', 'Temperature']])
    y = data['Voltage'].values

    # Time-series data preparation
    X, y_final = [], []
    for i in range(timesteps, len(X_scaled)):
        X.append(X_scaled[i-timesteps:i])
        y_final.append(y[i])

    X = np.array(X)
    y_final = np.array(y_final)

    X_train, X_test, y_train, y_test = train_test_split(X, y_final, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    model = Sequential()
    model.add(GRU(params['units1'], return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]), 
                  kernel_regularizer=l2(params['l2_reg'])))
    model.add(BatchNormalization())
    model.add(Dropout(params['dropout']))
    model.add(GRU(params['units2'], kernel_regularizer=l2(params['l2_reg'])))
    model.add(Dense(params['dense_units'], activation='relu', kernel_regularizer=l2(params['l2_reg'])))
    model.add(Dense(1))

    model.compile(optimizer=params['optimizer'], loss=params['loss'])
    history = model.fit(X_train, y_train, epochs=params['epochs'], batch_size=params['batch_size'], 
                        validation_data=(X_val, y_val), verbose=0)
    return history.history['val_loss'], model.evaluate(X_test, y_test, verbose=0)

# Bi-LSTM Model
def train_bilstm(data, params, timesteps=10):
    """
    Train a Bi-LSTM model with configurable parameters.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data[['Current', 'Energy', 'Temperature']])
    y = data['Voltage'].values

    # Time-series data preparation
    X, y_final = [], []
    for i in range(timesteps, len(X_scaled)):
        X.append(X_scaled[i-timesteps:i])
        y_final.append(y[i])

    X = np.array(X)
    y_final = np.array(y_final)

    X_train, X_test, y_train, y_test = train_test_split(X, y_final, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    model = Sequential()
    model.add(LSTM(params['units1'], return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]), 
                   kernel_regularizer=l2(params['l2_reg'])))
    model.add(BatchNormalization())
    model.add(Dropout(params['dropout']))
    model.add(LSTM(params['units2'], kernel_regularizer=l2(params['l2_reg'])))
    model.add(Dense(params['dense_units'], activation='relu', kernel_regularizer=l2(params['l2_reg'])))
    model.add(Dense(1))

    model.compile(optimizer=params['optimizer'], loss=params['loss'])
    history = model.fit(X_train, y_train, epochs=params['epochs'], batch_size=params['batch_size'], 
                        validation_data=(X_val, y_val), verbose=0)
    return history.history['val_loss'], model.evaluate(X_test, y_test, verbose=0)

# XGBoost Model
def train_xgboost(data, params):
    """
    Train an XGBoost model with configurable parameters.
    """
    X = data[['Current', 'Energy', 'Temperature']]
    y = data['Voltage']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = XGBRegressor(n_estimators=params['n_estimators'], max_depth=params['max_depth'], 
                         learning_rate=params['learning_rate'], reg_lambda=params['l2_reg'], 
                         random_state=42)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=10, verbose=False)
    test_loss = np.mean((y_test - model.predict(X_test)) ** 2)
    return test_loss

# ANN Model
def train_ann(data, params):
    """
    Train an ANN model with configurable parameters.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data[['Current', 'Energy', 'Temperature']])
    y = data['Voltage'].values

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    model = Sequential()
    model.add(Dense(params['dense_units'], activation='relu', input_shape=(X_train.shape[1],), 
                    kernel_regularizer=l2(params['l2_reg'])))
    model.add(Dropout(params['dropout']))
    model.add(Dense(params['dense_units2'], activation='relu', kernel_regularizer=l2(params['l2_reg'])))
    model.add(Dense(1))

    model.compile(optimizer=params['optimizer'], loss=params['loss'])
    history = model.fit(X_train, y_train, epochs=params['epochs'], batch_size=params['batch_size'], 
                        validation_data=(X_val, y_val), verbose=0)
    return history.history['val_loss'], model.evaluate(X_test, y_test, verbose=0)

# DINN Model
def train_dinn(data, params):
    """
    Train a DINN model with configurable parameters and Weibull loss.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data[['Current', 'Energy', 'Temperature']])
    y = data['Voltage'].values

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)

    model = Sequential()
    model.add(Dense(params['dense_units'], kernel_regularizer=l2(params['l2_reg']), input_shape=(X_train.shape[1],)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(params['dropout']))
    model.add(Dense(params['dense_units2'], kernel_regularizer=l2(params['l2_reg'])))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(params['dropout']))
    model.add(Dense(1))

    model.compile(optimizer=params['optimizer'], loss=lambda y_true, y_pred: weibull_loss(y_true, y_pred, 
                                                                                          shape=params['shape'], 
                                                                                          scale=params['scale']))
    history = model.fit(X_train, y_train, epochs=params['epochs'], batch_size=params['batch_size'], 
                        validation_data=(X_val, y_val), verbose=0)
    return history.history['val_loss'], model.evaluate(X_test, y_test, verbose=0)

# Train Models
print("Training GRU...")
gru_val_loss, gru_test_loss = train_gru(data, gru_params)

print("\nTraining Bi-LSTM...")
bilstm_val_loss, bilstm_test_loss = train_bilstm(data, bilstm_params)

print("\nTraining XGBoost...")
xgboost_test_loss = train_xgboost(data, xgboost_params)

print("\nTraining ANN...")
ann_val_loss, ann_test_loss = train_ann(data, ann_params)

print("\nTraining DINN...")
dinn_val_loss, dinn_test_loss = train_dinn(data, dinn_params)

# Print Results
print("\nModel Results:")
print(f"GRU - Validation Loss: {gru_val_loss[-1]}, Test Loss: {gru_test_loss}")
print(f"Bi-LSTM - Validation Loss: {bilstm_val_loss[-1]}, Test Loss: {bilstm_test_loss}")
print(f"XGBoost - Test Loss: {xgboost_test_loss}")
print(f"ANN - Validation Loss: {ann_val_loss[-1]}, Test Loss: {ann_test_loss}")
print(f"DINN - Validation Loss: {dinn_val_loss[-1]}, Test Loss: {dinn_test_loss}")
