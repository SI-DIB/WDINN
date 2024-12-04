import numpy as np
from itertools import product
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization, Activation, LSTM
from tensorflow.keras.regularizers import l2

# Hyperparameter grid for grid search
hyperparameter_grid = {
    "dense_units": [32, 64, 128],
    "dropout_rate": [0.3, 0.5, 0.6],
    "l2_reg": [0.001, 0.01, 0.1],
    "batch_size": [16, 32, 64],
    "optimizer": ["adam", "sgd", "rmsprop"],
    "weibull_shape": [1.5, 2.0, 2.5],
    "weibull_scale": [300, 500, 700],
}

# Weibull loss function
def weibull_loss(y_true, y_pred, beta, lambda_):
    failure_prob = 1 - tf.exp(-((y_pred / lambda_) ** beta))
    mse = tf.reduce_mean(tf.square(y_true - y_pred))
    loss = mse * (1 + failure_prob)
    return loss

# Function to create and train a model
def train_model(hyperparams, X_train, y_train, X_val, y_val, model_type="WDINN"):
    model = Sequential()
    l2_reg = l2(hyperparams["l2_reg"])

    if model_type in ["WDINN", "ANN"]:
        # Fully connected layers
        model.add(Dense(hyperparams["dense_units"], input_shape=(X_train.shape[1],), kernel_regularizer=l2_reg))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(hyperparams["dropout_rate"]))
        model.add(Dense(1))
    elif model_type in ["GRU", "LSTM", "Attention-LSTM"]:
        # Recurrent layers
        recurrent_layer = GRU if model_type == "GRU" else LSTM
        model.add(recurrent_layer(hyperparams["dense_units"], return_sequences=False, kernel_regularizer=l2_reg))
        model.add(BatchNormalization())
        model.add(Dropout(hyperparams["dropout_rate"]))
        model.add(Dense(1))
    
    # Compile the model
    if model_type == "WDINN":
        beta = hyperparams["weibull_shape"]
        lambda_ = hyperparams["weibull_scale"]
        model.compile(optimizer=hyperparams["optimizer"], loss=lambda y_true, y_pred: weibull_loss(y_true, y_pred, beta, lambda_))
    else:
        model.compile(optimizer=hyperparams["optimizer"], loss="mse")
    
    # Train the model
    history = model.fit(X_train, y_train, batch_size=hyperparams["batch_size"], epochs=10, validation_data=(X_val, y_val), verbose=0)
    return model, history

# Function to perform grid search
def grid_search(X, y, model_type="WDINN"):
    results = []
    
    # Generate all combinations of hyperparameters
    hyperparameter_combinations = list(product(*hyperparameter_grid.values()))
    print(f"Total combinations: {len(hyperparameter_combinations)}")
    
    for idx, hyperparam_values in enumerate(hyperparameter_combinations):
        # Map hyperparameter values to keys
        hyperparams = dict(zip(hyperparameter_grid.keys(), hyperparam_values))
        print(f"\nEvaluating combination {idx + 1}/{len(hyperparameter_combinations)}: {hyperparams}")
        
        # Split the data
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        model, history = train_model(hyperparams, X_train, y_train, X_val, y_val, model_type)
        
        # Evaluate the model
        val_loss = history.history["val_loss"][-1]
        results.append({"hyperparams": hyperparams, "val_loss": val_loss})
        print(f"Validation Loss: {val_loss}")
    
    # Sort results by validation loss
    results.sort(key=lambda x: x["val_loss"])
    return results

# Example usage
if __name__ == "__main__":
    # Replace with actual data
    X = np.random.rand(1000, 4)  # Features: Current, Voltage, Energy, Temperature
    y = np.random.rand(1000)  # Target: RUL

    # Perform grid search for WDINN
    results = grid_search(X, y, model_type="WDINN")
    
    print("\nBest Hyperparameters:")
    print(results[0])

