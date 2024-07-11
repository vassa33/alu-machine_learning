#!/usr/bin/env python3
"""
    Optimizes a machine learning model using GPyOpt
    PyCodeStyle: Ignore
"""

import numpy as np
import GPyOpt
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(-1, 28*28).astype('float32') / 255
x_test = x_test.reshape(-1, 28*28).astype('float32') / 255
y_train = (y_train >= 5).astype('int32')  # Binary classification: 0-4 vs 5-9
y_test = (y_test >= 5).astype('int32')

def create_model(learning_rate, num_units, dropout_rate, l2_weight, batch_size):
    model = keras.Sequential([
        keras.layers.Dense(num_units, activation='relu', kernel_regularizer=keras.regularizers.l2(l2_weight), input_shape=(784,)),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(num_units, activation='relu', kernel_regularizer=keras.regularizers.l2(l2_weight)),
        keras.layers.Dropout(dropout_rate),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

def objective_function(parameters):
    learning_rate, num_units, dropout_rate, l2_weight, batch_size = parameters[0]
    
    model = create_model(learning_rate, int(num_units), dropout_rate, l2_weight, int(batch_size))
    
    checkpoint_filename = f"best_model_lr{learning_rate:.4f}_units{int(num_units)}_dropout{dropout_rate:.2f}_l2{l2_weight:.4f}_batch{int(batch_size)}.h5"
    checkpoint_callback = keras.callbacks.ModelCheckpoint(checkpoint_filename, save_best_only=True, monitor='val_accuracy', mode='max')
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    
    history = model.fit(x_train, y_train, 
                        epochs=50, 
                        batch_size=int(batch_size),
                        validation_split=0.2,
                        callbacks=[checkpoint_callback, early_stopping],
                        verbose=0)
    
    val_accuracy = max(history.history['val_accuracy'])
    return -val_accuracy  # We minimize the negative accuracy (equivalent to maximizing accuracy)

# Define the domain of the hyperparameters
domain = [
    {'name': 'learning_rate', 'type': 'continuous', 'domain': (1e-4, 1e-2)},
    {'name': 'num_units', 'type': 'discrete', 'domain': (32, 64, 128, 256)},
    {'name': 'dropout_rate', 'type': 'continuous', 'domain': (0, 0.5)},
    {'name': 'l2_weight', 'type': 'continuous', 'domain': (1e-5, 1e-3)},
    {'name': 'batch_size', 'type': 'discrete', 'domain': (32, 64, 128, 256)}
]

# Run Bayesian Optimization
optimizer = GPyOpt.methods.BayesianOptimization(f=objective_function, domain=domain, maximize=False)
optimizer.run_optimization(max_iter=30)

# Plot the convergence
optimizer.plot_convergence()
plt.savefig('convergence_plot.png')

# Save the optimization report
with open('bayes_opt.txt', 'w') as f:
    f.write(f"Optimal parameters: {optimizer.x_opt}\n")
    f.write(f"Optimal value: {-optimizer.fx_opt}\n")
    f.write("\nOptimization History:\n")
    for i, (x, y) in enumerate(zip(optimizer.X, optimizer.Y)):
        f.write(f"Iteration {i+1}:\n")
        f.write(f"  Parameters: {x}\n")
        f.write(f"  Validation Accuracy: {-y[0]:.4f}\n")

print("Optimization completed. Results saved in 'bayes_opt.txt' and 'convergence_plot.png'.")