import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense

def model(idim):
    "Simple Keras model for testing purposes"
    ml_model = keras.Sequential([keras.layers.Dense(1024, activation='relu',input_shape=(idim,)),
                                 keras.layers.Dropout(0.3),
                                 keras.layers.Dense(512, activation='relu'),
                                 keras.layers.Dropout(0.3),
                                 keras.layers.Dense(256, activation='relu'),
                                 keras.layers.Dropout(0.3),
                                 keras.layers.Dense(1, activation='sigmoid')])
    ml_model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3),
                     loss=keras.losses.BinaryCrossentropy(), metrics=[keras.metrics.BinaryAccuracy(name='accuracy'), keras.metrics.AUC(name='auc')])
    return ml_model
