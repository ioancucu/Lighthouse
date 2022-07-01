from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from tensorflow import keras

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os



class CTCLayer(keras.layers.Layer):
    def __init__(self, name=None, **kwargs):
        super().__init__(name=name, **kwargs)
        self.loss_fn = keras.backend.ctc_batch_cost

    def call(self, y_true, y_pred):
        batch_len = tf.cast(tf.shape(y_true)[0], dtype="int64")
        input_length = tf.cast(tf.shape(y_pred)[1], dtype="int64")
        label_length = tf.cast(tf.shape(y_true)[1], dtype="int64")

        input_length = input_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        label_length = label_length * tf.ones(shape=(batch_len, 1), dtype="int64")
        loss = self.loss_fn(y_true, y_pred, input_length, label_length)
        self.add_loss(loss)

        # At test time, just return the computed predictions.
        return y_pred

class ProcessImages():

    def __init__(self):
        self.model = "C:\\Users\\cucui\\AppData\\Local\\Programs\\Python\\Python39\\ThisCyberware\\Lighthouse7\\Lighthouse7\\SavedModelHW\\handwrittenenus.h5"
        self.base_path = "C:\\Users\\cucui\\AppData\\Local\\Programs\\Python\\Python39\\ThisCyberware\\TrainingDatasets\\OCR\\IAM_Words\\IAM_Words\\"
    def initModel(self):
        new_model = tf.keras.models.load_model(self.model, custom_objects={"CTCLayer": CTCLayer})
        new_model.summary()
    def run(self):
        self.initModel()


    

pimage = ProcessImages()
pimage.run()

plt.show()
