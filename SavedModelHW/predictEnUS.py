from tensorflow.keras.layers.experimental.preprocessing import StringLookup
from tensorflow import keras

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import os
import glob

import cv2
from PIL import Image


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
        self.base_image_path = os.path.join(self.base_path, "words")

    def initModel(self):
        self.words_list = []

        words = open(f"{self.base_path}\\words.txt", "r").readlines()
        for line in words:
            if line[0] == "#":
                continue
            if line.split(" ")[1] != "err":  # We don't need to deal with errored entries.
                self.words_list.append(line)

        len(self.words_list)

       
        self.new_model = tf.keras.models.load_model(self.model, custom_objects={"CTCLayer": CTCLayer})
        self.new_model.summary()
        self.predictsModel = keras.models.Model(self.new_model.get_layer(name="image").input, self.new_model.get_layer(name="dense2").output)

    
    def run(self):
        self.initModel()
        self.predictModel()
    

    def predictModel(self):
        img_Path = "C:\\Users\\cucui\\AppData\\Local\Programs\\Python\\Python39\\ThisCyberware\\Lighthouse7\\Lighthouse7\\documents"
        test_img_paths = glob.glob(img_Path + "\\*.png")
        for batch in test_img_paths:
            img = tf.keras.utils.load_img(batch, target_size=(32, 128))
            img_array = tf.keras.utils.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0) # Create a batch

            preds = self.predictsModel.predict(img_array)
            pred_texts = self.decode_batch_predictions(preds)
           
    
    def preprocess_image(self,image_path, img_size=(128, 32)):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, 1)
        image = self.distortion_free_resize(image, img_size)
        image = tf.cast(image, tf.float32) / 255.0
        return image


    
            

    

pimage = ProcessImages()
pimage.run()

# plt.show()
