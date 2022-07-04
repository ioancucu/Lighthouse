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
        self.base_image_path = os.path.join(self.base_path, "words")

    def initModel(self):
        words_list = []

        words = open(f"{self.base_path}\\words.txt", "r").readlines()
        for line in words:
            if line[0] == "#":
                continue
            if line.split(" ")[1] != "err":  # We don't need to deal with errored entries.
                words_list.append(line)

        len(words_list)

        np.random.shuffle(words_list)
        split_idx = int(0.9 * len(words_list))
        self.train_samples = words_list[:split_idx]
        self.test_samples = words_list[split_idx:]

        val_split_idx = int(0.5 * len(self.test_samples))
        self.validation_samples = self.test_samples[:val_split_idx]
        self.test_samples = self.test_samples[val_split_idx:]
        self.train_img_paths, self.train_labels = self.get_image_paths_and_labels(self.train_samples)
        print (len(words_list))
        print (len(self.train_samples))
        print (len(self.validation_samples))
        print (len(self.test_samples))
        assert len(words_list) == len(self.train_samples) + len(self.validation_samples) + len(
            self.test_samples
        )
        
        self.new_model = tf.keras.models.load_model(self.model, custom_objects={"CTCLayer": CTCLayer})
        self.new_model.summary()
        self.predictsModel = keras.models.Model(self.new_model.get_layer(name="image").input, self.new_model.get_layer(name="dense2").output)

    def initMaxLen(self):
        self.max_len = 0
        self.characters = set()
        for label in self.train_labels:
            label = label.split(" ")[-1].strip()
            for char in label:
                self.characters.add(char)

            self.max_len = max(self.max_len, len(label))

        self.characters= sorted(list(self.characters))    
        self.char_to_num = StringLookup(vocabulary=list(self.characters), mask_token=None)
        self.num_to_char = StringLookup(vocabulary=self.char_to_num.get_vocabulary(), mask_token=None, invert=True)
  

    def run(self):
        self.initModel()
        self.initMaxLen()
        self.predictModel()
    def decode_batch_predictions(self, pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        # Use greedy search. For complex tasks, you can use beam search.
        results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][:, :self.max_len]
        # Iterate over the results and get back the text.
        output_text = []
        for res in results:
            res = tf.gather(res, tf.where(tf.math.not_equal(res, -1)))
            res = tf.strings.reduce_join(self.num_to_char(res)).numpy().decode("utf-8")
            output_text.append(res)
        return output_text   

    def clean_labels(self, labels):
        cleaned_labels = []
        for label in labels:
            label = label.split(" ")[-1].strip()
            cleaned_labels.append(label)
        return cleaned_labels

    def predictModel(self):
        
        test_img_paths, test_labels = self.get_image_paths_and_labels(self.test_samples)

        test_labels_cleaned = self.clean_labels(test_labels)
        test_ds = self.prepare_dataset(test_img_paths, test_labels_cleaned)
        print (test_ds)
        for batch in test_ds.take(1):
            batch_images = batch["image"]
            _, ax = plt.subplots(4, 4, figsize=(15, 8))
            print(batch_images)
            preds = self.predictsModel.predict(batch_images)
            pred_texts = self.decode_batch_predictions(preds)

            for i in range(16):
                img = batch_images[i]
                img = tf.image.flip_left_right(img)
                img = tf.transpose(img, perm=[1, 0, 2])
                img = (img * 255.0).numpy().clip(0, 255).astype(np.uint8)
                img = img[:, :, 0]

                title = f"Prediction: {pred_texts[i]}"
                ax[i // 4, i % 4].imshow(img, cmap="gray")
                ax[i // 4, i % 4].set_title(title)
                ax[i // 4, i % 4].axis("off")

    def prepare_dataset(self, image_paths, labels):
        AUTOTUNE = tf.data.AUTOTUNE
        batch_size = 64
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels)).map(
            self.process_images_labels, num_parallel_calls=AUTOTUNE
        )
        return dataset.batch(batch_size).cache().prefetch(AUTOTUNE)    
    
    def process_images_labels(self, image_path, label):
        image = self.preprocess_image(image_path)
        label = self.vectorize_label(label)
        return {"image": image, "label": label}   

    def preprocess_image(self,image_path, img_size=(128, 32)):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_png(image, 1)
        image = self.distortion_free_resize(image, img_size)
        image = tf.cast(image, tf.float32) / 255.0
        return image


    def vectorize_label(self, label):
        label = self.char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
        length = tf.shape(label)[0]
        pad_amount = self.max_len - length
        label = tf.pad(label, paddings=[[0, pad_amount]], constant_values=99)
        return label         

    def get_image_paths_and_labels(self,samples):
        paths = []
        corrected_samples = []
        for (i, file_line) in enumerate(samples):
            line_split = file_line.strip()
            line_split = line_split.split(" ")

            # Each line split will have this format for the corresponding image:
            # part1/part1-part2/part1-part2-part3.png
            image_name = line_split[0]
            partI = image_name.split("-")[0]
            partII = image_name.split("-")[1]
            img_path = os.path.join(self.base_image_path, partI, partI + "-" + partII, image_name + ".png")
            if os.path.getsize(img_path):
                paths.append(img_path)
                corrected_samples.append(file_line.split("\n")[0])

        return paths, corrected_samples

    def distortion_free_resize(self, image, img_size):
        w, h = img_size
        image = tf.image.resize(image, size=(h, w), preserve_aspect_ratio=True)

        # Check tha amount of padding needed to be done.
        pad_height = h - tf.shape(image)[0]
        pad_width = w - tf.shape(image)[1]

        # Only necessary if you want to do same amount of padding on both sides.
        if pad_height % 2 != 0:
            height = pad_height // 2
            pad_height_top = height + 1
            pad_height_bottom = height
        else:
            pad_height_top = pad_height_bottom = pad_height // 2

        if pad_width % 2 != 0:
            width = pad_width // 2
            pad_width_left = width + 1
            pad_width_right = width
        else:
            pad_width_left = pad_width_right = pad_width // 2

        image = tf.pad(
            image,
            paddings=[
                [pad_height_top, pad_height_bottom],
                [pad_width_left, pad_width_right],
                [0, 0],
            ],
        )

        image = tf.transpose(image, perm=[1, 0, 2])
        image = tf.image.flip_left_right(image)
        return image

    

pimage = ProcessImages()
pimage.run()

plt.show()
