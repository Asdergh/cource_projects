import tensorflow as tf
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import os

from matplotlib.animation import FuncAnimation


class FaceClassification():

    def __init__(self, input_shape, classification_type, 
                 model_depth, tree_type=False, tree_block_depth=6,
                 tree_blocks_count=3, max_pooling_mode=False, filters_size=(3, 3),
                 filters_count=32, pooling_size=(2, 2) ) -> None:
        
        self.input_shape = input_shape
        self.classification_type = classification_type
        self.model_depth = model_depth

        self.tree_type = tree_type
        self.tree_block_depth = tree_block_depth
        self.tree_blocks_count = tree_blocks_count

        self.max_pooling_mode = max_pooling_mode
        self.filters_size = filters_size
        self.filters_count = filters_count
        self.pooling_size = pooling_size

    def generate_data(self, base_dir):
        
        self.train_dir = os.path.join(base_dir, "train")
        self.validation_dir = os.path.join(base_dir, "validation")
        self.test_dir = os.path.join(base_dir, "test")

        self.train_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            self.train_dir,
            rotation_range=0.12,
            width_shift_range=0.12,
            height_shift_range=0.12,
            horizontal_flip=True
        )

        self.validation_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            self.validation_dir,
            rotation_range=0.12,
            width_shift_range=0.12,
            height_shift_range=0.12,
            vertical_flip=True
        )

        self.test_data_generator = tf.keras.preprocessing.image.ImageDataGenerator(
            self.test_dir
            rotation_range=0.12,
            width_shift_range=0.12,
            height_shift_range=0.12,
            horizontal_flip=True,
            vertical_flip=True
        )


    
    def generate_model(self) -> dict:

        self.model_discription = {}
        self.input_tensor = tf.keras.Input(shape=self.input_shape)


        if self.tree_type is None:

            if self.max_pooling_mode:

                for layer in range(0, self.model_depth, 2):

                    if len(self.model_discription):
                        curent_layer = tf.keras.layers.Conv2D(self.filters_count, self.filters_size, activation="relu")(self.input_tensor)

                    else:
                        curent_layer = tf.keras.layers.Conv2D(self.filters_count, self.filters_size, activation="relu")(self.model_discription[f"layer{layer - 1}"])
                    
                    sub_curent_layer = tf.keras.layers.MaxPool2D(self.pooling_size)
                    self.model_discription[f"layer{layer}"] = curent_layer
                    self.model_discription[f"layer{layer + 1}"] = sub_curent_layer
                
            else:

                for layer in range(0, self.model_depth):

                    if len(self.model_discription):

                        curent_layer = tf.keras.layers.Conv2D(self.filters_count, self.filters_size, activation="relu")(self.input_tensor)
                        
                    else:

                        curent_layer = tf.keras.layes.Conv2D(self.filters_count, self.filters_size, activation="relu")(self.model_discription[f"layer{layer}"])
                        
                    self.model_discription[f"layer{layer}"] = curent_layer
            
            self.model_discription[f"layer{self.model_depth + 1}"] = tf.keras.layers.Flatten()(self.model_discription[f"layer{self.model_depth}"])
            self.model_discription[f"layer{self.model_depth + 2}"] = tf.keras.layers.Dense(123, activation="relu")(self.model_discription[f"layer{self.model_depth + 1}"])
            self.model_discription[f"layer{self.model_depth + 3}"] = tf.keras.layers.Dense(1, activation="sigmoid")(self.model_discription[f"layer{self.model_depth + 2}"])

            self.model = tf.keras.Model(self.input_tensor, self.model_discription.values()[-1])
        
        else:

            for block in range(self.tree_blocks_count):

                if self.max_pooling_mode:

                    for layer in range(0, self.tree_block_depth, 2):
                        
                        if len(self.model_discription[f"block{block}"]) == 0:
                            curent_layer = tf.keras.layers.Conv2D(self.filters_count, self.filters_size, activation="relu")(self.input_tensor)
                        
                        else:
                            curent_layer = tf.keras.layes.Conv2D(self.filters_count, self.filters_size, activation="relu")(self.model_discription[f"block{block}"][f"layer{layer - 1}"])
                        
                        sub_curent_layer = tf.keras.layers.MaxPool2D(self.pooling_size)
                        self.model_discription[f"block{block}"][f"layer{layer}"] = curent_layer
                        self.model_discription[f"block{block}"][f"layers{layer + 1}"] = sub_curent_layer

                else:


                    for layer in range(0, self.tree_block_depth):
                        
                        if len(self.model_discription[f"block{block}"]) == 0:
                            curent_layer = tf.keras.layers.Conv2D(self.filters_count, self.filters_size, activation="relu")(self.input_tensor)
                        
                        else:
                            curent_layer = tf.keras.layes.Conv2D(self.filters_count, self.filters_size, activation="relu")(self.model_discription[f"block{block}"][f"layer{layer - 1}"])
                        
                        self.model_discription[f"block{block}"][f"layer{layer}"] = curent_layer

            layers_to_concatenate = [layers[-1] for layers in self.model_discription.values()]
            concatenated_layer = tf.keras.layers.concatenate(layers_to_concatenate, axis=1)
            self.model_discription[f"block{self.tree_blocks_count + 1}"][f"layer{0}"] = tf.keras.layers.Flatten()(concatenated_layer)
            self.model_discription[f"block{self.tree_blocks_count + 2}"][f"layer{1}"] = tf.keras.layers.Dense(123, activation="relu")\
                (self.model_discription[f"block{self.tree_blocks_count + 1}"][f"layer{0}"])
            self.model_discription[f"block{self.tree_blocks_count + 3}"][f"layer{2}"] = tf.keras.layers.Dense(1, activation="sigmoid")\
                (self.model_discription[f"block{self.tree_blocks_count + 1}"][f"layer{1}"])

            self.model = tf.keras.Model(self.input_tensor, self.model_discription[f"block{self.tree_blocks_count + 1}"][f"layer{2}"])

            self.model.compile(
                optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.01),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=tf.keras.metrics.Accuracy())

            return self.model_discription
    
    def fit_model(self):

        pass

                
                
            


        
