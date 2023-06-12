import tensorflow as tf
import numpy as np
import os
from Utilities.configuration import configuration


class CNNUtilities():
    def __init__(self, configuration: dict):
        self.configuration = configuration
        pass

    def load_the_image_from_the_dataset_folder(self, path: str, image_shape: tuple) -> (np.ndarray, np.ndarray, list):
        """
        WRITE HERE
        :param path: (str)
        :param image_shape: (tuple)
        :return:
            rbg_image_list: (np.array)
            depth_image_list: (np.array)
            label_image_list (list)
        """
        try:
            rgb_image_list, depth_image_list, label_image_list = list(), list(), list()

            for filename_I_need in os.listdir(f"{path}/rgb"):
                # load the image as obj
                rgb_image = tf.keras.utils.load_img(f"{path}/rgb/{filename_I_need}", target_size=image_shape)
                depth_image = tf.keras.utils.load_img(f"{path}/depth_color/{filename_I_need}", target_size=image_shape)
                label_image = tf.keras.utils.load_img(f"{path}/GT_color/{filename_I_need}", target_size=image_shape)

                rgb_image = tf.keras.utils.img_to_array(rgb_image)
                depth_image = tf.keras.utils.img_to_array(depth_image)
                label_image = tf.keras.utils.img_to_array(label_image)

                if self.configuration["VERBOSE"]:
                    print(f"[RGB] Loaded {rgb_image.shape} - [DEPTH] {depth_image.shape} [GT] {label_image.shape}")

                rgb_image_list.append(rgb_image.astype('uint8'))
                depth_image_list.append(depth_image.astype('uint8'))
                label_image_list.append(label_image.astype('uint8'))

            return np.array(rgb_image_list), np.array(depth_image_list), label_image_list
        except Exception as ex:
            print(f"[EXCEPTION] Main throws exception {ex}")