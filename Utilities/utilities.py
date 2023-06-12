import tensorflow as tf
import numpy as np
import os
from Utilities.configuration import configuration


class CNNUtilities:
    """
    WRITE HERE
    Methods:
        load_the_image_from_the_dataset_folder(self, path: str, image_shape: tuple) -> (np.ndarray, np.ndarray, np.ndarray)
    """
    def __init__(self, configuration: dict, segmentation_index_list: list):
        """
        CNN Constructor
        :param configuration: (dict) configuration dictionary
        :param segmentation_index_list: (list) segmentation index list
        """
        self.configuration = configuration
        self.segmentation_index_list = segmentation_index_list


    def load_the_image_from_the_dataset_folder(self, path: str, image_shape: tuple) -> (np.ndarray, np.ndarray, np.ndarray):
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

            # create the encoded labels
            encoded_label = self.__encode_the_labels(label_list=label_image_list)

            return np.array(rgb_image_list), np.array(depth_image_list), encoded_label
        except Exception as ex:
            print(f"[EXCEPTION] Main throws exception {ex}")

    def __encode_the_labels(self, label_list: list):
        try:
            encoded_label = list()
            for label in label_list:
                # get the shape
                h, w, c = label.shape
                label = np.dot(label.reshape(h * w, c)[:, ], [1, 4, 9])
                for i in range(len(self.segmentation_index_list)):
                    label[label == self.segmentation_index_list[i]] = i
                # Do one hot encoding
                label = (np.arange(len(self.segmentation_index_list)) == label[..., None]) * 1
                # encode
                encoded_label.append(label)
            return np.array(encoded_label)
        except Exception as ex:
            print(f"[EXCEPTION] Encode the labels throws exception {ex}")