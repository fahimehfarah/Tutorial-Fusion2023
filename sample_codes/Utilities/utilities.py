import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt

class CNNUtilities:
    """
    WRITE HERE
    Public Methods:
        load_the_image_from_the_dataset_folder(self, path: str, image_shape: tuple) -> (np.ndarray, np.ndarray, np.ndarray)

    Private Methods:
        __encode_the_labels(self, label_list: list) -> np.ndarray
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

    def __encode_the_labels(self, label_list: list) -> np.ndarray:
        """
        WRITE HERE
        :param label_list: (list)
        :return:
            encoded_label (np.ndarray)
        """
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

    def convert_the_prediction(self, prediction: np.ndarray, number_of_classes: int, height_of_predicted_image: int, width_of_pred_image: int) -> np.ndarray:
        try:
            temp = prediction.argmax(1)
            color_map = np.array([255, 255, 255, 0, 255, 0, 51, 102, 102, 0, 60, 0, 255, 120, 0, 170, 170, 170],
                                 dtype=np.uint8).reshape(number_of_classes, 3)
            result_holder = np.zeros((temp.shape[0], 3), dtype=np.uint8)
            class_count = [0, 0, 0, 0, 0, 0]

            for i in range(temp.shape[0]):
                if temp[i] == 0:
                    result_holder[i] = color_map[0]
                class_count[0] += 1
                if temp[i] == 1:
                    result_holder[i] = color_map[1]
                    class_count[1] += 1
                if temp[i] == 2:
                    result_holder[i] = color_map[2]
                class_count[2] += 1
                if temp[i] == 3:
                    result_holder[i] = color_map[3]
                class_count[3] += 1
                if temp[i] == 4:
                    result_holder[i] = color_map[4]
                class_count[4] += 1
                if temp[i] == 5:
                    result_holder[i] = color_map[5]
                class_count[5] += 1

            #if self.configuration["verbose"]:
                #print(f"[COUNT] The class count is {class_count}")

            result_holder = result_holder.reshape((height_of_predicted_image, width_of_pred_image, 3))
            return result_holder
        except Exception as ex:
            print(f"[EXCEPTION] Convert predictions throws exception {ex}")

    def plot_some_images(self, how_many_rows: int, how_many_cols: int, list_of_element_to_plot) -> None:

        f, ax = plt.subplots(how_many_rows, how_many_cols)
        counter = 0

        for row in range(how_many_rows):
            for col in range(how_many_cols):
                ax[row, col].imshow(list_of_element_to_plot[counter])
                counter += 1
