import os
import argparse
import tensorflow as tf
from Utilities.configuration import configuration, segmentation_index
from Utilities.utilities import CNNUtilities

if __name__ == '__main__':
    NOGPU = False
    if NOGPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
        for i in range(len(physical_devices)):
            tf.config.experimental.set_memory_growth(physical_devices[i], True)

    # create utility instance
    utilities = CNNUtilities(configuration=configuration,
                             segmentation_index_list=segmentation_index)

    # create args
    parser = argparse.ArgumentParser()
    parser.add_argument("-height", help="height of the image in pixel", type=int, required=True)
    parser.add_argument("-width", help="width of the image in pixel", type=int, required=True)
    parser.add_argument("-epochs", help="number of epochs", type=int, required=True)

    # parse args
    args = parser.parse_args()

    # check the args
    # override h
    if args.height:
        h = args.height
    else:
        h = configuration["image_height"]

    # override w
    if args.width:
        w = args.width
    else:
        w = configuration["image_width"]

    # override epochs
    if args.epochs:
        epochs = args.epochs
    else:
        epochs = configuration["epochs"]

    # define the var for the final shape
    image_shape = (h, w)

    # first step load the dataset
    train_rgb, train_depth, train_labels = utilities.load_the_image_from_the_dataset_folder(path="../Dataset/train",
                                                                                            image_shape=image_shape)

    print(f"[TRAIN] RGB {train_rgb.shape} NIR {train_depth.shape} LABEL {train_labels.shape}")

    test_rgb, test_depth, test_labels = utilities.load_the_image_from_the_dataset_folder(path="../Dataset/test", 
                                                                                         image_shape=image_shape)

    print(f"[TEST] RGB {test_rgb.shape} NIR {test_depth.shape} LABEL {test_labels.shape}")

    validation_rgb, validation_depth, validation_labels = utilities.load_the_image_from_the_dataset_folder(path="../Dataset/valid", 
                                                                                                           image_shape=image_shape)

    print(f"[VALID] RGB {validation_rgb.shape} NIR {validation_depth.shape} LABEL {validation_labels.shape}")

