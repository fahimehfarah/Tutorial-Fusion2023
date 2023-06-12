import os
import argparse
import tensorflow as tf
from Utilities.configuration import configuration, segmentation_index
from Utilities.utilities import CNNUtilities
from NNFactory.NNFactoryVGG import NNFactoryWithVGG

if __name__ == '__main__':
    NOGPU = True
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
        configuration["epochs"] = args.epochs
    else:
        configuration["epochs"] = 200

    # define the var for the final shape
    image_shape = (h, w)

    # first step load the datasets
    train_rgb, train_depth, train_labels = utilities.load_the_image_from_the_dataset_folder(path="../Dataset/train",
                                                                                            image_shape=image_shape)

    print(f"[TRAIN] RGB {train_rgb.shape} NIR {train_depth.shape} LABEL {train_labels.shape}")

    test_rgb, test_depth, test_labels = utilities.load_the_image_from_the_dataset_folder(path="../Dataset/test",
                                                                                         image_shape=image_shape)

    print(f"[TEST] RGB {test_rgb.shape} NIR {test_depth.shape} LABEL {test_labels.shape}")

    validation_rgb, validation_depth, validation_labels = utilities.load_the_image_from_the_dataset_folder(path="../Dataset/valid",
                                                                                                           image_shape=image_shape)

    print(f"[VALID] RGB {validation_rgb.shape} NIR {validation_depth.shape} LABEL {validation_labels.shape}")

    # create cnn config
    cnn_configuration = {
        'shape_stream_rgb': train_rgb.shape[1:],
        'kernel_size_stream_rgb': (train_rgb.shape[3], train_rgb.shape[3]),
        'shape_stream_nir': train_depth.shape[1:],
        'kernel_size_stream_nir': (train_depth.shape[3], train_depth.shape[3]),
        'list_of_conv_layers': [128, 256],
        "dropout": 0.2,
        'number_of_classes': len(segmentation_index)
    }

    print(f'[CNN CONFIGURATION] {cnn_configuration}')

    dictionary_of_training = {
        'input_rgb': train_rgb,
        'input_nir': train_depth
    }

    dictionary_of_test = {
        'input_rgb': test_rgb,
        'input_nir': test_depth
    }

    dictionary_of_validation = {
        'input_rgb': validation_rgb,
        'input_nir': validation_depth
    }

    # instance of the model
    cnn_handler = NNFactoryWithVGG(**cnn_configuration)

    # create a late fusion
    cnn_handler.early_fusion()

    # fit the model
    cnn_handler.fit_the_model(x_train=dictionary_of_training,
                              y_train=train_labels,
                              x_validation=dictionary_of_validation,
                              y_validation=validation_labels,
                              epochs=configuration["epochs"])
    # evaluate the model
    cnn_handler.evaluate_the_model(x_test=dictionary_of_test,
                                   y_test=test_labels)

    # predict
    final_predictions = cnn_handler.make_predictions(x_test=dictionary_of_test)

