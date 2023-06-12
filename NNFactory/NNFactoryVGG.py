import numpy as np
import tensorflow as tf


class NNFactoryWithVGG:
    def __init__(self, shape_stream_rgb, kernel_size_stream_rgb, shape_stream_nir, kernel_size_stream_nir,
                 list_of_conv_layers, dropout, number_of_classes):
        """
        Class constructor
        :param shape_stream_rgb:
        :param kernel_size_stream_rgb:
        :param shape_stream_nir:
        :param kernel_size_stream_nir:
        :param list_of_conv_layers:
        :param dropout:
        :param number_of_classes:
        """
        self.shape_stream_rgb = shape_stream_rgb
        self.shape_stream_nir = shape_stream_nir
        self.kernel_size_stream_rgb = kernel_size_stream_rgb
        self.kernel_size_stream_nir = kernel_size_stream_nir
        self.list_of_conv_layers = list_of_conv_layers
        self.dropout = dropout
        self.number_of_classes = number_of_classes
        self.model = None
        self.model_history = None

    def early_fusion(self):
        # define the model
        input_rgb = tf.keras.Input(self.shape_stream_rgb, name="input_rgb")
        input_nir = tf.keras.Input(self.shape_stream_nir, name="input_nir")

        # concatenate layers
        merged_branch = tf.keras.layers.Concatenate(axis=-1)([input_rgb, input_nir])

        unique_vgg = tf.keras.applications.VGG16(weights=None,
                                                 include_top=False,
                                                 input_shape=(self.shape_stream_rgb[0],self.shape_stream_rgb[1], self.shape_stream_rgb[2] + self.shape_stream_rgb[2] ))

        # change the model name cos you can not use two time
        unique_vgg._name = "vgg16_unique"
        # change the name because thwy should be unique
        for i, l in enumerate(unique_vgg.layers):
            l._name = f'layer_unique_{i}'

        merged_branch = unique_vgg(merged_branch)

        # do the others layer
        for unit in self.list_of_conv_layers:
            merged_branch = tf.keras.layers.Conv2D(unit, self.kernel_size_stream_rgb,
                                                strides=(1, 1),
                                                padding='same',
                                                activation='relu')(merged_branch)

            # make drop out
        if self.dropout is not None:
            merged_branch = tf.keras.layers.Dropout(self.dropout)(merged_branch)


        # deconv layer
        deconv_last_layer = tf.keras.layers.Conv2DTranspose(self.number_of_classes, (64, 64),
                                                            strides=(32, 32),
                                                            padding='same',
                                                            activation='relu',
                                                            kernel_initializer='glorot_normal')(merged_branch)

        # reshape the model
        reshape_layer = tf.keras.layers.Reshape((self.shape_stream_rgb[0] * self.shape_stream_rgb[1], self.number_of_classes))(deconv_last_layer)
        # output layer
        output_layer = tf.keras.layers.Activation('softmax')(reshape_layer)

        # get the model
        self.model = tf.keras.Model(inputs=[input_rgb, input_nir], outputs=[output_layer])

        # compile the model
        self.__compile_the_model()



    def middle_fusion(self):
        # define the model
        input_rgb = tf.keras.Input(self.shape_stream_rgb, name="input_rgb")
        # get the vgg
        vgg_model_rgb = tf.keras.applications.VGG16(weights='imagenet', include_top=False)

        # change the model name cos you can not use two time
        vgg_model_rgb._name = "vgg16_rgb"
        # change the name because thwy should be unique
        for i, l in enumerate(vgg_model_rgb.layers):
            l._name = f'layer_rgb_{i}'

        # connect input to first layer
        rgb_branch = vgg_model_rgb(input_rgb)

        # same for nir
        input_nir = tf.keras.Input(self.shape_stream_nir, name="input_nir")
        vgg_model_nir = tf.keras.applications.VGG16(weights='imagenet', include_top=False)

        # change the model name cos you can not use two time
        vgg_model_nir._name = "vgg16_nir"
        for i, l in enumerate(vgg_model_nir.layers):
            l._name = f"layer_nir_{i}"

        nir_branch = vgg_model_nir(input_nir)

        # we should concat here
        merged_branch = tf.keras.layers.Concatenate(axis=-1)([rgb_branch, nir_branch])

        # do the others layer
        for unit in self.list_of_conv_layers:
            merged_branch = tf.keras.layers.Conv2D(unit, self.kernel_size_stream_rgb,
                                                strides=(1, 1),
                                                padding='same',
                                                activation='relu')(merged_branch)

        # make drop out
        if self.dropout is not None:
            merged_branch = tf.keras.layers.Dropout(self.dropout)(merged_branch)

        # deconv layer
        deconv_last_layer = tf.keras.layers.Conv2DTranspose(self.number_of_classes, (64, 64),
                                                            strides=(32, 32),
                                                            padding='same',
                                                            activation='relu',
                                                            kernel_initializer='glorot_normal')(merged_branch)
        # reshape the model
        reshape_layer = tf.keras.layers.Reshape((self.shape_stream_rgb[0] * self.shape_stream_rgb[1], self.number_of_classes))(deconv_last_layer)
        # output layer
        output_layer = tf.keras.layers.Activation('softmax')(reshape_layer)

        # get the model
        self.model = tf.keras.Model(inputs=[input_rgb, input_nir], outputs=[output_layer])

        # compile the model
        self.__compile_the_model()

    def late_fusion(self):
        # define the model
        input_rgb = tf.keras.Input(self.shape_stream_rgb, name="input_rgb")
        # get the vgg
        vgg_mdel_rgb = tf.keras.applications.VGG16(weights='imagenet', include_top=False)

        # change the model name cos you can not use two time
        vgg_mdel_rgb._name = "vgg16_rgb"
        # change the name because thwy should be unique
        for i, l in enumerate(vgg_mdel_rgb.layers):
            l._name = f'layer_rgb_{i}'

        # connect input to first layer
        rgb_branch = vgg_mdel_rgb(input_rgb)

        # same for nir
        input_nir = tf.keras.Input(self.shape_stream_nir, name="input_nir")
        vgg_model_nir = tf.keras.applications.VGG16(weights='imagenet', include_top=False)

        # change the model name cos you can not use two time
        vgg_model_nir._name = "vgg16_nir"
        for i, l in enumerate(vgg_model_nir.layers):
            l._name = f"layer_nir_{i}"

        nir_branch = vgg_model_nir(input_nir)

        # do the others layer
        for unit in self.list_of_conv_layers:
            rgb_branch = tf.keras.layers.Conv2D(unit, self.kernel_size_stream_rgb,
                                                strides=(1, 1),
                                                padding='same',
                                                activation='relu')(rgb_branch)

            nir_branch = tf.keras.layers.Conv2D(unit, self.kernel_size_stream_nir,
                                                strides=(1, 1),
                                                padding='same',
                                                activation='relu')(nir_branch)

            # make drop out
        if self.dropout is not None:
            rgb_branch = tf.keras.layers.Dropout(self.dropout)(rgb_branch)
            nir_branch = tf.keras.layers.Dropout(self.dropout)(nir_branch)

        # we should concat here
        merged_branch = tf.keras.layers.Concatenate(axis=-1)([rgb_branch, nir_branch])

        # deconv layer
        deconv_last_layer = tf.keras.layers.Conv2DTranspose(self.number_of_classes, (64, 64),
                                                            strides=(32, 32),
                                                            padding='same',
                                                            activation='relu',
                                                            kernel_initializer='glorot_normal')(merged_branch)
        # reshape the model
        reshape_layer = tf.keras.layers.Reshape((self.shape_stream_rgb[0] * self.shape_stream_rgb[1], self.number_of_classes))(deconv_last_layer)
        # output layer
        output_layer = tf.keras.layers.Activation('softmax')(reshape_layer)

        # get the model
        self.model = tf.keras.Model(inputs=[input_rgb, input_nir], outputs=[output_layer])

        # compile the model
        self.__compile_the_model()

    def __compile_the_model(self) -> None:
        self.model.compile(tf.keras.optimizers.legacy.SGD(learning_rate=0.008, decay=1e-6, momentum=0.9, nesterov=True),
                            loss='categorical_crossentropy',
                            metrics=['accuracy'])

    def fit_the_model(self, x_train: dict, y_train: dict, x_validation: dict = None, y_validation: dict = None, epochs: int= 200) -> None:
        """

        :param x_train:
        :param y_train:
        :param x_validation:
        :param y_validation:
        :param epochs:
        """
        # check if the model is not None
        if self.model is None:
            raise Exception("[EXCEPTION] The model can not be null")

        history = self.model.fit(x_train,
                                y_train,
                                validation_data=(x_validation, y_validation),
                                epochs=epochs,
                                steps_per_epoch=100)

        self.model_history = history

    def evaluate_the_model(self, x_test: dict, y_test: dict) -> None:
        """

        :param x_test:
        :param y_test:
        """
        # check if the model is not None
        if self.model is None:
            raise Exception("[EXCEPTION] The model can not be null")

        # score the model
        scores = self.model.evaluate(x_test, y_test)
        print(f"[SCORES] The score is {scores[1] * 100}")

    def make_predictions(self, x_test: dict) -> np.ndarray:
        """
        This function is in charge to make prediuction from the trained model
        :param x_test: (dict)
        :return:
            predictions: np.ndarray
        """
        # check if the model is not None
        if self.model is None:
            raise Exception("[EXCEPTION] The model can not be null")
        prediction = self.model.predict(x_test)
        return prediction
