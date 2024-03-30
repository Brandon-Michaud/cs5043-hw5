import tensorflow as tf
from keras import Model
from keras.layers import (Input, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D,
                          GlobalMaxPooling2D, SpatialDropout2D, UpSampling2D, Concatenate, Activation)
import numpy as np


def convolution_module_2d(tensor, filters=32, kernel_size=(3, 3), strides=1, padding='same', bias=True,
                     kernel_regularizer=None, batch_normalization=True, activation='elu', p_spatial_dropout=None):
    '''
    Builds a single convolution module on top of given tensor

    :param tensor: The beginning tensor
    :param filters: Number of convolution filters
    :param kernel_size: Size of convolution filters
    :param strides: Length of strides
    :param padding: Padding type; same or valid
    :param bias: Use bias term in convolutions
    :param kernel_regularizer: L1 or L2 regularization
    :param batch_normalization: Use batch normalization
    :param activation: Activation function for convolution
    :param p_spatial_dropout: Probability for spatial dropout
    '''
    tensor = Conv2D(filters=filters, kernel_size=kernel_size, strides=strides,
                    padding=padding, use_bias=bias, kernel_regularizer=kernel_regularizer)(tensor)
    if batch_normalization:
        tensor = BatchNormalization()(tensor)
    tensor = Activation(activation)(tensor)
    tensor = SpatialDropout2D(p_spatial_dropout if p_spatial_dropout is not None else 0)(tensor)
    return tensor


def create_unet_classifier(image_size, nchannels, kernel_size=(3, 3), pool_size=(2, 2), depth=3, conv_per_layer=2,
                           p_spatial_dropout=None, lambda_l2=None, batch_normalization=True, lrate=0.001, n_classes=7,
                           loss=None, metrics=None, padding='same',
                           conv_activation='elu', skip=True):
    '''
    Creates a UNET model

    :param image_size: Size of 2x2 image
    :param nchannels: Number of channels in input image
    :param kernel_size: Size of convolution kernels to use
    :param pool_size: Size of max pooling and up sampling to use
    :param depth: Number of max pooling layers/up sampling layers in model
    :param conv_per_layer: Number of convolution modules per depth level
    :param p_spatial_dropout: Probability of spatial dropout
    :param lambda_l2: L2 regularization strength
    :param batch_normalization: Use batch normalization
    :param lrate: Learning rate
    :param n_classes: Number of output classes
    :param loss: Loss function
    :param metrics: Metrics to record during training and evaluation
    :param padding: Padding to use for convolutions; same or valid
    :param conv_activation: Activation function to use for convolution layers
    :param skip: Add skip connections to UNET
    '''
    # Input layer
    tensor = Input(shape=(image_size[0], image_size[1], nchannels), name='input')
    input_tensor = tensor

    # Determine sizes for convolution filters
    exp = int(np.floor(np.log2(nchannels))) + 1
    sizes = [2 ** i for i in range(exp, exp + depth)]

    # Stack for skip connections
    tensor_stack = []

    # Reduce feature map size
    for i, size in enumerate(sizes):
        # Down convolutions
        for j in range(conv_per_layer):
            factor = 1
            if j == conv_per_layer - 1:
                factor = 2
            tensor = convolution_module_2d(tensor, filters=size*factor, kernel_size=kernel_size, strides=1, padding=padding,
                                      bias=True, kernel_regularizer=tf.keras.regularizers.l2(l2=lambda_l2),
                                      batch_normalization=batch_normalization, activation=conv_activation,
                                      p_spatial_dropout=p_spatial_dropout)

        # Keep track of skip connection
        if skip:
            tensor_stack.append(tensor)

        # Reduce feature map size
        tensor = MaxPooling2D(pool_size=pool_size, strides=pool_size, padding=padding,
                              name=f'max_pool_{i}')(tensor)

    # Bottom convolutions
    for j in range(conv_per_layer):
        factor = 1
        if j == conv_per_layer - 1:
            factor = 2
        tensor = convolution_module_2d(tensor, filters=sizes[-1]*2/factor, kernel_size=kernel_size, strides=1, padding=padding,
                                  bias=True, kernel_regularizer=tf.keras.regularizers.l2(l2=lambda_l2),
                                  batch_normalization=batch_normalization, activation=conv_activation,
                                  p_spatial_dropout=p_spatial_dropout)

    # Increase feature map size
    for i, size in enumerate(reversed(sizes)):
        # Increase feature map size
        tensor = UpSampling2D(size=pool_size)(tensor)

        # Add skip connection
        if skip:
            tensor = Concatenate()([tensor, tensor_stack.pop()])

        # Up convolutions
        for j in range(conv_per_layer):
            factor = 1
            if j == conv_per_layer - 1:
                factor = 2
            tensor = convolution_module_2d(tensor, filters=size/factor, kernel_size=kernel_size, strides=1, padding=padding,
                                      bias=True, kernel_regularizer=tf.keras.regularizers.l2(l2=lambda_l2),
                                      batch_normalization=batch_normalization, activation=conv_activation,
                                      p_spatial_dropout=p_spatial_dropout)

    # Output
    tensor = Dense(n_classes, name='output')(tensor)
    if batch_normalization:
        tensor = BatchNormalization()(tensor)
    tensor = Activation('softmax')(tensor)
    output_tensor = tensor

    # Create model from data flow
    model = Model(inputs=input_tensor, outputs=output_tensor)

    # Optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=lrate, amsgrad=False)

    # Bind the optimizer and the loss function to the model
    if loss is None:
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(loss=loss, optimizer=opt, metrics=metrics)

    return model
