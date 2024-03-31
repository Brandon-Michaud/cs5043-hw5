import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Embedding


def create_simple_cnn(input_size,
                      n_classes,
                      n_tokens,
                      n_embedding,
                      conv_layers,
                      kernel_sizes,
                      dense_layers,
                      pool_size=2,
                      padding='valid',
                      activation_conv=None,
                      activation_dense=None,
                      lambda_regularization=None,
                      spatial_dropout=None,
                      dropout=None,
                      batch_normalization=False,
                      grad_clip=None,
                      lrate=0.0001,
                      loss=None,
                      metrics=None):
    # Regularization
    if lambda_regularization is not None:
        lambda_regularization = tf.keras.regularizers.l2(lambda_regularization)

    model = Sequential()
    model.add(Embedding(input_dim=n_tokens, output_dim=n_embedding, input_length=input_size))
    for filters, kernel_size in zip(conv_layers, kernel_sizes):
        model.add(Conv1D(filters=filters, kernel_size=kernel_size, strides=1,
                         padding=padding, use_bias=True, activation=activation_conv,
                         kernel_regularizer=tf.keras.regularizers.l2(l2=lambda_regularization)))
        model.add(MaxPooling1D(pool_size=pool_size, strides=pool_size, padding=padding))

    model.add(GlobalMaxPooling1D())

    for n_neurons in dense_layers:
        model.add(Dense(units=n_neurons,
                        activation=activation_dense,
                        use_bias=True,
                        kernel_regularizer=lambda_regularization))

    model.add(Dense(units=n_classes,
                    activation='softmax',
                    kernel_initializer='random_uniform',
                    kernel_regularizer=lambda_regularization))

    # The optimizer determines how the gradient descent is to be done
    opt = keras.optimizers.Adam(learning_rate=lrate, amsgrad=False, clipnorm=grad_clip)

    if loss is None:
        loss = tf.keras.losses.SparseCategoricalCrossentropy()
    model.compile(loss=loss, optimizer=opt, metrics=metrics)

    return model
