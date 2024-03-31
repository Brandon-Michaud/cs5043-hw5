import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Embedding


def create_simple_rnn(input_size,
                      n_classes,
                      n_tokens,
                      n_embedding,
                      rnn_layers,
                      dense_layers,
                      activation_rnn=None,
                      activation_dense=None,
                      return_sequences=False,
                      unroll=True,
                      lambda_regularization=None,
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
    for n_neurons in rnn_layers:
        model.add(SimpleRNN(n_neurons,
                            activation=activation_rnn,
                            use_bias=True,
                            return_sequences=return_sequences,
                            kernel_initializer='random_uniform',
                            bias_initializer='zeros',
                            kernel_regularizer=lambda_regularization,
                            unroll=unroll))
    for n_neurons in dense_layers:
        model.add(Dense(units=n_neurons,
                        activation=activation_dense,
                        use_bias=True,
                        kernel_initializer='random_uniform',
                        bias_initializer='zeros',
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
