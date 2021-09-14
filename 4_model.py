import pandas as pd
import pickle
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization,\
    MaxPool2D, AvgPool2D, Flatten, Permute, Conv1D, Conv2D
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers
kl =  tfp.distributions.kl_divergence
from functools import partial
efn = tf.keras.applications.efficientnet

def read_waveform_tfrecord(example, shape=[1, 220500, 1]):
    tfrecord_format = {
        'waveform': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }

    example = tf.io.parse_single_example(example, tfrecord_format)

    # Extract content
    waveform = example['waveform']
    label = example['label']

    # Process content
    waveform = tf.io.parse_tensor(waveform, out_type=tf.float32)
    waveform = tf.reshape(waveform, shape=shape)

    return waveform, label

def read_spectrogram_tfrecord(example):
    tfrecord_format = {
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64),
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
        }

    example = tf.io.parse_single_example(example, tfrecord_format)

    # Extract content
    height = example['height']
    width = example['width']
    depth = example['depth']
    image = example['image']
    label = example['label']

    # Process content
    image = tf.io.parse_tensor(image, out_type=tf.float32)
    image = tf.reshape(image, shape=[128, 431, 3])

    return image, label

def read_windowed_spectrogram_tfrecord(example):
    tfrecord_format = {
        'id': tf.io.FixedLenFeature([], tf.float32),
        'height': tf.io.FixedLenFeature([], tf.int64),
        'width': tf.io.FixedLenFeature([], tf.int64),
        'depth': tf.io.FixedLenFeature([], tf.int64),
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
        }

    example = tf.io.parse_single_example(example, tfrecord_format)

    # Extract content
    id = example['id']
    height = example['height']
    width = example['width']
    depth = example['depth']
    image = example['image']
    label = example['label']

    # Process content
    image = tf.io.parse_tensor(image, out_type=tf.float32)
    image = tf.reshape(image, shape=[128, 128, 2])

    return image, label

def load_dataset(filenames, reader=lambda example:\
        read_waveform_tfrecord(example)):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # disable order, increase speed
    dataset = tf.data.TFRecordDataset(
        filenames
    )  # automatically interleaves reads from multiple files
    dataset = dataset.with_options(
        ignore_order
    )  # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(
        partial(reader),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    # returns a dataset of (image, label)
    return dataset

def get_dataset(filenames, reader=lambda example:\
        read_waveform_tfrecord(example), batch_size = 16, classes=50):
    dataset = load_dataset(filenames, reader=reader)
    dataset = dataset.map(lambda x, y: (x, tf.one_hot(y, classes))) #OHE
    dataset = dataset.shuffle(8192)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    return dataset

def len_batched_tf_data(batched_data):
    cum_sum = 0
    for batch in batched_data:
        cum_sum += batch[0].shape[0]
    return cum_sum

def gen_efn_model(input_shape=(128, 431, 3), output_shape=50):
    """ Builds an neural network for environmental sound classification

    Args:
        input_shape (tuple): 3-tuple of the input image dimensions
        output_shape (int): Number of classes in classification problem

    Returns:
        keras model
    """
    # Define model components
    input = Input(shape=input_shape, dtype='float32', name='input')
    base_efn = efn.EfficientNetB4(include_top=False, pooling='avg')
    dropout = Dropout(0.5)
    output = Dense(output_shape, activation='softmax',
                   kernel_regularizer=regularizers.l2(1e-4))

    # Freeze EfficentNet weights
    base_efn.trainable = False

    # Build model and print summary
    model = Sequential([
        input,
        base_efn,
        dropout,
        output
    ])

    model.summary()

    model.compile(Adam(lr=1e-2),
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

    return model

def nll(y_true, y_pred):
    return -y_pred.log_prob(y_true)

def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    return lambda t : tfd.MultivariateNormalDiag(loc=tf.zeros(n),
                                                 scale_diag=tf.ones(n))
def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    return Sequential([
        tfpl.VariableLayer(tfpl.IndependentNormal.params_size(n)),
        tfpl.IndependentNormal(n)
    ])

def gen_bnn_model(prior=prior, posterior=posterior,
                  batch_size=16, input_shape=(128, 431, 3), output_shape=50,
                  loss=nll, optimizer=Adam(1e-2), metrics=['accuracy']):
    model = Sequential([
        Input(shape=input_shape, dtype='float32', name='input'),
        tfpl.Convolution2DReparameterization(8, (9, 9), (3, 3),
                                             activation='relu',
                                             bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
                                             bias_posterior_fn=tfpl.util.default_mean_field_normal_fn()),
        tfpl.Convolution2DReparameterization(16, (5, 5), (2, 2),
                                             activation='relu',
                                             bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
                                             bias_posterior_fn=tfpl.util.default_mean_field_normal_fn()),
        MaxPool2D(),
        tfpl.Convolution2DReparameterization(32, (3, 3), (1, 1),
                                             activation='relu',
                                             bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
                                             bias_posterior_fn=tfpl.util.default_mean_field_normal_fn()),
        MaxPool2D(),
        Permute((3, 2, 1)),
        tfpl.Convolution2DReparameterization(8, (3, 3), (1, 1),
                                             activation='relu',
                                             bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
                                             bias_posterior_fn=tfpl.util.default_mean_field_normal_fn()),
        MaxPool2D(),
        tfpl.Convolution2DReparameterization(16, (3, 3), (1, 1),
                                             activation='relu',
                                             bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
                                             bias_posterior_fn=tfpl.util.default_mean_field_normal_fn()),
        AvgPool2D(),
        Flatten(),
        Dropout(0.2),
        tfpl.DenseVariational(
            tfpl.OneHotCategorical.params_size(output_shape),
            make_posterior_fn=posterior,
            make_prior_fn=prior,
            kl_weight=1/batch_size,
            kl_use_exact=False
        ),
        tfpl.OneHotCategorical(output_shape,
                               convert_to_tensor_fn=tfd.Distribution.mode)
    ])

    model.summary()

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)

    return model

def gen_simple_cnn(input_shape=(128, 431, 3), output_shape=50,
                  loss=nll, optimizer=RMSprop(), metrics=['accuracy']):
    model = Sequential([
        Input(shape=input_shape, dtype='float32', name='input'),
        Conv2D(8, (9, 9), (3, 5), activation='relu'),
        MaxPool2D(),
        Conv2D(16, (5, 5), (2, 3), activation='relu'),
        MaxPool2D(),
        Flatten(),
        Dense(tfpl.OneHotCategorical.params_size(output_shape)),
        tfpl.OneHotCategorical(output_shape,
                               convert_to_tensor_fn=tfd.Distribution.mode)
    ])

    model.summary()

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)

    return model

def gen_simple_bnn(input_shape=(128, 431, 1), output_shape=50,
                  loss=nll, optimizer=RMSprop(), metrics=['accuracy'],
                   n=1200):

    divergence_fn = lambda q, p, _: tfd.kl_divergence(q, p) / n

    model = Sequential([
        Input(shape=input_shape, dtype='float32', name='input'),
        BatchNormalization(),
        tfpl.Convolution2DReparameterization(
            filters=8, kernel_size=16, strides=(4, 8),
            activation='relu',
            kernel_prior_fn=tfpl.default_multivariate_normal_fn,
            kernel_posterior_fn=tfpl.default_mean_field_normal_fn(
                is_singular=False),
            kernel_divergence_fn=divergence_fn,
            bias_prior_fn=tfpl.default_multivariate_normal_fn,
            bias_posterior_fn=tfpl.default_mean_field_normal_fn(
                is_singular=False),
            bias_divergence_fn=divergence_fn
        ),
        tfpl.Convolution2DReparameterization(
            filters=16, kernel_size=8, strides=(2, 4),
            activation='relu',
            kernel_prior_fn=tfpl.default_multivariate_normal_fn,
            kernel_posterior_fn=tfpl.default_mean_field_normal_fn(
                is_singular=False),
            kernel_divergence_fn=divergence_fn,
            bias_prior_fn=tfpl.default_multivariate_normal_fn,
            bias_posterior_fn=tfpl.default_mean_field_normal_fn(
                is_singular=False),
            bias_divergence_fn=divergence_fn
        ),
        MaxPool2D(pool_size=4),
        Flatten(),
        BatchNormalization(),
        Dropout(0.2),
        tfpl.DenseReparameterization(
            units=tfpl.OneHotCategorical.params_size(output_shape),
            activation=None,
            kernel_prior_fn=tfpl.default_multivariate_normal_fn,
            kernel_posterior_fn=tfpl.default_mean_field_normal_fn(
                is_singular=False),
            kernel_divergence_fn=divergence_fn,
            bias_prior_fn=tfpl.default_multivariate_normal_fn,
            bias_posterior_fn=tfpl.default_mean_field_normal_fn(
                is_singular=False),
            bias_divergence_fn=divergence_fn
        ),
        tfpl.OneHotCategorical(output_shape)
    ])

    model.summary()

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)

    return model

def gen_acdnet(input_shape=(1, 220500, 1), num_classes=50,
               loss='categorical_crossentropy',
               optimizer=Adam(),
               metrics=['accuracy']):
    model = Sequential([
        Input(shape=input_shape, dtype='float32', name='input'),
        Conv2D(filters=8, kernel_size=(1, 9), strides=(1, 2),
               activation='relu'),
        Conv2D(filters=64, kernel_size=(1, 5), strides=(1, 2),
               activation='relu'),
        MaxPool2D(pool_size=(1, 110), strides=(1, 110)),
        Permute((3, 2, 1)),
        Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),
               activation='relu'),
        MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
               activation='relu'),
        Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
               activation='relu'),
        MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),
               activation='relu'),
        Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),
               activation='relu'),
        MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),
               activation='relu'),
        Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1),
               activation='relu'),
        MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1),
               activation='relu'),
        Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1),
               activation='relu'),
        MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        Dropout(0.2),
        Conv2D(filters=num_classes, kernel_size=(1, 1), strides=(1, 1),
               activation='relu'),
        AvgPool2D(pool_size=(2, 4), strides=(2, 4)),
        Flatten(),
        Dense(units=num_classes, activation='softmax')
    ])


    model.summary()

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)

    return model

def gen_acdnet_insp(input_shape=(1, 220500, 1), num_classes=50,
                    loss='categorical_crossentropy',
                    optimizer=SGD(learning_rate=0.1, nesterov=0.9),
                    metrics=['accuracy'],
                    reg = 5e-4):
    model = Sequential([
        Input(shape=input_shape, dtype='float32', name='input'),
        BatchNormalization(),
        Conv2D(filters=16, kernel_size=(1, 9), strides=(1, 3),
               activation='relu',
               kernel_regularizer=regularizers.l2(reg)),
        Conv2D(filters=32, kernel_size=(1, 5), strides=(1, 3),
               activation='relu',
               kernel_regularizer=regularizers.l2(reg)),
        Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 3),
               activation='relu',
               kernel_regularizer=regularizers.l2(reg)),
        MaxPool2D(pool_size=(1, 100), strides=(1, 100)),
        BatchNormalization(),
        Permute((3, 2, 1)),
        Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1),
               activation='relu',
               kernel_regularizer=regularizers.l2(reg)),
        Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1),
               activation='relu',
               kernel_regularizer=regularizers.l2(reg)),
        MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),
               activation='relu',
               kernel_regularizer=regularizers.l2(reg)),
        Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),
               activation='relu',
               kernel_regularizer=regularizers.l2(reg)),
        MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
        BatchNormalization(),
        Dropout(0.5),
        Conv2D(filters=num_classes, kernel_size=(2, 2), strides=(2, 2),
               activation='relu',
               kernel_regularizer=regularizers.l2(reg)),
        AvgPool2D(pool_size=(6, 8), strides=(6, 8)),
        Flatten(),
        BatchNormalization(),
        Dense(units=num_classes, activation='softmax',
               kernel_regularizer=regularizers.l2(reg))
    ])
    model.summary()

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)

    return model

def gen_acdnet_insp_2(input_shape=(1, 33333, 1), num_classes=50,
                      loss='categorical_crossentropy',
                      optimizer=SGD(learning_rate=0.1, nesterov=0.9),
                      metrics=['accuracy'],
                      reg = 5e-4):
    model = Sequential([
        Input(shape=input_shape, dtype='float32', name='input'),
        BatchNormalization(),
        Conv2D(filters=8, kernel_size=(1, 9), strides=(1, 2),
               activation='relu',
               kernel_regularizer=regularizers.l2(reg)),
        Conv2D(filters=64, kernel_size=(1, 5), strides=(1, 2),
               activation='relu',
               kernel_regularizer=regularizers.l2(reg)),
        MaxPool2D(pool_size=(1, 49), strides=(1, 49)),
        BatchNormalization(),
        Permute((3, 2, 1)),
        Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),
               activation='relu',
               kernel_regularizer=regularizers.l2(reg),
               padding='same'),
        Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),
               activation='relu',
               kernel_regularizer=regularizers.l2(reg),
               padding='same'),
        MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        BatchNormalization(),
        Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
               activation='relu',
               kernel_regularizer=regularizers.l2(reg),
               padding='same'),
        Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1),
               activation='relu',
               kernel_regularizer=regularizers.l2(reg),
               padding='same'),
        MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        BatchNormalization(),
        Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),
               activation='relu',
               kernel_regularizer=regularizers.l2(reg),
               padding='same'),
        Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1),
               activation='relu',
               kernel_regularizer=regularizers.l2(reg),
               padding='same'),
        MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='same'),
        BatchNormalization(),
        Dropout(0.2),
        Conv2D(filters=num_classes, kernel_size=(1, 1), strides=(1, 1),
               activation='relu',
               kernel_regularizer=regularizers.l2(reg),
               padding='same'),
        AvgPool2D(pool_size=(8, 22), strides=(8, 22), padding='same'),
        Flatten(),
        BatchNormalization(),
        Dense(units=num_classes, activation='softmax',
               kernel_regularizer=regularizers.l2(reg))
    ])
    model.summary()

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)


    return model

def gen_simp(input_shape=(1, 220500, 1), num_classes=50,
                    loss='categorical_crossentropy',
                    optimizer=Adam(),
                    metrics=['accuracy'],
                    reg = 1e-2):
    model = Sequential([
        Input(shape=input_shape, dtype='float32', name='input'),
        BatchNormalization(),
        Conv2D(filters=16, kernel_size=(1, 21), strides=(1, 5),
               activation='relu',
               kernel_regularizer=regularizers.l2(reg)),
        Conv2D(filters=16, kernel_size=(1, 21), strides=(1, 5),
               activation='relu',
               kernel_regularizer=regularizers.l2(reg)),
        MaxPool2D(pool_size=(1, 9), strides=(1, 9)),
        BatchNormalization(),
        Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 3),
               activation='relu',
               kernel_regularizer=regularizers.l2(reg)),
        Conv2D(filters=32, kernel_size=(1, 7), strides=(1, 3),
               activation='relu',
               kernel_regularizer=regularizers.l2(reg)),
        MaxPool2D(pool_size=(1, 7), strides=(1, 7)),
        BatchNormalization(),
        Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1),
               activation='relu',
               kernel_regularizer=regularizers.l2(reg)),
        Conv2D(filters=64, kernel_size=(1, 3), strides=(1, 1),
               activation='relu',
               kernel_regularizer=regularizers.l2(reg)),
        MaxPool2D(pool_size=(1, 5), strides=(1, 5)),
        BatchNormalization(),
        Flatten(),
        Dropout(0.2),
        Dense(units=num_classes, activation='softmax',
              kernel_regularizer=regularizers.l2(reg))
    ])

    model.summary()

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)

    return model

def train_model(model, data, validation_data=None, epochs=1000,
                callbacks=None):
    model.fit(data,
              validation_data=validation_data,
              epochs=epochs,
              callbacks=callbacks)

def evaluate_model(model, data):
    results = model.evaluate(data)
    with open('Models/Results/result.txt', 'w') as output:
        output.write(str(results))

def train_acdnet():
    data_train = get_dataset(list(set().union(*[
        [f'Data/esc50_wav_acdnet_tfr/{dir}/fold_{i}.tfrecords' for i in [1, 2, 3, 4]]
        for dir in ['raw', 'aug']])),
                             reader=lambda example:\
        read_waveform_tfrecord(example, [1, 33333, 1]),
                             batch_size=64)
    data_val = get_dataset('Data/esc50_wav_acdnet_tfr/raw/fold_5.tfrecords',
                           reader=lambda example: \
                               read_waveform_tfrecord(example, [1, 33333, 1]),
                           batch_size=64)

    model = gen_acdnet_insp_2(reg=5e-4, optimizer=Adam())

    def scheduler(epoch, lr):
        if epoch < 10:
            return 0.01
        elif epoch < 400:
            return 0.1
        elif epoch < 800:
            return 0.01
        elif epoch < 1200:
            return 0.001
        elif epoch < 1600:
            return 0.0001
        else:
            return 0.00001

    train_model(model, data_train, data_val, epochs=1000)

def train_simp():
    data_train = get_dataset(list(set().union(*[
        [f'Data/esc50_wav_tfr/{dir}/fold_{i}.tfrecords' for i in [1, 2, 3, 4]]
        for dir in ['raw', 'aug']])),
                             reader=read_waveform_tfrecord,
                             batch_size=64)
    data_val = get_dataset('Data/esc50_wav_tfr/raw/fold_5.tfrecords',
                           reader=read_waveform_tfrecord,
                           batch_size=64)

    model = gen_simp()

    def scheduler(epoch, lr):
        return 0.01

    train_model(model, data_train, data_val, epochs=500,
                callbacks=[tf.keras.callbacks.LearningRateScheduler(scheduler)])

def gen_wind_mel_cnn(input_shape=(128, 128, 2), num_classes=50,
                     loss='categorical_crossentropy',
                     optimizer=SGD(learning_rate=0.002, nesterov=0.9),
                     metrics=['accuracy'],
                     reg = 1e-3,
                     dor=0.5):

    model = Sequential([
        Input(shape=input_shape, dtype='float32'),
        Conv2D(filters=80, kernel_size=(120, 5), strides=(1, 1),
               activation='relu',
               kernel_regularizer=regularizers.l2(reg)),
        MaxPool2D(pool_size=(9, 3), strides=(1, 3)),
        Dropout(rate=dor),
        Conv2D(filters=80, kernel_size=(1, 3), strides=(1, 1),
               activation='relu',
               kernel_regularizer=regularizers.l2(reg)),
        MaxPool2D(pool_size=(1, 3), strides=(1, 3)),
        Flatten(),
        Dense(units=5000, activation='relu',
              kernel_regularizer=regularizers.l2(reg)),
        Dropout(rate=dor),
        Dense(units=5000, activation='relu',
              kernel_regularizer=regularizers.l2(reg)),
        Dropout(rate=dor),
        Dense(units=num_classes, activation='softmax')
    ])

# Test

    model.summary()

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)

    return model

def train_wind_mel_cnn():
    data_train = get_dataset(list(set().union(*[
        [f'Data/esc50_mel_wind_tfr/{dir}/fold_{i}.tfrecords' for i in [1, 2, 3, 4]]
        for dir in ['raw', 'aug']])),
                             reader=read_windowed_spectrogram_tfrecord,
                             batch_size=1024)
    data_val = get_dataset('Data/esc50_mel_wind_tfr/raw/fold_5.tfrecords',
                           reader=read_windowed_spectrogram_tfrecord,
                           batch_size=1024)

    model = gen_wind_mel_cnn()

    train_model(model, data_train, data_val, epochs=100)

def gen_wind_mel_cnn_insp(input_shape=(128, 128, 2), num_classes=50,
                          loss='categorical_crossentropy',
                          optimizer=RMSprop(),
                          metrics=['accuracy'],
                          reg = 1e-4):

    model = Sequential([
        Input(shape=input_shape, dtype='float32'),
        BatchNormalization(),
        Conv2D(filters=16, kernel_size=15, strides=1,
               activation='elu',
               kernel_regularizer=regularizers.l2(reg)),
        MaxPool2D(pool_size=3, strides=3),
        BatchNormalization(),
        Dropout(rate=0.2),
        Conv2D(filters=32, kernel_size=7, strides=1,
               activation='elu',
               kernel_regularizer=regularizers.l2(reg)),
        MaxPool2D(pool_size=2, strides=2),
        BatchNormalization(),
        Dropout(rate=0.2),
        Conv2D(filters=32, kernel_size=5, strides=(1, 1),
               activation='elu',
               kernel_regularizer=regularizers.l2(reg)),
        AvgPool2D(pool_size=2, strides=2),
        BatchNormalization(),
        Dropout(rate=0.2),
        Conv2D(filters=64, kernel_size=3, strides=(1, 1),
               activation='elu',
               kernel_regularizer=regularizers.l2(reg)),
        AvgPool2D(pool_size=2, strides=2),
        Flatten(),
        BatchNormalization(),
        Dropout(rate=0.5),
        Dense(units=128, activation='elu',
              kernel_regularizer=regularizers.l2(reg)),
        BatchNormalization(),
        Dropout(rate=0.5),
        Dense(units=num_classes, activation='softmax')
    ])

    model.summary()

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)

    return model

def gen_wind_mel_bnn_insp(input_shape=(128, 128, 2), num_classes=50,
                          loss=nll,
                          optimizer=RMSprop(learning_rate=0.025),
                          metrics=['accuracy'],
                          reg = 0,
                          prior_scale=1,
                          train_size=1024):

    # Define prior
    def prior(kernel_size, bias_size, dtype=None):
        n = kernel_size + bias_size
        prior_model = Sequential(
            [
                tfp.layers.DistributionLambda(
                    lambda t: tfp.distributions.MultivariateNormalDiag(
                        loc=tf.zeros(n), scale_diag=tf.ones(n) * prior_scale
                    )
                )
            ]
        )
        return prior_model

    def posterior(kernel_size, bias_size, dtype=None):
        n = kernel_size + bias_size
        posterior_model = Sequential(
            [
                tfp.layers.VariableLayer(
                    tfp.layers.IndependentNormal.params_size(n),
                    dtype=dtype
                ),
                tfp.layers.IndependentNormal(n),
            ]
        )
        return posterior_model

    model = Sequential([
        Input(shape=input_shape, dtype='float32'),
        BatchNormalization(),
        Conv2D(filters=16, kernel_size=15, strides=1,
               activation='elu',
               kernel_regularizer=regularizers.l2(reg)),
        MaxPool2D(pool_size=3, strides=3),
        BatchNormalization(),
        Dropout(rate=0.2),
        Conv2D(filters=32, kernel_size=7, strides=1,
               activation='elu',
               kernel_regularizer=regularizers.l2(reg)),
        MaxPool2D(pool_size=2, strides=2),
        BatchNormalization(),
        Dropout(rate=0.2),
        Conv2D(filters=32, kernel_size=5, strides=(1, 1),
               activation='elu',
               kernel_regularizer=regularizers.l2(reg)),
        AvgPool2D(pool_size=2, strides=2),
        BatchNormalization(),
        Dropout(rate=0.2),
        Conv2D(filters=64, kernel_size=3, strides=(1, 1),
               activation='elu',
               kernel_regularizer=regularizers.l2(reg)),
        AvgPool2D(pool_size=2, strides=2),
        Flatten(),
        BatchNormalization(),
        Dropout(rate=0.5),
        Dense(units=128, activation='elu',
              kernel_regularizer=regularizers.l2(reg)),
        BatchNormalization(),
        Dropout(rate=0.5),
        tfp.layers.DenseReparameterization(
            tfpl.OneHotCategorical.params_size(num_classes),
            kernel_posterior_fn=tfpl.util.default_mean_field_normal_fn(),
            kernel_posterior_tensor_fn=(lambda d: d.sample()),
            kernel_prior_fn=tfp.layers.default_multivariate_normal_fn,
            kernel_divergence_fn=(lambda q, p, ignore: kl(q, p)/train_size) ,
            bias_posterior_fn=tfpl.util.default_mean_field_normal_fn(),
            bias_posterior_tensor_fn = (lambda d: d.sample()),
            bias_prior_fn = tfp.layers.default_multivariate_normal_fn,
            bias_divergence_fn = (lambda q, p, ignore: kl(q, p)/train_size)
    ),
        tfpl.OneHotCategorical(num_classes,
                               convert_to_tensor_fn=tfd.Distribution.mode)
    ])

    """
    tfpl.DenseVariational(
        128,
        activation='elu',
        make_posterior_fn=posterior,
        make_prior_fn=prior,
        kl_weight=1 / batch_size,
        kl_use_exact=False
    )
    tfpl.DenseVariational(
            tfpl.OneHotCategorical.params_size(num_classes),
            make_posterior_fn=posterior,
            make_prior_fn=prior,
            kl_weight=1/batch_size,
            kl_use_exact=False
        )
        
    tfp.layers.DenseReparameterization(
        units, activation=None, activity_regularizer=None, trainable=True,
        kernel_posterior_fn=tfp_layers_util.default_mean_field_normal_fn(),
        kernel_posterior_tensor_fn=(lambda d: d.sample()),
        kernel_prior_fn=tfp.layers.default_multivariate_normal_fn,
        kernel_divergence_fn=(lambda q, p, ignore: kl_lib.kl_divergence(q, p)), bias_pos
        terior_fn=tfp_layers_util.default_mean_field_normal_fn(is_singular=True),
        bias_posterior_tensor_fn=(lambda d: d.sample()), bias_prior_fn=None,
        bias_divergence_fn=(lambda q, p, ignore: kl_lib.kl_divergence(q, p)), **kwargs
    )
    """

    model.summary()

    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)

    return model

def train_wind_mel(batch_size, model_generator, epochs, fpath_id,
                   save_via_pickle=False):
    fold_list = list(range(1, 6))
    for fold in range(1, 6):
        # Make a list of folds that exclude the current validation fold
        train_fold_list = fold_list[:fold-1] + fold_list[fold:]

        # Use training fold list to load in training data
        data_train = get_dataset(
            [f'Data/esc50_mel_wind_tfr/aug/fold_{i}.tfrecords' for i in train_fold_list],
            reader=read_windowed_spectrogram_tfrecord,
            batch_size=1024)
        # Load validation data
        data_val = get_dataset(f'Data/esc50_mel_wind_tfr/raw/fold_{fold}.tfrecords',
                               reader=read_windowed_spectrogram_tfrecord,
                               batch_size=batch_size)

        # Generate model
        train_size = 0
        for batch in data_train:
            train_size += batch[0].shape[0]
        model = model_generator(train_size=train_size)

        # Train model and record history
        history = model.fit(data_train,
                  validation_data=data_val,
                  epochs=epochs)

        # Save history
        history_df = pd.DataFrame(history.history)
        history_df.to_csv(f'models/{fpath_id}/hist_fold_{fold}.csv')

        # Save model
        if not save_via_pickle:
            model.save(f'models/{fpath_id}/model_fold_{fold}.hp5')
        else:
            pickling_on = open(f'models/{fpath_id}/model_fold_{fold}.pickle',
                               "wb")
            pickle.dump(model, pickling_on)
            pickling_on.close()

if __name__ == '__main__':
    # Set GPU to use:
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"
    train_wind_mel(batch_size=1024,
                   model_generator=gen_wind_mel_bnn_insp,
                   epochs=1,
                   fpath_id='bnn',
                   save_via_pickle=True)