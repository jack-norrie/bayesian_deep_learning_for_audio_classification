import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization,\
    MaxPool2D, AvgPool2D, Flatten, Conv2D
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import RMSprop
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


def gen_wind_mel_cnn_insp(input_shape=(128, 128, 2), num_classes=50,
                          loss='categorical_crossentropy',
                          optimizer=RMSprop(),
                          metrics=['accuracy'],
                          reg=1e-4,
                          train_size=None):
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
                          loss='categorical_crossentropy',
                          optimizer=RMSprop(learning_rate=0.01),
                          metrics=['accuracy'],
                          reg=0,
                          prior_scale=1,
                          train_size=None):
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
            activation='softmax',
            kernel_posterior_fn=tfpl.util.default_mean_field_normal_fn(),
            kernel_posterior_tensor_fn=(lambda d: d.sample()),
            kernel_prior_fn=tfp.layers.default_multivariate_normal_fn,
            kernel_divergence_fn=(lambda q, p, ignore: kl(q, p) / train_size),
            bias_posterior_fn=tfpl.util.default_mean_field_normal_fn(),
            bias_posterior_tensor_fn=(lambda d: d.sample()),
            bias_prior_fn=tfp.layers.default_multivariate_normal_fn,
            bias_divergence_fn=(lambda q, p, ignore: kl(q, p) / train_size)
        )
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
                   save_model=True, make_preds=True, prob_model=False,
                   num_ensembles=1):
    fold_list = list(range(1, 6))
    for fold in range(1, 6):
        # Make a list of folds that exclude the current validation fold
        train_fold_list = fold_list[:fold - 1] + fold_list[fold:]

        # Use training fold list to load in training data
        data_train = get_dataset(
            [f'Data/esc50_mel_wind_tfr/aug/fold_{i}.tfrecords' for i in
             train_fold_list],
            reader=read_windowed_spectrogram_tfrecord,
            batch_size=1024)
        # Load validation data
        data_val = get_dataset(
            f'Data/esc50_mel_wind_tfr/raw/fold_{fold}.tfrecords',
            reader=read_windowed_spectrogram_tfrecord,
            batch_size=batch_size)

        train_size = 0
        for batch in data_train:
            # Generate model
            train_size += batch[0].shape[0]

        for ensemble in range(1, num_ensembles + 1):
            model = model_generator(train_size=train_size)

            # Train model and record history
            history = model.fit(data_train,
                                validation_data=data_val,
                                epochs=epochs)

            # Save history
            history_df = pd.DataFrame(history.history)
            history_df.to_csv(
                f'models/{fpath_id}/hist_fold_{ensemble}_{fold}.csv')

            # Save model
            if save_model:
                model.save(
                    f'models/{fpath_id}/model_fold_{ensemble}_{fold}.hp5')

            # Save predictions
            if make_preds:
                # unshuffle and unbatch validation set
                data_val_fresh = load_dataset(
                    f'Data/esc50_mel_wind_tfr/raw/fold_{fold}.tfrecords',
                    reader=read_windowed_spectrogram_tfrecord)
                preds = []
                for example in data_val_fresh.batch(1):
                    if not prob_model:
                        preds.append(model(example[0]).numpy()[0])
                    else:
                        # Make 100 predicitons for the input
                        example_preds = [model(example[0]) for _ in range(10)]
                        example_preds = np.stack(example_preds)
                        vpd = np.mean(example_preds, axis=0)
                        preds.append(vpd[0])
            preds = np.stack(preds)
            np.save(f'models/{fpath_id}/preds_fold_{ensemble}_{fold}.npy',
                    preds)


if __name__ == '__main__':
    # Set GPU to use:
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = '1'

    # CNN
    train_wind_mel(batch_size=1024,
                   model_generator=gen_wind_mel_cnn_insp,
                   epochs=100,
                   fpath_id='cnn',
                   save_model=False,
                   make_preds=True,
                   prob_model=False,
                   num_ensembles=1)

    # BNN
    train_wind_mel(batch_size=1024,
                   model_generator=gen_wind_mel_bnn_insp,
                   epochs=100,
                   fpath_id='bnn',
                   save_model=False,
                   make_preds=True,
                   prob_model=True,
                   num_ensembles=1)

    # CNN Ensemble
    train_wind_mel(batch_size=1024,
                   model_generator=gen_wind_mel_cnn_insp,
                   epochs=100,
                   fpath_id='cnn_ens',
                   save_model=False,
                   make_preds=True,
                   prob_model=False,
                   num_ensembles=5)

    # BNN Ensemble
    train_wind_mel(batch_size=1024,
                   model_generator=gen_wind_mel_bnn_insp,
                   epochs=100,
                   fpath_id='bnn_ens',
                   save_model=False,
                   make_preds=True,
                   prob_model=True,
                   num_ensembles=5)