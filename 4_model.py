import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization,\
    MaxPool2D, AvgPool2D, Flatten, Permute, Conv1D, Conv2D
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers
from functools import partial
efn = tf.keras.applications.efficientnet

def read_waveform_tfrecord(example):
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
    waveform = tf.reshape(waveform, shape=[1, 220500, 1])

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

def load_dataset(filenames, reader=read_waveform_tfrecord):
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

def get_dataset(filenames, reader=read_waveform_tfrecord, batch_size = 16, classes=50):
    dataset = load_dataset(filenames, reader=reader)
    dataset = dataset.map(lambda x, y: (x, tf.one_hot(y, classes))) #OHE
    dataset = dataset.shuffle(2048)
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
        Dropout(0.2),
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


def train_model(model, data, validation_data=None, epochs=100,
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
        [f'Data/esc50_wav_tfr/{dir}/fold_{i}.tfrecords' for i in [1, 2, 3, 4]]
        for dir in ['raw', 'aug']])),
                             reader=read_waveform_tfrecord,
                             batch_size=64)
    data_val = get_dataset('Data/esc50_wav_tfr/raw/fold_5.tfrecords',
                           reader=read_waveform_tfrecord,
                           batch_size=64)

    model = gen_acdnet_insp(reg=1.0)

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

    train_model(model, data_train, data_val, epochs=2000,
                callbacks=[tf.keras.callbacks.LearningRateScheduler(scheduler)])

if __name__ == '__main__':
    train_acdnet()