import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization,\
    MaxPool2D, AvgPool2D, Flatten, Permute, Conv2D
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam, RMSprop
import tensorflow_probability as tfp
tfd = tfp.distributions
tfpl = tfp.layers
from functools import partial
efn = tf.keras.applications.efficientnet

def read_tfrecord(example):
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

def load_dataset(filenames):
    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False  # disable order, increase speed
    dataset = tf.data.TFRecordDataset(
        filenames
    )  # automatically interleaves reads from multiple files
    dataset = dataset.with_options(
        ignore_order
    )  # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(
        partial(read_tfrecord),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    # returns a dataset of (image, label)
    return dataset

def get_dataset(filenames, classes=50):
    dataset = load_dataset(filenames)
    dataset = dataset.map(lambda x, y: (x[:, :, 0], tf.one_hot(y, classes))) #OHE
    dataset = dataset.shuffle(2048)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(16)
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
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])

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
        MaxPool2D(pool_size=8),
        Flatten(),
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

def train_model(model, data, validation_data=None, epochs=100):
    model.fit(data,
              validation_data=validation_data,
              epochs=epochs)

def evaluate_model(model, data):
    results = model.evaluate(data)
    with open('Models/Results/result.txt', 'w') as output:
        output.write(str(results))


if __name__ == '__main__':
    data_train = get_dataset([f'Data/esc50_multi_tfr/fold_{i}.tfrecords'
                              for i in [1, 2, 3]])
    data_val = get_dataset('Data/esc50_multi_tfr/fold_4.tfrecords')
    data_test = get_dataset('Data/esc50_multi_tfr/fold_5.tfrecords')
    model = gen_simple_bnn()
    train_model(model, data_train, data_val, epochs=100)
    evaluate_model(model, data_test)



