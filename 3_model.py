import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization,\
    MaxPool2D, AvgPool2D, Flatten, Permute
from tensorflow.keras.models import Sequential
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
import tensorflow_probability as tfp
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

def get_dataset(filenames):
    dataset = load_dataset(filenames)
    dataset = dataset.shuffle(2048)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(16)
    return dataset

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

    return model

def gen_bnn_model(input_shape=(128, 431, 3), output_shape=50):
    model = Sequential([
        Input(shape=input_shape, dtype='float32', name='input'),
        BatchNormalization(),
        tfpl.Convolution2DReparameterization(8, (9, 9), (3, 3),
                                             activation='relu'),
        tfpl.Convolution2DReparameterization(16, (5, 5), (2, 2),
                                             activation='relu'),
        MaxPool2D(),
        tfpl.Convolution2DReparameterization(32, (3, 3), (1, 1),
                                             activation='relu'),
        MaxPool2D(),
        Permute((3, 2, 1)),
        tfpl.Convolution2DReparameterization(8, (3, 3), (1, 1),
                                             activation='relu'),
        MaxPool2D(),
        tfpl.Convolution2DReparameterization(16, (3, 3), (1, 1),
                                             activation='relu'),
        AvgPool2D(),
        Flatten(),
        Dropout(0.2),
        tfpl.DenseReparameterization(output_shape, activation='softmax')
    ])
    model.summary()
    return model

def train_model(model, data, validation_data=None, epochs=100):
    model.compile(Adam(lr=1e-4),
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_accuracy'])
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
    model = gen_bnn_model()
    train_model(model, data_train, data_val, epochs=100)
    evaluate_model(model, data_test)
