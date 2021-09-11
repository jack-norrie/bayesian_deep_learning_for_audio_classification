import tensorflow as tf
from functools import partial


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

    return image, label, id

def load_dataset(filenames, reader=lambda example:\
        read_windowed_spectrogram_tfrecord(example)):
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

if __name__ == '__main__':
    import os
    fold = 1
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    data_val = load_dataset(f'Data/esc50_mel_wind_tfr/raw/fold_{fold}.tfrecords')

    for example in data_val:
        print(example)