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

    model_fold = tf.keras.models.load_model("f'models/cnn/hist_fold_{fold}.csv'")

    num_examples = 0
    num_correct = 0
    current_id = None
    cuurent_label = None
    for example in data_val.take(20):
        feature = example[0]
        label = example[1]
        id = example[2]

        # Check to see if new example has entered
        if id != current_id:

            # Evaluate previous id fully - will not enter on first iteration
            if current_id:
                prediction_probs /= tf.float32(num_ids)
                prediction = tf.math.argmax(prediction_probs)

                # Incriment correct predictino counter if prediction correct
                if prediction == cuurent_label:
                    num_correct += 1



            # reset and incriment variables
            num_examples += 1
            current_id = id
            cuurent_label = label
            num_ids = 1
            prediction_probs =  model_fold(feature)
        else:
            num_ids += 1
            prediction_probs += model_fold(feature)

    print(f"Fold {fold}: {num_correct} / {num_examples} "
          f"= {num_correct/num_examples:.4f}")
