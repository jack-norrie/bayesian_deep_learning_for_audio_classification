import tensorflow as tf
import numpy as np
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

def test_wind_mel_model(model_path, data_val):
    """ Tests models trained on segmented log-mel spectrograms.

    Args:
        model_path (str): File location for the model.
        data_val (tf.Dataset): Dataset to test on.

    Returns:
        Accuracy of model evaluated over the supplied dataset.
    """
    model_fold = tf.keras.models.load_model(model_path)

    num_examples = 0
    num_correct = 0
    current_id = None
    cuurent_label = None
    for example in data_val:
        feature = tf.expand_dims(example[0], 0)
        label = example[1]
        id = example[2]

        # Check to see if new example has entered
        if id != current_id:

            # Evaluate previous id fully - will not enter on first iteration
            if current_id:
                prediction_probs /= tf.cast(num_ids, tf.float32)
                prediction = tf.math.argmax(prediction_probs, axis=1).numpy()[0]
                # Incriment correct predictino counter if prediction correct
                if prediction == cuurent_label:
                    num_correct += 1

            # reset and incriment variables
            num_examples += 1
            current_id = id
            cuurent_label = label
            num_ids = 1
            prediction_probs = model_fold(feature)
        else:
            num_ids += 1
            prediction_probs += model_fold(feature)

    accuracy = num_correct / num_examples

    return accuracy

def cv(model_path_stem):
    """ Performs cross validation on a segmented log-mel spectrogram trained model.

    Args:
        model_path_stem: Path of models trained on respective training folds.

    """
    fold_accs = []
    for fold in range(1, 6):
        data_val = load_dataset(
            f'Data/esc50_mel_wind_tfr/raw/fold_{fold}.tfrecords')
        model_path = f'{model_path_stem}/model_fold_{fold}.hp5'
        fold_acc = test_wind_mel_model(model_path, data_val)
        print(f"Fold {fold}: {fold_acc:.4f}")
        fold_accs.append(fold_acc)
    cv_acc = np.mean(fold_accs)
    print(f"The cross validation accuracy is {cv_acc:.4f}")

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    cv('models/cnn')


