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

def test_wind_mel_model(preds_paths, data_val):
    """ Tests models trained on segmented log-mel spectrograms.

    Args:
        model_path (str): File location for the model.
        data_val (tf.Dataset): Dataset to test on.

    Returns:
        Accuracy of model evaluated over the supplied dataset.
    """
    # Load model predicitions - allowing for possibility of ensemble
    model_preds = np.stack([np.load(pred_path) for pred_path in preds_paths])
    model_preds = np.mean(model_preds, axis=0)

    # Get ids and true labels
    labels = []
    ids = []
    for example in data_val:
        labels.append(example[1])
        ids.append(example[2])

    # Calculate accuracy and label-predication pairs
    num_examples = 0
    num_correct = 0
    current_id = None
    current_label = None
    c_matrix = np.zeros((50, 50))
    for i in range(len(ids)):
        label = labels[i]
        id = ids[i]

        # Check to see if new example has entered
        if id != current_id:

            # Evaluate previous id fully - will not enter on first iteration
            if current_id:
                current_prediction_probs /= num_ids
                prediction = np.argmax(current_prediction_probs)

                # update lab_pred counts
                c_matrix[int(current_label), int(prediction)] += 1

                # Increment correct prediction counter if prediction correct
                if prediction == current_label:
                    num_correct += 1

            # reset and increment variables
            num_examples += 1
            current_id = id
            current_label = label
            num_ids = 1
            current_prediction_probs = model_preds[i]
        else:
            num_ids += 1
            current_prediction_probs += model_preds[i]

    accuracy = num_correct / num_examples

    print(f"{num_correct} / {num_examples} = {accuracy:.4f}")

    return accuracy, c_matrix

def cv(preds_path_stem, num_ensemble=1):
    """ Performs cross validation on a segmented log-mel spectrogram trained model.

    Args:
        model_path_stem: Path of models trained on respective training folds.

    """
    fold_accs = []
    fold_c_matricies = []
    for fold in range(1, 6):
        data_val = load_dataset(
            f'Data/esc50_mel_wind_tfr/raw/fold_{fold}.tfrecords')
        pred_paths=[f'{preds_path_stem}preds_fold_{i}_{fold}.npy'
                    for i in range(1, num_ensemble+1)]
        fold_acc, fold_c_matrix = test_wind_mel_model(pred_paths, data_val)
        fold_accs.append(fold_acc)
        fold_c_matricies.append(fold_c_matrix)
    cv_acc = np.mean(fold_accs)
    cv_acc_std = np.std(fold_accs)
    c_matrix = np.sum(fold_c_matricies, axis=0) / np.sum(fold_c_matricies)
    np.save(f'{preds_path_stem}cmatrix_{num_ensemble}.npy', c_matrix)
    print(f"The cross validation accuracy is {cv_acc:.4f} "
          f"+/- 1.96 * {cv_acc_std:.4f}")

if __name__ == '__main__':
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"

    print("\nCNN")
    cv('models/cnn/', num_ensemble=1)
    print("\nCNN Ensemble")
    cv('models/cnn_ens/', num_ensemble=5)

    print("\nBNN")
    cv('models/bnn/', num_ensemble=1)
    print("\nBNN Ensemble")
    cv('models/bnn_ens/', num_ensemble=5)

    print("\nBNN low learning rate")
    cv('models/bnn_low/', num_ensemble=1)
    print("\nBNN low learning rate Ensemble")
    cv('models/bnn_low_ens/', num_ensemble=5)


