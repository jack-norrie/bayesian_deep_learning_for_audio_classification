import numpy as np
import tensorflow as tf
import librosa
from functools import partial
import matplotlib.pyplot as plt

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

def trim_pad(signal, desired_length):
    """Trims or zero-pads a signal to a specified length"""
    current_length = len(signal)
    if current_length < desired_length:
        pad = np.zeros(desired_length - current_length)
        return np.concatenate([signal, pad])
    else:
        return signal[:desired_length]

def speedup_factor_range(signal, low=0.8, high=1.2):
    """Speeds-up/slows-down a signal by a random factor within some range."""
    # Perform stretching via a randomised speed up or slow down.
    speedup_factor = np.random.uniform(low, high)
    signal_streth = librosa.effects.time_stretch(signal, speedup_factor)

    # Trim/pad the stretched signal to the original length
    original_length = len(signal)
    return trim_pad(signal_streth, original_length)

def semitone_shift_range(signal, sr=44100, low=-2, high=2):
    """Shifts signal by random number of semitones"""
    semitone_shift_steps = np.random.uniform(low, high)
    signal_shifted = librosa.effects.pitch_shift(signal, sr,
                                                 semitone_shift_steps)
    return signal_shifted

def volume_gain_range(signal, low=-3, high=3):
    """Shifts volume by random number of decibels"""
    decibel_shift = np.random.uniform(low, high)
    signal_volume_shifted = signal * np.exp(0.115129 * decibel_shift)
    return signal_volume_shifted

def snr_noiser(signal, low=0, high=10):
    """Adds noise at a randomly generated signal to noise ratio"""
    # Generate SNR in db and find associated rms of noise
    snr_db = np.random.uniform(low, high)
    squared_rms_signal = np.mean(signal ** 2)
    rms_noise = np.sqrt(squared_rms_signal / np.exp(0.2302585093 * snr_db))

    # Since using guassian noise rms == std
    signal_length = len(signal)
    noise = np.random.normal(0, rms_noise, signal_length)
    noisy_signal = signal + noise
    return noisy_signal

def time_shift_range(signal, sr=44100, low=-0.1, high=0.1):
    """Shifts the signal either forward or backwards"""
    time_shift = np.random.uniform(low, high)
    sample_shift = int(time_shift * sr)
    if sample_shift >= 0:
        signal_time_shifted = np.concatenate([np.zeros(sample_shift),
                                              signal[:-sample_shift]])
    else:
        signal_time_shifted = np.concatenate([signal[-sample_shift:],
                                              np.zeros(-sample_shift)])
    return  signal_time_shifted

def sgn(waveform):
    """Performs standard signal augmentation."""
    aug_waveform = waveform
    if np.random.choice([0, 1]):
        aug_waveform = speedup_factor_range(aug_waveform)
    if np.random.choice([0, 1]):
        aug_waveform = semitone_shift_range(aug_waveform)
    if np.random.choice([0, 1]):
        aug_waveform = volume_gain_range(aug_waveform)
    if np.random.choice([0, 1]):
        aug_waveform = time_shift_range(aug_waveform)
    if np.random.choice([0, 1]):
        aug_waveform = snr_noiser(aug_waveform)
    return aug_waveform

def data_augmentor(data, augmentor, augment_factor=9,
                   output_shape=[1, 220500, 1]):
    augmented_data = []
    for sample in data:
        feature = np.squeeze(sample[0].numpy())
        for _ in range(augment_factor):
            augmented_data.append(augmentor(feature).reshape(output_shape))
    return np.stack(augmented_data)

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))): # if value ist tensor
        value = value.numpy() # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a floast_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def serialize_array(array):
    array = tf.io.serialize_tensor(tf.cast(array, tf.float32))
    return array

def parse_single_waveform(waveform, label):
    # define the dictionary -- the structure -- of our single example
    data = {
        'waveform': _bytes_feature(serialize_array(waveform)),
        'label': _int64_feature(label)
    }
    # create an Example, wrapping the single features
    out = tf.train.Example(features=tf.train.Features(feature=data))
    return out

def write_waveforms_to_tfr_short(waveforms, labels, filename):
    """Writes images to a TFRecord"""
    filename= filename+".tfrecords"
    writer = tf.io.TFRecordWriter(filename)
    count = 0

    for index in range(len(waveforms)):

        # get the data we to write
        current_waveform = waveforms[index]
        current_label = labels[index]

        out = parse_single_waveform(waveform=current_waveform,
                                    label=current_label)
        writer.write(out.SerializeToString())
        count += 1

    writer.close()
    print(f"Wrote {count} waveforms to TFRecord")
    return count

data = load_dataset([f'Data/esc50_wav_tfr/raw/fold_{i}.tfrecords'
                           for i in [1]],
                          reader=read_waveform_tfrecord)
data_aug = data_augmentor(data, sgn, 9, (1, 220500, 1))
