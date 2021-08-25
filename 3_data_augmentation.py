import numpy as np
import tensorflow as tf
import librosa
from scipy.io import wavfile
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

def volume_gain_range(signal, low=-5, high=5):
    """Shifts volume by random number of decibels"""
    decibel_shift = np.random.uniform(low, high)
    signal_volume_shifted = signal * np.exp(0.115129 * decibel_shift)
    return signal_volume_shifted

def snr_noiser(signal, low=9.5, high=17):
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

def time_shift_range(signal, sr=44100, low=-0.25, high=0.25):
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

def sgn(waveform, sr=44100):
    """Performs standard signal augmentation."""
    aug_waveform = waveform
    if np.random.choice([0, 1]):
        aug_waveform = speedup_factor_range(aug_waveform)
    if np.random.choice([0, 1]):
        aug_waveform = semitone_shift_range(aug_waveform, sr)
    if np.random.choice([0, 1]):
        aug_waveform = volume_gain_range(aug_waveform)
    if np.random.choice([0, 1]):
        aug_waveform = time_shift_range(aug_waveform, sr)
    if np.random.choice([0, 1]):
        aug_waveform = snr_noiser(aug_waveform)
    return aug_waveform

def data_augmentor(fpath_in, fpath_out,
                   augmentor, augment_factor=9,
                   output_shape=[1, 220500, 1]):
    """ Performs data augmentation on a specified file and outputs the result.

    Args:
        fpath_in (str): Input file path to read TFRecord.
        fpath_out (str): Output file path to write TFRecord.
        augmentor (function): A feature wise data augmnetation function.
        augment_factor (int): Number of augmented samples to generate per sample.
        output_shape (array): Shape to reshape output to
    """
    data = load_dataset(fpath_in, reader=read_waveform_tfrecord)
    augmented_features = []
    augmented_labels = []
    for sample in data:
        feature = np.squeeze(sample[0].numpy())
        augmented_labels += [sample[1].numpy()] * augment_factor
        for _ in range(augment_factor):
            # Sometimes there are issues with the augmentor
            while True:
                try:
                    augmented_feature = augmentor(feature).reshape(output_shape)
                    augmented_features.append(augmented_feature)
                    break
                except:
                    print("Error encountered during augmentation - "
                          "discarding augmented sample")

    augmented_features = np.stack(augmented_features)
    augmented_labels = np.stack(augmented_labels)
    write_waveforms_to_tfr_short(augmented_features, augmented_labels, fpath_out)

def visualise_augmentation(feature, augmentor, augment_factor=9,
                           x_vals=None, ncols=5, figsize=(10, 5)):
    # Perform augmentation and plot setup
    features = [feature]
    for _ in range(augment_factor):
        features.append(augmentor(feature))
    if x_vals is None:
        x_vals = np.arange(0, len(feature), 1)

    # Make plot grid
    nrows = -(-(augment_factor + 1) // ncols)  # number of rows for plot grid
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             sharex=True, sharey=True,
                             figsize=figsize, squeeze=False)
    fig.tight_layout(pad=5, w_pad=0.1, h_pad=1.5)
    fig.suptitle(f"{9 + 1}-Fold Data Augmentation")
    fig.supxlabel("Time (s)", fontsize=12)
    fig.supylabel("Normalised Sound Intensity", fontsize=12)
    for i in range(nrows):
        for j in range(ncols):
            if i * ncols + j <= augment_factor:
                if i * ncols + j == 0:
                    axes[i, j].plot(x_vals,
                                    features[i * ncols + j],
                                    alpha=0.8, color='#1f77b4')
                else:
                    axes[i, j].plot(x_vals,
                                    features[i * ncols + j],
                                    alpha=0.8, color='lightseagreen')
            else:
                break

def generate_augmentation_visualisation():
    data = load_dataset('Data/esc50_wav_tfr/raw/fold_1.tfrecords',
                        reader=read_waveform_tfrecord)
    feature = np.squeeze(next(iter(data.shuffle(1024)))[0].numpy())
    del data
    visualise_augmentation(feature, sgn, x_vals=np.linspace(0, 5, len(feature)))

def generate_augmentation_examples(augmentor, fpath,
                                   sr = 44100, num_examples=5, augment_factor=9):
    data = load_dataset('Data/esc50_wav_tfr/raw/fold_1.tfrecords',
                        reader=read_waveform_tfrecord)
    data_sub = data.shuffle(1024).take(num_examples); del data
    for i, waveform in enumerate(data_sub):
        waveform = np.squeeze(waveform[0].numpy())
        wavfile.write(fpath + f'/{i+1}_1.wav', 44100, waveform)
        for j in range(2, augment_factor + 2):
            wavfile.write(fpath + f'/{i+1}_{j}.wav', 44100, augmentor(waveform))


if __name__ == '__main__':
    for i in range(1, 6):
        """
        # Augment pure waveforms
        data_augmentor(fpath_in=f'Data/esc50_wav_tfr/raw/fold_{i}.tfrecords',
                       fpath_out=f'Data/esc50_wav_tfr/aug/fold_{i}.tfrecords',
                       augmentor=sgn,
                       augment_factor=29,
                       output_shape=[1, 220500, 1])
        """

        # Augment ACDNet waveforms
        data_augmentor(fpath_in=f'Data/esc50_wav_acdnet_tfr/raw/fold_{i}.tfrecords',
                       fpath_out=f'Data/esc50_wav_acdnet_tfr/aug/fold_{i}.tfrecords',
                       augmentor=lambda wav: sgn(wav, sr=20000),
                       augment_factor=29,
                       output_shape=[1, 33333, 1])

        """
        generate_augmentation_visualisation()
        generate_augmentation_examples(augmentor=sgn,
                                       fpath='Figures/aug_clips/sgn')
        """



