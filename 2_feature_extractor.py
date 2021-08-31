import numpy as np
import tensorflow as tf
import librosa
from functools import partial

def get_log_mel_spectrograms(waveforms, sample_rate=44100, normalise=True):
    """ Generates the log-mel-spectrograms from a matrix of waveforms.

    Args:
        waveforms (numpy.ndarray): Matrix of waveforms.
        sample_rate (int): Sample rate of waveforms.

    Returns:
        numpy.ndarray: 3 dimensional numpy array with first dimension indexing
            the waveform that generated log-mel-spectrograms contained in the
            proceeding next 2 dimensions.

    """
    n = waveforms.shape[0]
    log_mel_spectrograms_list = []
    for i in range(n):
        s = librosa.feature.melspectrogram(y=waveforms[i], sr=sample_rate)
        s_log = librosa.power_to_db(s, ref=np.max)
        if normalise:
            log_mel_spectrograms_list.append(librosa.util.normalize(s_log))
        else:
            log_mel_spectrograms_list.append(s_log)
    log_mel_spectrograms = np.stack(log_mel_spectrograms_list)
    return log_mel_spectrograms

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

def read_waveform_tfrecord(example, output_shape):
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
    waveform = tf.reshape(waveform, shape=output_shape)

    return waveform, label

def load_dataset(filenames,
                 reader=lambda example : read_waveform_tfrecord(example,
                                                                [1, 220500, 1])):
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

def fold_wav_extractor(fold, fpath, sample_rate=44100):
    """Extracts waveforms for a certain fold of the data"""
    waveforms = np.load(fpath + f'/X_{fold}.npy')
    labels = np.load(fpath + f'/y_{fold}.npy')
    return waveforms, np.squeeze(labels.astype(np.int64))

def wav_extractor(in_fpath='Data/esc50_tabulated',
                  out_fpath='Data/esc50_wav_tfr/raw/fold',
                  num_folds=5):
    """Extracts and writes waveforms into TFRecords"""
    for fold in range(1, num_folds+1):
        waveforms, labels = \
            fold_wav_extractor(fold, in_fpath)
        write_waveforms_to_tfr_short(waveforms,
                                        labels,
                                        f'{out_fpath}_{fold}')

def wav_downsampled_extractor(in_fpath='Data/esc50_tabulated',
                              out_fpath='Data/esc50_wav_acdnet_tfr/raw/fold',
                              num_folds=5,
                              in_sr=44100,
                              out_sr=20000,
                              trunc_factor = 3):
    """Extracts and writes downsampled and truncated waveforms into TFRecords"""
    for fold in range(1, num_folds+1):
        waveforms, labels = \
            fold_wav_extractor(fold, in_fpath)

        # Down-sample waveforms to 20kHz
        waveforms = np.array([librosa.resample(wav, in_sr, out_sr)
                              for wav in waveforms])
        # Truncate
        waveforms = np.array([wav[:len(wav)//trunc_factor] for wav in waveforms])

        write_waveforms_to_tfr_short(waveforms,
                                     labels,
                                     f'{out_fpath}_{fold}')

def parse_single_image(image, label):
    # define the dictionary -- the structure -- of our single example
    data = {
        'height': _int64_feature(image.shape[0]),
        'width': _int64_feature(image.shape[1]),
        'depth': _int64_feature(image.shape[2]),
        'image': _bytes_feature(serialize_array(image)),
        'label': _int64_feature(label)
    }
    # create an Example, wrapping the single features
    out = tf.train.Example(features=tf.train.Features(feature=data))

    return out

def write_spectrograms_to_tfr_short(images, labels, filename):
    """Writes images to a TFRecord"""
    filename= filename+".tfrecords"
    writer = tf.io.TFRecordWriter(filename) #create a writer that'll store data to disk
    count = 0

    for index in range(len(images)):

        # get the data we to write
        current_image = images[index]
        current_label = labels[index]

        out = parse_single_image(image=current_image, label=current_label)
        writer.write(out.SerializeToString())
        count += 1

    writer.close()
    print(f"Wrote {count} spectrograms to TFRecord")
    return count

def fold_mel_extractor(fold, fpath, sr=44100):
    """Extracts log-mel-spectrograms for a certain fold of the data"""
    waveforms = np.load(fpath + f'/X_{fold}.npy')
    labels = np.load(fpath + f'/y_{fold}.npy')
    log_mel_spectrograms = get_log_mel_spectrograms(waveforms,
                                                    sample_rate=sr)
    return log_mel_spectrograms[..., np.newaxis],\
           np.squeeze(labels.astype(np.int64))

def mel_extractor(in_fpath='Data/esc50_tabulated',
                  out_fpath='Data/esc50_mel_tfr/raw/fold',
                  num_folds=5):
    """Extracts and writes mel spectrograms into TFRecords"""
    for fold in range(1, num_folds+1):
        log_mel_spectrograms, labels =\
            fold_mel_extractor(fold, in_fpath)
        write_spectrograms_to_tfr_short(log_mel_spectrograms,
                                  labels,
                                  f'{out_fpath}_{fold}')
def parse_single_windowed_image(image, label, id):
    # define the dictionary -- the structure -- of our single example
    data = {
        'id': _int64_feature(id),
        'height': _int64_feature(image.shape[0]),
        'width': _int64_feature(image.shape[1]),
        'depth': _int64_feature(image.shape[2]),
        'image': _bytes_feature(serialize_array(image)),
        'label': _int64_feature(label)
    }
    # create an Example, wrapping the single features
    out = tf.train.Example(features=tf.train.Features(feature=data))

    return out


def windowed_mel_delta_extractor(fpath_in, fpath_out, sr=44100):
    """Extracts and windows mel spectrograms from a tfrecords"""
    for fold in range(1, 6):
        data = load_dataset(fpath_in + f"fold_{fold}.tfrecords")
        write_count = 0
        for example in data:
            # Reshape
            waveform = example[0].numpy().reshape(-1)
            label = example[1]

            # Get log-Mel-spectrogram
            s = librosa.feature.melspectrogram(y=waveform,
                                               sr=sr,
                                               win_length=1024,
                                               n_mels=128)
            s_log = librosa.power_to_db(s, ref=np.max)

            # Get first derivative of log-Mel-spectrogram
            s_log_deltas = librosa.feature.delta(s_log)

            # Normalise log-Mel-spectrogram and deltas
            s_log_norm = librosa.util.normalize(s_log)
            s_log_deltas_norm = librosa.util.normalize(s_log_deltas)

            # Frame spectrograms
            s_log_norm_framed = librosa.util.frame(s_log_norm,
                                                   50, 25).\
                transpose((2, 0, 1))
            s_log_deltas_norm_framed = librosa.util.frame(s_log_deltas_norm,
                                                          50, 25).\
                transpose((2, 0, 1))

            # Check for silent frames
            n_frames = len(s_log_norm_framed)
            filter = [True] * n_frames
            for i in range (n_frames):
                silent_pixels = np.equal(s_log_norm_framed[i], -1)
                if silent_pixels.all():
                    filter[i] = False
            s_log_norm_framed = s_log_norm_framed[filter]
            s_log_deltas_norm_framed = s_log_deltas_norm_framed[filter]

            # Stack spectrograms and deltas
            framed_features = np.stack([s_log_norm_framed,
                                        s_log_deltas_norm_framed],
                                       axis=-1)

            # Save spectrograms and generate a unique id for testing purposes
            id = np.random.uniform(0, 1)
            fpath_out = fpath_out + f"fold_{fold}.tfrecords"
            writer = tf.io.TFRecordWriter(fpath_out)
            for frame in framed_features:
                out = parse_single_windowed_image(image=frame,
                                                  label=label,
                                                  id=id)
                writer.write(out.SerializeToString())
                write_count += 1
        print(f'Finished fold {fold}, wrote {write_count} examples to disk.')


def fold_multi_mel_extractor(fold, fpath, sample_rate=44100):
    """Extracts 3-channel mel spectrograms for a certain fold of the data"""
    waveforms = np.load(fpath + f'/X_{fold}.npy')
    labels = np.load(fpath + f'/y_{fold}.npy')
    spec_wavs =[]
    for waveform in waveforms:
        r = waveform
        h, p = librosa.effects.hpss(waveform)
        spec_wav_comps = []
        for wav_comp in [r, h, p]:
            spec_wav_comps.append(
                get_log_mel_spectrograms(wav_comp[np.newaxis, ...],
                                         sample_rate=sample_rate)
            )
        spec_wav = np.stack(spec_wav_comps, axis=-1)
        spec_wavs.append(spec_wav)
    return np.stack(np.squeeze(spec_wavs), axis=0),\
           np.squeeze(labels.astype(np.int64))

def multi_mel_extractor(in_fpath='Data/esc50_tabulated',
                  out_fpath='Data/esc50_multi_mel_tfr/raw/fold',
                  num_folds=5):
    """Extracts and writes multi-channel mel spectrograms into TFRecords"""
    for fold in range(1, num_folds+1):
        multi_log_mel_spectrograms, labels =\
            fold_multi_mel_extractor(fold, in_fpath)
        write_spectrograms_to_tfr_short(multi_log_mel_spectrograms,
                                  labels,
                                  f'{out_fpath}_{fold}')


if __name__ == '__main__':
    windowed_mel_delta_extractor('Data/esc50_wav_tfr/raw/',
                                 'Data/esc50_mel_wind_tfr/raw/')
    """
    wav_extractor()
    wav_downsampled_extractor()
    mel_extractor()
    multi_mel_extractor()
    """

