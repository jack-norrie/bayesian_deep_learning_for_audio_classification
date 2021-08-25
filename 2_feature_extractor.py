import numpy as np
import tensorflow as tf
import librosa

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

def fold_mel_extractor(fold, fpath, sample_rate=44100):
    """Extracts mel spectrograms for a certain fold of the data"""
    waveforms = np.load(fpath + f'/X_{fold}.npy')
    labels = np.load(fpath + f'/y_{fold}.npy')
    log_mel_spectrograms = get_log_mel_spectrograms(waveforms,
                                                    sample_rate=sample_rate)
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
    wav_extractor()
    wav_downsampled_extractor()
    mel_extractor()
    multi_mel_extractor()
