import numpy as np
import tensorflow as tf
import librosa

def get_log_mel_spectrograms(waveforms, sample_rate=44100):
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

def write_images_to_tfr_short(images, labels, filename):
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
    print(f"Wrote {count} elements to TFRecord")
    return count

def fold_feature_extractor(fold, fpath, sample_rate=44100):
    """Extracts mel spectrograms for a certain fold of the data"""
    waveforms = np.load(fpath + f'/X_{fold}.npy')
    labels = np.load(fpath + f'/y_{fold}.npy')
    log_mel_spectrograms = get_log_mel_spectrograms(waveforms,
                                                    sample_rate=sample_rate)
    return log_mel_spectrograms[..., np.newaxis],\
           np.squeeze(labels.astype(np.int64))

def fold_multi_channel_feature_extractor(fold, fpath, sample_rate=44100):
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

if __name__ == '__main__':
    for fold in range(1, 5+1):
        multi_log_mel_spectrograms, labels =\
            fold_multi_channel_feature_extractor(fold, 'Data/esc50_tabulated')
        write_images_to_tfr_short(multi_log_mel_spectrograms,
                                  labels,
                                  f'Data/esc50_multi_tfr/fold_{fold}')


