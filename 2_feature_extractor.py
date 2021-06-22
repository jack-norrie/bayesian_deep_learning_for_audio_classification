import numpy as np
import matplotlib.pyplot as plt
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

def feature_extractor(fpath, sample_rate=44100):

    waverforms = np.load(fpath)
    log_mel_spectrograms = get_log_mel_spectrograms(waverforms,
                                                    sample_rate=sample_rate)
    return log_mel_spectrograms

S_dB = feature_extractor('Data/esc50_tabulated/X_1.npy')
