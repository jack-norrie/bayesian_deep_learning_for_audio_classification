import numpy as np
import scipy as sp
import librosa
import librosa.display
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
plt.style.use('ggplot')

def extract_readable_labs(meta, lab_col_name, readable_lab_col_name):
    """Extracts a dictionary of readable labels from the metadata.

    Args:
        meta (pandas.DataFrame): A dataframe of metadata for the dataset.
        lab_col_name (str): Name of the column containing the labels.
        readable_lab_col_name (str): Name of the column containing the
            readable labels.

    Returns:
        dict: A dictionary with labels for keys and redable labels for values.
    """
    return dict(esc50_meta.groupby(lab_col_name).first()[readable_lab_col_name])

def first_occurance(y):
    """Finds first observation associated with first occurance of each class.

    Args:
        y (numpy.ndarray): 1D array of labels associated with inputs.

    Returns:
        numpy.ndarray:  2D array of unique labels and their associated
            first occurrence.
    """
    labs = np.unique(y)
    first_pos = np.zeros(len(labs), dtype=int)
    for i, lab in enumerate(labs):
        first_pos[i] = np.where(y == lab)[0][0]
    return np.array((labs, first_pos))

def plot_waveforms(x, y, label_dict, ncols=5, sample_rate=44100,
                   figsize=(10, 12)):
    """Plots a grid of supplied waveforms.

    Args:
        x (numpy.ndarray): A matrix of wave samples, columns containg samples and
            rows represnting different waveforms.
        y (numpy.ndarray): A vector of labels associated with the waveforms.
        label_dict (dict): A dictionary of readable label names.
        ncols (int): Number of columns in the plot grid.
        sample_rate (float): Number of samples per unit time.
        figsize (tuple): 2-tuple of figure dimensions (Width, Height).

    Returns:
        matplotlib.figure.Figure: Grid of waveform plots.
    """
    nrows = -(-x.shape[0]//ncols) # -(-x//y) will perform ceilling division
    num_samples = x.shape[1]
    timestamps = np.arange(0, num_samples, 1) / sample_rate

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             sharex=True, sharey=True,
                             figsize=figsize)
    fig.tight_layout(pad=5, w_pad=0.1, h_pad=1.5)
    fig.supxlabel("Time (s)")
    fig.supylabel("LPCM-16 Value")
    for i in range(ncols):
        for j in range(nrows):
            axes[j, i].plot(timestamps, x[i * nrows + j], alpha=0.5)
            axes[j, i].set_title(label_dict[y[i * nrows + j, 0]].
                                 replace("_", " ").title())
    return fig

def plot_ffts(x, y, label_dict, ncols=5, sample_rate=44100,
                      log_scale=True, figsize=(10, 12)):
    """Plots a grid of the supplied waveforms' fast fourier transforms.

        Args:
            x (numpy.ndarray): A matrix of wave samples, columns containg samples and
                rows represnting different waveforms.
            y (numpy.ndarray): A vector of labels associated with the waveforms.
            label_dict (dict): A dictionary of readable label names.
            ncols (int): Number of columns in the plot grid.
            sample_rate (float): Number of samples per unit time.
            log_scale (bool): Should the spectrogram have a decibel scale.
            figsize (tuple): 2-tuple of figure dimensions (Width, Height).

        Returns:
            matplotlib.figure.Figure: Grid of FFT plots.
        """
    nrows = -(-x.shape[0]//ncols) # -(-x//y) will perform ceilling division
    num_samples = x.shape[1]

    # Perform FFT and find associated frequencies for the FFT
    x_fft = np.fft.fft(x)
    x_fft = np.fft.fftshift(x_fft)[:, int(num_samples / 2):]
    freqs = np.fft.fftfreq(num_samples, 1 / sample_rate)
    freqs = np.fft.fftshift(freqs)[int(num_samples / 2):]

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             sharex=True, sharey=True,
                             figsize=figsize)
    fig.tight_layout(pad=5, w_pad=0.1, h_pad=1.5)
    fig.supxlabel("Frequency (Hz)")
    if log_scale:
        fig.supylabel("Waverform Fourier Transform (dB)")
    else:
        fig.supylabel("Waverform Fourier Transform")
    for i in range(ncols):
        for j in range(nrows):
            if log_scale:
                axes[j, i].plot(freqs,
                                10 * np.log10(abs(x_fft[i * nrows + j])),
                                alpha=0.5)
            else:
                axes[j, i].plot(freqs,
                                abs(x_fft[i * nrows + j]),
                                alpha=0.5)
            axes[j, i].set_title(label_dict[y[i * nrows + j, 0]].
                                 replace("_", " ").title())
    return fig

def plot_spectrograms(x, y, label_dict, ncols=5, sample_rate=44100,
                   figsize=(10, 12)):
    """Plots a grid of the supplied waveforms' spectrograms.

    Args:
        x (numpy.ndarray): A matrix of wave samples, columns containg samples and
            rows represnting different waveforms.
        y (numpy.ndarray): A vector of labels associated with the waveforms.
        label_dict (dict): A dictionary of readable label names.
        ncols (int): Number of columns in the plot grid.
        sample_rate (float): Number of samples per unit time.
        figsize (tuple): 2-tuple of figure dimensions (Width, Height).

    Returns:
        matplotlib.figure.Figure: Grid of spectrogram plots.
    """
    nrows = -(-x.shape[0]//ncols) # -(-x//y) will perform ceilling division

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             sharex=True, sharey=True,
                             figsize=figsize)
    fig.tight_layout(pad=5, w_pad=0.1, h_pad=1.5)
    fig.supxlabel("Time (s)")
    fig.supylabel("Frequency (Hz)")
    for i in range(ncols):
        for j in range(nrows):
            f, t, Sxx = signal.spectrogram(x[i * nrows + j], 44100)
            axes[j, i].pcolormesh(t, f, Sxx,
                                  shading='gouraud',
                                  norm=LogNorm(clip=True))
            axes[j, i].set_title(label_dict[y[i * nrows + j, 0]].
                                 replace("_", " ").title())
    return fig



esc50_meta = pd.read_csv('Data/ESC-50-master/meta/esc50.csv')
x = np.load('Data/esc50_tabulated/X_1.npy')
y = np.load('Data/esc50_tabulated/y_1.npy')

label_dict = extract_readable_labs(esc50_meta, 'target', 'category')
first_occ = first_occurance(y)
x_first_occ = x[first_occ[1]]
y_first_occ = y[first_occ[1]]
plot_waveforms(x_first_occ, y_first_occ, label_dict).\
    savefig('Figures/waveforms.PNG')
plot_ffts(x_first_occ, y_first_occ, label_dict).\
    savefig('Figures/waveforms_ffts.PNG')
plot_spectrograms(x_first_occ, y_first_occ, label_dict).\
    savefig('Figures/waveforms_spectrograms.PNG')

