import numpy as np
import librosa
import librosa.display
from scipy import signal
import pandas as pd
import matplotlib as mpl
from cycler import cycler
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import tensorflow as tf
from tensorflow.keras.activations import linear, elu, relu, softplus
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
                             figsize=figsize, squeeze=False)
    fig.tight_layout(pad=5, w_pad=0.1, h_pad=1.5)
    fig.supxlabel("Time (s)")
    for i in range(ncols):
        for j in range(nrows):
            if i * nrows + j >= x.shape[0]:
                break
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
            f, t, Sxx = signal.spectrogram(x[i * nrows + j], sample_rate)
            axes[j, i].pcolormesh(t, f, Sxx,
                                  shading='gouraud',
                                  norm=LogNorm(clip=True))
            axes[j, i].set_title(label_dict[y[i * nrows + j, 0]].
                                 replace("_", " ").title())
    return fig

def plot_mel_spectrograms(x, y, label_dict, ncols=5, sample_rate=44100,
                   figsize=(12, 18), frac=0.04, aspect=50):
    """Plots a grid of the supplied waveforms' mel-spectrograms.

    Args:
        x (numpy.ndarray): A matrix of wave samples, columns containg samples and
            rows represnting different waveforms.
        y (numpy.ndarray): A vector of labels associated with the waveforms.
        label_dict (dict): A dictionary of readable label names.
        ncols (int): Number of columns in the plot grid.
        sample_rate (float): Number of samples per unit time.
        figsize (tuple): 2-tuple of figure dimensions (Width, Height).

    Returns:
        matplotlib.figure.Figure: Grid of mel-spectrogram plots.
    """
    nrows = -(-x.shape[0]//ncols) # -(-x//y) will perform ceilling division

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols,
                             sharex=True, sharey=True,
                             figsize=figsize, squeeze=False)
    fig.tight_layout(pad=5, w_pad=0.1, h_pad=1.5)
    fig.supxlabel("Time (s)", fontsize=20)
    fig.supylabel("Frequency (Hz)", fontsize=20)
    for i in range(ncols):
        for j in range(nrows):
            S = librosa.feature.melspectrogram(y=x[i * nrows + j].astype(float),
                                               sr=sample_rate, n_mels=128)
            S_dB = librosa.power_to_db(S, ref=np.max)
            img = librosa.display.specshow(S_dB, x_axis='s', y_axis='mel',
                                     sr=sample_rate, ax=axes[j, i])
            axes[j, i].locator_params(axis='x', nbins=4)
            axes[j, i].set_xlabel("")
            axes[j, i].set_ylabel("")
            axes[j, i].set_title(label_dict[y[i * nrows + j, 0]].
                                 replace("_", " ").title(),
                                 fontsize=17)
    fig.colorbar(img, ax=axes, format="%+2.f dB",
                 aspect=aspect, fraction=frac)
    return fig

def plot_hpss(x, sample_rate=44100, figsize=(12, 4), frac=0.15, aspect=20):
    """ Plots mel-spectrograms of HPSS decomposition

    Args:
        x (numpy.ndarray): Waveform samples

    Returns:
        matplotlib.figure.Figure: HPSS components' mel-spectrogram plots.

    """
    # Perform HPSS to extract raw, harmonic and precussive components
    r = x
    h, p = librosa.effects.hpss(x)

    # wrapper to reuse mel_spectrogram code
    waveforms = np.stack([r, h, p])
    y = np.array([[0], [1], [2]])
    label_dict = {0:'Raw', 1:'Harmonic', 2:'percussive'}
    return plot_mel_spectrograms(waveforms, y, label_dict, ncols=3,
                                 sample_rate=sample_rate,
                                 figsize=figsize,
                                 frac=frac, aspect=aspect)

def activation_plot(range=[-1.5, 1.5], figsize=[8, 4]):
    fig, axes = plt.subplots(nrows=1, ncols=4,
                             sharey=True, figsize=figsize)
    title = ["Linear", "ReLU", "ELU", 'Softplus']
    for i, func in enumerate([linear, relu, elu, softplus]):
        a = tf.linspace(*range, 100)
        with tf.GradientTape() as t:
            t.watch(a)
            h = func(a)
        dh_da =  t.gradient(h, a)
        axes[i].plot(a.numpy(), h.numpy())
        axes[i].plot(a.numpy(), dh_da.numpy(), ls="--")
        axes[i].set_title(title[i])

    return fig

def plot_learning_curves(history_path_stem, num_folds=5, num_ensembles=1,
                         figsize=[6, 12]):

    fig, axes = plt.subplots(nrows=num_folds, ncols=1,  squeeze=False,
                             figsize=figsize)

    fig.tight_layout(pad=3, h_pad=2.5)

    for i in range(1, num_folds+1):
        history_paths = [f'{history_path_stem}hist_fold_{j}_{i}.csv' \
                         for j in range(1, num_ensembles+1)]
        histories = [pd.read_csv(path) for path in history_paths]

        if len(histories) > 1:
            alpha=0.25
        else:
            alpha = 1

        # Set current axis and a twin axis for accuracies
        c_ax = axes[i-1, 0]
        c_ax.set_title(f"Training Fold {i}")
        c_ax.set_ylim(0, 5)
        c_ax.set_yticks(np.linspace(0, 5, 5))
        c_ax_t = c_ax.twinx()
        c_ax_t.set_yticks(np.linspace(0, 1, 5))
        c_ax_t.set_ylim(0, 1)

        for history in histories:
            epochs = [i for i in range(1, 101)]
            c_ax.plot(epochs, history['loss'],
                      color='red', alpha=alpha)
            c_ax.plot(epochs, history['val_loss'],
                      color='orange', alpha=alpha)
            c_ax_t.plot(epochs, history['accuracy'],
                        color='blue', alpha=alpha)
            c_ax_t.plot(epochs, history['val_accuracy'],
                        color='purple', alpha=alpha)
    return fig

def plot_c_matrix(c_matrix_path):
    c_matrix = np.load(c_matrix_path) * 50

    fig, ax = plt.subplots(1, 1)
    heat = ax.imshow(c_matrix)
    ax.xaxis.tick_top()
    ax.set_xlabel("Predicted")
    ax.xaxis.set_label_position('top')
    ax.set_ylabel("Actual")
    ax.grid(b=None)
    plt.colorbar(heat)

    return fig


def plot_mel_filter(figsize=[8, 3]):
    mels = librosa.filters.mel(sr=44100, n_fft=1024, n_mels=8)
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_title("Mel Filter Bank")
    ax.set_xlabel("Frequency Index")
    ax.set_ylabel("Weight")
    for mel in mels:
        ax.plot(mel)
    return fig





if __name__ == '__main__':
    # Load  in relevant data.
    esc50_meta = pd.read_csv('Data/ESC-50-master/meta/esc50.csv')
    x = np.load('Data/esc50_tabulated/X_1.npy')
    y = np.load('Data/esc50_tabulated/y_1.npy')

    # Extract readable labels and first occurrences of each label.
    label_dict = extract_readable_labs(esc50_meta, 'target', 'category')
    first_occ = first_occurance(y)
    x_first_occ = x[first_occ[1]]
    y_first_occ = y[first_occ[1]]

    # Make Plots
    plot_waveforms(x_first_occ, y_first_occ, label_dict).\
        savefig('Figures/waveforms.PNG')
    plot_ffts(x_first_occ, y_first_occ, label_dict).\
        savefig('Figures/waveforms_ffts.PNG')
    plot_spectrograms(x_first_occ, y_first_occ, label_dict).\
        savefig('Figures/waveforms_spectrograms.PNG')
    plot_mel_spectrograms(x_first_occ, y_first_occ, label_dict).\
        savefig('Figures/waveforms_mel_spectrograms.PNG')
    plot_hpss(x_first_occ[35]).\
        savefig('Figures/hpss.PNG')
    plt.style.use('default')
    mpl.rcParams['axes.prop_cycle'] = cycler('color', [ '#9467bd', '#1f77b4',
                                                       '#2ca02c', '#d62728',
                                                        '#8c564b', '#ff7f0e',
                                                       '#e377c2', '#7f7f7f',
                                                       '#bcbd22', '#17becf'])
    plot_waveforms(np.array([x_first_occ[20]]),
                   y_first_occ[20][..., np.newaxis],
                   label_dict,
                   ncols=1,
                   figsize=(10, 4.25)).\
        savefig('Figures/single_waveform.PNG')

    activation_plot(range=[-1.25, 1.25], figsize=[8, 3]).savefig('Figures/activation.PNG')

    plot_mel_filter(figsize=[12, 4.5]).savefig('Figures/mel.PNG')

    for stem in [('models/cnn/', 1),
                 ('models/cnn_ens/', 5),
                 ('models/bnn_low/', 1),
                 ('models/bnn_low_ens/', 5)]:
        plot_learning_curves(history_path_stem=stem[0],
                             num_folds=5,
                             num_ensembles=stem[1],
                             figsize=(10, 16)).\
            savefig(f'{stem[0]}learning_curve')

    plot_c_matrix('models/cnn/cmatrix_1.npy').savefig('models/cm_cnn.png')
    plot_c_matrix('models/cnn_ens/cmatrix_5.npy').savefig('models/cm_cnn5.png')
    plot_c_matrix('models/bnn_low/cmatrix_1.npy').savefig('models/cm_bnn.png')
    plot_c_matrix('models/bnn_low_ens/cmatrix_5.npy').savefig('models/cm_bnn5.png')




