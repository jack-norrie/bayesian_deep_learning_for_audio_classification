import numpy as np
import librosa
from os import listdir

def load_wavs(fpath, sample_rate=44100):
    """Loads .wav files from a specified filepath.

    The specified filepath should contain only .wav files.

    Args:
        fpath (str): The filepath containg the .wav files.

    Returns:
        wav_dict (dict): Dictionary of .wav files with keys specified by
            filenames and values of the form (sample_rate, samples).
    """
    wav_dict = {}
    file_names = listdir(fpath)
    for file_name in file_names:
        try:
            wav_dict[file_name] = librosa.load(fpath + '/' + file_name,
                                               sr=sample_rate)
        except:
            print(f"Exception occured for {fpath + '/' + file_name}")
    return wav_dict

def check_compatible(wav_dict):
    """Checks sample rate and sample sizes are equal across wav files.

    Args:
        wav_dict (dict): Dictionary of .wav files with keys specified by
            filenames and values of the form (sample_rate, samples).

    Returns:
        bool: True for tabulation compatible, False otherwise.
    """
    rates = []
    wav_sizes = []
    for file_name in wav_dict.keys():
        rates.append(wav_dict[file_name][1])
        wav_sizes.append(len(wav_dict[file_name][0]))
    unique_rates = np.unique(rates)
    unique_sizes = np.unique(wav_sizes)
    if len(unique_rates) == 1 and len(unique_sizes) == 1:
        print("The .wav files are compatable and can thus be tabulated\n" +
              f"Distinct sample rates: {unique_rates}\n" +
              f"Distinct sample sizes: {unique_sizes}")
        return True
    else:
        print("The .wav files are incompatable and cannot thus be tabulated\n" +
              f"Distinct sample rates: {unique_rates}\n" +
              f"Distinct sample sizes: {unique_sizes}")
        return False

def tabulate_wavs(wav_dict, lab_pos, fold_pos):
    """Tabulates a dictionary of .wavs and extracts labels, fold and filename.

    Args:
        wav_dict (dict): Dictionary of .wav files with keys specified by
            filenames and values of the form (sample_rate, samples).
        lab_pos (int): Position of label in filename (indexing starts at zero).
        fold_pos (int): Position of fold in filename (indexing starts at zero).

    Returns:
        4-tuple containing an array of the  tabulated waves, an array of the
            associated labels, an array of the associated folds and an
             array of the assoicated filenames.
    """
    # Extract number of samples per .wav adn total number of .wavs
    wav_dict_vals = list(wav_dict.values())
    wav_len = len(wav_dict_vals[0][0])
    num_wav = len(wav_dict_vals)

    # Initialise arrays to store quantities of interest
    wavs = np.zeros((num_wav, wav_len), dtype=np.float32)
    wavs_labs = np.zeros((num_wav, 1), dtype=np.int8)
    wavs_folds = np.zeros((num_wav, 1), dtype=np.int8)
    wavs_fname = []

    # Populate arrays of interest
    file_names = wav_dict.keys()
    for i, file_name in enumerate(file_names):
        # Extract .wav samples
        wavs[i] = wav_dict[file_name][0]

        # Remove .wav part and then split by '-'
        file_name_trunc = file_name.split('.')[0]
        file_name_parts = file_name_trunc.split('-')

        # Extract label, folds and filename information
        wavs_labs[i] = file_name_parts[lab_pos]
        wavs_folds[i] = file_name_parts[fold_pos]
        wavs_fname.append(file_name)

    return wavs, wavs_labs, wavs_folds, np.array(wavs_fname, dtype=str)

def folds_to_csv(folder_path, wavs, wavs_labs, wavs_folds, wavs_fname):
    """Writes tabulated .wavs samples and labels to .csv.

    Args:
        folder_path (str): Folder for data to be written to.
        wavs (array): Array of .wav samples.
        wavs_labs (array): Array of labels.
        wavs_folds (array): Array of fold memberships.

    """
    folds = np.unique(wavs_folds)
    for fold in folds:
        fold_idx = np.where(wavs_folds == fold)[0]
        np.save(folder_path+f'/X_{fold}.npy',
                   wavs[fold_idx])
        np.save(folder_path + f'/y_{fold}.npy',
                   wavs_labs[fold_idx])
        np.save(folder_path + f'/z_{fold}.npy',
                wavs_fname[fold_idx])


def esc_wav_processor(input_fpath, output_fpath, lab_pos, fold_pos):
    """ Chains the previous functions to form a pre-processor.

    Args:
        input_fpath (str): Filepath containg the .wav files.
        output_fpath (str): Filepath for the matrix of waveforms the .wav files.
        lab_pos (int): Position of label in filename (indexing starts at zero).
        fold_pos (int): Position of fold in filename (indexing starts at zero).

    """
    # Import .wav files as dictionary
    wav_dict = load_wavs(input_fpath)

    # Check dictionary of .wavs can be tabulated:
    if not check_compatible(wav_dict):
        raise Exception(".wav files have an incompatible number of samples or"
                        "sample rate, thus they cannot be tabulated.")

    # Store the audio files in a tabular data structure:
    wavs, wavs_labs, wavs_folds, wavs_fname = \
        tabulate_wavs(wav_dict, lab_pos, fold_pos)

    # Write tabulated .wavs to seperate csv files for each fold
    folds_to_csv(output_fpath, wavs, wavs_labs, wavs_folds, wavs_fname)

if __name__ == '__main__':
    # Process ESC50
    esc_wav_processor('Data/ESC-50-master/audio',
                      'Data/esc50_tabulated',
                      3, 0)


