# bayesian_deep_learning_for_audio_classification
This repository contains the code associated with the MSc research project: Bayesian Neural Network Audio Classifiers. This project was produced in partial fulfilment of the requirments for my MSc in Statistics (Data Science) from Imperial College London. 

Before the findings of this report, which is contained in this repository as report.pdf, can be replicated it is important to set up ones working environemnt correctly. The frist step in this process is setting up a Pyhton 3.7 virtual environment with the packages and package versions shown in requirments.txt. Next, one needs to setup their working directory to accomidate the I/O operations associated with the scripts in this repo. The exact preperation requeired will depend on wether one is trying to replicate the results used by the models investigated during experimentation, these are stored in experiments.py, or simply the main findings of this report. In order to replicate the the main findings one must:
1. Download the ESC-50 dataset from https://github.com/karolpiczak/ESC-50, clonning the repository into a folder called "Data". 
2. Make a new directory called Data/esc50_tabulated
3. Make a new directory called Data/esc50_wav_tfr/raw & Data/esc50_wav_tfr/aug
4. Make a new directory called Data/esc50_mel_wind_tfr/raw & Data/esc50_mel_wind_tfr
5. Make a new directory called models/cnn, models/bnn, models/cnn_ens, models/bnn_ens

With ones worjing environment correctly setup it is simply a case of running through the following scripts sequentially:
1_data_extractor.py
2_feature_extractor.py
3_data_augmentation.py
4_model.py
5_test.py

Fianlly, the plots.py script is responsible for producing all the visualisations displayed within the report.
