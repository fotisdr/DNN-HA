## DNN-based hearing-aid strategy for real-time processing

This repository contains a deep-neural-network hearing-aid (DNN-HA) processing strategy that can provide individualised sound processing for the audiogram of a listener using a single DNN model architecture. The supporting paper will be presented in ICASSP 2023 – 2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), scheduled for 4-9 June 2023 in Rhodes Island, Greece:

> "A DNN-based hearing-aid strategy for real-time processing: One size fits all" by Fotios Drakopoulos, Arthur Van Den Broucke, Sarah Verhulst

The DNN-HA strategy uses an audio signal and an audiogram as inputs to process sound in real-time, such as to provide optimal HA processing that compensates for the elevated hearing thresholds of an individual listener. The `CNN-HA-12layers` folder contains the trained DNN-HA model, while an example script `test_DNN-HA_wavfile.py` is provided for running and evaluating the model. This repository also contains this `README.md` document, a license file, and the supplementary files that are necessary for the execution of the model.

The DNN-HA model was trained using the differentiable closed-loop framework that we describe [here](https://doi.org/10.48550/arXiv.2207.07091). The framework is based on CoNNear, a DNN version of a biophysically realistic model of the auditory system. The code for the CoNNear periphery model can be found [here](https://github.com/HearingTechnology/CoNNear_periphery). More details about the DNN-HA model architecture and the training procedure will be made available upon publication of the corresponding paper as part of ICASSP 2023. 

## How to use the hearing-aid model

The `test_DNN-HA_wavfile.py` script provides a usage example of the DNN-HA model and can be used to process an example sentence (`00131.wav`) from the Flemish Matrix presented at 70 dB SPL. Within this script, the audiogram of a listener ([L44](https://github.com/fotisdr/DNN-HA/blob/main/test_DNN-HA_wavfile.py#L44)) can be specified across 8 frequencies (125-8000 Hz) and be used to process a wavfile ([L45](https://github.com/fotisdr/DNN-HA/blob/main/test_DNN-HA_wavfile.py#L45)) presented at the desired intensity level ([L47](https://github.com/fotisdr/DNN-HA/blob/main/test_DNN-HA_wavfile.py#L47)). The script visualises the audio waveform before and after processing, and also the unprocessed and processed spectra using [PyOctaveBand](https://github.com/jmrplens/PyOctaveBand). By default, the processed sound is also saved as a wavfile under the `wavfiles` folder. The script also provides the choice to add noise at a desired SNR ([L48](https://github.com/fotisdr/DNN-HA/blob/main/test_DNN-HA_wavfile.py#L48)) and to process sound in frames ([L53](https://github.com/fotisdr/DNN-HA/blob/main/test_DNN-HA_wavfile.py#L53)), such that the DNN-HA model can be evaluated for real-time and/or low-latency processing. More information can be found in the `test_DNN-HA_wavfile.py` script.

To run the example script and the DNN-HA model in Python, Numpy, Scipy and Tensorflow are necessary. We used a conda environment (v4.14.0) that included the following versions: 
+ Python 3.9.13
+ Numpy 1.21.5
+ Scipy 1.7.3
+ Tensorflow 2.7.0

----
## Citation
If you use this code, please cite the corresponding paper or this repository:

Fotios Drakopoulos, Arthur Van Den Broucke, & Sarah Verhulst. (2023). DNN-HA: A DNN-based hearing-aid strategy for real-time processing (v1.0). Zenodo. https://doi.org/10.5281/zenodo.7717218

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7717218.svg)](https://doi.org/10.5281/zenodo.7717218)

##
For questions, please reach out to one of the corresponding authors:

* Fotios Drakopoulos: f.drakopoulos@ucl.ac.uk
* Sarah Verhulst: s.verhulst@ugent.be

> This work was supported by European Research Council ERC-StG-678120 (RobSpear) and FWO grant G063821N Machine Hearing 2.0.
