# Lab 5

Minimal kNN experiment for the ASVspoof speech deepfake project.

The downloaded ASVspoof files are kept locally in `data/`. They are not
tracked by Git because the folder is about 33 GB.

The prepared feature file is already included:

~~~text
features/features.csv
~~~

It contains 45 librosa features for 200 utterances:

- 100 bonafide
- 100 spoof

Run from the repository root:

~~~bash
python -m pip install -r lab5/requirements.txt
python -m lab5.extract_samples
python -m lab5.build_features
python -m lab5.run_experiment
python -m pytest -q lab5/tests
~~~

`extract_samples.py` reads the ASVspoof key file and copies 100 bonafide and
100 spoof files from the audio archives into `data/sample_audio/`.

`build_features.py` sends the sample metadata to `features.py`. Librosa then
creates the MFCC, spectral, pitch, voicing, and duration features in
`features/features.csv`.

`run_experiment.py` performs the train/test split, runs custom and
scikit-learn kNN, compares several values of k, and writes the figures to
`lab5/results/`.

## Folder structure

~~~text
lab5/
|-- README.md
|-- requirements.txt
|-- features.py
|-- extract_samples.py
|-- build_features.py
|-- run_experiment.py
|-- data/
|   |-- audio/              full ASVspoof archives
|   |-- keys/               labels and protocol files
|   |-- sample_audio/       balanced 200-file sample
|   +-- sample_metadata.csv
|-- features/
|   +-- features.csv
|-- knn/
|   |-- encoding.py
|   |-- imputation.py
|   |-- distances.py
|   |-- sorting.py
|   |-- neighbors.py
|   |-- preprocessing.py
|   |-- classifier.py
|   +-- __init__.py
+-- tests/
    +-- test_knn.py
~~~

The dataset downloader is not included because the full archives are already
present locally. The sample extraction and feature conversion scripts are
included. The final classification experiment can also run directly from the
prepared feature matrix if the raw data is not available.
