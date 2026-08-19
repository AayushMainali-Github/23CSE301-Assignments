# Lab 5

Minimal kNN experiment for the ASVspoof speech deepfake project.

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
python -m lab5.run_experiment
python -m pytest -q lab5/tests
~~~

run_experiment.py performs the train/test split, runs custom and
scikit-learn kNN, compares several values of k, and writes the figures to
lab5/results/.

## Folder structure

~~~text
lab5/
|-- README.md
|-- requirements.txt
|-- features.py
|-- run_experiment.py
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

The dataset downloader and archive extraction scripts are intentionally not
included. The full audio dataset is also not needed to run the final
classification experiment because the prepared feature matrix is included.
