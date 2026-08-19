import csv
import io
import subprocess
import warnings
from pathlib import Path

import librosa
import numpy as np


def summary(values):
    # keep only normal numeric values
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    # return zero when a feature has no usable values
    if len(values) == 0:
        return [0.0, 0.0]

    # save the average and the spread of the feature
    return [float(np.mean(values)), float(np.std(values))]


def names():
    # make the names in the same order as the values below
    result = []

    # each MFCC gives one mean and one standard deviation
    for i in range(1, 14):
        result += [f"mfcc_{i}_mean", f"mfcc_{i}_std"]

    # these are the short time spectral features
    for name in ["centroid", "bandwidth", "rolloff", "flatness", "zcr", "rms"]:
        result += [f"{name}_mean", f"{name}_std"]

    # these are the pitch and voicing features
    result += ["f0_mean", "f0_std", "f0_min", "f0_max", "f0_median"]
    result += ["voiced_ratio", "duration"]
    return result


def load_audio(path, sr, seconds):
    # try to read the audio normally with librosa
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return librosa.load(path, sr=sr, mono=True, duration=seconds)
    except Exception:
        # use the flac command only if librosa cannot read a flac file
        if Path(path).suffix.lower() != ".flac":
            raise

        # decode the file into memory and pass it back to librosa
        command = ["flac", "--decode", "--decode-through-errors",
                   "--stdout", str(path)]
        output = subprocess.run(command, check=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.DEVNULL)
        return librosa.load(io.BytesIO(output.stdout), sr=sr,
                            mono=True, duration=seconds)


def extract(path, sr, seconds):
    # load one audio file as mono audio
    y, sr = load_audio(path, sr, seconds)

    # these settings control the small audio frames
    n_fft = 2048
    hop = 512

    # calculate the 13 MFCC rows
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13,
                                n_fft=n_fft, hop_length=hop)

    # calculate simple spectral features for every frame
    spectral = [
        librosa.feature.spectral_centroid(y=y, sr=sr,
                                          n_fft=n_fft, hop_length=hop)[0],
        librosa.feature.spectral_bandwidth(y=y, sr=sr,
                                           n_fft=n_fft, hop_length=hop)[0],
        librosa.feature.spectral_rolloff(y=y, sr=sr,
                                         n_fft=n_fft, hop_length=hop)[0],
        librosa.feature.spectral_flatness(y=y, n_fft=n_fft,
                                          hop_length=hop)[0],
        librosa.feature.zero_crossing_rate(y, hop_length=hop)[0],
        librosa.feature.rms(y=y, frame_length=n_fft, hop_length=hop)[0],
    ]

    # turn every time-varying feature into a few fixed values
    result = []
    for row in mfcc:
        result += summary(row)
    for row in spectral:
        result += summary(row)

    # estimate the fundamental frequency for each frame
    f0 = librosa.yin(y, fmin=65, fmax=400, sr=sr,
                     frame_length=n_fft, hop_length=hop)

    # remove pitch values that are outside the normal voice range
    energy = spectral[-1]
    voiced = np.isfinite(f0) & (f0 > 65) & (f0 < 400)
    voiced = voiced & (energy > np.percentile(energy, 20))
    f0 = f0[voiced]

    # if no pitch was found, use zeros for the five pitch values
    if len(f0) == 0:
        result += [0.0] * 5
    else:
        # save basic statistics of the detected pitch
        result += [float(np.mean(f0)), float(np.std(f0)),
                   float(np.min(f0)), float(np.max(f0)),
                   float(np.median(f0))]

    # voiced ratio tells how much of the audio looks voiced
    # duration is the length of the audio that was loaded
    result += [float(np.mean(voiced)), float(len(y) / sr)]
    return np.nan_to_num(result).tolist()


def make_csv(metadata, output):
    # read the metadata file and extract features for each audio path
    rows = []
    with open(metadata, newline="") as file:
        for row in csv.DictReader(file):
            values = extract(row["audio_path"], 16000, 3)

            # keep the useful information beside the feature values
            item = {"utt_id": row["utt_id"],
                    "label": row["label"],
                    "audio_path": row["audio_path"]}
            item.update(dict(zip(names(), values)))
            rows.append(item)

    # write one row per audio file
    with open(output, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
