from pathlib import Path

from .features import make_csv


METADATA_FILE = Path("lab5/data/sample_metadata.csv")
OUTPUT_FILE = Path("lab5/features/features.csv")


def main():
    # the metadata file tells the extractor where every sample is
    if not METADATA_FILE.exists():
        raise FileNotFoundError(
            "run extract_samples.py before building features")

    # convert the sample audio into the 45-feature table
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    make_csv(str(METADATA_FILE), str(OUTPUT_FILE))
    print("feature file:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
