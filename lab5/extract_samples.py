import csv
import tarfile
from pathlib import Path


KEY_FILE = Path("lab5/data/keys/keys/DF/CM/trial_metadata.txt")
AUDIO_DIR = Path("lab5/data/audio")
SAMPLE_DIR = Path("lab5/data/sample_audio")
METADATA_FILE = Path("lab5/data/sample_metadata.csv")
LIMIT = 100


def read_labels():
    # take the first 100 files from each class
    rows = []
    counts = {"bonafide": 0, "spoof": 0}

    with open(KEY_FILE) as file:
        for line in file:
            parts = line.split()

            # ignore empty or incomplete lines
            if len(parts) < 6:
                continue

            utt_id = parts[1]
            label = parts[5]

            if label not in counts:
                continue

            if counts[label] >= LIMIT:
                continue

            rows.append([utt_id, label])
            counts[label] += 1

            if counts["bonafide"] == LIMIT and counts["spoof"] == LIMIT:
                break

    return rows


def extract(rows):
    # make the sample folder if it is not already there
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    wanted = {}

    for utt_id, label in rows:
        wanted[utt_id] = label

    found = {}

    # look through the four ASVspoof archive files
    for archive in sorted(AUDIO_DIR.glob("*.tar.gz")):
        with tarfile.open(archive, "r:gz") as file:
            for member in file:
                if not member.isfile():
                    continue

                utt_id = Path(member.name).stem
                if utt_id not in wanted or utt_id in found:
                    continue

                source = file.extractfile(member)
                output = SAMPLE_DIR / (utt_id + ".flac")

                # copy only the selected audio file
                with open(output, "wb") as target:
                    target.write(source.read())

                found[utt_id] = wanted[utt_id]

        if len(found) == len(wanted):
            break

    # make sure the requested sample was found in the archives
    if len(found) != len(wanted):
        missing = sorted(set(wanted) - set(found))
        raise ValueError("audio files not found: " + str(missing[:5]))

    # save the labels and paths used by the feature script
    with open(METADATA_FILE, "w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["utt_id", "label", "audio_path"])

        for utt_id, label in rows:
            writer.writerow([
                utt_id,
                label,
                str(SAMPLE_DIR / (utt_id + ".flac")),
            ])


def main():
    # read the labels and extract a balanced 200-file sample
    rows = read_labels()
    extract(rows)
    print("sample files:", len(rows))
    print("metadata file:", METADATA_FILE)


if __name__ == "__main__":
    main()
