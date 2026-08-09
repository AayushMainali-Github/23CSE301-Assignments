# DMToolkit - Lab 04

Standalone GenAI-assisted package for repeating Lab 03 experiments, unit testing both the original and AI-assisted implementations, and comparing K-means performance.

## Run
```bash
pip install -r requirements.txt
python -m pytest -v
python experiments/run_marketing.py
python experiments/compare_kmeans.py
```

Generated outputs are written to `outputs/figures` and `outputs/results`.
