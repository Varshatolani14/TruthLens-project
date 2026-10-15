"""Run the reproducible TruthLens pipeline from the project root."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command):
    print("\n$", " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", help="Run only the small smoke-test dataset.")
    parser.add_argument("--skip-download", action="store_true", help="Use already downloaded raw files.")
    args = parser.parse_args()

    if not args.skip_download and not args.sample:
        run([sys.executable, "scripts/download_data.py"])

    preprocess = [sys.executable, "src/preprocess.py"]
    baseline = [sys.executable, "src/train_baseline.py"]
    evaluate = [sys.executable, "src/evaluate.py"]

    if args.sample:
        preprocess.append("--sample")
        baseline.append("--sample")
        evaluate.append("--sample")

    run(preprocess)
    run(baseline)

    if args.sample:
        run(evaluate)
        print("\nSmoke pipeline completed. Do not use these metrics in the report.")
        return

    run([sys.executable, "src/train_transformer.py"])
    run(evaluate)
    run([sys.executable, "src/explain.py"])
    run([sys.executable, "scripts/build_submission_docs.py"])
    print("\nFull pipeline completed.")


if __name__ == "__main__":
    main()
