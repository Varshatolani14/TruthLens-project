"""Download and validate the official ISOT fake-news dataset."""

from pathlib import Path
import sys
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)

DATA_URL = "https://onlineacademiccommunity.uvic.ca/isot/wp-content/uploads/sites/7295/2023/03/News-_dataset.zip"
ZIP_PATH = RAW / "isot_news_dataset.zip"


def validate():
    for name in ("True.csv", "Fake.csv"):
        path = RAW / name
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}")
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            header = f.readline().lower()
        if "text" not in header:
            raise ValueError(f"{name} does not look like the expected ISOT CSV")


def main():
    try:
        if not ZIP_PATH.exists():
            print("Downloading the official ISOT dataset...")
            urllib.request.urlretrieve(DATA_URL, ZIP_PATH)

        with zipfile.ZipFile(ZIP_PATH) as zf:
            zf.extractall(RAW)

        validate()
        print("Dataset is ready in data/raw/")
    except Exception as exc:
        print("Automatic download failed:")
        print(f"  {exc}")
        print("\nDownload the dataset from the official ISOT page:")
        print("  https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/")
        print("Then place True.csv and Fake.csv in data/raw/ and run this script again.")
        sys.exit(1)


if __name__ == "__main__":
    main()
