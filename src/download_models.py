"""Download public Darknet YOLOv3 assets into the original src/Data folder.

Run: python -m src.download_models
The ~237 MB weights are never added to Git; inspect licenses upstream.
"""
import argparse
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "Data"
ASSETS = {
    "yolov3.cfg": ("https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3.cfg", 1000),
    "coco.names": ("https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names", 100),
    "yolov3.weights": ("https://pjreddie.com/media/files/yolov3.weights", 200_000_000),
}


def download_models(model_dir: Path = MODEL_DIR) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    for filename, (url, min_size) in ASSETS.items():
        destination = model_dir / filename
        if destination.is_file() and destination.stat().st_size >= min_size:
            print(f"Exists: {destination}")
            continue
        temporary = model_dir / (filename + ".tmp")
        print(f"Downloading {filename} from {url} ...", flush=True)
        try:
            with urlopen(url, timeout=60) as response, temporary.open("wb") as out:
                while block := response.read(1024 * 1024):
                    out.write(block)
            if temporary.stat().st_size < min_size:
                raise ValueError(f"Downloaded {filename} appears truncated")
            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)
    print("Model assets ready in src/Data/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dir", type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    download_models(args.dir)
