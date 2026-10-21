"""Console and optional file logging for the traffic monitor."""
import logging
from pathlib import Path

logger = logging.getLogger("traffic_monitor")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    logger.addHandler(handler)


def enable_file_logging(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "app.log"
    if any(isinstance(h, logging.FileHandler) and h.baseFilename == str(target.resolve()) for h in logger.handlers):
        return
    file_handler = logging.FileHandler(target, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(file_handler)
