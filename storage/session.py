import json
import logging
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from api.models import Item

logger = logging.getLogger(__name__)


def _sanitise(text: str, max_len: int = 40) -> str:
    text = text.strip()
    text = re.sub(r"[^\w\s,.-]", "", text, flags=re.UNICODE)
    text = re.sub(r"[\s,]+", "_", text)
    return text[:max_len]


class Session:
    def __init__(
        self,
        base_dir: str,
        lon: float,
        lat: float,
        address: Optional[str] = None,
    ):
        self.lon = lon
        self.lat = lat
        self.address = address
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        if address:
            loc_str = _sanitise(address)
        else:
            loc_str = f"lon{lon:.4f}_lat{lat:.4f}"
        folder_name = f"{timestamp}_{loc_str}"
        self.folder = Path(base_dir) / folder_name
        self.images_dir = self.folder / "images"
        self.metadata_dir = self.folder / "metadata"
        self.folder.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(exist_ok=True)
        self.metadata_dir.mkdir(exist_ok=True)
        logger.info("Session folder: %s", self.folder)

    def save_item_metadata(self, item: Item) -> None:
        """Write item metadata JSON before downloading image."""
        path = self.metadata_dir / f"{item.id}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(item.to_dict(), f, indent=2, ensure_ascii=False)

    def image_path(self, item_id: str) -> Path:
        return self.images_dir / f"{item_id}.tif"

    def save_image_bytes(self, item_id: str, data: bytes) -> Path:
        """Write raw image bytes; skip if file already exists."""
        path = self.image_path(item_id)
        if path.exists():
            logger.info("Skipping %s, already saved", item_id)
            return path
        with open(path, "wb") as f:
            f.write(data)
        return path

    def write_session_json(
        self,
        query_info: dict,
        items: list[Item],
    ) -> None:
        direction_counts: Counter = Counter(item.direction for item in items)
        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query_info,
            "images_found": len(items),
            "images_saved": len(items),
            "direction_counts": dict(direction_counts),
            "session_folder": str(self.folder),
        }
        path = self.folder / "session.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    @staticmethod
    def load_from_path(path: str) -> tuple[list[Item], dict[str, np.ndarray]]:
        """
        Load items and decoded images from an existing session folder.
        Returns (items, images) where images maps item_id → BGR ndarray.
        """
        folder = Path(path)
        metadata_dir = folder / "metadata"
        images_dir = folder / "images"

        items: list[Item] = []
        images: dict[str, np.ndarray] = {}

        for meta_file in sorted(metadata_dir.glob("*.json")):
            with open(meta_file, encoding="utf-8") as f:
                feature = json.load(f)
            try:
                item = Item.from_feature(feature)
            except (KeyError, ValueError) as exc:
                logger.warning("Skipping malformed metadata %s: %s", meta_file.name, exc)
                continue

            img_path = images_dir / f"{item.id}.tif"
            if not img_path.exists():
                logger.warning("Image missing for %s, skipping", item.id)
                continue

            img_data = img_path.read_bytes()
            arr = np.frombuffer(img_data, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
                if img is not None and img.ndim == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            del img_data, arr  # free compressed bytes immediately; decoded array is in img
            if img is None:
                logger.warning("Could not decode image %s", img_path)
                continue

            items.append(item)
            images[item.id] = img

        logger.info("Loaded %d images from session: %s", len(items), path)
        return items, images
