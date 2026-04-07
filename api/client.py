import logging
from typing import Optional

import cv2
import numpy as np
import requests

from config import BASE_URL, DAWA_URL
from api.models import Item

logger = logging.getLogger(__name__)


class SkraafotoClient:
    def __init__(self, token: str):
        self.token = token
        self.session = requests.Session()
        self.session.headers.update({"token": token})

    def search(
        self,
        bbox: list[float],
        collection: str,
        max_images: int = 30,
    ) -> list[Item]:
        """POST /search with pagination; returns up to max_images Items."""
        url = f"{BASE_URL}/search"
        body: dict = {
            "bbox": bbox,
            "collections": [collection],
            "limit": min(max_images, 100),
        }

        items: list[Item] = []
        seen_ids: set[str] = set()

        while True:
            logger.debug("POST %s  body=%s", url, body)
            resp = self.session.post(url, json=body)
            resp.raise_for_status()
            data = resp.json()

            new_in_page = 0
            for feature in data.get("features", []):
                fid = feature.get("id", "")
                if fid in seen_ids:
                    continue
                seen_ids.add(fid)
                try:
                    items.append(Item.from_feature(feature))
                    new_in_page += 1
                except (KeyError, ValueError) as exc:
                    logger.warning("Skipping malformed feature %s: %s", fid, exc)

            if len(items) >= max_images:
                items = items[:max_images]
                break

            # Pagination: next link uses POST with an updated body containing "pt"
            next_body = None
            for link in data.get("links", []):
                if link.get("rel") == "next" and link.get("method") == "POST":
                    next_body = link.get("body")
                    break

            if not next_body or new_in_page == 0:
                break

            # Use the body provided by the next link (it contains the pt cursor)
            body = next_body

        logger.info("Search returned %d items", len(items))
        return items

    def download_image(self, href: str) -> Optional[np.ndarray]:
        """Download a COG image and decode it to a BGR numpy array."""
        logger.debug("Downloading %s", href)
        resp = self.session.get(href, timeout=120)
        resp.raise_for_status()
        data = resp.content
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            logger.warning("cv2.imdecode failed for %s; trying IMREAD_UNCHANGED", href)
            img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
            if img is not None and img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return img

    def download_image_bytes(self, href: str) -> bytes:
        """Download raw bytes (for saving to disk)."""
        logger.debug("Downloading bytes %s", href)
        resp = self.session.get(href, timeout=120)
        resp.raise_for_status()
        return resp.content

    @staticmethod
    def geocode(address: str) -> tuple[float, float]:
        """Return (lon, lat) for a Danish address via DAWA API (no token needed)."""
        resp = requests.get(
            DAWA_URL,
            params={"q": address, "format": "geojson"},
            timeout=30,
        )
        resp.raise_for_status()
        features = resp.json().get("features", [])
        if not features:
            raise ValueError(f"Address not found: {address!r}")
        coords = features[0]["geometry"]["coordinates"]
        return float(coords[0]), float(coords[1])
