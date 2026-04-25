import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from api.models import Item

logger = logging.getLogger(__name__)

TARGET_LONG_EDGE = 4000  # Resize images to this long edge for triangulation (~0.25 m GSD)


@dataclass
class Camera:
    item_id: str
    direction: str
    K: np.ndarray        # 3×3 intrinsic matrix
    R: np.ndarray        # 3×3 rotation (world→camera)
    t: np.ndarray        # 3×1 camera centre in local ENU metres
    P: np.ndarray        # 3×4 projection matrix
    image: np.ndarray    # BGR image array
    img_size: tuple      # (W, H) in pixels — stored separately to avoid touching image array
    session_path: Optional[Path] = None


def build_intrinsics(item: Item) -> np.ndarray:
    """Build 3×3 intrinsic matrix from pers:interior_orientation."""
    interior = item.interior
    # pixel_spacing in mm/pixel
    fx = interior.focal_length / interior.pixel_spacing[0]
    fy = interior.focal_length / interior.pixel_spacing[1]
    w, h = interior.sensor_array_dimensions
    # principal_point_offset in mm → pixels.
    # The offset is in the photogrammetric image frame (y up); pixel frame has y down,
    # so ppy is negated when converting to cy.
    ppx_px = interior.principal_point_offset[0] / interior.pixel_spacing[0]
    ppy_px = interior.principal_point_offset[1] / interior.pixel_spacing[1]
    cx = w / 2.0 + ppx_px
    cy = h / 2.0 - ppy_px
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float64)
    return K


def _rotation_from_matrix(rotation_matrix_flat: list[float]) -> np.ndarray:
    """Reshape flat 9-element row-major list to 3×3 rotation matrix."""
    return np.array(rotation_matrix_flat, dtype=np.float64).reshape(3, 3)


def build_cameras(
    items: list[Item],
    images: dict[str, "np.ndarray"],
    session_folder: Optional[Path] = None,
) -> tuple[list[Camera], np.ndarray]:
    """Returns (cameras, origin_utm) where origin_utm is the mean camera
    position in UTM EPSG:25832 [E, N, h] used as the local ENU origin."""
    """
    Build Camera objects for all items that have matching images.
    Camera positions (pers:perspective_center) are in UTM EPSG:25832 [E, N, h].
    We convert to a local ENU frame by subtracting the mean camera position.
    """
    import cv2

    # Filter to items with available images
    valid = [(item, images[item.id]) for item in items if item.id in images]
    if not valid:
        raise ValueError("No images available to build cameras.")

    # Compute scene origin in UTM (mean of all camera positions)
    utm_positions = np.array(
        [item.perspective_center for item, _ in valid], dtype=np.float64
    )   # N×3: [E, N, h]
    origin_utm = utm_positions.mean(axis=0)

    cameras: list[Camera] = []
    for item, img in valid:
        utm = np.array(item.perspective_center, dtype=np.float64)
        # Local ENU: subtract origin (UTM is already ~planar in metres)
        t_local = utm - origin_utm   # [dE, dN, dH]

        K = build_intrinsics(item)

        # The API provides R_c2w in the photogrammetric convention:
        # camera +Z points away from the scene (toward the sky for nadir cameras),
        # and camera +Y points up in the image plane.
        # OpenCV convention requires camera +Z into the scene and +Y down in the image.
        # Conversion: flip both Y and Z axes in camera space.
        R_c2w = _rotation_from_matrix(item.rotation_matrix)
        _flip = np.diag([1.0, -1.0, -1.0])
        R = _flip @ R_c2w.T   # world→camera (OpenCV convention)

        # Resize image to TARGET_LONG_EDGE for speed
        if TARGET_LONG_EDGE is not None:
            h_img, w_img = img.shape[:2]
            long_edge = max(h_img, w_img)
            if long_edge > TARGET_LONG_EDGE:
                scale = TARGET_LONG_EDGE / long_edge
                new_w = int(round(w_img * scale))
                new_h = int(round(h_img * scale))
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                K = K.copy()
                K[0, :] *= scale   # fx, cx
                K[1, :] *= scale   # fy, cy

        # Projection matrix P = K @ [R | -R@t]
        t_col = t_local.reshape(3, 1)
        P = K @ np.hstack([R, -R @ t_col])

        session_path = None
        if session_folder is not None:
            session_path = Path(session_folder) / "images" / f"{item.id}.tif"

        cameras.append(Camera(
            item_id=item.id,
            direction=item.direction,
            K=K,
            R=R,
            t=t_local,
            P=P,
            image=img,
            img_size=(img.shape[1], img.shape[0]),
            session_path=session_path,
        ))

    logger.info("Built %d cameras", len(cameras))
    return cameras, origin_utm
