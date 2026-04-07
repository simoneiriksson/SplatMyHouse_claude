import logging
import math
from dataclasses import dataclass

import numpy as np

import cv2

from reconstruction.camera import Camera, TARGET_LONG_EDGE

logger = logging.getLogger(__name__)

# Match stereo.py constants — used for pre-screening pair disparity.
_SGBM_LONG_EDGE = 800
_MAX_SGBM_DISP = 512    # pairs with estimated disparity above this are skipped
_MIN_SCENE_DEPTH = 900.0  # conservative minimum camera-to-ground depth (m)


def _estimate_pair_disparity(cam_i: Camera, cam_j: Camera) -> float:
    """
    Return the expected maximum SGBM-scale disparity for a pair using the
    actual stereoRectify output.  Returns inf if rectification fails.

    This is called during pair selection to pre-screen pairs that SGBM
    cannot process (disparity range exceeds image width).
    """
    img_size = (
        min(cam_i.image.shape[1], cam_j.image.shape[1]),
        min(cam_i.image.shape[0], cam_j.image.shape[0]),
    )
    K1, K2 = cam_i.K, cam_j.K
    R1, R2 = cam_i.R, cam_j.R
    t1, t2 = cam_i.t, cam_j.t
    R_rel = R2 @ R1.T
    t_rel = (R2 @ (t1 - t2)).reshape(3, 1)
    dist_zeros = np.zeros(5)
    try:
        _, _, _, P2_rect, _, _, _ = cv2.stereoRectify(
            K1, dist_zeros, K2, dist_zeros,
            img_size, R_rel, t_rel,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
        )
    except cv2.error:
        return float("inf")
    Tx = abs(float(P2_rect[0, 3]))
    Ty = abs(float(P2_rect[1, 3]))
    disp_coeff = max(Tx, Ty)
    sgbm_scale = _SGBM_LONG_EDGE / max(img_size)
    return disp_coeff * sgbm_scale / _MIN_SCENE_DEPTH


def _direction_bonus(d1: str, d2: str) -> float:
    if d1 == d2 and d1 == "":
        return 0.0
    oblique = {"north", "south", "east", "west"}
    if d1 == d2:
        return 1.2   # same direction, adjacent strip
    if "nadir" in (d1, d2):
        return 1.5   # nadir + oblique: best for ground + wall coverage
    if {d1, d2} <= oblique:
        return 1.1   # cross-oblique: good for building corners
    return 1.0


def _baseline_score(baseline: float, min_b: float, max_b: float) -> float:
    peak = 300.0
    if baseline <= min_b or baseline >= max_b:
        return 0.0
    if baseline <= peak:
        return (baseline - min_b) / (peak - min_b)
    return (max_b - baseline) / (max_b - peak)


@dataclass
class Pair:
    i: int
    j: int
    score: float
    baseline: float
    cam_i: Camera
    cam_j: Camera


def _ground_footprint(cam: Camera, ground_z: float = -1000.0) -> np.ndarray:
    """
    Compute the approximate ground intersection of the camera's principal ray.
    Returns a 2-element array [E, N] in ENU metres.

    ground_z: Z coordinate of the ground plane in local ENU (negative for
    cameras flying above origin).  Defaults to -1000 m which is a conservative
    estimate for Skråfoto (typically 500–1500 m AGL).
    """
    d = cam.R.T[:, 2]   # principal-ray direction in world space
    t = cam.t
    # Ray: p = t + lam * d, solve for lam where p[2] == ground_z
    dz = d[2]
    if abs(dz) < 1e-6:
        # Nearly horizontal ray — footprint is far away; return camera XY
        return t[:2].copy()
    lam = (ground_z - t[2]) / dz
    if lam < 0:
        # Ray points away from ground — return camera XY
        return t[:2].copy()
    return t[:2] + lam * d[:2]


def compute_pairs(
    cameras: list[Camera],
    min_baseline: float,
    max_baseline: float,
    max_pairs: int,
    debug: bool = False,
    scene_center: "np.ndarray | None" = None,
    ground_z: float = -1000.0,
) -> list[Pair]:
    """
    Evaluate all camera pairs and return top-scoring ones within baseline range.

    Diversity is enforced by capping how many pairs of each direction
    combination (e.g. nadir↔nadir, nadir↔north, north↔east) can be selected,
    so the final set covers all viewing-angle combinations available.
    """
    n = len(cameras)
    candidates: list[Pair] = []
    rejected: list[str] = []

    if debug:
        baselines = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                b = float(np.linalg.norm(cameras[i].t - cameras[j].t))
                baselines[i, j] = baselines[j, i] = b
        header = "  ".join(f"{c.direction[:3]:>5}" for c in cameras)
        logger.debug("Baseline matrix (m):\n        %s", header)
        for i, cam in enumerate(cameras):
            row = "  ".join(f"{baselines[i,j]:5.0f}" for j in range(n))
            logger.debug("%5s   %s", cam.direction[:5], row)

    for i in range(n):
        for j in range(i + 1, n):
            cam_i, cam_j = cameras[i], cameras[j]
            baseline = float(np.linalg.norm(cam_i.t - cam_j.t))

            if baseline < min_baseline or baseline > max_baseline:
                rejected.append(
                    f"  [{cam_i.direction}/{cam_i.item_id[-8:]} ↔ "
                    f"{cam_j.direction}/{cam_j.item_id[-8:]}] "
                    f"baseline={baseline:.0f}m REJECTED (out of range)"
                )
                continue

            # Angle between optical axes — allow up to 75° to include
            # cross-oblique pairs (north↔east ≈ 60°, previously rejected)
            z1 = cam_i.R[2, :]
            z2 = cam_j.R[2, :]
            cos_angle = float(np.clip(z1 @ z2, -1, 1))
            angle_deg = math.degrees(math.acos(cos_angle))
            if angle_deg > 75.0:
                rejected.append(
                    f"  [{cam_i.direction}/{cam_i.item_id[-8:]} ↔ "
                    f"{cam_j.direction}/{cam_j.item_id[-8:]}] "
                    f"angle={angle_deg:.1f}° REJECTED (> 75°)"
                )
                continue

            # SGBM disparity pre-screen using actual stereoRectify output.
            # Pairs whose disparity range exceeds SGBM's practical limit only
            # produce noise and waste compute time — skip them early.
            d_est = _estimate_pair_disparity(cam_i, cam_j)
            if d_est > _MAX_SGBM_DISP:
                rejected.append(
                    f"  [{cam_i.direction}/{cam_i.item_id[-8:]} ↔ "
                    f"{cam_j.direction}/{cam_j.item_id[-8:]}] "
                    f"est_disparity={d_est:.0f}px REJECTED (> {_MAX_SGBM_DISP})"
                )
                continue

            bs = _baseline_score(baseline, min_baseline, max_baseline)
            db = _direction_bonus(cam_i.direction, cam_j.direction)

            # Proximity bonus: prefer pairs whose ground footprints are near target.
            # Uses the corrected ground plane so oblique cameras are scored by
            # where they actually look, not where they are positioned.
            prox = 1.0
            if scene_center is not None:
                fp_i = _ground_footprint(cam_i, ground_z=ground_z)
                fp_j = _ground_footprint(cam_j, ground_z=ground_z)
                mid = (fp_i + fp_j) / 2.0
                cx, cy = float(scene_center[0]), float(scene_center[1])
                dist_to_target = float(np.linalg.norm(mid - np.array([cx, cy])))
                prox = float(np.exp(-0.5 * (dist_to_target / 500.0) ** 2))
                logger.debug(
                    "  %s/%s ↔ %s/%s  footprint_mid=(%.0f,%.0f) dist=%.0fm prox=%.2f",
                    cam_i.direction, cam_i.item_id[-8:],
                    cam_j.direction, cam_j.item_id[-8:],
                    mid[0], mid[1], dist_to_target, prox,
                )

            score = bs * db * prox
            candidates.append(Pair(i=i, j=j, score=score, baseline=baseline,
                                   cam_i=cam_i, cam_j=cam_j))

    if not candidates and rejected:
        logger.warning("All pairs were filtered out. Rejection reasons:")
        for r in rejected:
            logger.warning(r)
        return []

    candidates.sort(key=lambda p: p.score, reverse=True)

    # ── Diversity cap ─────────────────────────────────────────────────────────
    # Limit each direction combination (e.g. nadir↔north) to at most
    # max_per_combo slots so nadir↔nadir pairs cannot flood the selection.
    max_per_combo = max(2, max_pairs // 6)
    combo_counts: dict[frozenset, int] = {}
    selected: list[Pair] = []
    for pair in candidates:
        key = frozenset({pair.cam_i.direction, pair.cam_j.direction})
        if combo_counts.get(key, 0) < max_per_combo:
            selected.append(pair)
            combo_counts[key] = combo_counts.get(key, 0) + 1
        if len(selected) >= max_pairs:
            break

    logger.info(
        "Selected %d / %d candidate pairs (cap %d/combo):",
        len(selected), len(candidates), max_per_combo,
    )
    for k, pair in enumerate(selected, 1):
        fp_info = ""
        if scene_center is not None:
            fp_i = _ground_footprint(pair.cam_i, ground_z=ground_z)
            fp_j = _ground_footprint(pair.cam_j, ground_z=ground_z)
            mid = (fp_i + fp_j) / 2.0
            cx, cy = float(scene_center[0]), float(scene_center[1])
            dist = float(np.linalg.norm(mid - np.array([cx, cy])))
            fp_info = f"  footprint_dist={dist:.0f}m"
        logger.info(
            "  Pair %02d: %s/%s ↔ %s/%s  baseline=%.0fm  score=%.3f%s",
            k,
            pair.cam_i.direction, pair.cam_i.item_id[-8:],
            pair.cam_j.direction, pair.cam_j.item_id[-8:],
            pair.baseline, pair.score, fp_info,
        )

    # Log direction-combination summary
    summary: dict[str, int] = {}
    for p in selected:
        key = f"{p.cam_i.direction}↔{p.cam_j.direction}"
        summary[key] = summary.get(key, 0) + 1
    logger.info("Direction mix: %s", "  ".join(f"{k}:{v}" for k, v in sorted(summary.items())))

    return selected
