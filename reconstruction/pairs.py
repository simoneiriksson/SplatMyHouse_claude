import logging
import math
from dataclasses import dataclass

import numpy as np

import cv2

from reconstruction.camera import Camera

logger = logging.getLogger(__name__)

# Match stereo.py constants — used for pre-screening pair disparity.
_SGBM_LONG_EDGE = 800
_MAX_SGBM_DISP = 608      # pairs with estimated disparity above this are skipped
_ALTITUDE_AGL = 1300.0    # typical Skraafotos altitude above ground (m)


def _estimate_pair_disparity(cam_i: Camera, cam_j: Camera) -> float:
    """
    Return the expected maximum SGBM-scale disparity for a pair using the
    actual stereoRectify output.  Returns inf if rectification fails.

    stereoRectify can inflate f_rect by 2-3× for oblique cameras (it zooms
    into the overlap region).  stereo.py compensates by re-running with a
    smaller newImageSize when d_est is too large.  This function mirrors that
    logic so pairs rejected here are actually unworkable, not just large.

    The minimum scene depth is derived from the camera tilt angle:
      - Nadir cameras (looking straight down): depth ≈ altitude AGL
      - Oblique cameras at angle θ from nadir: depth ≈ altitude / cos(θ)
    """
    img_size = (
        min(cam_i.img_size[0], cam_j.img_size[0]),
        min(cam_i.img_size[1], cam_j.img_size[1]),
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

    # Minimum scene depth: altitude AGL / cos(tilt).
    tilt_cos = abs((R1[2, 2] + R2[2, 2]) / 2.0)
    tilt_cos = max(tilt_cos, 0.3)   # clamp for nearly-horizontal cameras
    min_depth = _ALTITUDE_AGL / tilt_cos

    return disp_coeff * sgbm_scale / min_depth


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


def _scene_overlap_fraction(
    cam_i: Camera,
    cam_j: Camera,
    scene_center: np.ndarray,
    scene_radius: float,
    ground_z: float,
    n_ring: int = 4,
    n_angle: int = 20,
) -> float:
    """
    Fraction of uniformly sampled scene-circle points (at ground_z) that project
    inside both cameras' image boundaries.  Returns a value in [0, 1].

    This is a better pair-selection signal than footprint-midpoint proximity:
    same-strip nadir pairs (along-track baseline) share only a strip of the
    scene circle, while cross-strip pairs (cross-track baseline) cover a
    2-D patch — and score higher here.
    """
    cx, cy = float(scene_center[0]), float(scene_center[1])
    radii = np.linspace(scene_radius / n_ring, scene_radius, n_ring)
    angles = np.linspace(0, 2 * np.pi, n_angle, endpoint=False)
    pts: list = [[cx + r * np.cos(a), cy + r * np.sin(a), float(ground_z)]
                 for r in radii for a in angles]
    pts.append([cx, cy, float(ground_z)])
    pts_arr = np.array(pts, dtype=np.float64)
    pts4 = np.hstack([pts_arr, np.ones((len(pts_arr), 1))])

    def _in_image(cam: Camera) -> np.ndarray:
        P = cam.K @ np.hstack([cam.R, -(cam.R @ cam.t.reshape(3, 1))])
        uvw = (P @ pts4.T).T
        in_front = uvw[:, 2] > 0
        uv = np.full((len(pts_arr), 2), -1.0)
        uv[in_front] = uvw[in_front, :2] / uvw[in_front, 2:3]
        W, H = float(cam.img_size[0]), float(cam.img_size[1])
        return (in_front
                & (uv[:, 0] >= 0) & (uv[:, 0] <= W)
                & (uv[:, 1] >= 0) & (uv[:, 1] <= H))

    both = _in_image(cam_i) & _in_image(cam_j)
    return float(both.sum()) / len(pts_arr)


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
    scene_radius: "float | None" = None,
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

            # ── Perpendicular-baseline filter ──────────────────────────────────
            # For oblique cameras, a baseline mostly parallel to the optical axis
            # (along-track) causes cv2.stereoRectify to produce a rectified frame
            # where the scene projects far outside the visible image window.
            # SGBM then matches unrelated image content → garbage 3D points.
            # Reject any pair where the baseline is > 60% along the average
            # optical-axis direction; score the rest by the perpendicular component.
            t_vec = cam_j.t - cam_i.t
            opt_i = cam_i.R.T[:, 2]   # cam_i optical axis in world (OpenCV +Z into scene)
            opt_j = cam_j.R.T[:, 2]
            avg_opt = (opt_i + opt_j) / 2.0
            avg_opt /= float(np.linalg.norm(avg_opt))
            along_axis = abs(float(t_vec @ avg_opt))
            perp_baseline = math.sqrt(max(0.0, baseline ** 2 - along_axis ** 2))
            # For south/north cameras (45° tilt), a purely N-S baseline has
            # perp_fraction = cos(45°) = 0.707 which is along-track and produces
            # degenerate rectification.  A purely E-W baseline gives perp=1.0 (ideal).
            # Threshold 0.8 rejects along-track oblique pairs while accepting cross-track
            # pairs and all nadir pairs (which always have perp_fraction ≈ 1.0).
            if perp_baseline < 0.8 * baseline:
                rejected.append(
                    f"  [{cam_i.direction}/{cam_i.item_id[-8:]} ↔ "
                    f"{cam_j.direction}/{cam_j.item_id[-8:]}] "
                    f"perp_baseline={perp_baseline:.0f}m ({100*perp_baseline/baseline:.0f}% of {baseline:.0f}m) "
                    f"REJECTED (along-track pair)"
                )
                continue

            # ── Altitude-difference filter ─────────────────────────────────────
            # Pairs where one camera is much higher/lower than the other produce
            # degenerate rectification: the scene falls outside one camera's image
            # even though both cameras look in the same direction.  Reject if the
            # altitude difference exceeds 40% of the total baseline.
            dz = abs(float(cam_j.t[2] - cam_i.t[2]))
            if dz > 0.4 * baseline:
                rejected.append(
                    f"  [{cam_i.direction}/{cam_i.item_id[-8:]} ↔ "
                    f"{cam_j.direction}/{cam_j.item_id[-8:]}] "
                    f"dz={dz:.0f}m ({100*dz/baseline:.0f}% of {baseline:.0f}m) "
                    f"REJECTED (altitude mismatch)"
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

            # Score by the perpendicular (geometrically effective) baseline.
            # This prefers pairs with good depth sensitivity over along-track pairs.
            bs = _baseline_score(perp_baseline, min_baseline, max_baseline)
            db = _direction_bonus(cam_i.direction, cam_j.direction)

            # Scene-coverage score: fraction of the scene circle visible in both
            # cameras.  This prefers cross-strip pairs (2-D patch coverage) over
            # same-strip pairs (narrow along-track strip) and naturally handles
            # proximity — cameras that don't see the target score near zero.
            # Falls back to footprint-midpoint proximity when scene_radius is absent.
            coverage = 1.0
            if scene_center is not None and scene_radius is not None:
                coverage = _scene_overlap_fraction(
                    cam_i, cam_j, scene_center, scene_radius, ground_z,
                )
                logger.debug(
                    "  %s/%s ↔ %s/%s  scene_overlap=%.2f",
                    cam_i.direction, cam_i.item_id[-8:],
                    cam_j.direction, cam_j.item_id[-8:],
                    coverage,
                )
            elif scene_center is not None:
                fp_i = _ground_footprint(cam_i, ground_z=ground_z)
                fp_j = _ground_footprint(cam_j, ground_z=ground_z)
                mid = (fp_i + fp_j) / 2.0
                cx, cy = float(scene_center[0]), float(scene_center[1])
                dist_to_target = float(np.linalg.norm(mid - np.array([cx, cy])))
                coverage = float(np.exp(-0.5 * (dist_to_target / 500.0) ** 2))

            score = bs * db * coverage
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
