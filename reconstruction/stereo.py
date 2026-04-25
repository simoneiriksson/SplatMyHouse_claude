import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from reconstruction.camera import Camera

logger = logging.getLogger(__name__)

MIN_VALID_POINTS = 500
MAX_SCENE_RADIUS = 5000.0  # metres — broad validity check

# SGBM runs on images scaled to this long edge for speed.
# Feature matching, rectification (H1/H2), and triangulation all happen at
# the full TARGET_LONG_EDGE (set in camera.py).  Only SGBM uses this smaller
# scale; the resulting disparity is scaled back before triangulation.
SGBM_LONG_EDGE = 800


def _calibrated_rectify(
    cam1: Camera,
    cam2: Camera,
    img_size: tuple[int, int],
) -> Optional[tuple]:
    """
    Compute rectification homographies from known camera geometry.

    Uses cv2.stereoRectify (calibrated), which is more robust than the
    uncalibrated version for nadir pairs where feature matches can be
    degenerate (collinear).

    Returns a 13-element tuple:
      (H1, H2, disp_axis, cam_swapped, disp_coeff,
       roi1, roi2, R1_rect, R2_rect, f_rect, cx_rect, cy_rect, rect_size)

    where:
      - H1, H2 map ORIGINAL-image pixels → RECTIFIED-image (rect_size) pixels
      - rect_size: (W, H) of the rectified images (may differ from img_size when
        stereoRectify down-scales to control disparity range)
      - disp_axis: 0 = horizontal stereo, 1 = vertical stereo
    Returns None on failure.
    """
    K1, R1, t1 = cam1.K, cam1.R, cam1.t
    K2, R2, t2 = cam2.K, cam2.R, cam2.t

    # Relative pose: p_cam2 = R_rel @ p_cam1 + t_rel
    R_rel = R2 @ R1.T
    t_rel = (R2 @ (t1 - t2)).reshape(3, 1)

    dist_zeros = np.zeros(5)

    try:
        R1_rect, R2_rect, P1_rect, P2_rect, _Q, roi1, roi2 = cv2.stereoRectify(
            K1, dist_zeros, K2, dist_zeros,
            img_size, R_rel, t_rel,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0,
        )
    except cv2.error as exc:
        logger.warning("stereoRectify failed: %s", exc)
        return None

    rect_size = img_size

    # New (rectified) intrinsic matrix from the projection matrix
    K1_new = P1_rect[:3, :3]
    K2_new = P2_rect[:3, :3]

    # Detect disparity axis: horizontal (0) or vertical (1).
    # For nadir cameras with a N-S baseline the rectified pair is vertical
    # (epipolar lines run left-right but displacement is top-bottom).
    Tx = float(P2_rect[0, 3])
    Ty = float(P2_rect[1, 3])
    if abs(Ty) > abs(Tx):
        disp_axis = 1
        # Vertical stereo: SGBM expects d = v1 - v2 > 0 → need Ty < 0.
        # If Ty > 0, cam2 is "above" cam1 in rectified frame; swap so d > 0.
        cam_swapped = Ty > 0
    else:
        disp_axis = 0
        # Horizontal stereo: SGBM expects d = u1 - u2 > 0 → need Tx < 0.
        # If Tx > 0, cam2 is to the left of cam1; swap so d > 0.
        cam_swapped = Tx > 0
    logger.debug("Pair: Tx=%.0f Ty=%.0f → disp_axis=%d cam_swapped=%s",
                 Tx, Ty, disp_axis, cam_swapped)

    # Expected-disparity coefficient: |P2_rect[axis, 3]| = f_rect * baseline.
    # Dividing by depth (metres) gives the full-res rectified disparity.
    # We return this so process_pair can set numDisparities appropriately.
    disp_coeff = abs(Ty if disp_axis == 1 else Tx)  # f_rect × baseline (px·m)
    f_rect = float(P1_rect[0, 0])
    cx_rect = float(P1_rect[0, 2])
    cy_rect = float(P1_rect[1, 2])

    # Homographies for warping images into the rectified frame.
    H1 = K1_new @ R1_rect @ np.linalg.inv(K1)
    H2 = K2_new @ R2_rect @ np.linalg.inv(K2)
    return (H1, H2, disp_axis, cam_swapped, disp_coeff,
            roi1, roi2, R1_rect, R2_rect, f_rect, cx_rect, cy_rect, rect_size)


def _target_crop_window(
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    H: np.ndarray,
    scene_center: np.ndarray,
    scene_radius: float,
    ground_z: float,
    rect_size: tuple[int, int],
    margin_px: int = 100,
) -> Optional[tuple[int, int, int, int]]:
    """
    Return (x0, y0, x1, y1) in rectified-image pixels that tightly covers the
    scene cylinder (scene_center ± scene_radius, ground_z … ground_z+100 m).
    Returns None when the window would cover ≥90 % of the image (no useful crop).
    """
    cx, cy = float(scene_center[0]), float(scene_center[1])
    angles = np.linspace(0, 2 * np.pi, 24, endpoint=False)
    pts = []
    for z in [ground_z, ground_z + 50.0, ground_z + 100.0]:
        for a in angles:
            pts.append([cx + scene_radius * np.cos(a),
                        cy + scene_radius * np.sin(a), z])
    pts = np.array(pts, dtype=np.float64)

    P = K @ np.hstack([R, -(R @ t.reshape(3, 1))])
    pts4 = np.hstack([pts, np.ones((len(pts), 1))])
    uvw = (P @ pts4.T).T
    in_front = uvw[:, 2] > 0
    if in_front.sum() < 4:
        return None
    uv = uvw[in_front, :2] / uvw[in_front, 2:3]

    uv3 = np.hstack([uv, np.ones((len(uv), 1))])
    rh = (H @ uv3.T).T
    uv_r = rh[:, :2] / rh[:, 2:3]

    x0 = max(0, int(uv_r[:, 0].min()) - margin_px)
    x1 = min(rect_size[0], int(uv_r[:, 0].max()) + margin_px)
    y0 = max(0, int(uv_r[:, 1].min()) - margin_px)
    y1 = min(rect_size[1], int(uv_r[:, 1].max()) + margin_px)
    if x1 <= x0 or y1 <= y0:
        return None
    if (x1 - x0) * (y1 - y0) >= 0.9 * rect_size[0] * rect_size[1]:
        return None
    return x0, y0, x1, y1


def process_pair(
    cam1: Camera,
    cam2: Camera,
    debug_dir: Optional[Path] = None,
    pair_idx: int = 0,
    max_points: int = 100_000,
    scene_center: "np.ndarray | None" = None,
    scene_radius: "float | None" = None,
    ground_z: float = -1300.0,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """
    Stereo reconstruct one camera pair.

    Pipeline:
      1. Calibrated stereo rectification using known K, R, t
      2. SGBM at SGBM_LONG_EDGE (fast); disparity scaled back to full res
      3. Per-pixel triangulation via H1⁻¹, H2⁻¹ → P1, P2 (metric ENU)

    Returns (xyz_Nx3 metres, rgb_Nx3 [0,1]) or None.
    """
    img1, img2 = cam1.image, cam2.image
    h1_px, w1_px = img1.shape[:2]
    h2_px, w2_px = img2.shape[:2]
    # Use the smaller of the two images as the working resolution
    img_size = (min(w1_px, w2_px), min(h1_px, h2_px))  # (W, H)

    def _to_size(img, sz):
        if (img.shape[1], img.shape[0]) == sz:
            return img
        return cv2.resize(img, sz, interpolation=cv2.INTER_AREA)

    img1_use = _to_size(img1, img_size)
    img2_use = _to_size(img2, img_size)

    # Scale intrinsics if images were resized
    def _scale_K(K, orig_wh, new_wh):
        if orig_wh == new_wh:
            return K
        sx = new_wh[0] / orig_wh[0]
        sy = new_wh[1] / orig_wh[1]
        K_s = K.copy()
        K_s[0, :] *= sx
        K_s[1, :] *= sy
        return K_s

    # Build temporary camera copies with scaled K if needed
    from copy import copy as _copy
    cam1_s = _copy(cam1)
    cam2_s = _copy(cam2)
    cam1_s.K = _scale_K(cam1.K, (w1_px, h1_px), img_size)
    cam2_s.K = _scale_K(cam2.K, (w2_px, h2_px), img_size)

    # ------------------------------------------------------------------ #
    # 1. Calibrated stereo rectification
    # ------------------------------------------------------------------ #
    result = _calibrated_rectify(cam1_s, cam2_s, img_size)
    if result is None:
        logger.warning("Pair %d: calibrated rectification failed", pair_idx)
        return None
    H1, H2, disp_axis, cam_swapped, disp_coeff, roi1, roi2, R1_rect, R2_rect, f_rect, cx_rect, cy_rect, rect_size = result

    # If cam2 is "left" of cam1 in rectified space, SGBM would need negative
    # disparity (minDisparity=0 blocks all valid matches).  Swap the pair so
    # cam1 is always the left image and disparity is positive.
    if cam_swapped:
        H1, H2 = H2, H1
        img1_use, img2_use = img2_use, img1_use
        cam1_s, cam2_s = cam2_s, cam1_s
        cam1, cam2 = cam2, cam1
        roi1, roi2 = roi2, roi1
        R1_rect, R2_rect = R2_rect, R1_rect
        logger.debug("Pair %d: cameras swapped to ensure positive disparity", pair_idx)

    # Log ROI coverage so we can see how much of the rectified image is valid.
    # With alpha=0, pixels outside the ROI are black; for oblique cameras the
    # valid region can be a small fraction of the full image.
    def _roi_frac(roi):
        _, _, w, h = roi
        return w * h / (rect_size[0] * rect_size[1])
    logger.info("Pair %d: rectified ROI coverage cam1=%.0f%% cam2=%.0f%%",
                pair_idx, _roi_frac(roi1) * 100, _roi_frac(roi2) * 100)


    # Warp to rect_size (which may be smaller than img_size when stereoRectify
    # down-scaled to control disparity range).
    rect1 = cv2.warpPerspective(img1_use, H1, rect_size)
    rect2 = cv2.warpPerspective(img2_use, H2, rect_size)

    if debug_dir is not None:
        cv2.imwrite(str(debug_dir / f"debug_pair_{pair_idx:02d}_left.jpg"), rect1)
        cv2.imwrite(str(debug_dir / f"debug_pair_{pair_idx:02d}_right.jpg"), rect2)

    # ── Target crop ───────────────────────────────────────────────────────
    # Project the scene cylinder (scene_center ± scene_radius, ground_z …
    # ground_z+100 m) through cam1's projection + H1 to find the bounding
    # box in the rectified image that covers the target area.  SGBM only
    # runs on this sub-image, which eliminates off-target matches (the main
    # cause of south-camera "bands" outside the scene_radius crop).
    crop_x, crop_y = 0, 0
    active_w, active_h = rect_size[0], rect_size[1]

    # Target crop: restrict SGBM to the scene cylinder projection.
    # Skip for nadir-nadir pairs — they already look straight at the target
    # and their full image overlap should be used.  Oblique cameras have wide
    # footprints where most of the image is off-target, so the crop helps.
    both_nadir = (cam1.direction == "nadir" and cam2.direction == "nadir")
    if not both_nadir and scene_center is not None and scene_radius is not None:
        win = _target_crop_window(
            cam1_s.K, cam1_s.R, cam1_s.t, H1,
            scene_center, scene_radius, ground_z, rect_size,
        )
        if win is not None:
            x0, y0, x1, y1 = win
            rect1 = rect1[y0:y1, x0:x1]
            rect2 = rect2[y0:y1, x0:x1]
            crop_x, crop_y = x0, y0
            active_w, active_h = x1 - x0, y1 - y0
            logger.info(
                "Pair %d: target crop → %dx%d (%.0f%% of %dx%d)",
                pair_idx, active_w, active_h,
                100.0 * active_w * active_h / (rect_size[0] * rect_size[1]),
                rect_size[0], rect_size[1],
            )

    # ------------------------------------------------------------------ #
    # 2. SGBM at reduced scale (SGBM_LONG_EDGE)
    #
    # SGBM only handles horizontal (X-axis) disparity.  For nadir cameras
    # with a N-S baseline, rectification produces VERTICAL stereo (disparity
    # in Y).  We handle this by transposing both rectified images so the Y
    # baseline maps to the X direction, then running SGBM normally.
    # ------------------------------------------------------------------ #
    max_rect_edge = max(rect_size)
    sgbm_scale = SGBM_LONG_EDGE / max_rect_edge
    sgbm_w = max(1, int(round(active_w * sgbm_scale)))
    sgbm_h = max(1, int(round(active_h * sgbm_scale)))
    sgbm_size = (sgbm_w, sgbm_h)

    rect1_s = cv2.resize(rect1, sgbm_size, interpolation=cv2.INTER_AREA)
    rect2_s = cv2.resize(rect2, sgbm_size, interpolation=cv2.INTER_AREA)

    if disp_axis == 1:
        # Vertical stereo: transpose so Y-baseline → X-direction for SGBM
        rect1_s = rect1_s.transpose(1, 0, 2)
        rect2_s = rect2_s.transpose(1, 0, 2)

    g1_s = cv2.cvtColor(rect1_s, cv2.COLOR_BGR2GRAY)
    g2_s = cv2.cvtColor(rect2_s, cv2.COLOR_BGR2GRAY)

    # Compute expected maximum disparity at SGBM scale.
    # disp_coeff = f_rect * baseline (px·m).  Dividing by a conservative
    # minimum depth (900 m ≈ low-flying camera) gives the max expected
    # full-resolution rectified disparity; scaling by sgbm_scale converts
    # it to SGBM-image pixels.
    # Minimum depth: altitude AGL / cos(tilt), matching pairs.py logic.
    tilt_cos = abs((cam1.R[2, 2] + cam2.R[2, 2]) / 2.0)
    tilt_cos = max(tilt_cos, 0.3)
    MIN_DEPTH_M = 1300.0 / tilt_cos
    MAX_NUM_DISP = 608   # cap; beyond this SGBM is unreliable/slow
    d_expected_sgbm = disp_coeff * sgbm_scale / MIN_DEPTH_M
    if d_expected_sgbm > MAX_NUM_DISP:
        logger.warning(
            "Pair %d: expected disparity %.0f px exceeds max %d — pair geometry "
            "incompatible with dense SGBM, skipping",
            pair_idx, d_expected_sgbm, MAX_NUM_DISP,
        )
        return None
    block_size = 5
    num_disp = max(64, int(np.ceil(d_expected_sgbm / 16)) * 16 + 32)
    sgbm = cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disp,
        blockSize=block_size,
        P1=8 * 3 * block_size ** 2,
        P2=32 * 3 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=150,
        speckleRange=2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    disp_small = sgbm.compute(g1_s, g2_s).astype(np.float32) / 16.0

    # Scale disparity back to full img_size coordinate system.
    # For vertical stereo the disp_small is in transposed space (W×H);
    # we need to un-transpose back to H×W (but keeping the disp values).
    if disp_axis == 1:
        disp_small = disp_small.T   # back to H×W
    disp_full = cv2.resize(disp_small, (active_w, active_h),
                           interpolation=cv2.INTER_LINEAR) / sgbm_scale

    if debug_dir is not None:
        disp_vis = cv2.normalize(disp_small, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        cv2.imwrite(str(debug_dir / f"debug_pair_{pair_idx:02d}_disp.png"),
                    cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET))

    # ------------------------------------------------------------------ #
    # 3. Triangulate via H1⁻¹, H2⁻¹, P1, P2
    # ------------------------------------------------------------------ #
    # ROI validity mask: stereoRectify with alpha=0 fills pixels outside the
    # valid overlap region with black (0).  For oblique cameras this border
    # can be large.  SGBM matches black-on-black regions, producing spurious
    # small-positive disparities that triangulate to garbage positions.
    # Masking to the ROI intersection eliminates these false matches.
    H_px, W_px = active_h, active_w
    roi_mask = np.zeros((H_px, W_px), dtype=bool)
    x1, y1, w1, h1 = roi1
    x2, y2, w2, h2 = roi2
    # Intersection of both valid regions, shifted by crop offset
    ix = max(x1, x2) - crop_x;  iy = max(y1, y2) - crop_y
    ix2 = min(x1 + w1, x2 + w2) - crop_x;  iy2 = min(y1 + h1, y2 + h2) - crop_y
    ix = max(0, ix);  iy = max(0, iy)
    ix2 = min(active_w, ix2);  iy2 = min(active_h, iy2)
    if ix2 > ix and iy2 > iy:
        roi_mask[iy:iy2, ix:ix2] = True
    overlap_frac = roi_mask.sum() / roi_mask.size
    logger.info("Pair %d: valid overlap region %.0f%% of image", pair_idx, overlap_frac * 100)

    valid_mask = (disp_full > 1.0) & roi_mask
    if int(valid_mask.sum()) < MIN_VALID_POINTS:
        logger.warning("Pair %d: only %d valid disparity pixels",
                       pair_idx, int(valid_mask.sum()))
        return None

    # Subsample if huge
    max_pts = max_points
    rows, cols = np.where(valid_mask)
    if len(rows) > max_pts:
        idx = np.random.choice(len(rows), max_pts, replace=False)
        rows, cols = rows[idx], cols[idx]

    us = cols.astype(np.float64) + crop_x
    vs = rows.astype(np.float64) + crop_y
    ds = disp_full[rows, cols].astype(np.float64)

    # ── Direct pinhole reprojection (avoids ill-conditioned H1_inv) ────────
    # With CALIB_ZERO_DISPARITY, cx and cy are identical in both rectified
    # cameras.  For horizontal stereo (disp_axis=0):
    #   d = u1 - u2  →  Z = disp_coeff / d,  X = (u-cx)*Z/f,  Y = (v-cy)*Z/f
    # For vertical stereo (disp_axis=1):
    #   d = v1 - v2  →  same depth formula (disp_coeff uses |Ty| instead of |Tx|)
    # Both cases reduce to Z = disp_coeff / d in the rectified cam1 frame.
    # We then rotate back to world ENU via R1_rect and cam1's pose.
    Z_rect = disp_coeff / ds                          # depth along rect. Z axis
    X_rect = (us - cx_rect) * Z_rect / f_rect
    Y_rect = (vs - cy_rect) * Z_rect / f_rect

    pts_rect = np.stack([X_rect, Y_rect, Z_rect], axis=1)  # N×3 in rectified cam1 frame

    # Rectified cam1 frame → cam1 original frame → world ENU
    pts_cam1 = (R1_rect.T @ pts_rect.T).T            # undo rectification rotation
    xyz = (cam1.R.T @ pts_cam1.T).T + cam1.t         # world ENU (N×3)
    nonzero = np.ones(len(xyz), dtype=bool)           # all finite by construction

    # ------------------------------------------------------------------ #
    # 4. Filter outliers
    # ------------------------------------------------------------------ #
    dist = np.linalg.norm(xyz, axis=1)

    # Points must be in front of both cameras (positive depth in camera space).
    depth1 = (cam1.R @ (xyz - cam1.t).T)[2]
    depth2 = (cam2.R @ (xyz - cam2.t).T)[2]

    keep = nonzero & (depth1 > 0) & (depth2 > 0) & (dist < MAX_SCENE_RADIUS)

    # Distance-from-origin outlier removal.
    if keep.sum() > 10:
        d_vals = dist[keep]
        med = float(np.median(d_vals))
        std = float(np.std(d_vals))
        keep &= dist < med + 3 * std

    # Z-mode filter: find the dominant depth level via histogram and discard
    # points that are far from it.  For nadir cameras the ground plane creates
    # a massive spike; debris triangulates to scattered depths far below that.
    # The ±200 m window is wide enough for buildings / oblique cameras but
    # rejects false matches that triangulate kilometres underground.
    if keep.sum() > 200:
        z_vals = xyz[keep, 2]
        hist_counts, hist_edges = np.histogram(z_vals, bins=80)
        peak_idx = int(np.argmax(hist_counts))
        z_mode = float((hist_edges[peak_idx] + hist_edges[peak_idx + 1]) / 2)

        # Sanity check: the mode should be below the cameras but not by more
        # than ~2500 m.  If it's deeper, the pair matched noise (no real
        # overlap) and should be rejected entirely.
        cam_z = float(min(cam1.t[2], cam2.t[2]))
        max_reasonable_depth = 2500.0  # metres — generous upper bound
        if z_mode < cam_z - max_reasonable_depth:
            logger.warning(
                "Pair %d: Z-mode=%.1f is %.0f m below cameras (cam_z=%.1f) "
                "— likely noise-only pair, rejecting",
                pair_idx, z_mode, cam_z - z_mode, cam_z,
            )
            return None

        # Asymmetric: buildings are above ground (up to 100 m); below-ground is noise.
        keep &= (xyz[:, 2] >= z_mode - 30.0) & (xyz[:, 2] <= z_mode + 100.0)
        logger.debug("Pair %d: Z-mode=%.1f, kept Z in [%.1f, %.1f]",
                     pair_idx, z_mode, z_mode - 30.0, z_mode + 100.0)

    n_valid = int(keep.sum())
    if n_valid < MIN_VALID_POINTS:
        logger.warning("Pair %d: only %d points after 3D filter", pair_idx, n_valid)
        return None

    xyz_out = xyz[keep].astype(np.float32)
    bgr = rect1[rows[keep], cols[keep]]
    rgb_out = bgr[:, ::-1].astype(np.float32) / 255.0

    logger.info("Pair %d: %d 3D points reconstructed", pair_idx, n_valid)

    if debug_dir is not None:
        try:
            import open3d as o3d
            pcd_debug = o3d.geometry.PointCloud()
            pcd_debug.points = o3d.utility.Vector3dVector(xyz_out.astype(np.float64))
            pcd_debug.colors = o3d.utility.Vector3dVector(rgb_out.astype(np.float64))
            o3d.io.write_point_cloud(
                str(debug_dir / f"debug_pair_{pair_idx:02d}.ply"), pcd_debug)
        except Exception as exc:
            logger.debug("Could not save debug PLY: %s", exc)

    return xyz_out, rgb_out
