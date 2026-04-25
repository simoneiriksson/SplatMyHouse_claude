import logging
from pathlib import Path
from typing import Optional

import numpy as np

from reconstruction.camera import Camera
from reconstruction.pairs import compute_pairs, _ground_footprint
from reconstruction.stereo import process_pair
from reconstruction.pointcloud import merge_pointclouds

logger = logging.getLogger(__name__)


def run(
    cameras: list[Camera],
    max_pairs: int,
    min_baseline: float,
    max_baseline: float,
    debug: bool = False,
    session_folder: Optional[Path] = None,
    scene_center: "np.ndarray | None" = None,
    scene_radius: float | None = None,
    max_points: int = 100_000,
    progress_callback=None,
    ground_z: float = -1000.0,
) -> tuple["open3d.geometry.PointCloud", dict]:  # type: ignore[name-defined]
    """
    Orchestrate the full reconstruction pipeline.
    Returns (pcd, stats_dict).

    progress_callback: optional callable(step, message, done, total)
      step    — string label for the current phase
      message — human-readable detail
      done    — units completed so far (int)
      total   — total units for this phase (int)
    """
    debug_dir: Optional[Path] = None
    if debug and session_folder is not None:
        debug_dir = Path(session_folder) / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)

    def _cb(step, message, done, total):
        if progress_callback:
            progress_callback(step, message, done, total)

    # Pair selection
    _cb("pairs", "Selecting stereo pairs…", 0, 1)
    pairs = compute_pairs(
        cameras,
        min_baseline=min_baseline,
        max_baseline=max_baseline,
        max_pairs=max_pairs,
        debug=debug,
        scene_center=scene_center,
        ground_z=ground_z,
    )
    _cb("pairs", f"{len(pairs)} pairs selected", 1, 1)

    if not pairs:
        raise RuntimeError(
            "No stereo pairs selected. Try adjusting --min-baseline / --max-baseline."
        )

    # Debug: top-down camera layout
    if debug and session_folder is not None:
        _save_camera_layout(cameras, pairs, debug_dir)

    # Process each pair
    xyz_list: list[np.ndarray] = []
    rgb_list: list[np.ndarray] = []
    points_before_merge = 0
    processed = 0
    pair_stats: list[dict] = []
    # Per-pair arrays for the interactive viewer (subsampled to save memory)
    _VIEW_MAX = 15_000
    pair_view_xyz: list[np.ndarray] = []
    pair_view_rgb: list[np.ndarray] = []

    for idx, pair in enumerate(pairs):
        _cb("stereo", f"Pair {idx + 1}/{len(pairs)}: {pair.cam_i.direction} ↔ {pair.cam_j.direction}", idx, len(pairs))
        n_pts = 0
        n_in_scene = 0
        ok = False
        view_xyz = view_rgb = None

        # Footprint midpoint distance to scene centre — tells user where this pair looks
        fp_dist: float | None = None
        if scene_center is not None:
            fp_i = _ground_footprint(pair.cam_i, ground_z=ground_z)
            fp_j = _ground_footprint(pair.cam_j, ground_z=ground_z)
            mid = (fp_i + fp_j) / 2.0
            fp_dist = float(np.linalg.norm(mid - scene_center[:2]))

        try:
            result = process_pair(
                pair.cam_i, pair.cam_j,
                debug_dir=debug_dir, pair_idx=idx, max_points=max_points,
                scene_center=scene_center, scene_radius=scene_radius, ground_z=ground_z,
            )
            if result is not None:
                xyz, rgb = result
                n_pts = len(xyz)
                xyz_list.append(xyz)
                rgb_list.append(rgb)
                points_before_merge += n_pts
                processed += 1
                ok = True

                # Subsample for the viewer — crop to scene_radius first so the
                # per-pair viewer shows exactly what each pair contributes to the
                # final PLY (not the full unclipped reconstruction).
                xyz_v, rgb_v = xyz, rgb
                if scene_center is not None and scene_radius is not None:
                    xy_dist = np.linalg.norm(xyz[:, :2] - scene_center[:2], axis=1)
                    in_r = xy_dist <= scene_radius
                    xyz_v = xyz[in_r]
                    rgb_v = rgb[in_r]
                n_in_scene = len(xyz_v)

                if n_in_scene > _VIEW_MAX:
                    sel = np.random.choice(n_in_scene, _VIEW_MAX, replace=False)
                    view_xyz = xyz_v[sel]
                    view_rgb = rgb_v[sel]
                elif n_in_scene > 0:
                    view_xyz = xyz_v.copy()
                    view_rgb = rgb_v.copy()
        except Exception as exc:
            logger.warning("Pair %d failed: %s", idx, exc, exc_info=True)
        pair_stats.append({
            "idx": idx + 1,
            "dir_i": pair.cam_i.direction,
            "id_i": pair.cam_i.item_id[-8:],
            "dir_j": pair.cam_j.direction,
            "id_j": pair.cam_j.item_id[-8:],
            "baseline": pair.baseline,
            "score": pair.score,
            "points": n_pts,
            "points_in_scene": n_in_scene,
            "fp_dist": fp_dist,
            "ok": ok,
        })
        pair_view_xyz.append(view_xyz)
        pair_view_rgb.append(view_rgb)
    _cb("stereo", f"{processed}/{len(pairs)} pairs succeeded", len(pairs), len(pairs))

    logger.info("Pairs processed: %d / %d selected", processed, len(pairs))

    if not xyz_list:
        raise RuntimeError("All stereo pairs failed — no points reconstructed.")

    _cb("merge", "Merging & filtering point cloud…", 0, 1)
    camera_positions = [c.t for c in cameras]
    pcd = merge_pointclouds(
        xyz_list, rgb_list,
        camera_positions=camera_positions,
        scene_center=scene_center,
        scene_radius=scene_radius,
    )
    _cb("merge", f"{len(pcd.points):,} points", 1, 1)

    stats = {
        "pairs_selected": len(pairs),
        "pairs_processed": processed,
        "points_before_merge": points_before_merge,
        "points_after_merge": len(pcd.points),
        "pair_stats": pair_stats,
        "pair_view_xyz": pair_view_xyz,
        "pair_view_rgb": pair_view_rgb,
    }
    return pcd, stats


def _save_camera_layout(cameras: list[Camera], pairs, debug_dir: Path) -> None:
    """Save top-down matplotlib plot of cameras and pair connections."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        direction_colors = {
            "nadir": "black",
            "north": "blue",
            "south": "red",
            "east": "green",
            "west": "orange",
        }

        fig, ax = plt.subplots(figsize=(10, 10))

        # Draw pair connections
        for pair in pairs:
            x = [pair.cam_i.t[0], pair.cam_j.t[0]]
            y = [pair.cam_i.t[1], pair.cam_j.t[1]]
            ax.plot(x, y, "k-", alpha=0.2, linewidth=0.8)

        # Draw cameras
        for cam in cameras:
            color = direction_colors.get(cam.direction, "grey")
            ax.scatter(cam.t[0], cam.t[1], c=color, s=60, zorder=5)
            ax.annotate(cam.direction[:1].upper(), (cam.t[0], cam.t[1]),
                        fontsize=7, ha="center", va="bottom")

        ax.set_aspect("equal")
        ax.set_xlabel("East (m)")
        ax.set_ylabel("North (m)")
        ax.set_title("Camera layout (top-down ENU)")

        # Legend
        for d, c in direction_colors.items():
            ax.scatter([], [], c=c, label=d, s=40)
        ax.legend(loc="upper right", fontsize=8)

        out_path = debug_dir / "debug_cameras.png"
        fig.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved camera layout to %s", out_path)
    except Exception as exc:
        logger.warning("Could not save camera layout: %s", exc)
