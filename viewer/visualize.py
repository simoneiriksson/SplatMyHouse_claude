import logging
from collections import Counter
from pathlib import Path
from typing import Optional

import numpy as np

from reconstruction.camera import Camera

logger = logging.getLogger(__name__)


def show(
    pcd: "open3d.geometry.PointCloud",  # type: ignore[name-defined]
    cameras: list[Camera],
    location_label: str,
    session_path: Optional[Path] = None,
    save_path: Optional[str] = None,
) -> None:
    """
    Display the point cloud in an Open3D interactive window.
    Falls back to saving a PLY if Open3D display is unavailable.
    """
    try:
        import open3d as o3d
    except ImportError:
        logger.error("Open3D not available. Install open3d>=0.18.")
        _fallback_save(pcd, save_path)
        return

    n_points = len(pcd.points)
    direction_counts = Counter(c.direction for c in cameras)

    print(f"Loaded {n_points} points from {len(cameras)} cameras")
    print("Cameras: " + ", ".join(f"{v} {k}" for k, v in sorted(direction_counts.items())))
    if session_path:
        print(f"Session folder: {session_path}")
    print("Controls: R=reset view, Q=quit, S=save screenshot")

    geometries: list = [pcd]

    # Scene origin
    origin = np.zeros(3)
    if len(cameras) > 0:
        origin = np.mean([c.t for c in cameras], axis=0)

    direction_rgb = {
        "nadir":  [1.0, 0.0, 0.0],
        "north":  [0.0, 0.0, 1.0],
        "south":  [1.0, 0.2, 0.2],
        "east":   [0.0, 0.8, 0.0],
        "west":   [1.0, 0.5, 0.0],
    }

    lines_points: list[list[float]] = []
    lines_edges: list[list[int]] = []
    lines_colors: list[list[float]] = []

    for cam in cameras:
        # Small red sphere at camera position
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=2.0)
        sphere.translate(cam.t.astype(np.float64))
        color = direction_rgb.get(cam.direction, [0.5, 0.5, 0.5])
        sphere.paint_uniform_color(color)
        geometries.append(sphere)

        # Line from camera to origin
        idx_base = len(lines_points)
        lines_points.append(cam.t.tolist())
        lines_points.append(origin.tolist())
        lines_edges.append([idx_base, idx_base + 1])
        lines_colors.append([0.6, 0.6, 0.6])

    # Add camera-to-origin lines
    if lines_points:
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(lines_points)
        line_set.lines = o3d.utility.Vector2iVector(lines_edges)
        line_set.colors = o3d.utility.Vector3dVector(lines_colors)
        geometries.append(line_set)

    # Ground plane footprint rectangle (thin grey box)
    pts_arr = np.asarray(pcd.points) if len(pcd.points) > 0 else np.zeros((1, 3))
    if len(pts_arr) > 1:
        min_xy = pts_arr.min(axis=0)
        max_xy = pts_arr.max(axis=0)
        ground_z = float(pts_arr[:, 2].min()) - 2
        corners = np.array([
            [min_xy[0], min_xy[1], ground_z],
            [max_xy[0], min_xy[1], ground_z],
            [max_xy[0], max_xy[1], ground_z],
            [min_xy[0], max_xy[1], ground_z],
        ])
        box_edges = [[0, 1], [1, 2], [2, 3], [3, 0]]
        box_ls = o3d.geometry.LineSet()
        box_ls.points = o3d.utility.Vector3dVector(corners)
        box_ls.lines = o3d.utility.Vector2iVector(box_edges)
        box_ls.colors = o3d.utility.Vector3dVector([[0.7, 0.7, 0.7]] * 4)
        geometries.append(box_ls)

    # Launch viewer
    try:
        o3d.visualization.draw_geometries(
            geometries,
            window_name=f"Skråfoto 3D – {location_label}",
            width=1280,
            height=800,
        )
    except Exception as exc:
        logger.warning("Open3D visualisation failed (%s); saving PLY instead.", exc)
        _fallback_save(pcd, save_path)


def save_ply(
    pcd: "open3d.geometry.PointCloud",  # type: ignore[name-defined]
    path: str,
) -> None:
    try:
        import open3d as o3d
        o3d.io.write_point_cloud(path, pcd)
        logger.info("Point cloud saved to %s", path)
    except Exception as exc:
        logger.error("Failed to save PLY: %s", exc)


def _fallback_save(pcd, save_path: Optional[str]) -> None:
    if save_path:
        save_ply(pcd, save_path)
    else:
        fallback = "output_pointcloud.ply"
        save_ply(pcd, fallback)
        print(f"Point cloud saved to {fallback}")
