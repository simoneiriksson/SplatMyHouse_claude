import logging

import numpy as np

logger = logging.getLogger(__name__)


def merge_pointclouds(
    xyz_list: list[np.ndarray],
    rgb_list: list[np.ndarray],
    camera_positions: list[np.ndarray] | None = None,
    scene_center: np.ndarray | None = None,
    scene_radius: float | None = None,
) -> "open3d.geometry.PointCloud":  # type: ignore[name-defined]
    """
    Concatenate per-pair point clouds, optionally crop to an XY cylinder of
    scene_radius metres around scene_center (ENU), downsample, remove outliers,
    estimate normals.  Returns an open3d PointCloud.
    """
    import open3d as o3d

    if not xyz_list:
        raise ValueError("No point clouds to merge.")

    xyz_all = np.concatenate(xyz_list, axis=0).astype(np.float64)
    rgb_all = np.concatenate(rgb_list, axis=0).astype(np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz_all)
    pcd.colors = o3d.utility.Vector3dVector(np.clip(rgb_all, 0, 1))

    # Crop to XY cylinder around the target before any filtering
    if scene_center is not None and scene_radius is not None:
        pts = np.asarray(pcd.points)
        cx, cy = float(scene_center[0]), float(scene_center[1])
        dist_xy = np.sqrt((pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2)
        keep_idx = np.where(dist_xy <= scene_radius)[0]
        pcd = pcd.select_by_index(keep_idx.tolist())
        logger.info(
            "Points after XY crop (r=%.0f m around target): %d",
            scene_radius, len(pcd.points),
        )

    before_count = len(pcd.points)
    logger.info("Points before downsample: %d", before_count)

    # Voxel downsample
    pcd = pcd.voxel_down_sample(voxel_size=0.25)
    logger.info("Points after voxel downsample (0.25 m): %d", len(pcd.points))

    # Statistical outlier removal
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
    after_count = len(pcd.points)
    logger.info("Points after outlier removal: %d", after_count)

    # Z-mode clip: find the dominant depth level via histogram peak and keep
    # only points within ±300 m of it.  IQR-based clipping breaks when debris
    # constitutes >25 % of the data (q25 lands in the debris zone, inflating
    # IQR to the point where the floor is useless).  The histogram peak is
    # robust: the correct ground plane always has the highest point density.
    pts_z = np.asarray(pcd.points)[:, 2]
    if len(pts_z) > 0:
        hist_counts, hist_edges = np.histogram(pts_z, bins=100)
        peak_idx = int(np.argmax(hist_counts))
        z_mode = float((hist_edges[peak_idx] + hist_edges[peak_idx + 1]) / 2)
        z_floor = z_mode - 300.0
        z_ceil  = z_mode + 300.0
        keep_z = np.where((pts_z >= z_floor) & (pts_z <= z_ceil))[0]
        pcd = pcd.select_by_index(keep_z.tolist())
        logger.info("Points after Z-mode clip (mode=%.0f, ±300 m): %d",
                    z_mode, len(pcd.points))

    # Estimate normals
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=2.0, max_nn=30)
    )

    # Orient normals towards mean camera position
    if camera_positions:
        mean_cam = np.mean(camera_positions, axis=0).astype(np.float64)
        pcd.orient_normals_towards_camera_location(mean_cam)
    else:
        pts = np.asarray(pcd.points)
        if len(pts) > 0:
            mean_pt = pts.mean(axis=0)
            mean_pt[2] += 500  # above the scene
            pcd.orient_normals_towards_camera_location(mean_pt)

    return pcd
