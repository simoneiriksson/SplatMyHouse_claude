# CLAUDE.md

## New session onboarding

Read `prompt.md` before doing anything else. It is the single source of truth for this project — it contains the original build specification, the API reference, and a history of every prompt used in prior sessions. Understanding it gives full context for why the code is structured the way it is.

## Project overview

Aerial 3D point-cloud reconstruction from Danish Skraafotos imagery (Dataforsyningen API). Fetches oblique aerial images for a location, builds camera models from embedded photogrammetric metadata, selects stereo pairs, runs SGBM dense matching, and merges depth maps into a PLY point cloud.

## Key architecture decisions

- **Camera convention**: the API provides rotation matrices in the photogrammetric convention (camera +Z away from scene). A Y+Z flip (`_flip = diag([1,-1,-1])`) converts to OpenCV convention in `reconstruction/camera.py`.
- **Stereo rectification**: uses `cv2.stereoRectify` (calibrated), not uncalibrated. Detects vertical vs horizontal stereo from `P2_rect` and transposes images when needed so SGBM always searches horizontally.
- **Disparity sign**: detects whether cam2 is "left" of cam1 in the rectified frame and swaps camera/image order to ensure positive disparity for SGBM (`minDisparity=0`).
- **Pair pre-screening**: `compute_pairs` in `reconstruction/pairs.py` calls `stereoRectify` for each candidate pair and rejects any whose expected SGBM-scale disparity exceeds 512 px. This filters out geometrically incompatible pairs (oblique cameras with large transverse baselines) before any image processing.
- **Coordinate system**: local ENU with origin at mean camera position in UTM EPSG:25832.

## Prompt history discipline

After every session that changes the codebase, append a dated entry to the `## Prompt history` section at the bottom of `prompt.md` describing what was done.
