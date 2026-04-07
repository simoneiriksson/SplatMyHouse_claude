# skraafoto3d

Reconstruct a 3D point cloud from Danish oblique aerial imagery using multi-view stereo.

## Prerequisites

- Python 3.11+
- A free API token from [dataforsyningen.dk](https://dataforsyningen.dk) (create a user account, then generate a token in the profile settings)

## Installation

```bash
pip install -r requirements.txt
```

## Quick start

```bash
export DATAFORSYNINGEN_TOKEN=your_token_here

# By address (geocoded via DAWA)
python main.py --address "Rådhuspladsen 1, København"

# By coordinates
python main.py --lonlat 12.5683,55.6761

# Save point cloud without opening the viewer
python main.py --lonlat 12.5683,55.6761 --save pointcloud.ply --no-viewer
```

## Session folder layout

Every run creates a dated folder under `./data/`:

```
data/20240315_143022_Rådhuspladsen_1_København/
├── images/
│   └── <item_id>.tif       # raw COG images (exactly as downloaded)
├── metadata/
│   └── <item_id>.json      # full STAC Item JSON with camera calibration
├── session.json            # summary: query params, image counts by direction
└── debug/                  # (only with --debug)
    ├── debug_cameras.png
    ├── debug_pair_00_left.jpg
    ├── debug_pair_00_right.jpg
    ├── debug_pair_00_disp.png
    └── debug_pair_00.ply
```

### Re-run reconstruction from a saved session

```bash
python main.py --from-session ./data/20240315_143022_Rådhuspladsen_1_København \
               --max-pairs 10 --save output.ply --no-viewer
```

This skips all API calls and re-runs only the stereo pipeline from cached images.

## Multi-image strategy

The Skråfoto archive covers Denmark with overlapping flight strips from five directions
(nadir, north, south, east, west). For any point in space, 15–30 images are typically
available from adjacent strips and along-track overlap.

**Pair selection** works as follows:

1. All pairs are evaluated for baseline distance (camera separation in metres).
2. Pairs outside the range `[min_baseline, max_baseline]` are discarded.
3. Pairs whose optical axes diverge by more than 60° are discarded.
4. Remaining pairs are scored with a tent function peaking at 200 m, multiplied by a
   direction bonus (nadir+oblique = 1.4, same-direction strip = 1.2, cross-oblique = 0.8).
5. The top `max_pairs` pairs are used for reconstruction.

## Coordinate system

All 3D points are expressed in a **local ENU (East-North-Up)** frame:

- **Origin**: mean of all downloaded camera positions, converted from ECEF
- **Units**: metres
- **Axes**: X = East, Y = North, Z = Up

This keeps numerical precision high and makes the coordinates interpretable.

## Tuning guide

| Parameter | Default | Effect |
|---|---|---|
| `--max-pairs` | 20 | More pairs → denser cloud, slower run |
| `--min-baseline` | 50 m | Lower → more pairs from close strips |
| `--max-baseline` | 800 m | Higher → include widely-spaced cameras |
| `--max-images` | 30 | Cap on API downloads |
| SGBM `blockSize` | 7 | Smaller → more detail but noisier; edit `stereo.py` |

For **dense urban areas**: use default settings.  
For **open terrain**: increase `--min-baseline` (100–150 m) to avoid texture-less pairs.  
For **speed**: reduce `--max-pairs` to 5–10.

## Known limitations

- No bundle adjustment — camera poses come directly from the STAC metadata OPK angles.
  GPS/IMU errors accumulate; neighbouring strips may be slightly misregistered.
- Rectification uses `stereoRectifyUncalibrated` (planar homography). This breaks down
  for large baselines or large convergence angles.
- Metric scale in the reconstructed depth maps is approximate (derived from
  focal length × baseline, with no ground-truth check).
- Cloud cover, seasonal changes, or mixed-year imagery can cause inconsistent textures.

## Suggested next steps

- **COLMAP bundle adjustment**: export images + calibration to COLMAP for robust
  multi-view reconstruction with loop closure.
- **Gaussian Splatting**: use the point cloud as an initialisation for 3D Gaussian
  Splatting for photorealistic novel-view synthesis.
- **DTM fusion**: align the point cloud with the Danish Height Model (DHM) from
  Kortforsyningen for absolute elevation registration.
