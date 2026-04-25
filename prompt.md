# skraafoto3d — Master Prompt File
#
# This file is the single source of truth for the skraafoto3d application.
# Every time a new feature is added or changed via a Claude Code session,
# append a new dated section at the bottom of this file under the heading
# "## Prompt history" so the file could theoretically be used to regenerate
# the entire codebase from scratch.
#
# Format for appended entries:
#   ## Prompt history
#   ### YYYY-MM-DD — <short title>
#   <the prompt text used in that session>
#
# The initial build prompt follows immediately below.
# ============================================================

Build a Python application that fetches oblique aerial images from the Danish Skråfoto
STAC API and reconstructs a 3D point cloud using multi-view stereo with OpenCV and Open3D.

## Overview

The app fetches all available images of a location from the Dataforsyningen Skråfoto API.
For a given coordinate, the API returns multiple overlapping frames per direction (north/
south/east/west/nadir) from adjacent flight strips and along-track overlap — typically
15–30 images total for a small area. The app uses the embedded camera calibration metadata
to establish camera poses, automatically selects good stereo pairs based on baseline
distance, performs dense stereo matching on each pair, and merges all depth maps into a
single 3D point cloud that can be viewed interactively.

Every API request saves all downloaded images and their metadata to a dedicated folder
so results are fully reproducible and inspectable without re-querying the API.

---

## Project structure

skraafoto3d/
├── main.py                  # CLI entry point
├── config.py                # Token, API base URL, constants
├── api/
│   ├── __init__.py
│   ├── client.py            # STAC API queries and image download
│   └── models.py            # Dataclasses for Item, CameraModel
├── reconstruction/
│   ├── __init__.py
│   ├── camera.py            # Parse pers:* metadata → intrinsic/extrinsic matrices
│   ├── pairs.py             # Baseline analysis and pair selection
│   ├── stereo.py            # Pairwise rectification + SGBM disparity → depth
│   ├── pointcloud.py        # Depth map → 3D points, merge, filter
│   └── pipeline.py          # Orchestrates the full reconstruction
├── storage/
│   ├── __init__.py
│   └── session.py           # Per-request folder creation, image + metadata saving
├── viewer/
│   ├── __init__.py
│   └── visualize.py         # Open3D interactive viewer
├── requirements.txt
└── README.md

---

## Skråfoto REST API reference

All endpoints share:
  Base URL:  https://api.dataforsyningen.dk/rest/skraafoto_api/v1.0
  Auth:      HTTP header  token: <TOKEN>
             Token is read from environment variable DATAFORSYNINGEN_TOKEN.
             Never put the token in query parameters.
  Format:    application/geo+json  (GeoJSON / JSON responses)

There are three separate services:

  1. STAC API      https://api.dataforsyningen.dk/rest/skraafoto_api/v1.0
                   Metadata search and item lookup. Used by this app for all queries.

  2. COG Server    https://cdn.dataforsyningen.dk/skraafoto_server
                   Delivers raw Cloud Optimized GeoTIFF images via HTTP range requests.
                   The full-resolution asset href in each STAC item points here.

  3. Cogtiler      https://api.dataforsyningen.dk/rest/skraafoto_cogtiler/v1.0
                   Translates COG → JPEG tiles for browsers. Not needed for this app
                   since we download full COG images directly.

### Endpoint: Landing page
  GET /
  Returns API metadata, version, and links to conformance and collections.

### Endpoint: Conformance
  GET /conformance
  Returns an array of conformance class URIs. Used by STAC clients to discover
  supported extensions. Not needed in normal use.

### Endpoint: List collections
  GET /collections
  Returns all available skråfoto collections (one per flight year).
  Known collections as of 2024: skraafotos2019, skraafotos2021, skraafotos2023
  Response shape:
    { "collections": [ { "id": "skraafotos2023", "title": "...", ... }, ... ] }

### Endpoint: Get single collection
  GET /collections/{collectionid}
  Returns metadata for one collection including spatial and temporal extent.
  Example:
    GET /collections/skraafotos2023

### Endpoint: Get items in collection  ← use this for single-year queries
  GET /collections/{collectionid}/items
  Query parameters:
    bbox          [W,S,E,N] in WGS84, e.g. 12.565,55.671,12.572,55.678
    bbox-crs      default: http://www.opengis.net/def/crs/OGC/1.3/CRS84
                  alternative: http://www.opengis.net/def/crs/EPSG/0/25832
    crs           coordinate system for response geometries (same options as bbox-crs)
    datetime      ISO8601 interval, e.g. 2023-01-01T00:00:00Z/2023-12-31T23:59:59Z
    limit         integer, default 10, max 1000
    filter        CQL2 text filter (see Filter Extension below)
    filter-lang   cql2-text  or  cql2-json
    sortby        e.g. +datetime or -datetime
  Response: GeoJSON FeatureCollection of STAC Items
  Note: Prefer this endpoint over /search when querying a single known collection,
        as it consumes fewer server resources.

### Endpoint: Search  ← use this to query across multiple collections
  GET  /search   (same parameters as collection items, passed as query string)
  POST /search   (same parameters passed as JSON body)
  Additional body parameters for POST:
    ids           array of specific item IDs to retrieve
    collections   array of collection IDs to search within
                  e.g. ["skraafotos2023"]
    intersects    GeoJSON geometry (alternative to bbox)
  Example POST body:
    {
      "bbox": [12.565, 55.671, 12.572, 55.678],
      "collections": ["skraafotos2023"],
      "filter": "direction IN ('north','south','east','west','nadir')",
      "filter-lang": "cql2-text",
      "limit": 50
    }
  Response: GeoJSON FeatureCollection with a "context" extension object:
    {
      "type": "FeatureCollection",
      "context": { "returned": 23, "limit": 50, "matched": 23 },
      "features": [ <STAC Item>, ... ]
    }

### Endpoint: Get single item
  GET /collections/{collectionid}/items/{itemid}
  Returns one STAC Item by its exact ID.
  Example:
    GET /collections/skraafotos2021/items/2021_83_36_4_0008_00004522

### Filter Extension (CQL2)
  Supported filter fields on items:
    direction       string: 'north' | 'south' | 'east' | 'west' | 'nadir'
    datetime        timestamp
    gsd             ground sample distance in metres (float)
    pers:kappa      float (degrees)
    pers:omega      float (degrees)
    pers:phi        float (degrees)
  Example filters:
    direction = 'nadir'
    direction IN ('north','south','east','west','nadir')
    gsd < 0.15
  Queryable fields can be discovered at:
    GET /queryables
    GET /collections/{collectionid}/queryables

### Sort Extension
  sortby parameter accepts field names prefixed with + (ascending) or - (descending).
  Examples:  sortby=+datetime   sortby=-gsd

### CRS Extension
  All geometry inputs and outputs default to WGS84 (CRS84).
  EPSG:25832 (UTM zone 32N, the Danish national grid) is also supported.
  Specify with:  crs=http://www.opengis.net/def/crs/EPSG/0/25832

### Pagination
  Responses include a "next" link when more results are available:
    response["links"] → look for { "rel": "next", "href": "..." }
  Follow the next href to retrieve the subsequent page. Implement pagination in
  client.py so the app collects all matching items, not just the first page.

### STAC Item structure
  Each feature in a response is a GeoJSON Feature with:

  item["id"]                    unique item identifier
  item["type"]                  "Feature"
  item["geometry"]              GeoJSON Point or Polygon (image footprint)
  item["bbox"]                  [W, S, E, N]
  item["collection"]            collection ID string

  item["assets"]["full"]
    ["href"]                    URL to full-resolution COG image on CDN server
    ["type"]                    "image/tiff; application=geotiff; profile=cloud-optimized"

  item["assets"]["thumbnail"]
    ["href"]                    URL to JPEG thumbnail via Cogtiler
    ["type"]                    "image/jpeg"

  item["properties"]["datetime"]          ISO8601 acquisition timestamp
  item["properties"]["direction"]         'north'|'south'|'east'|'west'|'nadir'
  item["properties"]["gsd"]               ground sample distance in metres
  item["properties"]["license"]           license string

  item["properties"]["pers:interior_orientation"]
    {
      "camera_id":                string,
      "focal_length":             float (mm),
      "pixel_spacing":            [sx, sy] (mm/pixel),
      "principal_point_offset":   [ppx, ppy] (pixels),
      "sensor_array_dimensions":  [width, height] (pixels)
    }

  item["properties"]["pers:exterior_orientation"]
    {
      "camera_center_ecef":   [X, Y, Z] (metres, ECEF),
      "omega":                float (degrees),
      "phi":                  float (degrees),
      "kappa":                float (degrees)
    }

  item["links"]                 array of relation links (self, collection, root, next)

### Address geocoding (Dataforsyningen DAWA)
  This service does NOT require a token.
  GET https://api.dataforsyningen.dk/adresser?q=<address>&format=geojson
  Extract: response["features"][0]["geometry"]["coordinates"]  → [lon, lat]

### Image download
  GET the href from item["assets"]["full"]["href"]
  Header: token: <TOKEN>
  The server supports HTTP range requests (COG format) but for this app download
  the full file. Response is a binary TIFF/JPEG stream; decode with cv2.imdecode
  after reading bytes into a numpy array.

---

## Session storage (storage/session.py)

Every time the app runs a request, it creates a dedicated folder to save all
raw data before reconstruction begins. This makes runs fully reproducible — the
reconstruction pipeline can be re-run from the saved folder without hitting the API.

### Folder naming
  Base storage directory: ./data/  (configurable via --data-dir CLI argument)
  Folder name: <timestamp>_<sanitised_location>
    timestamp:          YYYYMMDD_HHMMSS  (UTC, time of the API query)
    sanitised_location: address string with spaces→underscores and special chars
                        stripped, truncated to 40 characters; or "lon{lon}_lat{lat}"
                        if --lonlat was used.
  Examples:
    ./data/20240315_143022_Rådhuspladsen_1_København/
    ./data/20240315_143022_lon12.5683_lat55.6761/

### Folder contents
  images/
    <item_id>.tif          raw downloaded image bytes (exactly as received from API)
  metadata/
    <item_id>.json         full STAC Item JSON for that image (pretty-printed)
  session.json             summary file written after all downloads complete:
    {
      "timestamp":     "2024-03-15T14:30:22Z",
      "query": {
        "lonlat":      [12.5683, 55.6761],
        "address":     "Rådhuspladsen 1, København",  (or null)
        "bbox":        [W, S, E, N],
        "collection":  "skraafotos2023"
      },
      "images_found":    23,
      "images_saved":    23,
      "direction_counts": { "nadir": 5, "north": 4, "south": 5, "east": 4, "west": 5 },
      "session_folder":  "./data/20240315_143022_Rådhuspladsen_1_København"
    }

### Reuse / cache behaviour
  Add a --from-session PATH argument that skips the API entirely and loads
  images and metadata from an existing session folder. This allows re-running
  reconstruction with different parameters without re-downloading.
  When --from-session is used, log: "Loaded N images from session: <path>"

### Implementation notes
  - Write each image file immediately after download (not buffered to end of batch)
    so a partial run still saves what it got
  - The metadata JSON must be saved before image download begins, so camera
    parameters are always present even if the image download fails
  - If a file already exists in the session folder (e.g. partial re-run),
    skip re-downloading it and log: "Skipping <item_id>, already saved"

---

## Camera model (pers:* extension)

### Interior orientation
Derived intrinsic matrix K (3×3):
  fx = focal_length / pixel_spacing[0]
  fy = focal_length / pixel_spacing[1]
  cx = sensor_array_dimensions[0]/2 + principal_point_offset[0]
  cy = sensor_array_dimensions[1]/2 + principal_point_offset[1]

  K = [[fx,  0, cx],
       [ 0, fy, cy],
       [ 0,  0,  1]]

Scale K when the image is resized: fx, fy, cx, cy all scale by the same factor
  (new_long_edge / original_long_edge).

### Exterior orientation
Build rotation matrix R from OPK angles:
  Rω = Rx(omega), Rφ = Ry(phi), Rκ = Rz(kappa)
  R_opk = Rκ @ Rφ @ Rω

Convert all camera positions from ECEF to a local ENU (East-North-Up) frame:
  - Compute scene_origin_ecef as the mean of all camera_center_ecef positions
  - Derive R_ecef2enu from the geodetic coordinates of scene_origin_ecef
  - For each camera: t_local = R_ecef2enu @ (t_ecef - scene_origin_ecef)

Store per-camera:
  K  (3×3 float64)
  R  (3×3 float64, world→camera rotation)
  t  (3×1 float64, camera centre in local ENU metres)
  P  = K @ [R | -R@t]   (3×4 projection matrix)
  image (numpy array, BGR)
  direction label
  item_id
  session_path  (path to the saved .tif file for this image)

---

## Pair selection (reconstruction/pairs.py)

### Compute baselines
For every possible pair (i, j):
  baseline = np.linalg.norm(cam_i.t - cam_j.t)   # metres in local ENU

### Baseline filter
  MIN_BASELINE = 50 m
  MAX_BASELINE = 800 m

### Angle filter
Compute angle between optical axes (world-space Z column of each R).
Reject pairs where this angle exceeds 60 degrees.

### Scoring
  baseline_score: tent function peaking at 200 m
    if baseline <= 200: (baseline - MIN) / (200 - MIN)
    if baseline >  200: (MAX - baseline) / (MAX - 200)

  direction_bonus:
    nadir + oblique (any):        1.4
    same direction, adj strip:    1.2
    cross-oblique (e.g. N+S):     0.8
    identical image:              0.0

  score = baseline_score * direction_bonus

### Selection
Sort by score descending.
Select top N where N = min(candidates, max_pairs).
max_pairs default: 20, configurable via --max-pairs.
Log a summary table:
  "Pair 01: nadir_frame3 ↔ north_frame2  baseline=187m  score=0.94"

---

## Stereo reconstruction (reconstruction/stereo.py)

For each selected pair (cam1, cam2):

### 1. Rectification
Derive F analytically from projection matrices:
  R_rel = R2 @ R1.T
  t_rel = R2 @ (t1 - t2)
  tx    = skew-symmetric matrix of t_rel
  F     = inv(K2).T @ tx @ R_rel @ inv(K1)

Find sparse correspondences:
  cv2.SIFT_create(nfeatures=2000) + cv2.BFMatcher(cv2.NORM_L2) + ratio test 0.75

Apply cv2.stereoRectifyUncalibrated(pts1, pts2, F, imgSize) → H1, H2
Warp: cv2.warpPerspective(img, H, imgSize)

### 2. Disparity
cv2.StereoSGBM_create:
  minDisparity=0, numDisparities=128, blockSize=7
  P1=8*3*blockSize**2, P2=32*3*blockSize**2
  disp12MaxDiff=1, uniquenessRatio=10
  speckleWindowSize=100, speckleRange=32
  mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
Divide raw disparity by 16.0.

### 3. Depth → 3D points
Build Q from rectified geometry.
cv2.reprojectImageTo3D(disparity, Q) → XYZ in rectified space.
Back-transform to local ENU via H1 inverse and cam1 extrinsics.

Validity mask:
  disparity > 1.0, depth > 0, depth < 500 m
  Remove points > 3 std from depth median
Sample colour from rectified img1.
Skip pair if < 1000 valid points.

---

## Point cloud merging (reconstruction/pointcloud.py)

1. Concatenate all pair XYZ + RGB arrays
2. Convert to open3d.geometry.PointCloud
3. Voxel downsample: voxel_size = 0.5 m
4. Statistical outlier removal: nb_neighbors=20, std_ratio=2.0
5. Estimate normals: radius=2.0, max_nn=30, orient towards mean camera position
6. Log final point count

---

## Viewer (viewer/visualize.py)

Open3D interactive window:
  - Point cloud with per-point colours
  - Camera positions as small red spheres (radius 2 m), labelled by direction
  - Lines from each camera centre to scene origin
  - Flight strip footprints as thin grey rectangles on the ground plane
  - Window title: "Skråfoto 3D – <address or lon/lat>"
  Console on launch:
    "Loaded N points from M stereo pairs"
    "Cameras: X nadir, X north, X south, X east, X west"
    "Session folder: <path>"
    "Controls: R=reset view, Q=quit, S=save screenshot"

---

## Debug mode (--debug)

When --debug is passed:
  - Save rectified pair images:  debug_pair_<i>_left.jpg / _right.jpg
  - Save colour-mapped disparity: debug_pair_<i>_disp.png
  - Save per-pair point cloud:   debug_pair_<i>.ply
  - Print full baseline matrix as formatted table
  - Plot top-down camera layout in matplotlib (ENU X/Y), coloured by direction,
    with pair connections drawn as lines; save as debug_cameras.png in session folder

All debug outputs go into a debug/ subfolder inside the session folder.

---

## CLI (main.py)

  python main.py --address "Rådhuspladsen 1, København"
  python main.py --lonlat 12.5683,55.6761
  python main.py --lonlat 12.5683,55.6761 --save pointcloud.ply
  python main.py --from-session ./data/20240315_143022_lon12.5683_lat55.6761

Arguments:
  --address TEXT          Danish address, geocoded via DAWA API (no token needed)
  --lonlat FLOAT,FLOAT    WGS84 lon,lat — skips geocoding
  --token TEXT            API token (overrides DATAFORSYNINGEN_TOKEN env var)
  --save PATH             Save merged point cloud as PLY
  --no-viewer             Skip Open3D window
  --collection TEXT       STAC collection (default: skraafotos2023)
  --max-pairs INT         Max stereo pairs (default: 20)
  --min-baseline FLOAT    Min pair baseline metres (default: 50)
  --max-baseline FLOAT    Max pair baseline metres (default: 800)
  --max-images INT        Cap total images downloaded (default: 30)
  --data-dir PATH         Base directory for session storage (default: ./data)
  --from-session PATH     Load existing session folder, skip API entirely
  --debug                 Save intermediate debug outputs to session/debug/
  --verbose               Set log level to DEBUG

---

## requirements.txt

opencv-contrib-python>=4.9
open3d>=0.18
numpy>=1.26
requests>=2.31
scipy>=1.12
tqdm>=4.66
click>=8.1
matplotlib>=3.8

---

## Error handling and logging

- Use Python logging (INFO default, DEBUG with --verbose)
- If API returns < 2 images after bbox widening, exit with raw response shown
- If all pairs filtered out, print full candidate list with rejection reasons
- Wrap each pair reconstruction in try/except; one bad pair must not abort run
- If Open3D unavailable, fall back to saving PLY and printing point count
- End-of-run summary:
    "Session folder: <path>"
    "Images saved: N"
    "Pairs processed: M / N_selected"
    "Points before merge: X  after merge: Y"
    "Output saved to: <path>"  (if --save used)

---

## README.md

1. Prerequisites: Python 3.11+, free token from dataforsyningen.dk
2. Installation: pip install -r requirements.txt
3. Quick start — copy-paste example for Rådhuspladsen, Copenhagen
4. Session folder layout and how to use --from-session for re-runs
5. Multi-image strategy: flight strips, overlap, pair selection
6. Coordinate system: local ENU, origin = mean camera position
7. Tuning guide:
   - --max-pairs: more = denser cloud, slower
   - --min-baseline / --max-baseline: urban vs open terrain
   - SGBM blockSize: smaller = detail/noisy, larger = smooth
8. Known limitations: sparse angular coverage, no bundle adjustment,
   planar rectification breaks for large baseline/angle pairs
9. Suggested next steps: COLMAP bundle adjustment, Gaussian Splatting

---
## Prompt history

### 2024-XX-XX — Initial build
(This file is the initial prompt. Future sessions append below this line.)

### 2026-04-05 — Higher reconstruction resolution
Improved output resolution from ~1 m to ~0.25 m by: (1) increasing TARGET_LONG_EDGE
from 2000 to 4000 px in camera.py (effective lateral GSD 0.25 m vs 0.71 m), (2) adding
separate SGBM_LONG_EDGE=800 in stereo.py so SGBM runs fast on small images while
triangulation uses full 4000 px precision, (3) reducing voxel_size from 0.5 m to 0.25 m
in pointcloud.py, (4) increasing max triangulation points from 50k to 100k per pair, and
(5) wrapping stereoRectifyUncalibrated in try/except to handle degenerate match sets.

### 2026-04-05 — Notebook UI and interactive viewer
Added `notebook.ipynb`: a Jupyter notebook with an ipywidgets form (Address / Lon,Lat /
Load existing session toggle, advanced settings accordion) that runs the full reconstruction
pipeline and renders the output PLY in an embedded Plotly 3-D scatter viewer with
per-point colours, percentile outlier clipping, and dark-theme layout.
Added `plotly>=5.18` and `ipywidgets>=8.0` to requirements.txt.

### 2026-04-05 — Full greenfield implementation
Built the entire skraafoto3d application from scratch following the PROMPT.md specification.
Created all 17 files: requirements.txt, config.py, api/{__init__,models,client}.py,
storage/{__init__,session}.py, reconstruction/{__init__,camera,pairs,stereo,pointcloud,pipeline}.py,
viewer/{__init__,visualize}.py, main.py, and README.md.
Key implementation choices: ECEF→ENU coordinate conversion via geodetic decomposition,
OPK→rotation as Rκ@Rφ@Rω, uncalibrated stereo rectification with SIFT+BFMatcher,
SGBM disparity with 3σ depth filtering, open3d voxel downsample + statistical outlier removal,
Click CLI with mutually-exclusive --address/--lonlat/--from-session modes, and full debug mode
with per-pair outputs and matplotlib camera layout plot.

### 2026-04-06 — Camera convention fix, calibrated rectification, pair scoring
Rewrote camera.py to use correct OpenCV convention: API provides photogrammetric rotation
(camera +Z away from scene), converted with `_flip = diag([1,-1,-1])` so cam.R[2,:] is the
optical axis pointing into the scene. Switched stereo rectification from uncalibrated
(stereoRectifyUncalibrated + SIFT) to calibrated (cv2.stereoRectify + K/R/t), eliminating
dependency on feature matching for geometry. Replaced straight-line camera layout with
ground-footprint-based pair scoring: each camera's principal ray is traced to the ground plane
(ground_z=-1000m default) and pairs are ranked by how close the footprint midpoint is to the
target (Gaussian proximity bonus, σ=500m). Added --scene-radius for XY cylinder crop and
--scene-center / --ground-z CLI args. Updated Z filtering to Z-mode histogram peak with
symmetric ±200m window. Added progress bars (tqdm) throughout the pipeline.

### 2026-04-07/08 — Disparity sign swap, oblique pair wall geometry, git init
Fixed a systematic bug where SGBM produced zero valid matches for all oblique pairs: when cam2
is "left" of cam1 in rectified space, valid disparities are negative but SGBM only searches
positive (minDisparity=0). Detection: sign of P2_rect[axis,3] from stereoRectify — positive Tx
(or Ty for vertical stereo) means swap is needed. Fixed in _calibrated_rectify (stereo.py) which
now returns a cam_swapped flag; process_pair swaps both homographies and camera objects when set.
Added tilt-aware minimum depth for disparity pre-screening: min_depth = 1300m / cos(tilt) so
oblique cameras (45°) use ~1838m instead of the flat 900m estimate that was falsely rejecting
valid within-strip pairs. Replaced symmetric Z clip (±300m) with asymmetric [-30m, +100m] around
Z-mode in both per-pair (stereo.py) and global (pointcloud.py) filtering — buildings extend above
ground, never below, so tight below-ground clamping eliminates SGBM noise without clipping walls.
Added Z-mode depth sanity check: pairs whose reconstructed Z-mode is >2500m below camera altitude
are rejected as noise-only. Results: 14/15 pairs succeed across all 5 directions (nadir, south,
east, west, north); 212k clean points; 54.7% above ground level confirming wall/rooftop geometry.
Initialized git repository, added .gitignore (data/, *.ply, .env, etc.), created CLAUDE.md.
### 2026-04-15 — Fix oblique stereo debris: perpendicular baseline filter + SGBM feasibility
Diagnosed why non-nadir pairs produced garbage 3D points when nadir-nadir direction was excluded.
Root cause 1: along-track (N-S) baselines for south oblique cameras make cv2.stereoRectify place
cy_rect≈10478px (scene at v=-240, outside the 2999px-tall rectified image). SGBM matches different
neighbourhoods in left and right images → scatter debris. Fix: perpendicular baseline filter in
compute_pairs (pairs.py) — reject pairs where perp_baseline < 0.8 * baseline. South cameras with
N-S baselines have perp_fraction=cos(45°)=0.707 < 0.8, so they are rejected; E-W baselines have
perp_fraction≈1.0 → accepted; nadir pairs always pass.
Root cause 2: for large E-W baselines (>400m), stereoRectify with alpha=0 zooms into the overlap
region, inflating f_rect by up to 9×, producing SGBM-scale disparities of 2000-6000px >> 512 limit.
A previous session attempted to fix this with adaptive newImageSize, but this is mathematically
ineffective: reducing newImageSize also increases sgbm_scale proportionally (sgbm_scale =
SGBM_LONG_EDGE / max(rect_size)), so SGBM disparity = f_rect*baseline*sgbm_scale/depth is invariant
to newImageSize. Fix: removed the adaptive newImageSize logic from _calibrated_rectify (stereo.py)
and reverted _estimate_pair_disparity (pairs.py) to return the actual d_est without the incorrect
'return _TARGET_D_EST' shortcut that made infeasible pairs appear feasible. The practical max
feasible baseline for south-south pairs is ~350m (d_est ~450px at SGBM scale).
Also fixed rect_size propagation in stereo.py: disp_full resize and ROI mask now use rect_size
(= img_size) consistently.
Added tests/test_oblique_stereo.py with 4 geometric validation tests: (1) south footprints contain
target, (2) along-track pairs rejected, (3) cross-track south pair produces valid cloud near target,
(4) output points within footprint bbox. All 4 pass: the south pair (strips 142↔208, perp=277m,
d_est=434px) yields 11,500+ points with centroid 192m from Rådhuspladsen and 0% outside bbox.
### 2026-04-16 — Fix separated point-cloud clumps for south oblique stereo
Diagnosed root cause of south-south point clouds appearing as separated clumps: the only SGBM-
feasible south pair (142↔208) had a 296m altitude difference out of a 331m baseline. stereoRectify
placed the scene principal point at cy=−4363 (4363px above the image top), leaving only the bottom
8% of the rectified image as valid overlap. Result: tiny 145m×120m coverage area → separated clumps.
Fix: raised the SGBM disparity cap from 512 to 608 in both reconstruction/stereo.py (MAX_NUM_DISP)
and reconstruction/pairs.py (_MAX_SGBM_DISP). This admits the same-altitude cross-track south pairs
(strips 208↔209, dz≈4m, perp=600m, d_est=577px) which were previously rejected at the 512px cap.
Same-altitude pairs have no cy≈−4363 problem (matching flight height → scene centered in both
images), yielding full ROI coverage.
Also added an altitude-difference pre-filter in compute_pairs: pairs where |Δz| > 0.4 × baseline
are rejected before calling stereoRectify, avoiding the degenerate cy geometry and reducing
unnecessary computation.
Added img_size field to Camera dataclass so _estimate_pair_disparity can read image dimensions
without materialising the image array in memory.
Updated tests/test_oblique_stereo.py: raised d_est filter threshold from 450 to 608 to match the
new cap; fixed pair-scoring in tests 3 and 4 to weight footprint proximity (score = perp /
(1 + fd/200)) so the test selects the pair whose cameras actually see the target rather than the
pair with the marginally largest baseline. All 4 tests pass: 208↔209 pair (perp=600m, d_est=577px)
yields 21,930 points with centroid 240m from Rådhuspladsen, 0% outside footprint bbox.
### 2026-04-25 — Target-crop: use camera geometry to focus SGBM on scene area
Root cause of south-camera "band outside nadir cloud": SGBM runs on the full rectified image and
matches off-target content (fields, roads far from the target). The output 3D points lie outside
the scene_radius crop applied later in merge_pointclouds, giving the appearance of a separate band.
Also, east/north cameras in this dataset have ground footprints ~1300m from the target (they look
away from it), so almost all their points are discarded at merge time; fp_dist column now makes
this transparent in the notebook UI.

Fix: added _target_crop_window() to reconstruction/stereo.py. Projects the scene cylinder
(scene_center ± scene_radius, ground_z … ground_z+100 m) through cam1's full projection matrix and
then through H1 (rectification homography) to compute a tight bounding box in rectified-image
pixels. If the box covers <90% of the full rectified image it is used to crop rect1/rect2 before
SGBM so the stereo matcher only processes the target area.

Changes to process_pair() in reconstruction/stereo.py:
- New parameters: scene_center (np.ndarray|None), scene_radius (float|None), ground_z (float=-1300)
- After warpPerspective debug saves: call _target_crop_window; crop rect1/rect2; track crop_x,
  crop_y, active_w, active_h (default to 0/0/rect_size when no crop)
- SGBM sizing: split sgbm_scale_cap (for disparity cap check, uses max(rect_size)) from sgbm_scale
  (for actual SGBM, uses max(active_w, active_h)) so cap check stays consistent with pairs.py
- disp_full resize: uses (active_w, active_h) instead of rect_size
- ROI mask: uses (active_h, active_w) shape; ROI coordinates shifted by (-crop_x, -crop_y) and
  clamped to [0, active_w/h]
- Triangulation: us = cols + crop_x, vs = rows + crop_y (convert crop-relative to rectified coords)
- Color sampling: unchanged (rect1 is already cropped)

Updated reconstruction/pipeline.py: process_pair() call now passes scene_center, scene_radius,
ground_z forwarded from run().

Prior session also added:
- fp_dist (footprint midpoint distance to scene_center) in pair_stats
- Per-pair viewer in pipeline.py crops xyz to scene_radius before subsampling, so view matches PLY
- notebook.ipynb: pair legend shows fp=Xm, table has pts(scene) column, merged PLY shown after grid
