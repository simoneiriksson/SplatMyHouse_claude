#!/usr/bin/env python3
"""Skråfoto 3D — CLI entry point."""

import logging
import sys
from pathlib import Path

import click
import cv2
import numpy as np

import config
from api.client import SkraafotoClient
from storage.session import Session
from reconstruction.camera import build_cameras
from reconstruction.pipeline import run as run_pipeline
from viewer.visualize import show, save_ply


def _lonlat_to_utm32n(lon: float, lat: float) -> tuple[float, float]:
    """Forward projection from WGS84 to UTM Zone 32N (EPSG:25832)."""
    import math
    a, f = 6378137.0, 1 / 298.257223563
    e2 = 2 * f - f ** 2
    k0, lon0, E0 = 0.9996, math.radians(9.0), 500_000.0
    phi, lam = math.radians(lat), math.radians(lon) - lon0
    N = a / math.sqrt(1 - e2 * math.sin(phi) ** 2)
    T, C = math.tan(phi) ** 2, e2 / (1 - e2) * math.cos(phi) ** 2
    A = math.cos(phi) * lam
    M = a * (
        (1 - e2 / 4 - 3 * e2 ** 2 / 64) * phi
        - (3 * e2 / 8 + 3 * e2 ** 2 / 32) * math.sin(2 * phi)
        + (15 * e2 ** 2 / 256) * math.sin(4 * phi)
    )
    E = k0 * N * (A + (1 - T + C) * A ** 3 / 6) + E0
    N_utm = k0 * (M + N * math.tan(phi) * (A ** 2 / 2 + (5 - T + 9 * C) * A ** 4 / 24))
    return E, N_utm


def _bbox_from_lonlat(lon: float, lat: float, radius_m: float = 200.0) -> list[float]:
    """Compute a WGS84 bbox ±radius_m around lon/lat."""
    import math
    lat_deg = radius_m / 111_320.0
    lon_deg = radius_m / (111_320.0 * math.cos(math.radians(lat)))
    return [lon - lon_deg, lat - lat_deg, lon + lon_deg, lat + lat_deg]


@click.command()
@click.option("--address", default=None, help="Danish address (geocoded via DAWA API).")
@click.option("--lonlat", default=None, help="WGS84 lon,lat e.g. 12.5683,55.6761")
@click.option("--token", default=None, help="API token (overrides DATAFORSYNINGEN_TOKEN env).")
@click.option("--save", default=None, metavar="PATH", help="Save point cloud as PLY.")
@click.option("--no-viewer", is_flag=True, default=False, help="Skip Open3D window.")
@click.option("--collection", default=config.DEFAULT_COLLECTION, show_default=True,
              help="STAC collection ID.")
@click.option("--max-pairs", default=config.MAX_PAIRS, show_default=True, type=int,
              help="Max stereo pairs.")
@click.option("--min-baseline", default=config.MIN_BASELINE, show_default=True, type=float,
              help="Min pair baseline (metres).")
@click.option("--max-baseline", default=config.MAX_BASELINE, show_default=True, type=float,
              help="Max pair baseline (metres).")
@click.option("--max-images", default=config.MAX_IMAGES, show_default=True, type=int,
              help="Cap total images downloaded.")
@click.option("--data-dir", default="./data", show_default=True,
              help="Base directory for session storage.")
@click.option("--from-session", default=None, metavar="PATH",
              help="Load existing session folder, skip API.")
@click.option("--scene-radius", default=config.SCENE_RADIUS, show_default=True, type=float,
              help="Crop point cloud to this XY radius (metres) around the target location.")
@click.option("--debug", is_flag=True, default=False,
              help="Save debug outputs to session/debug/.")
@click.option("--verbose", is_flag=True, default=False, help="Set log level to DEBUG.")
def main(
    address, lonlat, token, save, no_viewer, collection,
    max_pairs, min_baseline, max_baseline, max_images,
    data_dir, from_session, scene_radius, debug, verbose,
):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger("main")

    # ------------------------------------------------------------------ #
    # Validate inputs
    # ------------------------------------------------------------------ #
    mode_count = sum([address is not None, lonlat is not None, from_session is not None])
    if mode_count == 0:
        raise click.UsageError(
            "Provide one of: --address, --lonlat, or --from-session."
        )
    if mode_count > 1:
        raise click.UsageError(
            "--address, --lonlat, and --from-session are mutually exclusive."
        )

    # ------------------------------------------------------------------ #
    # Case 1: load from existing session
    # ------------------------------------------------------------------ #
    if from_session is not None:
        items, images = Session.load_from_path(from_session)
        if len(items) < 2:
            log.error("Session %s has fewer than 2 usable images.", from_session)
            sys.exit(1)
        session_folder = Path(from_session)
        location_label = session_folder.name

        cameras, origin_utm = build_cameras(items, images, session_folder=session_folder)

        # Compute scene centre in local ENU from the session's query lonlat
        scene_center_enu = _scene_center_from_session(from_session, origin_utm, log)

        pcd, stats = run_pipeline(
            cameras,
            max_pairs=max_pairs,
            min_baseline=min_baseline,
            max_baseline=max_baseline,
            debug=debug,
            session_folder=session_folder,
            scene_center=scene_center_enu,
            scene_radius=scene_radius,
        )
        _print_summary(session_folder, len(items), stats, save)
        if save:
            save_ply(pcd, save)
        if not no_viewer:
            show(pcd, cameras, location_label, session_path=session_folder, save_path=save)
        return

    # ------------------------------------------------------------------ #
    # Case 2: fetch from API
    # ------------------------------------------------------------------ #
    api_token = config.get_token(token)
    client = SkraafotoClient(api_token)

    if address is not None:
        lon, lat = client.geocode(address)
        log.info("Geocoded '%s' → lon=%.6f lat=%.6f", address, lon, lat)
        location_label = address
    else:
        try:
            lon_s, lat_s = lonlat.split(",")
            lon, lat = float(lon_s.strip()), float(lat_s.strip())
        except ValueError:
            raise click.UsageError("--lonlat must be two floats separated by a comma.")
        location_label = f"lon{lon:.4f}_lat{lat:.4f}"

    bbox = _bbox_from_lonlat(lon, lat)
    log.info("Searching bbox: %.6f,%.6f,%.6f,%.6f", *bbox)

    items = client.search(bbox, collection=collection, max_images=max_images)

    if len(items) < 2:
        log.error(
            "API returned only %d image(s) — cannot run stereo. "
            "Try widening the search area or using a different collection.",
            len(items),
        )
        sys.exit(1)

    # Create session and save metadata + images
    session = Session(base_dir=data_dir, lon=lon, lat=lat, address=address)

    images: dict[str, "np.ndarray"] = {}
    saved = 0
    for item in items:
        session.save_item_metadata(item)
        img_path = session.image_path(item.id)
        if img_path.exists():
            log.info("Skipping %s, already saved", item.id)
        else:
            try:
                raw = client.download_image_bytes(item.full_href)
                session.save_image_bytes(item.id, raw)
                saved += 1
            except Exception as exc:
                log.warning("Download failed for %s: %s", item.id, exc)
                continue

        # Decode from saved file
        raw_bytes = img_path.read_bytes()
        arr_np = np.frombuffer(raw_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr_np, cv2.IMREAD_COLOR)
        if img is not None:
            images[item.id] = img
        else:
            log.warning("Could not decode image for %s", item.id)

    query_info = {
        "lonlat": [lon, lat],
        "address": address,
        "bbox": bbox,
        "collection": collection,
    }
    session.write_session_json(query_info, items)

    if len(images) < 2:
        log.error("Fewer than 2 images decoded successfully.")
        sys.exit(1)

    cameras, origin_utm = build_cameras(items, images, session_folder=session.folder)

    # Compute scene centre in local ENU from the query lon/lat
    target_e, target_n = _lonlat_to_utm32n(lon, lat)
    scene_center_enu = np.array([target_e - origin_utm[0], target_n - origin_utm[1], 0.0])
    log.info("Scene centre in ENU: (%.1f, %.1f) m", scene_center_enu[0], scene_center_enu[1])

    pcd, stats = run_pipeline(
        cameras,
        max_pairs=max_pairs,
        min_baseline=min_baseline,
        max_baseline=max_baseline,
        debug=debug,
        session_folder=session.folder,
        scene_center=scene_center_enu,
        scene_radius=scene_radius,
    )

    _print_summary(session.folder, len(items), stats, save)

    if save:
        save_ply(pcd, save)

    if not no_viewer:
        show(pcd, cameras, location_label, session_path=session.folder, save_path=save)


def _scene_center_from_session(session_path, origin_utm: np.ndarray, log) -> np.ndarray:
    """Read the query lonlat from session.json and return the target in local ENU."""
    import json
    sess_json = Path(session_path) / "session.json"
    if not sess_json.exists():
        log.warning("session.json not found; using ENU origin as scene centre")
        return np.zeros(3)
    data = json.loads(sess_json.read_text())
    lonlat = data.get("query", {}).get("lonlat")
    if lonlat is None:
        log.warning("No lonlat in session.json; using ENU origin as scene centre")
        return np.zeros(3)
    lon, lat = lonlat
    target_e, target_n = _lonlat_to_utm32n(lon, lat)
    center = np.array([target_e - origin_utm[0], target_n - origin_utm[1], 0.0])
    log.info("Scene centre in ENU: (%.1f, %.1f) m", center[0], center[1])
    return center


def _print_summary(session_folder, n_images, stats, save_path):
    print("")
    print("=" * 50)
    print(f"Session folder:     {session_folder}")
    print(f"Images saved:       {n_images}")
    print(f"Pairs processed:    {stats['pairs_processed']} / {stats['pairs_selected']}")
    print(f"Points before merge: {stats['points_before_merge']}")
    print(f"Points after merge:  {stats['points_after_merge']}")
    if save_path:
        print(f"Output saved to:    {save_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
