from dataclasses import dataclass
from typing import Any


@dataclass
class InteriorOrientation:
    camera_id: str
    focal_length: float          # mm
    pixel_spacing: list[float]   # [sx, sy] mm/pixel
    principal_point_offset: list[float]  # [ppx, ppy] mm
    sensor_array_dimensions: list[int]   # [width, height] pixels

    @classmethod
    def from_dict(cls, d: dict) -> "InteriorOrientation":
        return cls(
            camera_id=d.get("camera_id", ""),
            focal_length=float(d["focal_length"]),
            pixel_spacing=[float(v) for v in d["pixel_spacing"]],
            principal_point_offset=[float(v) for v in d["principal_point_offset"]],
            sensor_array_dimensions=[int(v) for v in d["sensor_array_dimensions"]],
        )


@dataclass
class Item:
    id: str
    collection: str
    geometry: dict[str, Any]
    bbox: list[float]
    datetime: str
    direction: str
    gsd: float

    # Full-resolution image URL (from properties.asset:data)
    full_href: str
    thumbnail_href: str

    # Interior orientation
    interior: InteriorOrientation

    # Exterior orientation — flat fields from the actual API
    omega: float                       # degrees
    phi: float                         # degrees
    kappa: float                       # degrees
    perspective_center: list[float]    # [X, Y, Z] in UTM EPSG:25832
    rotation_matrix: list[float]       # 9-element row-major 3×3

    @classmethod
    def from_feature(cls, feature: dict) -> "Item":
        props = feature["properties"]

        # Assets may be in feature["assets"] OR in properties as "asset:data"
        full_href = (
            props.get("asset:data")
            or (feature.get("assets", {}).get("full", {}).get("href", ""))
        )
        thumbnail_href = (
            props.get("asset:thumbnail")
            or (feature.get("assets", {}).get("thumbnail", {}).get("href", ""))
        )

        return cls(
            id=feature["id"],
            collection=feature.get("collection", ""),
            geometry=feature.get("geometry", {}),
            bbox=feature.get("bbox", []),
            datetime=props.get("datetime", ""),
            direction=props.get("direction", ""),
            gsd=float(props.get("gsd", 0.0)),
            full_href=full_href,
            thumbnail_href=thumbnail_href,
            interior=InteriorOrientation.from_dict(props["pers:interior_orientation"]),
            omega=float(props["pers:omega"]),
            phi=float(props["pers:phi"]),
            kappa=float(props["pers:kappa"]),
            perspective_center=[float(v) for v in props["pers:perspective_center"]],
            rotation_matrix=[float(v) for v in props["pers:rotation_matrix"]],
        )

    def to_dict(self) -> dict:
        """Serialisable dict that round-trips through from_feature."""
        return {
            "id": self.id,
            "type": "Feature",
            "collection": self.collection,
            "geometry": self.geometry,
            "bbox": self.bbox,
            "properties": {
                "datetime": self.datetime,
                "direction": self.direction,
                "gsd": self.gsd,
                "asset:data": self.full_href,
                "asset:thumbnail": self.thumbnail_href,
                "pers:interior_orientation": {
                    "camera_id": self.interior.camera_id,
                    "focal_length": self.interior.focal_length,
                    "pixel_spacing": self.interior.pixel_spacing,
                    "principal_point_offset": self.interior.principal_point_offset,
                    "sensor_array_dimensions": self.interior.sensor_array_dimensions,
                },
                "pers:omega": self.omega,
                "pers:phi": self.phi,
                "pers:kappa": self.kappa,
                "pers:perspective_center": self.perspective_center,
                "pers:rotation_matrix": self.rotation_matrix,
            },
        }
