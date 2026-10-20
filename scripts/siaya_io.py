"""Windowed Vantor reads and atomic checkpoints for the full Siaya map."""

import json
from pathlib import Path

import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import transform_geom
from rasterio.windows import from_bounds
from shapely.geometry import box, mapping, shape
from shapely.strtree import STRtree

CRS = "EPSG:32636"
PIXEL_M = 0.5
CLOUD_MARGIN_M = 100


def save_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False))
    tmp.replace(path)


def save_tile(path, **arrays):
    tmp = Path(str(path) + ".tmp")
    with tmp.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    tmp.replace(path)


def intersects(a, b):
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


class Strip:
    """RGB and validity share a grid; cloud shadows are not detected."""

    def __init__(self, record):
        self.record = record
        self.dataset = rasterio.open(record["rgb"])
        ds = self.dataset
        if str(ds.crs) != CRS or ds.res != (PIXEL_M, PIXEL_M):
            raise ValueError(f"Unexpected grid: {record['rgb']}")
        if ds.count != 3 or ds.dtypes != ("uint8",) * 3:
            raise ValueError("Expected RGB uint8 VRT")
        cloud = json.loads(Path(record["clouds"]).read_text())
        features = cloud.get("features", [cloud])
        polygons = []
        for feature in features:
            geom = feature.get("geometry")
            if not geom or not geom.get("coordinates"):
                continue
            projected = shape(transform_geom("EPSG:4326", CRS, geom)).buffer(
                CLOUD_MARGIN_M
            )
            if not projected.is_empty:
                polygons.extend(
                    projected.geoms
                    if projected.geom_type == "MultiPolygon"
                    else [projected]
                )
        self.clouds = STRtree(polygons)

    def read(self, bounds):
        """Read only this window and rasterize intersecting buffered cloud polygons."""
        ds = self.dataset
        win = from_bounds(*bounds, ds.transform)
        if any(
            abs(v - round(v)) > 1e-5
            for v in (win.col_off, win.row_off, win.width, win.height)
        ):
            raise ValueError("Window is not on the shared pixel grid")
        win = win.round_offsets().round_lengths()
        image = ds.read(window=win, boundless=True, fill_value=0)
        valid = ds.read_masks(window=win, boundless=True).min(0) > 0
        transform = ds.window_transform(win)
        selected = self.clouds.query(box(*bounds), predicate="intersects")
        if len(selected):
            shapes = [mapping(self.clouds.geometries[i]) for i in selected]
            valid &= ~geometry_mask(
                shapes, valid.shape, transform, invert=True, all_touched=True
            )
        return image, valid

    def close(self):
        self.dataset.close()
