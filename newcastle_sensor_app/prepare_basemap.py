# Needed for reducing whole basemap down for kriging background

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from rasterio.enums import Resampling
from rasterio.windows import Window, from_bounds

APP_DIR = Path(__file__).resolve().parent
INPUT_FILE = APP_DIR / "newcastle_basemap.tif"
OUTPUT_FILE = APP_DIR / "newcastle_basemap_small.tif"

BNG_TRANSFORMER = Transformer.from_crs("EPSG:4326", "EPSG:27700", always_xy=True,)


def sensor_bounds():
    # Return a padded BNG rectangle around the app's sensors

    sensors = pd.read_csv("naming.csv", encoding="utf-8-sig")
    sensors.columns = sensors.columns.str.strip()

    sensor_names = sensors["sensor_name"].astype(str).str.strip()
    is_monitor = sensor_names.str.startswith("PER_AIRMON_MONITOR")
    is_reference = ~sensor_names.str.startswith("PER_AIRMON_")
    sensors = sensors.loc[is_monitor | is_reference].copy()

    sensors["latitude"] = pd.to_numeric(sensors["latitude"], errors="coerce")
    sensors["longitude"] = pd.to_numeric(sensors["longitude"], errors="coerce")
    sensors = sensors.dropna(subset=["latitude", "longitude"])

    if sensors.empty:
        raise ValueError("No valid sensor coordinates were found in naming.csv")

    eastings, northings = BNG_TRANSFORMER.transform(sensors["longitude"].to_numpy(), sensors["latitude"].to_numpy(),)

    return (
        float(np.min(eastings)) - 5000,
        float(np.min(northings)) - 5000,
        float(np.max(eastings)) + 5000,
        float(np.max(northings)) + 5000,
    )


def prepare_basemap():
    # Crop and reduce the full raster

    requested_left, requested_bottom, requested_right, requested_top = sensor_bounds()

    with rasterio.open(INPUT_FILE) as source:

        # Restrict the requested sensor rectangle to the raster.
        left = max(requested_left, source.bounds.left)
        bottom = max(requested_bottom, source.bounds.bottom)
        right = min(requested_right, source.bounds.right)
        top = min(requested_top, source.bounds.top)

        if left >= right or bottom >= top:
            raise ValueError("The sensor area does not overlap the Digimap raster")

        window = from_bounds(left, bottom, right, top, source.transform)
        window = window.round_offsets().round_lengths()
        window = window.intersection(Window(0, 0, source.width, source.height))

        reduction = min(
            1.0,
            2000 / window.width,
            2000 / window.height,
        )
        output_width = max(1, round(window.width * reduction))
        output_height = max(1, round(window.height * reduction))

        image = source.read(window=window,
                            out_shape=(source.count, output_height, output_width),
                            resampling=Resampling.bilinear,)

        crop_transform = source.window_transform(window)
        output_transform = crop_transform * crop_transform.scale(window.width / output_width, window.height / output_height,)

        profile = source.profile.copy()
        profile.update(width=output_width, height=output_height, transform=output_transform,
                       compress="deflate", tiled=True, blockxsize=256, blockysize=256,)

        colour_map = None
        if source.count == 1:
            try:
                colour_map = source.colormap(1)
            except ValueError:
                pass

    with rasterio.open(OUTPUT_FILE, "w", **profile) as destination:
        destination.write(image)
        if colour_map:
            destination.write_colormap(1, colour_map)

    size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"Created {OUTPUT_FILE.name}")
    print(f"Dimensions: {output_width} x {output_height} pixels")
    print(f"File size: {size_mb:.1f} MB")


if __name__ == "__main__":
    prepare_basemap()
