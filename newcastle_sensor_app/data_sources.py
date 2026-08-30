from datetime import datetime
from functools import lru_cache
from io import StringIO
from pathlib import Path

import pandas as pd
import rdata
import warnings
import requests
import uo_pyfetch


DEFRA_RDATA_BASE = "https://uk-air.defra.gov.uk/openair/R_data"
AQE_RDATA_BASE = ("https://airqualityengland.co.uk/assets/openair/R_data")
UO_BASE = "https://api.v2.urbanobservatory.ac.uk"
NEWCASTLE_BBOX = [-1.85, 54.85, -1.35, 55.15]
HTTP_TIMEOUT = 10


def load_sensor_registry(path: Path) -> pd.DataFrame:
    registry = pd.read_csv(path, encoding="utf-8-sig")
    registry.columns = registry.columns.str.strip()
    registry["sensor_name"] = registry["sensor_name"].astype(str).str.strip()
    registry["new_name"] = registry["new_name"].astype(str).str.strip()
    registry["provider"] = registry["sensor_name"].map(_provider_from_name)
    registry["code"] = (registry["code"]
                        .astype(str)
                        .str.strip())
    registry["type"] = (registry["type"]
                        .astype(str)
                        .str.strip())   
    return registry.rename(columns={"new_name": "display_name"})


def _provider_from_name(sensor_name: str) -> str:
    if sensor_name.startswith("PER_AIRMON_MESH"):
        return "UO-Mesh"
    if sensor_name.startswith("PER_AIRMON_MONITOR"):
        return "UO-Mon"
    return "DEFRA/Local"


def get_sensor_metadata(registry: pd.DataFrame) -> pd.DataFrame:
    """Join the app registry to live UO coordinates.

    DEFRA/local rows remain in the table. Add their coordinates to naming.csv
    later, or join them in a future DEFRA adapter (DONE)
    """
    result = registry.copy()
    if "latitude" not in result:
        result["latitude"] = pd.NA
    if "longitude" not in result:
        result["longitude"] = pd.NA

    try:
        # Fetch only the local area. Requesting every UO sensor can make the
        # initial Shiny session appear to hang.
        response = requests.get(
            f"{UO_BASE}/sensors/csv",
            params={
                "limit": 5000,
                "offset": 0,
                "bbox_p1_x": NEWCASTLE_BBOX[0],
                "bbox_p1_y": NEWCASTLE_BBOX[1],
                "bbox_p2_x": NEWCASTLE_BBOX[2],
                "bbox_p2_y": NEWCASTLE_BBOX[3],
            },
            timeout=HTTP_TIMEOUT,
        )
        response.raise_for_status()
        uo_sensors = pd.read_csv(StringIO(response.text))
        coords = (
            uo_sensors[
                ["Sensor_Name", "Sensor_Centroid_Latitude", "Sensor_Centroid_Longitude"]
            ]
            .drop_duplicates("Sensor_Name")
            .rename(
                columns={
                    "Sensor_Name": "sensor_name",
                    "Sensor_Centroid_Latitude": "uo_latitude",
                    "Sensor_Centroid_Longitude": "uo_longitude",
                }
            )
        )
        result = result.merge(coords, on="sensor_name", how="left")
        result["latitude"] = result["uo_latitude"].combine_first(result["latitude"])
        result["longitude"] = result["uo_longitude"].combine_first(result["longitude"])
        result = result.drop(columns=["uo_latitude", "uo_longitude"])
    except Exception as exc:
        print(f"Could not load Urban Observatory sensor coordinates: {exc}")

    try:
        defra_meta = _get_defra_metadata()
        result = result.merge(defra_meta, on="code", how="left")
        result["latitude"] = result["defra_latitude"].combine_first(result["latitude"])
        result["longitude"] = result["defra_longitude"].combine_first(result["longitude"])
        result = result.drop(columns=["defra_latitude", "defra_longitude"])
    except Exception as exc:
        print(f"Could not load DEFRA sensor coordinates: {exc}")

    return result


def get_uo_readings(sensor_name: str, start: datetime, end: datetime, variable: str = "PM2.5",) -> pd.DataFrame:
    try:
        response = requests.get(
            f"{UO_BASE}/sensors/{sensor_name}/data/csv",
            params={
                "start": start.isoformat(),
                "end": end.isoformat(),
                "variables": variable,
                "limit": 100_000,
                "offset": 0,
            },
            timeout=HTTP_TIMEOUT,
        )
        response.raise_for_status()
        readings = pd.read_csv(StringIO(response.text))
    except Exception as exc:
        print(f"Could not load readings for {sensor_name}: {exc}")
        return pd.DataFrame()

    if readings.empty:
        return readings

    readings = readings.copy()
    readings["Timestamp"] = pd.to_datetime(readings["Timestamp"], utc=True, errors="coerce")
    readings["Value"] = pd.to_numeric(readings["Value"], errors="coerce")
    if "Flagged" in readings.columns:
        readings = readings.loc[~readings["Flagged"].fillna(False).astype(bool)]
    return readings.dropna(subset=["Timestamp", "Value"]).sort_values("Timestamp")


def get_defra_readings(site_code: str, start: datetime, end: datetime, variable: str = "PM2.5", source_type: str = "AURN",) -> pd.DataFrame:
    frames = []
    for year in range(start.year, end.year + 1):
        try:
            yearly = _get_defra_year(site_code.upper(), year, source_type).copy()
        except (requests.RequestException, ValueError) as exc:
            print(f"Could not load DEFRA data for {site_code}, {year}: {exc}")
            continue

        if "date" not in yearly.columns or variable not in yearly.columns:
            continue
        frame = yearly[["date", variable]].rename(
            columns={"date": "Timestamp", variable: "Value"}
        )
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    readings = pd.concat(frames, ignore_index=True)
    # The R files store POSIX timestamps as seconds.
    if pd.api.types.is_numeric_dtype(readings["Timestamp"]):
        readings["Timestamp"] = pd.to_datetime(readings["Timestamp"], unit="s", utc=True, errors="coerce")
    else:
        readings["Timestamp"] = pd.to_datetime(readings["Timestamp"], utc=True, errors="coerce")
    readings["Value"] = pd.to_numeric(readings["Value"], errors="coerce")
    start_utc = pd.Timestamp(start)
    if start_utc.tzinfo is None:
        start_utc = start_utc.tz_localize("UTC")
    else:
        start_utc = start_utc.tz_convert("UTC")
    end_utc = pd.Timestamp(end)
    if end_utc.tzinfo is None:
        end_utc = end_utc.tz_localize("UTC")
    else:
        end_utc = end_utc.tz_convert("UTC")

    return (
        readings.loc[readings["Timestamp"].between(start_utc, end_utc)]
        .dropna(subset=["Timestamp", "Value"])
        .sort_values("Timestamp")
    )


@lru_cache(maxsize=32)
def _get_defra_year(site_code: str, year: int, source_type: str,) -> pd.DataFrame:
    if source_type.strip().lower() == "local authority":
        base_url = AQE_RDATA_BASE
    else:
        base_url = DEFRA_RDATA_BASE
    object_name = f"{site_code}_{year}"
    url = f"{base_url}/{object_name}.RData"

    return _read_remote_rdata(
        url,
        object_name=object_name,
    )


@lru_cache(maxsize=1)
def _get_defra_metadata() -> pd.DataFrame:
    metadata = _read_remote_rdata(f"{DEFRA_RDATA_BASE}/AURN_metadata.RData")
    return (
        metadata[["site_id", "latitude", "longitude"]]
        .drop_duplicates("site_id")
        .rename(
            columns={
                "site_id": "code",
                "latitude": "defra_latitude",
                "longitude": "defra_longitude",
            }
        )
    )


def _read_remote_rdata(url: str, object_name: str | None = None,) -> pd.DataFrame:

    response = requests.get(url, timeout=HTTP_TIMEOUT,)
    response.raise_for_status()
    # Parse directly from the downloaded bytes.
    # This avoids Windows locking NamedTemporaryFile.
    parsed = rdata.parser.parse_data(response.content)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning,)
        objects = rdata.conversion.convert(parsed)
    if not objects:
        raise ValueError(f"No data frame found in {url}")
    if object_name and object_name in objects:
        return objects[object_name]

    return next(iter(objects.values()))
