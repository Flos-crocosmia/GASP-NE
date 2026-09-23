from datetime import datetime
from html import unescape
from io import StringIO
from pathlib import Path
import re
from urllib.parse import urljoin
import warnings

import pandas as pd
import rdata
import requests

UO_URL = "https://api.v2.urbanobservatory.ac.uk"
AURN_URL = "https://uk-air.defra.gov.uk/openair/R_data"
LOCAL_URL = "https://airqualityengland.co.uk/assets/openair/R_data"
AQE_URL = "https://www.airqualityengland.co.uk"

HTTP_TIMEOUT = 20
HEADERS = {"User-Agent": "GASP-NE-Air-Quality/1.0"}


def load_sensor_registry(path: Path) -> pd.DataFrame:
    # Read naming.csv and keep only UO Monitor and reference sensors.

    sensors = pd.read_csv(path, encoding="utf-8-sig")
    sensors.columns = sensors.columns.str.strip()

    for column in ["sensor_name", "new_name", "code", "type"]:
        sensors[column] = sensors[column].astype(str).str.strip()
        
    is_monitor = sensors["sensor_name"].str.startswith("PER_AIRMON_MONITOR")
    is_reference = ~sensors["sensor_name"].str.startswith("PER_AIRMON_")
    sensors = sensors.loc[is_monitor | is_reference].copy()

    sensors["provider"] = "DEFRA/Local"
    sensors.loc[is_monitor, "provider"] = "UO-Mon"

    sensors["latitude"] = pd.to_numeric(sensors["latitude"], errors="coerce")
    sensors["longitude"] = pd.to_numeric(sensors["longitude"], errors="coerce")

    return sensors.rename(columns={"new_name": "display_name"})


def get_uo_readings(sensor_name: str, start: datetime, end: datetime, variable: str = "PM2.5",) -> pd.DataFrame:
    # Download data for one Urban Observatory Monitor sensor.

    url = f"{UO_URL}/sensors/{sensor_name}/data/csv"
    parameters = {"start": start.isoformat(), "end": end.isoformat(), "variables": variable,
                  "limit": 100_000,
                  "offset": 0,}
    try:
        response = requests.get(url, params=parameters, headers=HEADERS, timeout=HTTP_TIMEOUT,)
        response.raise_for_status()
        readings = pd.read_csv(StringIO(response.text))

    except (requests.RequestException, pd.errors.ParserError) as error:
        print(f"Could not load UO data for {sensor_name}: {error}")
        return empty_readings()

    if "Flagged" in readings.columns:
        flagged = readings["Flagged"].astype(str).str.lower().isin(["true", "1"])
        readings = readings.loc[~flagged]

    return clean_readings(readings, start, end)


def get_defra_readings(site_code: str, start: datetime, end: datetime, variable: str = "PM2.5", source_type: str = "AURN",) -> pd.DataFrame:
    # Download data from an AURN or local-authority monitoring site.

    site_code = str(site_code).strip().upper()
    is_local = str(source_type).strip().lower() == "local authority"

    # The STS sites do not have annual RData files.
    if is_local and site_code.startswith("STS"):
        return get_aqe_readings(site_code, start, end, variable)

    base_url = LOCAL_URL if is_local else AURN_URL
    yearly_data = []

    for year in range(start.year, end.year + 1):
        object_name = f"{site_code}_{year}"
        url = f"{base_url}/{object_name}.RData"
        try:
            response = requests.get(url, headers=HEADERS, timeout=HTTP_TIMEOUT,)
            response.raise_for_status()
            # Parse the downloaded R file directly from memory. This avoids
            # temporary-file permission errors on Windows.
            parsed_file = rdata.parser.parse_data(response.content)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                objects = rdata.conversion.convert(parsed_file)
            if object_name in objects:
                table = objects[object_name]
            elif objects:
                table = next(iter(objects.values()))
            else:
                continue
        except (requests.RequestException, ValueError) as error:
            print(f"Could not load {site_code} data for {year}: {error}")
            continue
        if "date" not in table.columns or variable not in table.columns:
            continue
        table = table[["date", variable]].rename(
            columns={"date": "Timestamp", variable: "Value"}
        )
        # R sometimes stores dates as seconds
        if pd.api.types.is_numeric_dtype(table["Timestamp"]):
            table["Timestamp"] = pd.to_datetime(table["Timestamp"], unit="s", utc=True, errors="coerce",)
        yearly_data.append(table)

    # Some newer local sites only provide data through the AQE download page
    if not yearly_data and is_local:
        return get_aqe_readings(site_code, start, end, variable)

    if not yearly_data:
        return empty_readings()

    readings = pd.concat(yearly_data, ignore_index=True)
    return clean_readings(readings, start, end)


def get_aqe_readings(site_code: str, start: datetime, end: datetime, variable: str = "PM2.5",) -> pd.DataFrame:
    # Download PM2.5 data using the Air Quality England download form
    if variable != "PM2.5":
        return empty_readings()
    site_page_url = f"{AQE_URL}/site/data"
    download_page_url = f"{AQE_URL}/site/data.php"
    try:
        with requests.Session() as session:
            session.headers.update(HEADERS)
            # Page 1 supplies the form identifiers for this monitoring site.
            first_page = session.get(site_page_url, params={"site_id": site_code}, timeout=HTTP_TIMEOUT,)
            first_page.raise_for_status()
            query_id = find_form_value(first_page.text, "f_query_id")
            local_authority_id = find_form_value(first_page.text, "la_id")
            # Page 2 applies the requested date range.
            second_page = session.get(download_page_url,
                                      params={"site_id": site_code,
                                              "f_date_started": start.strftime("%d/%m/%Y"),
                                              "f_date_ended": end.strftime("%d/%m/%Y"),
                                              "f_query_id": query_id,
                                              "la_id": local_authority_id,
                                              "action": "step2",
                                              "data": "",
                                              "submit": "Next",},
                                      timeout=HTTP_TIMEOUT,)
            second_page.raise_for_status()
            query_id = find_form_value(second_page.text, "f_query_id")
            # Page 3 selects PM2.5 and creates a temporary CSV link.
            third_page = session.get(download_page_url, 
                                     params={"site_id": site_code,
                                             "parameter_id[]": "PM25",
                                             "f_query_id": query_id,
                                             "f_date_started": start.date().isoformat(),
                                             "f_date_ended": end.date().isoformat(),
                                             "la_id": local_authority_id,
                                             "action": "download",
                                             "data": "",
                                             "submit": "Download Data",},
                                     timeout=HTTP_TIMEOUT,)
            third_page.raise_for_status()
            csv_match = re.search(r'href=["\']([^"\']+\.csv)["\']',
                                  third_page.text, flags=re.IGNORECASE,)
            if csv_match is None:
                raise ValueError("AQE did not create a CSV download link")
            csv_url = urljoin(download_page_url, unescape(csv_match.group(1)))
            csv_response = session.get(csv_url, timeout=HTTP_TIMEOUT)
            csv_response.raise_for_status()
        table = pd.read_csv(StringIO(csv_response.text), skiprows=5)

    except (requests.RequestException, ValueError, pd.errors.ParserError) as error:
        print(f"Could not load Air Quality England data for {site_code}: {error}")
        return empty_readings()

    required_columns = {"End Date", "End Time", "PM25"}
    if not required_columns.issubset(table.columns):
        print(f"Unexpected Air Quality England columns for {site_code}")
        return empty_readings()

    readings = pd.DataFrame()
    readings["Timestamp"] = pd.to_datetime(table["End Date"], format="%d/%m/%Y", 
                                           utc=True, errors="coerce",) + pd.to_timedelta(table["End Time"], errors="coerce")
    readings["Value"] = table["PM25"]

    return clean_readings(readings, start, end)


def find_form_value(page_html: str, field_name: str) -> str:
    # Find a hidden value in an Air Quality England form.
    pattern = (rf'name=["\']{re.escape(field_name)}["\']' rf'[^>]*value=["\']([^"\']+)["\']')
    match = re.search(pattern, page_html, flags=re.IGNORECASE)

    if match is None:
        raise ValueError(f"AQE page did not contain {field_name}")

    return unescape(match.group(1))


def clean_readings(readings: pd.DataFrame, start: datetime, end: datetime,) -> pd.DataFrame:
    # Convert readings to the Timestamp/Value format
    required_columns = {"Timestamp", "Value"}
    if readings.empty or not required_columns.issubset(readings.columns):
        return empty_readings()
    readings = readings[["Timestamp", "Value"]].copy()
    readings["Timestamp"] = pd.to_datetime(readings["Timestamp"], utc=True, errors="coerce",)
    readings["Value"] = pd.to_numeric(readings["Value"], errors="coerce")
    readings = readings.dropna().sort_values("Timestamp")
    start = as_utc(start)
    end = as_utc(end)
    readings = readings.loc[readings["Timestamp"].between(start, end)]

    return readings.reset_index(drop=True)


def as_utc(value: datetime) -> pd.Timestamp:
    # Convert a datetime to a UTC timestamp
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def empty_readings() -> pd.DataFrame:
    # Return an empty result with the columns expected by the app
    return pd.DataFrame(columns=["Timestamp", "Value"])
