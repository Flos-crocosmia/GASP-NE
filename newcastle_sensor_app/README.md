# Newcastle sensor explorer

A Shiny for Python app for exploring Newcastle-area air-quality sensors.

The initial version:

- reads the selected sensor names from `naming.csv`
- retrieves Urban Observatory coordinates and PM2.5 readings live
- displays UO Monitor, DEFRA and Locally managed sensors on an interactive map
- updates a side panel when a marker is clicked
- offers 24-hour, 7-day, and 30-day time windows and choice of custom time interval for PM2.5 readings
- downloads DEFRA/openair yearly files and normalises them to the same chart format
- retrieves available AURN coordinates from DEFRA metadata
- Trend line, 24 hour rolling mean, on time series.

## Run locally

Create and activate a virtual environment, then run:

```bash
python -m pip install -r requirements.txt
shiny run --reload app.py
```

Open the local address printed by Shiny (normally `http://127.0.0.1:8000`).

## DEFRA and local data

The seven DEFRA/local records are identified by their `code` values in
`naming.csv`. `data_sources.py` follows the openair file convention
`CODE_YYYY.RData`, reads it with `pyreadr`, and filters the result to the selected
period. Downloads are cached while the app process is running.

The standard AURN metadata file supplies coordinates where available. If a
locally managed site is absent from that file, add `latitude` and `longitude`
columns to `naming.csv`; the app treats those values as a fallback.

Recent DEFRA values can be provisional and later ratified. The app displays the
values supplied by the source and does not relabel them as ratified.

External requests use a ten-second timeout. The initial UO metadata request is
restricted to a Newcastle/Tyneside bounding box rather than requesting the
entire Urban Observatory catalogue, so a slow service cannot indefinitely hold
the page on its loading state.
