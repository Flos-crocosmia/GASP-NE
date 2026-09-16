# Newcastle sensor explorer

A Shiny for Python app for exploring Newcastle-area air-quality sensors.

The initial version:

- Reads the selected sensor names from `naming.csv`
- Retrieves Urban Observatory coordinates and PM2.5 readings live
- Retrieves DEFRA and Locally managed automatic monitor PM2.5 readings (coords are in `naming.csv`)
- Displays UO Monitor, DEFRA and Locally managed sensors on an interactive map
- 2 tabs for a side panel
- Explorer side panel updates when a marker is clicked
    - Offers 24-hour, 7-day, and 30-day time windows and choice of custom time interval for PM2.5 readings and gives summaries.
    - Downloads DEFRA/openair yearly files and normalises them to the same chart format
    - Popup for currently chosen timeseries plot with then options for trend line, 24 hour rolling mean.
    - Demonstrative prediction on timeseries (just a random walk for now and is getting replaced with kriging interpolation and eventually a full spatial GP)
- Validation side panel allows choice in UO-Mon sensor and returns a scatter with it's closest DEFRA/Local for validation
    - Options for 7, 30 and 90 day windows and gives summaries.

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
