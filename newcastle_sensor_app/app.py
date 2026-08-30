from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import plotly.express as px
from ipyleaflet import CircleMarker, Map, basemaps
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_plotly, render_widget

from data_sources import (
    get_defra_readings,
    get_sensor_metadata,
    get_uo_readings,
    load_sensor_registry,
)


APP_DIR = Path(__file__).parent
registry = load_sensor_registry(APP_DIR / "naming.csv")

app_ui = ui.page_fillable(
    ui.tags.style(
        """
        .sensor-meta {line-height: 1.55; margin-bottom: 1rem;}
        .sensor-meta code {font-size: 0.82rem;}
        .status-note {color: #666; font-size: 0.9rem;}
        """
    ),
    ui.layout_sidebar(
        ui.sidebar(
            ui.h3(ui.output_text("panel_title")),
            ui.output_ui("sensor_details"),
            ui.input_select(
                "period",
                "Time period",
                choices={"24": "Last 24 hours", "168": "Last 7 days", "720": "Last 30 days"},
                selected="168",
            ),
            ui.input_select(
                "variable",
                "Measurement",
                choices={"PM2.5": "PM2.5"},
                selected="PM2.5",
            ),
            ui.output_ui("data_status"),
            output_widget("sensor_chart"),
            width=430,
            open="always",
        ),
        ui.card(
            ui.card_header("Newcastle air-quality sensors"),
            output_widget("sensor_map"),
            full_screen=True,
        ),
    ),
    title="Newcastle sensor explorer",
)


def server(input, output, session):
    selected_sensor = reactive.value(None)

    @reactive.calc
    def sensors():
        return get_sensor_metadata(registry)

    @render_widget
    def sensor_map():
        sensor_df = sensors()
        map_widget = Map(
            center=(54.9783, -1.6178),
            zoom=12,
            basemap=basemaps.OpenStreetMap.Mapnik,
            scroll_wheel_zoom=True,
        )

        colours = {"UO-Mesh": "#d95f02", "UO-Mon": "#1b9e77", "DEFRA/Local": "#386cb0"}

        for row in sensor_df.dropna(subset=["latitude", "longitude"]).itertuples():
            marker = CircleMarker(
                location=(row.latitude, row.longitude),
                radius=8,
                color=colours.get(row.provider, "#555555"),
                fill_color=colours.get(row.provider, "#555555"),
                fill_opacity=0.85,
                weight=2,
                title=row.display_name,
            )

            sensor_id = row.sensor_name

            def choose_sensor(_sensor_id=sensor_id, **_kwargs):
                selected_sensor.set(_sensor_id)

            marker.on_click(choose_sensor)
            map_widget.add(marker)

        return map_widget

    @reactive.calc
    def selected_row():
        sensor_name = selected_sensor.get()
        if sensor_name is None:
            return None
        matches = sensors().loc[sensors()["sensor_name"] == sensor_name]
        return None if matches.empty else matches.iloc[0]

    @reactive.calc
    def selected_data():
        row = selected_row()
        if row is None:
            return pd.DataFrame()

        hours = int(input.period())
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=hours)
        if row["provider"].startswith("UO-"):
            return get_uo_readings(
                sensor_name=row["sensor_name"],
                start=start,
                end=end,
                variable=input.variable(),
            )
        return get_defra_readings(
            site_code=row["code"],
            start=start,
            end=end,
            variable=input.variable(),
            source_type=row["type"],
        )

    @render.text
    def panel_title():
        row = selected_row()
        return "Select a sensor" if row is None else row["display_name"]

    @render.ui
    def sensor_details():
        row = selected_row()
        if row is None:
            return ui.p("Click a marker to view its details and measurements.", class_="status-note")

        coords = "Coordinates unavailable"
        if pd.notna(row["latitude"]) and pd.notna(row["longitude"]):
            coords = f'{row["latitude"]:.6f}, {row["longitude"]:.6f}'

        return ui.div(
            ui.div(ui.strong("Network: "), row["provider"]),
            ui.div(ui.strong("Sensor reference: "), ui.code(row["sensor_name"])),
            ui.div(ui.strong("Coordinates: "), coords),
            ui.div(ui.strong("Coordinate reference system: "), "WGS 84 (EPSG:4326)"),
            class_="sensor-meta",
        )

    @render.ui
    def data_status():
        row = selected_row()
        if row is None:
            return None
        data = selected_data()
        if data.empty:
            return ui.p(
                "No readings were returned for this sensor and period. Some locally managed site codes may need a separate feed.",
                class_="status-note",
            )
        return ui.p(f"{len(data):,} readings returned.", class_="status-note")

    @render_plotly
    def sensor_chart():
        data = selected_data()
        row = selected_row()
        if row is None or data.empty:
            fig = px.line()
            fig.update_layout(
                annotations=[dict(text="No data to display", x=0.5, y=0.5, showarrow=False)],
                xaxis_visible=False,
                yaxis_visible=False,
                margin=dict(l=10, r=10, t=20, b=10),
            )
            return fig

        fig = px.line(data, x="Timestamp", y="Value", labels={"Value": input.variable()})
        fig.update_traces(line_color="#1b9e77", line_width=1.5)
        fig.update_layout(
            title=f'{input.variable()} — {row["display_name"]}',
            hovermode="x unified",
            margin=dict(l=20, r=10, t=50, b=20),
        )
        return fig


app = App(app_ui, server)
