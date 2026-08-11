from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
from branca.colormap import linear
from ipyleaflet import (
    CircleMarker,
    DivIcon,
    FullScreenControl,
    LayersControl,
    Map,
    Marker,
    Popup,
    ScaleControl,
    WidgetControl,
    basemaps,
)
from ipywidgets import HTML
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_widget
import pyreader

# -----------------------------------------------------------------------------
# Demo data
# Replace load_sensor_data() with the two API calls when they are ready.
# The rest of the app only expects these columns:
# sensor_id, sensor_name, network, timestamp, latitude, longitude, value
# -----------------------------------------------------------------------------
SENSORS = pd.DataFrame(
    [
        ("UO-01", "Civic Centre", "Urban Observatory", 54.9783, -1.6178),
        ("UO-02", "Haymarket", "Urban Observatory", 54.9775, -1.6132),
        ("UO-03", "Ouseburn", "Urban Observatory", 54.9745, -1.5834),
        ("UO-04", "Jesmond", "Urban Observatory", 54.9914, -1.6077),
        ("UO-05", "Fenham", "Urban Observatory", 54.9853, -1.6540),
        ("UO-06", "Byker", "Urban Observatory", 54.9779, -1.5741),
        ("REF-01", "Newcastle Centre", "Council / Reference", 54.9747, -1.6108),
        ("REF-02", "Cradlewell", "Council / Reference", 54.9910, -1.5955),
        ("REF-03", "Gosforth", "Council / Reference", 55.0068, -1.6207),
        ("REF-04", "Elswick", "Council / Reference", 54.9661, -1.6404),
    ],
    columns=["sensor_id", "sensor_name", "network", "latitude", "longitude"],
)


def load_sensor_data() -> pd.DataFrame:
    """Return deterministic hourly demo readings for the previous seven days."""
    end = pd.Timestamp.now().floor("h")
    hours = pd.date_range(end=end, periods=7 * 24, freq="h")
    rows = []
    rng = np.random.default_rng(2026)

    for sensor_number, sensor in SENSORS.iterrows():
        sensor_offset = rng.normal(0, 1.8)
        for timestamp in hours:
            morning_peak = 5.5 * np.exp(-((timestamp.hour - 8) / 2.8) ** 2)
            evening_peak = 4.0 * np.exp(-((timestamp.hour - 18) / 3.2) ** 2)
            daily_cycle = 1.5 * np.sin(2 * np.pi * timestamp.dayofyear / 7)
            value = 7.0 + morning_peak + evening_peak + daily_cycle + sensor_offset
            value += rng.normal(0, 0.8)
            rows.append((*sensor, timestamp, max(0, value)))

    return pd.DataFrame(
        rows,
        columns=[
            "sensor_id", "sensor_name", "network", "latitude", "longitude",
            "timestamp", "value",
        ],
    )


SENSOR_DATA = load_sensor_data()
AVAILABLE_DATES = sorted(SENSOR_DATA["timestamp"].dt.date.unique())
NETWORKS = sorted(SENSOR_DATA["network"].unique())


COLOUR_SCALE = linear.YlOrRd_09.scale(0, 25)
COLOUR_SCALE.caption = "PM2.5 concentration (µg/m³)"


app_ui = ui.page_fluid(
    ui.tags.style(
        """
        :root {
            --navy: #18324a;
            --blue: #2d6f91;
            --pale: #f3f6f8;
            --ink: #1f2b33;
        }
        body { background: var(--pale); color: var(--ink); }
        .app-header {
            color: white; background: linear-gradient(110deg, #18324a, #2d6f91);
            padding: 1rem 1.6rem; margin: 0 -12px 1rem -12px;
            box-shadow: 0 2px 8px rgba(0,0,0,.15);
        }
        .app-header h2 { margin: 0; font-weight: 650; }
        .app-header p { margin: .25rem 0 0; opacity: .88; }
        .control-card, .metric-card, .info-card {
            background: white; border: 1px solid #dce4e8; border-radius: 10px;
            box-shadow: 0 2px 8px rgba(20,45,60,.07); padding: 1rem;
        }
        .control-card { height: calc(100vh - 155px); overflow-y: auto; }
        .map-card {
            background: white; border: 1px solid #dce4e8; border-radius: 10px;
            overflow: hidden; box-shadow: 0 2px 8px rgba(20,45,60,.07);
        }
        .metric-row { display: flex; gap: .7rem; margin-bottom: .8rem; }
        .metric-card { flex: 1; padding: .65rem .85rem; }
        .metric-label { color: #687985; font-size: .78rem; text-transform: uppercase; }
        .metric-value { color: var(--navy); font-size: 1.35rem; font-weight: 650; }
        .model-note {
            border-left: 4px solid #66a3bd; background: #edf6f9;
            padding: .7rem .8rem; margin-top: 1rem; font-size: .88rem;
        }
        .form-group { margin-bottom: 1rem; }
        """
    ),
    ui.div(
        ui.h2("Newcastle Air Quality Explorer"),
        ui.p("Interactive hourly sensor observations across Newcastle upon Tyne"),
        class_="app-header",
    ),
    ui.layout_columns(
        ui.div(
            ui.h4("Map controls"),
            ui.input_checkbox_group(
                "networks", "Sensor networks", NETWORKS, selected=NETWORKS
            ),
            ui.input_date(
                "selected_date", "Date", value=max(AVAILABLE_DATES),
                min=min(AVAILABLE_DATES), max=max(AVAILABLE_DATES),
            ),
            ui.input_slider("selected_hour", "Hour", 0, 23, 12, step=1),
            ui.output_text("selected_timestamp"),
            ui.hr(),
            ui.input_checkbox("show_labels", "Show sensor names", False),
            ui.div(
                ui.strong("Spatial prediction surface"), ui.br(),
                "The hourly Gaussian-process prediction layer will appear here in the next phase.",
                class_="model-note",
            ),
            class_="control-card",
        ),
        ui.div(
            ui.div(
                ui.div(
                    ui.span("Visible sensors", class_="metric-label"),
                    ui.div(ui.output_text("sensor_count"), class_="metric-value"),
                    class_="metric-card",
                ),
                ui.div(
                    ui.span("Mean PM2.5", class_="metric-label"),
                    ui.div(ui.output_text("mean_value"), class_="metric-value"),
                    class_="metric-card",
                ),
                ui.div(
                    ui.span("Highest reading", class_="metric-label"),
                    ui.div(ui.output_text("max_value"), class_="metric-value"),
                    class_="metric-card",
                ),
                class_="metric-row",
            ),
            ui.div(output_widget("sensor_map", height="calc(100vh - 245px)"), class_="map-card"),
        ),
        col_widths=(3, 9),
    ),
)


def server(input, output, session):
    @reactive.calc
    def selected_data() -> pd.DataFrame:
        chosen_networks = input.networks() or []
        chosen_date = pd.Timestamp(input.selected_date()).date()
        chosen_hour = int(input.selected_hour())
        selected = SENSOR_DATA[
            SENSOR_DATA["network"].isin(chosen_networks)
            & (SENSOR_DATA["timestamp"].dt.date == chosen_date)
            & (SENSOR_DATA["timestamp"].dt.hour == chosen_hour)
        ].copy()
        return selected

    @render.text
    def selected_timestamp():
        chosen = datetime.combine(input.selected_date(), datetime.min.time())
        chosen += timedelta(hours=int(input.selected_hour()))
        return chosen.strftime("Showing: %A %d %B %Y, %H:00")

    @render.text
    def sensor_count():
        return str(len(selected_data()))

    @render.text
    def mean_value():
        data = selected_data()
        return "—" if data.empty else f"{data['value'].mean():.1f} µg/m³"

    @render.text
    def max_value():
        data = selected_data()
        return "—" if data.empty else f"{data['value'].max():.1f} µg/m³"

    @render_widget
    def sensor_map():
        map_widget = Map(
            center=(54.9783, -1.6178),
            zoom=12,
            basemap=basemaps.CartoDB.Positron,
            scroll_wheel_zoom=True,
        )
        map_widget.add(ScaleControl(position="bottomleft"))
        map_widget.add(FullScreenControl())
        map_widget.add(LayersControl(position="topright"))

        data = selected_data()
        for _, sensor in data.iterrows():
            colour = COLOUR_SCALE(float(sensor["value"]))
            marker = CircleMarker(
                location=(sensor["latitude"], sensor["longitude"]),
                radius=10 if sensor["network"] == "Council / Reference" else 8,
                color="#ffffff",
                weight=2,
                fill_color=colour,
                fill_opacity=0.95,
            )
            popup_html = HTML(
                value=f"""
                <div style='min-width:190px'>
                  <strong>{sensor['sensor_name']}</strong><br>
                  <span style='color:#60727e'>{sensor['network']}</span><hr style='margin:.4rem 0'>
                  <strong>{sensor['value']:.1f} µg/m³</strong> PM2.5<br>
                  <small>{sensor['timestamp']:%d %b %Y, %H:%M}</small>
                </div>
                """
            )
            marker.popup = Popup(child=popup_html, close_button=True)
            map_widget.add(marker)

            if input.show_labels():
                label = Marker(
                    location=(sensor["latitude"], sensor["longitude"]),
                    icon=DivIcon(
                        html=(
                            "<div style='white-space:nowrap;background:rgba(255,255,255,.88);"
                            "padding:2px 5px;border-radius:3px;font-size:11px;"
                            "box-shadow:0 1px 3px rgba(0,0,0,.25);transform:translate(10px,-8px)'>"
                            f"{sensor['sensor_name']}</div>"
                        ),
                        icon_size=(0, 0),
                    ),
                    draggable=False,
                )
                map_widget.add(label)

        legend = HTML(
            value="""
            <div style='background:white;padding:7px 9px;border-radius:5px;
                        box-shadow:0 1px 5px rgba(0,0,0,.3);font-size:11px'>
              <strong>PM2.5 (µg/m³)</strong><br>
              <div style='width:150px;height:10px;margin:4px 0;
                          background:linear-gradient(to right,#ffffcc,#fed976,#fd8d3c,#e31a1c,#800026)'></div>
              <span>0</span><span style='float:right'>25+</span>
            </div>
            """
        )
        map_widget.add(WidgetControl(widget=legend, position="bottomright"))
        return map_widget


app = App(app_ui, server)