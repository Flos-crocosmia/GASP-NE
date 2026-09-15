from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from ipyleaflet import CircleMarker, Map, basemaps
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_plotly, render_widget

from data_sources import (
    get_defra_readings,
    get_sensor_metadata,
    get_uo_readings,
    load_sensor_registry,
)


APP_DIR = Path(__file__).resolve().parent
PM25_THRESHOLD = 10.0
DEMO_FORECAST_HOURS = 24
CIVIC_CENTRE_CODE = "NEWC"

registry = load_sensor_registry(APP_DIR / "naming.csv")

SENSOR_CHOICES = {
    row.sensor_name: row.display_name
    for row in registry.itertuples()
}

UO_MON_CHOICES = {
    row.sensor_name: row.display_name
    for row in registry.loc[registry["provider"] == "UO-Mon"].itertuples()
}

DEFAULT_SENSOR = next(iter(SENSOR_CHOICES), None)
DEFAULT_UO_MON = next(iter(UO_MON_CHOICES), None)


def clean_readings(readings: pd.DataFrame) -> pd.DataFrame:
    """Return readings in the common Timestamp/Value format."""
    if readings is None or readings.empty:
        return pd.DataFrame(columns=["Timestamp", "Value"])

    cleaned = readings[["Timestamp", "Value"]].copy()
    cleaned["Timestamp"] = pd.to_datetime(
        cleaned["Timestamp"],
        utc=True,
        errors="coerce",
    )
    cleaned["Value"] = pd.to_numeric(
        cleaned["Value"],
        errors="coerce",
    )

    return (
        cleaned
        .dropna(subset=["Timestamp", "Value"])
        .sort_values("Timestamp")
    )


def load_row_readings(
    row: pd.Series | dict,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    """Load one registry row from its appropriate source."""
    if row["provider"].startswith("UO-"):
        readings = get_uo_readings(
            sensor_name=row["sensor_name"],
            start=start,
            end=end,
            variable="PM2.5",
        )
    else:
        readings = get_defra_readings(
            site_code=row["code"],
            start=start,
            end=end,
            variable="PM2.5",
            source_type=row["type"],
        )

    return clean_readings(readings)


def empty_figure(message: str, height: int = 500) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
    )
    fig.update_layout(
        height=height,
        xaxis_visible=False,
        yaxis_visible=False,
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def haversine_distance_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    radius_km = 6371.0088
    phi1 = np.radians(lat1)
    phi2 = np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = (
        np.sin(delta_phi / 2) ** 2
        + np.cos(phi1)
        * np.cos(phi2)
        * np.sin(delta_lambda / 2) ** 2
    )

    return float(
        2 * radius_km * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    )


def pairwise_distance_matrix_km(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
) -> np.ndarray:
    lat1 = np.radians(latitudes[:, None])
    lat2 = np.radians(latitudes[None, :])
    delta_lat = lat2 - lat1
    delta_lon = np.radians(longitudes[None, :] - longitudes[:, None])

    a = (
        np.sin(delta_lat / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(delta_lon / 2) ** 2
    )

    return 2 * 6371.0088 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def find_logo(filename: str) -> Path:
    """Find a logo either beside app.py or in the conventional www folder."""
    candidates = [
        APP_DIR / filename,
        APP_DIR / "www" / filename,
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Could not find {filename}. Checked: "
        + ", ".join(str(path) for path in candidates)
    )


app_ui = ui.page_fluid(
    ui.tags.style(
        """
        body {
            background: #f4f6f8;
        }

        .app-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0.85rem 1.25rem;
            margin: 0 -12px 0.5rem -12px;
            border-bottom: 1px solid #d9dee3;
            background: #ffffff;
        }

        .app-heading h1 {
            margin: 0;
            color: #1f3b4d;
            font-size: 1.8rem;
            font-weight: 650;
        }

        .app-heading p {
            margin: 0.2rem 0 0 0;
            color: #66727c;
            font-size: 0.95rem;
        }

        .app-logos {
            display: flex;
            align-items: center;
            gap: 1.25rem;
            min-width: 340px;
            justify-content: flex-end;
        }

        .app-logos .shiny-image-output {
            display: inline-flex !important;
            align-items: center;
            justify-content: center;
        }

        .app-logos img {
            max-width: 100%;
            max-height: 55px;
            object-fit: contain;
        }

        .sensor-meta {
            line-height: 1.55;
            margin-bottom: 1rem;
        }

        .sensor-meta code {
            font-size: 0.82rem;
        }

        .status-note {
            color: #66727c;
            font-size: 0.9rem;
        }

        .summary-box {
            background: #f7f8f9;
            border: 1px solid #e3e7ea;
            border-radius: 7px;
            padding: 0.8rem;
            margin-bottom: 1rem;
            font-size: 0.9rem;
            line-height: 1.55;
        }

        .plot-note {
            color: #66727c;
            margin: 0.25rem 0 0.75rem 0;
            font-size: 0.9rem;
        }

        .tab-pane {
            padding-top: 0.75rem;
        }

        @media (max-width: 800px) {
            .app-header {
                align-items: flex-start;
                flex-direction: column;
                gap: 0.75rem;
            }

            .app-logos {
                min-width: 0;
                justify-content: flex-start;
            }
        }
        """
    ),

    ui.tags.header(
        ui.div(
            ui.h1("GASP-NE Air Quality Application"),
            ui.p(
                "Monitoring, validation and spatial analysis of PM2.5 "
                "across Newcastle upon Tyne."
            ),
            class_="app-heading",
        ),
        ui.div(
            ui.output_image(
                "ncl_logo",
                width="170px",
                height="55px",
                inline=True,
            ),
            ui.output_image(
                "epsrc_logo",
                width="170px",
                height="55px",
                inline=True,
            ),
            class_="app-logos",
        ),
        class_="app-header",
    ),

    ui.navset_tab(
        ui.nav_panel(
            "Sensor map",
            ui.card(
                ui.card_body(
                    ui.p(
                        "Select a sensor on the map to open its full temporal view. "
                        "Urban Observatory Monitor sites are green and DEFRA/local "
                        "sites are blue."
                    ),
                    output_widget("sensor_map", height="680px"),
                ),
                full_screen=True,
            ),
            value="map",
        ),

        ui.nav_panel(
            "Temporal",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_select(
                        "temporal_sensor",
                        "Sensor",
                        choices=SENSOR_CHOICES,
                        selected=DEFAULT_SENSOR,
                    ),
                    ui.h4(ui.output_text("panel_title")),
                    ui.output_ui("sensor_details"),
                    ui.input_select(
                        "period",
                        "Time period",
                        choices={
                            "24": "Last 24 hours",
                            "168": "Last 7 days",
                            "720": "Last 30 days",
                            "custom": "Custom interval",
                        },
                        selected="168",
                    ),
                    ui.panel_conditional(
                        "input.period === 'custom'",
                        ui.input_date_range(
                            "custom_dates",
                            "Custom date range",
                            start=(
                                datetime.now(timezone.utc) - timedelta(days=7)
                            ).date(),
                            end=datetime.now(timezone.utc).date(),
                        ),
                    ),
                    ui.input_checkbox_group(
                        "trend_lines",
                        "Plot options",
                        choices={
                            "rolling": "24-hour rolling mean",
                            "linear": "Linear trend",
                        },
                        selected=[],
                    ),
                    ui.output_ui("data_status"),
                    ui.output_ui("time_series_summary"),
                    width=340,
                    open="always",
                ),

                ui.navset_card_tab(
                    ui.nav_panel(
                        "Series and demonstration forecast",
                        output_widget("sensor_chart", height="650px"),
                        value="series",
                    ),
                    ui.nav_panel(
                        "Means and variation",
                        ui.div(
                            ui.input_select(
                                "structure_period",
                                "Analysis history",
                                choices={
                                    "30": "Last 30 days",
                                    "90": "Last 90 days",
                                    "365": "Last year",
                                },
                                selected="365",
                            ),
                            style="max-width: 240px; margin: 0.75rem;",
                        ),
                        ui.layout_columns(
                            ui.card(
                                ui.card_header(
                                    "Annual overview: daily mean and rolling variation"
                                ),
                                output_widget(
                                    "annual_variation_chart",
                                    height="520px",
                                ),
                            ),
                            ui.card(
                                ui.card_header(
                                    "Diurnal mean and variation"
                                ),
                                output_widget(
                                    "diurnal_variation_chart",
                                    height="520px",
                                ),
                            ),
                            col_widths=(7, 5),
                        ),
                        value="structure",
                    ),
                    id="temporal_tabs",
                ),
            ),
            value="temporal",
        ),

        ui.nav_panel(
            "Validation",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.h4("Monitor validation"),
                    ui.input_select(
                        "validation_uo_sensor",
                        "Urban Observatory Monitor",
                        choices=UO_MON_CHOICES,
                        selected=DEFAULT_UO_MON,
                    ),
                    ui.input_select(
                        "validation_period",
                        "Comparison period",
                        choices={
                            "7": "Past 7 days",
                            "30": "Past 30 days",
                            "90": "Past 90 days",
                        },
                        selected="7",
                    ),
                    ui.output_ui("nearest_validation_pair"),
                    ui.output_ui("validation_summary"),
                    width=340,
                    open="always",
                ),
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Hourly measurement comparison"),
                        output_widget("validation_scatter", height="540px"),
                    ),
                    ui.card(
                        ui.card_header("Aligned hourly time series"),
                        output_widget("validation_timeseries", height="540px"),
                    ),
                    col_widths=(6, 6),
                ),
            ),
            value="validation",
        ),

        ui.nav_panel(
            "Spatial GP",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.h4("Background/local decomposition"),
                    ui.input_select(
                        "gp_period",
                        "Aggregation period",
                        choices={
                            "1": "Past 24 hours",
                            "7": "Past 7 days",
                            "30": "Past 30 days",
                        },
                        selected="7",
                    ),
                    ui.input_slider(
                        "gp_length_scale",
                        "Spatial length scale (km)",
                        min=1,
                        max=30,
                        value=8,
                        step=1,
                    ),
                    ui.input_slider(
                        "gp_nugget",
                        "Nugget variance (fraction)",
                        min=0.01,
                        max=1.0,
                        value=0.15,
                        step=0.01,
                    ),
                    ui.p(
                        "The background at each site is predicted with that site "
                        "held out. Local contribution = observed mean − kriged "
                        "background.",
                        class_="plot-note",
                    ),
                    ui.output_ui("gp_status"),
                    ui.output_ui("civic_centre_validation"),
                    width=340,
                    open="always",
                ),
                ui.layout_columns(
                    ui.card(
                        ui.card_header("Spatial GP covariance matrix"),
                        output_widget("gp_covariance_chart", height="600px"),
                    ),
                    ui.card(
                        ui.card_header(
                            "Observed, background and local contributions"
                        ),
                        output_widget("gp_decomposition_chart", height="600px"),
                    ),
                    col_widths=(6, 6),
                ),
            ),
            value="spatial",
        ),

        id="main_tabs",
        selected="map",
    ),

    title="GASP-NE App",
)


def server(input, output, session):
    selected_sensor = reactive.value(DEFAULT_SENSOR)

    @render.image
    def ncl_logo():
        return {
            "src": str(find_logo("ncl_logo.png")),
            "height": "55px",
            "alt": "Newcastle University logo",
        }

    @render.image
    def epsrc_logo():
        return {
            "src": str(find_logo("epsrc_logo.jpg")),
            "height": "55px",
            "alt": "EPSRC logo",
        }

    @reactive.calc
    def sensors():
        result = get_sensor_metadata(registry)
        result["latitude"] = pd.to_numeric(result["latitude"], errors="coerce")
        result["longitude"] = pd.to_numeric(result["longitude"], errors="coerce")
        return result

    @reactive.effect
    @reactive.event(input.temporal_sensor)
    def _select_temporal_sensor():
        sensor_name = input.temporal_sensor()
        if sensor_name:
            selected_sensor.set(sensor_name)

    @render_widget
    def sensor_map():
        sensor_df = sensors()
        map_widget = Map(
            center=(54.9783, -1.6178),
            zoom=11,
            basemap=basemaps.OpenStreetMap.Mapnik,
            scroll_wheel_zoom=True,
        )

        colours = {
            "UO-Mon": "#1b9e77",
            "DEFRA/Local": "#386cb0",
        }

        visible_sensors = sensor_df.dropna(
            subset=["latitude", "longitude"]
        )

        for row in visible_sensors.itertuples():
            marker_colour = colours.get(row.provider, "#555555")
            marker = CircleMarker(
                location=(float(row.latitude), float(row.longitude)),
                radius=8,
                color=marker_colour,
                fill_color=marker_colour,
                fill_opacity=0.85,
                weight=2,
                title=row.display_name,
            )

            sensor_id = row.sensor_name

            def choose_sensor(_sensor_id=sensor_id, **_kwargs):
                selected_sensor.set(_sensor_id)
                ui.update_select(
                    "temporal_sensor",
                    selected=_sensor_id,
                    session=session,
                )
                ui.update_navset(
                    "main_tabs",
                    selected="temporal",
                    session=session,
                )

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
    def selected_interval():
        if input.period() == "custom":
            selected_dates = input.custom_dates()
            if (
                not selected_dates
                or len(selected_dates) != 2
                or selected_dates[0] is None
                or selected_dates[1] is None
            ):
                return None

            start_date, end_date = selected_dates
            start = datetime.combine(
                start_date,
                datetime.min.time(),
                tzinfo=timezone.utc,
            )
            end = datetime.combine(
                end_date + timedelta(days=1),
                datetime.min.time(),
                tzinfo=timezone.utc,
            )
        else:
            end = datetime.now(timezone.utc)
            start = end - timedelta(hours=int(input.period()))

        return start, end

    @reactive.calc
    def selected_data():
        row = selected_row()
        interval = selected_interval()
        if row is None or interval is None:
            return pd.DataFrame(columns=["Timestamp", "Value"])

        start, end = interval
        return load_row_readings(row, start, end)

    @reactive.calc
    def temporal_analysis_data():
        row = selected_row()
        if row is None:
            return pd.DataFrame(columns=["Timestamp", "Value"])

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=int(input.structure_period()))
        return load_row_readings(row, start, end)

    @render.text
    def panel_title():
        row = selected_row()
        return "Select a sensor" if row is None else row["display_name"]

    @render.ui
    def sensor_details():
        row = selected_row()
        if row is None:
            return ui.p("Select a sensor to view its details.", class_="status-note")

        coords = "Coordinates unavailable"
        if pd.notna(row["latitude"]) and pd.notna(row["longitude"]):
            coords = f'{row["latitude"]:.6f}, {row["longitude"]:.6f}'

        return ui.div(
            ui.div(ui.strong("Network: "), row["provider"]),
            ui.div(
                ui.strong("Sensor reference: "),
                ui.code(row["sensor_name"]),
            ),
            ui.div(ui.strong("Coordinates: "), coords),
            ui.div(
                ui.strong("Coordinate reference system: "),
                "WGS 84 (EPSG:4326)",
            ),
            class_="sensor-meta",
        )

    @render.ui
    def data_status():
        data = selected_data()
        if data.empty:
            return ui.p(
                "No readings were returned for this sensor and period.",
                class_="status-note",
            )
        return ui.p(f"{len(data):,} readings returned.", class_="status-note")

    @render.ui
    def time_series_summary():
        data = selected_data()
        if data.empty:
            return None

        values = data["Value"]
        exceedance_count = int((values >= PM25_THRESHOLD).sum())
        exceedance_percent = 100 * exceedance_count / len(values)

        trend_text = "Not enough data"
        if len(data) >= 2 and data["Timestamp"].nunique() >= 2:
            elapsed_days = (
                data["Timestamp"] - data["Timestamp"].iloc[0]
            ).dt.total_seconds() / 86400
            slope, _ = np.polyfit(elapsed_days, values, 1)
            trend_text = f"{slope:+.2f} µg/m³ per day"

        return ui.div(
            ui.div(ui.strong("Mean: "), f"{values.mean():.2f} µg/m³"),
            ui.div(ui.strong("Median: "), f"{values.median():.2f} µg/m³"),
            ui.div(ui.strong("Maximum: "), f"{values.max():.2f} µg/m³"),
            ui.div(
                ui.strong("Standard deviation: "),
                f"{values.std():.2f} µg/m³",
            ),
            ui.div(ui.strong("Linear trend: "), trend_text),
            ui.div(
                ui.strong(f"Readings ≥ {PM25_THRESHOLD:g} µg/m³: "),
                f"{exceedance_count:,} ({exceedance_percent:.1f}%)",
            ),
            class_="summary-box",
        )

    @render_plotly
    def sensor_chart():
        data = selected_data()
        row = selected_row()
        if row is None or data.empty:
            return empty_figure("No data to display", height=650)

        plot_data = data.copy()
        fig = px.line(
            plot_data,
            x="Timestamp",
            y="Value",
            labels={
                "Timestamp": "Time",
                "Value": "PM2.5 (µg/m³)",
            },
        )
        fig.update_traces(
            line_color="#1b9e77",
            line_width=1.5,
            name="Observed",
            showlegend=True,
        )

        fig.add_hline(
            y=PM25_THRESHOLD,
            line_dash="dot",
            line_color="#c2185b",
            line_width=2,
            annotation_text=f"{PM25_THRESHOLD:g} µg/m³ threshold",
            annotation_position="top left",
        )

        above_threshold = plot_data["Value"] >= PM25_THRESHOLD
        if above_threshold.any():
            differences = plot_data["Timestamp"].diff().dropna()
            normal_interval = (
                differences.median()
                if not differences.empty
                else pd.Timedelta(minutes=30)
            )
            groups = above_threshold.ne(above_threshold.shift()).cumsum()
            for _, section in plot_data.loc[above_threshold].groupby(
                groups.loc[above_threshold]
            ):
                fig.add_vrect(
                    x0=section["Timestamp"].iloc[0] - normal_interval / 2,
                    x1=section["Timestamp"].iloc[-1] + normal_interval / 2,
                    fillcolor="#f48fb1",
                    opacity=0.18,
                    line_width=0,
                    layer="below",
                )

        selected_trends = input.trend_lines() or []
        if "rolling" in selected_trends:
            rolling_data = (
                plot_data
                .set_index("Timestamp")["Value"]
                .rolling("24h", min_periods=2)
                .mean()
            )
            fig.add_scatter(
                x=rolling_data.index,
                y=rolling_data.values,
                mode="lines",
                name="24-hour rolling mean",
                line={"color": "#ff8c00", "width": 2.5},
            )

        if (
            "linear" in selected_trends
            and len(plot_data) >= 2
            and plot_data["Timestamp"].nunique() >= 2
        ):
            elapsed_days = (
                plot_data["Timestamp"] - plot_data["Timestamp"].iloc[0]
            ).dt.total_seconds() / 86400
            slope, intercept = np.polyfit(elapsed_days, plot_data["Value"], 1)
            fig.add_scatter(
                x=plot_data["Timestamp"],
                y=intercept + slope * elapsed_days,
                mode="lines",
                name=f"Linear trend ({slope:+.2f} µg/m³/day)",
                line={"color": "#6a3d9a", "width": 2, "dash": "dash"},
            )

        if len(plot_data) >= 2:
            differences = plot_data["Timestamp"].diff().dropna()
            differences = differences[differences > pd.Timedelta(0)]
            forecast_interval = (
                differences.median()
                if not differences.empty
                else pd.Timedelta(hours=1)
            )
            forecast_interval = max(
                pd.Timedelta(minutes=1),
                min(forecast_interval, pd.Timedelta(hours=6)),
            )
            forecast_steps = max(
                1,
                int(
                    pd.Timedelta(hours=DEMO_FORECAST_HOURS)
                    / forecast_interval
                ),
            )

            observed_differences = plot_data["Value"].diff().dropna()
            step_sigma = observed_differences.std()
            if pd.isna(step_sigma) or step_sigma <= 0:
                step_sigma = max(plot_data["Value"].std() * 0.05, 0.1)

            last_timestamp = plot_data["Timestamp"].iloc[-1]
            last_value = float(plot_data["Value"].iloc[-1])
            seed_text = f'{row["sensor_name"]}-{last_timestamp}'
            seed = int.from_bytes(
                hashlib.sha256(seed_text.encode("utf-8")).digest()[:8],
                byteorder="little",
            )
            random_generator = np.random.default_rng(seed)
            random_steps = random_generator.normal(
                loc=0.0,
                scale=step_sigma,
                size=(1000, forecast_steps),
            )
            simulated_paths = last_value + np.cumsum(random_steps, axis=1)
            simulated_paths = np.maximum(simulated_paths, 0.0)

            forecast_values = simulated_paths[0]
            lower_interval = np.percentile(simulated_paths, 2.5, axis=0)
            upper_interval = np.percentile(simulated_paths, 97.5, axis=0)
            forecast_times = pd.date_range(
                start=last_timestamp + forecast_interval,
                periods=forecast_steps,
                freq=forecast_interval,
            )

            connected_times = [last_timestamp, *forecast_times]
            connected_values = [last_value, *forecast_values]
            connected_lower = [last_value, *lower_interval]
            connected_upper = [last_value, *upper_interval]

            fig.add_scatter(
                x=connected_times,
                y=connected_lower,
                mode="lines",
                line={"width": 0},
                hoverinfo="skip",
                showlegend=False,
            )
            fig.add_scatter(
                x=connected_times,
                y=connected_upper,
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor="rgba(227, 26, 28, 0.18)",
                name="95% demo prediction interval",
            )
            fig.add_scatter(
                x=connected_times,
                y=connected_values,
                mode="lines",
                name="Demo projection",
                line={"color": "#e31a1c", "width": 2},
            )
            fig.add_vline(
                x=last_timestamp,
                line_color="#555555",
                line_width=1.5,
                line_dash="dot",
                annotation_text="Demo projection",
                annotation_position="top right",
            )

        fig.update_layout(
            height=650,
            hovermode="x unified",
            margin=dict(l=30, r=20, t=70, b=30),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="left",
                x=0,
            ),
        )
        return fig

    @render_plotly
    def annual_variation_chart():
        data = temporal_analysis_data()
        if data.empty:
            return empty_figure("No data available for the annual overview")

        daily = (
            data
            .set_index("Timestamp")["Value"]
            .resample("1D")
            .mean()
            .dropna()
        )
        if daily.empty:
            return empty_figure("Not enough data for daily means")

        rolling_mean = daily.rolling("30D", min_periods=3).mean()
        rolling_std = daily.rolling("30D", min_periods=3).std()
        lower = np.maximum(rolling_mean - rolling_std, 0.0)
        upper = rolling_mean + rolling_std

        fig = go.Figure()
        fig.add_scatter(
            x=daily.index,
            y=daily.values,
            mode="lines",
            name="Daily mean",
            line={"color": "#a7adb2", "width": 1},
        )
        fig.add_scatter(
            x=rolling_mean.index,
            y=lower,
            mode="lines",
            line={"width": 0},
            hoverinfo="skip",
            showlegend=False,
        )
        fig.add_scatter(
            x=rolling_mean.index,
            y=upper,
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(27, 158, 119, 0.18)",
            name="Rolling ±1 SD",
        )
        fig.add_scatter(
            x=rolling_mean.index,
            y=rolling_mean,
            mode="lines",
            name="30-day rolling mean",
            line={"color": "#1b9e77", "width": 2.5},
        )
        fig.update_layout(
            height=520,
            xaxis_title="Date",
            yaxis_title="PM2.5 (µg/m³)",
            hovermode="x unified",
            margin=dict(l=40, r=20, t=30, b=40),
        )
        return fig

    @render_plotly
    def diurnal_variation_chart():
        data = temporal_analysis_data()
        if data.empty:
            return empty_figure("No data available for the diurnal profile")

        hourly = (
            data
            .assign(hour=data["Timestamp"].dt.hour)
            .groupby("hour")["Value"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        hourly["std"] = hourly["std"].fillna(0.0)
        lower = np.maximum(hourly["mean"] - hourly["std"], 0.0)
        upper = hourly["mean"] + hourly["std"]

        fig = go.Figure()
        fig.add_scatter(
            x=hourly["hour"],
            y=lower,
            mode="lines",
            line={"width": 0},
            hoverinfo="skip",
            showlegend=False,
        )
        fig.add_scatter(
            x=hourly["hour"],
            y=upper,
            mode="lines",
            line={"width": 0},
            fill="tonexty",
            fillcolor="rgba(56, 108, 176, 0.18)",
            name="Hourly ±1 SD",
        )
        fig.add_scatter(
            x=hourly["hour"],
            y=hourly["mean"],
            mode="lines+markers",
            name="Hourly mean",
            line={"color": "#386cb0", "width": 2.5},
        )
        fig.update_layout(
            height=520,
            xaxis=dict(
                title="Hour of day (UTC)",
                tickmode="linear",
                dtick=2,
                range=[0, 23],
            ),
            yaxis_title="PM2.5 (µg/m³)",
            hovermode="x unified",
            margin=dict(l=40, r=20, t=30, b=40),
        )
        return fig

    @reactive.calc
    def validation_pair():
        uo_sensor_name = input.validation_uo_sensor()
        sensor_df = sensors()

        selected_matches = sensor_df.loc[
            sensor_df["sensor_name"] == uo_sensor_name
        ]
        references = sensor_df.loc[
            (sensor_df["provider"] == "DEFRA/Local")
            & sensor_df["latitude"].notna()
            & sensor_df["longitude"].notna()
        ].copy()

        if selected_matches.empty or references.empty:
            return None

        selected = selected_matches.iloc[0]
        if pd.isna(selected["latitude"]) or pd.isna(selected["longitude"]):
            return None

        references["distance_km"] = references.apply(
            lambda row: haversine_distance_km(
                float(selected["latitude"]),
                float(selected["longitude"]),
                float(row["latitude"]),
                float(row["longitude"]),
            ),
            axis=1,
        )
        nearest = references.sort_values("distance_km").iloc[0]
        return selected, nearest

    @reactive.calc
    def validation_data():
        pair = validation_pair()
        if pair is None:
            return pd.DataFrame(), None

        uo_row, reference_row = pair
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=int(input.validation_period()))

        uo_data = load_row_readings(uo_row, start, end)
        reference_data = load_row_readings(reference_row, start, end)
        if uo_data.empty or reference_data.empty:
            return pd.DataFrame(), pair

        uo_hourly = (
            uo_data
            .set_index("Timestamp")["Value"]
            .resample("1h")
            .mean()
            .rename("UO-Mon")
        )
        reference_hourly = (
            reference_data
            .set_index("Timestamp")["Value"]
            .resample("1h")
            .mean()
            .rename("DEFRA/Local")
        )
        comparison = (
            pd.concat([reference_hourly, uo_hourly], axis=1)
            .dropna()
            .reset_index()
        )
        return comparison, pair

    @render.ui
    def nearest_validation_pair():
        pair = validation_pair()
        if pair is None:
            return ui.p("No valid reference station was found.", class_="status-note")

        uo_row, reference_row = pair
        return ui.div(
            ui.strong("Automatically selected pair"),
            ui.div(uo_row["display_name"]),
            ui.div("↔"),
            ui.div(reference_row["display_name"]),
            ui.div(f'Distance: {reference_row["distance_km"]:.3f} km'),
            class_="summary-box",
        )

    @render.ui
    def validation_summary():
        comparison, _ = validation_data()
        if comparison.empty:
            return ui.p(
                "The pair has no overlapping valid hourly readings in this period.",
                class_="status-note",
            )

        correlation = comparison[["DEFRA/Local", "UO-Mon"]].corr().iloc[0, 1]
        difference = comparison["UO-Mon"] - comparison["DEFRA/Local"]
        rmse = np.sqrt(np.mean(difference**2))

        return ui.div(
            ui.div(
                ui.strong("Paired hours: "),
                f"{len(comparison):,}",
            ),
            ui.div(
                ui.strong("Pearson correlation: "),
                f"{correlation:.3f}",
            ),
            ui.div(
                ui.strong("Mean UO−reference difference: "),
                f"{difference.mean():+.2f} µg/m³",
            ),
            ui.div(ui.strong("RMSE: "), f"{rmse:.2f} µg/m³"),
            class_="summary-box",
        )

    @render_plotly
    def validation_scatter():
        comparison, pair = validation_data()
        if comparison.empty or pair is None:
            return empty_figure("No overlapping validation data", height=540)

        uo_row, reference_row = pair
        fig = px.scatter(
            comparison,
            x="DEFRA/Local",
            y="UO-Mon",
            hover_data=["Timestamp"],
            labels={
                "DEFRA/Local": f'{reference_row["display_name"]} (µg/m³)',
                "UO-Mon": f'{uo_row["display_name"]} (µg/m³)',
            },
        )
        fig.update_traces(
            marker={"color": "#386cb0", "size": 8, "opacity": 0.65},
            name="Hourly observations",
        )

        x_values = comparison["DEFRA/Local"].to_numpy()
        y_values = comparison["UO-Mon"].to_numpy()
        if len(comparison) >= 2 and np.ptp(x_values) > 0:
            slope, intercept = np.polyfit(x_values, y_values, 1)
            fitted_x = np.linspace(x_values.min(), x_values.max(), 100)
            fig.add_scatter(
                x=fitted_x,
                y=intercept + slope * fitted_x,
                mode="lines",
                name="Linear fit",
                line={"color": "#e31a1c", "width": 2, "dash": "dash"},
            )

        overall_min = min(x_values.min(), y_values.min())
        overall_max = max(x_values.max(), y_values.max())
        fig.add_scatter(
            x=[overall_min, overall_max],
            y=[overall_min, overall_max],
            mode="lines",
            name="Perfect agreement",
            line={"color": "#555555", "width": 1.5, "dash": "dot"},
        )
        fig.update_layout(
            height=540,
            margin=dict(l=40, r=20, t=40, b=50),
            legend=dict(orientation="h", y=1.05, x=0),
        )
        return fig

    @render_plotly
    def validation_timeseries():
        comparison, pair = validation_data()
        if comparison.empty or pair is None:
            return empty_figure("No overlapping validation data", height=540)

        uo_row, reference_row = pair
        fig = go.Figure()
        fig.add_scatter(
            x=comparison["Timestamp"],
            y=comparison["DEFRA/Local"],
            mode="lines",
            name=reference_row["display_name"],
            line={"color": "#386cb0", "width": 1.8},
        )
        fig.add_scatter(
            x=comparison["Timestamp"],
            y=comparison["UO-Mon"],
            mode="lines",
            name=uo_row["display_name"],
            line={"color": "#1b9e77", "width": 1.8},
        )
        fig.update_layout(
            height=540,
            xaxis_title="Time",
            yaxis_title="PM2.5 (µg/m³)",
            hovermode="x unified",
            margin=dict(l=40, r=20, t=40, b=50),
            legend=dict(orientation="h", y=1.05, x=0),
        )
        return fig

    @reactive.calc
    def gp_sensor_means():
        sensor_df = sensors().loc[
            sensors()["provider"].isin(["UO-Mon", "DEFRA/Local"])
            & sensors()["latitude"].notna()
            & sensors()["longitude"].notna()
        ].copy()

        if sensor_df.empty:
            return pd.DataFrame()

        end = datetime.now(timezone.utc)
        start = end - timedelta(days=int(input.gp_period()))

        def fetch_mean(row_dict: dict):
            try:
                readings = load_row_readings(row_dict, start, end)
                if readings.empty:
                    return None
                return {
                    **row_dict,
                    "observed_mean": float(readings["Value"].mean()),
                    "observation_count": int(len(readings)),
                }
            except Exception as exc:
                print(
                    f'Could not include {row_dict["display_name"]} in GP: {exc}'
                )
                return None

        rows = sensor_df.to_dict("records")
        records = []
        workers = min(6, len(rows))

        with ui.Progress(min=0, max=len(rows)) as progress:
            progress.set(message="Loading sensor means for spatial GP")
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(fetch_mean, row) for row in rows]
                for count, future in enumerate(as_completed(futures), start=1):
                    result = future.result()
                    if result is not None:
                        records.append(result)
                    progress.set(count, message="Loading sensor means for spatial GP")

        if not records:
            return pd.DataFrame()

        return pd.DataFrame(records).sort_values("display_name").reset_index(drop=True)

    @reactive.calc
    def gp_results():
        data = gp_sensor_means()
        if len(data) < 3:
            return None

        latitudes = data["latitude"].to_numpy(dtype=float)
        longitudes = data["longitude"].to_numpy(dtype=float)
        observations = data["observed_mean"].to_numpy(dtype=float)
        distances = pairwise_distance_matrix_km(latitudes, longitudes)

        length_scale = float(input.gp_length_scale())
        amplitude = max(float(np.var(observations, ddof=1)), 0.25)
        signal_covariance = amplitude * np.exp(
            -0.5 * (distances / length_scale) ** 2
        )
        nugget_variance = max(float(input.gp_nugget()) * amplitude, 1e-6)
        covariance = signal_covariance + nugget_variance * np.eye(len(data))

        backgrounds = np.empty(len(data))
        prediction_sds = np.empty(len(data))

        for index in range(len(data)):
            keep = np.arange(len(data)) != index
            training_values = observations[keep]
            training_mean = float(training_values.mean())
            training_covariance = covariance[np.ix_(keep, keep)]
            cross_covariance = signal_covariance[index, keep]

            try:
                weights = np.linalg.solve(
                    training_covariance,
                    training_values - training_mean,
                )
                covariance_solution = np.linalg.solve(
                    training_covariance,
                    cross_covariance,
                )
            except np.linalg.LinAlgError:
                stable_covariance = (
                    training_covariance + 1e-8 * np.eye(len(training_values))
                )
                weights = np.linalg.solve(
                    stable_covariance,
                    training_values - training_mean,
                )
                covariance_solution = np.linalg.solve(
                    stable_covariance,
                    cross_covariance,
                )

            backgrounds[index] = max(
                training_mean + cross_covariance @ weights,
                0.0,
            )
            latent_variance = max(
                signal_covariance[index, index]
                - cross_covariance @ covariance_solution,
                0.0,
            )
            prediction_sds[index] = np.sqrt(latent_variance + nugget_variance)

        result_data = data.copy()
        result_data["background"] = backgrounds
        result_data["local_contribution"] = observations - backgrounds
        result_data["prediction_sd"] = prediction_sds

        return {
            "data": result_data,
            "covariance": covariance,
            "distances": distances,
            "amplitude": amplitude,
            "nugget_variance": nugget_variance,
        }

    @render.ui
    def gp_status():
        result = gp_results()
        if result is None:
            return ui.p(
                "At least three sensors with current data are required.",
                class_="status-note",
            )
        return ui.p(
            f'{len(result["data"])} sensors included.',
            class_="status-note",
        )

    @render.ui
    def civic_centre_validation():
        result = gp_results()
        if result is None:
            return None

        civic = result["data"].loc[result["data"]["code"] == CIVIC_CENTRE_CODE]
        if civic.empty:
            return ui.p(
                "Newcastle Centre (NEWC) was not available for validation.",
                class_="status-note",
            )

        row = civic.iloc[0]
        return ui.div(
            ui.strong("Civic Centre hold-out validation"),
            ui.div(f'Observed mean: {row["observed_mean"]:.2f} µg/m³'),
            ui.div(f'Kriged background: {row["background"]:.2f} µg/m³'),
            ui.div(
                f'Local contribution: {row["local_contribution"]:+.2f} µg/m³'
            ),
            ui.div(f'Prediction SD: {row["prediction_sd"]:.2f} µg/m³'),
            class_="summary-box",
        )

    @render_plotly
    def gp_covariance_chart():
        result = gp_results()
        if result is None:
            return empty_figure("Not enough data for a covariance matrix", height=600)

        labels = result["data"]["display_name"].tolist()
        fig = go.Figure(
            data=go.Heatmap(
                z=result["covariance"],
                x=labels,
                y=labels,
                colorscale="Viridis",
                colorbar={"title": "Covariance"},
                hovertemplate=(
                    "%{y}<br>%{x}<br>Covariance: %{z:.3f}<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            height=600,
            xaxis={"tickangle": -45},
            margin=dict(l=170, r=30, t=30, b=170),
        )
        return fig

    @render_plotly
    def gp_decomposition_chart():
        result = gp_results()
        if result is None:
            return empty_figure("Not enough data for GP decomposition", height=600)

        data = result["data"].sort_values("background")
        fig = go.Figure()
        fig.add_bar(
            x=data["display_name"],
            y=data["local_contribution"],
            name="Local contribution",
            marker_color="#e78ac3",
            opacity=0.75,
        )
        fig.add_scatter(
            x=data["display_name"],
            y=data["background"],
            mode="lines+markers",
            name="Kriged background",
            line={"color": "#386cb0", "width": 2},
        )
        fig.add_scatter(
            x=data["display_name"],
            y=data["observed_mean"],
            mode="markers",
            name="Observed mean",
            marker={"color": "#1b9e77", "size": 9, "symbol": "diamond"},
        )
        fig.add_hline(y=0, line_color="#777777", line_width=1)
        fig.update_layout(
            height=600,
            xaxis={"title": "Sensor", "tickangle": -45},
            yaxis_title="PM2.5 (µg/m³)",
            hovermode="x unified",
            margin=dict(l=50, r=20, t=30, b=170),
            legend=dict(orientation="h", y=1.06, x=0),
        )
        return fig


app = App(app_ui, server)
