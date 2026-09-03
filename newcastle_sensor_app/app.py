from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import hashlib

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
PM25_THRESHOLD = 10.0
DEMO_FORECAST_HOURS = 24

registry = load_sensor_registry(APP_DIR / "naming.csv")

app_ui = ui.page_fillable(
    ui.tags.style(
        """
        .sensor-meta {line-height: 1.55; margin-bottom: 1rem;}
        .sensor-meta code {font-size: 0.82rem;}
        .status-note {color: #666; font-size: 0.9rem;}
        .summary-box {background-color: #f7f7f7;
                        border-radius: 6px;
                        padding: 0.75rem;
                        margin-bottom: 1rem;
                        font-size: 0.9rem;}
        .summary-box h5 {margin-top: 0;
                        margin-bottom: 0.5rem;}
        """
        ),

    ui.layout_sidebar(
        ui.sidebar(
            ui.h3(ui.output_text("panel_title")),
            ui.output_ui("sensor_details"),
            ui.input_select("period",
                            "Time period",
                            choices={
                                "24": "Last 24 hours",
                                "168": "Last 7 days",
                                "720": "Last 30 days",
                                "custom": "Custom interval",
                            },
                            selected="168",),
            ui.panel_conditional("input.period === 'custom'",
                ui.input_date_range("custom_dates", "Custom date range",
                    start=(datetime.now(timezone.utc) - timedelta(days=7)).date(),
                    end=datetime.now(timezone.utc).date(),
                    ),),
            ui.input_select("variable", "Measurement",
                choices={"PM2.5": "PM2.5",},
                selected="PM2.5",
            ),
            ui.input_checkbox_group("trend_lines", "Chart options",
                choices={"rolling": "24-hour rolling mean",
                        "linear": "Linear trend",},
                selected=[],
            ),
            ui.output_ui("data_status"),
            ui.output_ui("time_series_summary"),
            output_widget("sensor_chart"), width=430, open="always",
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
        map_widget = Map(center=(54.9783, -1.6178), zoom=12,
            basemap=(basemaps.OpenStreetMap.Mapnik),
            scroll_wheel_zoom=True,
        )
        colours = {"UO-Mesh": "#d95f02",
                    "UO-Mon": "#1b9e77",
                    "DEFRA/Local": "#386cb0",}
        visible_sensors = sensor_df.dropna(subset=["latitude","longitude",])

        for row in visible_sensors.itertuples():
            marker_colour = colours.get(row.provider, "#555555",)
            marker = CircleMarker(location=(float(row.latitude),
                                            float(row.longitude),),
                radius=8,
                color=marker_colour,
                fill_color=marker_colour,
                fill_opacity=0.85,
                weight=2,
                title=row.display_name,
            )

            sensor_id = row.sensor_name
            def choose_sensor(_sensor_id=sensor_id,**_kwargs,):
                selected_sensor.set(_sensor_id)

            marker.on_click(choose_sensor)
            map_widget.add(marker)

        return map_widget

    @reactive.calc
    def selected_row():
        sensor_name = (selected_sensor.get())
        if sensor_name is None:
            return None
        sensor_df = sensors()
        matches = sensor_df.loc[sensor_df["sensor_name"]== sensor_name]
        if matches.empty:
            return None
        
        return matches.iloc[0]

    @reactive.calc
    def selected_interval():
        if input.period() == "custom":
            selected_dates = (input.custom_dates())
            if (not selected_dates or len(selected_dates) != 2 or selected_dates[0] is None or selected_dates[1] is None):
                return None

            start_date, end_date = (selected_dates)
            start = datetime.combine(start_date,
                                    datetime.min.time(),
                                    tzinfo=timezone.utc,)
            # Include the whole final day.
            end = datetime.combine(end_date + timedelta(days=1),
                                    datetime.min.time(),
                                    tzinfo=timezone.utc,)
        else:
            hours = int(input.period())
            end = datetime.now(timezone.utc)
            start = end - timedelta(hours=hours)

        return start, end

    @reactive.calc
    def selected_data():
        row = selected_row()
        interval = selected_interval()
        if row is None or interval is None:
            return pd.DataFrame()

        start, end = interval
        if row["provider"].startswith("UO-"):
            readings = get_uo_readings(sensor_name=(row["sensor_name"]),
                                       start=start,
                                       end=end,
                                       variable=input.variable(),)
        else:
            readings = get_defra_readings(site_code=row["code"],
                                          start=start,
                                          end=end,
                                          variable=input.variable(),
                                          source_type=row["type"],)
        if readings.empty:
            return readings

        readings = readings.copy()
        readings["Timestamp"] = (pd.to_datetime(readings["Timestamp"],
                                                utc=True,
                                                errors="coerce",)
                                                )
        readings["Value"] = (pd.to_numeric(readings["Value"],
                                           errors="coerce",)
                                           )
        return (readings.dropna(subset=["Timestamp","Value",])
                .sort_values("Timestamp"))

    @render.text
    def panel_title():
        row = selected_row()
        if row is None:
            return "Select a sensor"

        return row["display_name"]

    @render.ui
    def sensor_details():
        row = selected_row()
        if row is None:
            return ui.p(
                ("Click a marker to view its "
                 "details and measurements."),
                class_="status-note",
            )
        coords = ("Coordinates unavailable")

        if (pd.notna(row["latitude"]) and pd.notna(row["longitude"])):
            coords = (f'{row["latitude"]:.6f},'
                      f'{row["longitude"]:.6f}')

        return ui.div(ui.div(ui.strong("Network: "),row["provider"],),
                      ui.div(ui.strong("Sensor reference: "),
                             ui.code(row["sensor_name"]),
                             ),
                      ui.div(ui.strong("Coordinates: "), coords,),
                      ui.div(ui.strong(("Coordinate reference "
                                        "system: ")),
                                        "WGS 84 (EPSG:4326)",),
                    class_="sensor-meta",)

    @render.ui
    def data_status():
        row = selected_row()
        if row is None:
            return None
        data = selected_data()
        if data.empty:
            return ui.p(("No readings were returned "
                         "for this sensor and period."),
                class_="status-note",)

        return ui.p(f"{len(data):,} readings returned.",
                    class_="status-note",)

    @render.ui
    def time_series_summary():
        data = selected_data()

        if data.empty:
            return None
        
        values = (pd.to_numeric(data["Value"], errors="coerce",)
            .dropna())
        
        if values.empty:
            return None
        
        exceedance_count = int((values>= PM25_THRESHOLD).sum())
        exceedance_percent = (100*exceedance_count/ len(values))
        trend_text = ("Not enough data")
        valid_trend_data = (data[["Timestamp", "Value"]]
            .dropna()
            .sort_values("Timestamp"))
        
        if (len(valid_trend_data) >= 2 and valid_trend_data["Timestamp"].nunique() >= 2):
            elapsed_days = (valid_trend_data["Timestamp"] - valid_trend_data["Timestamp"].iloc[0]).dt.total_seconds() / 86400
            slope, _ = np.polyfit(elapsed_days, 
                                  valid_trend_data["Value"], 1,)
            trend_text = (f"{slope:+.2f} "
                          "µg/m³ per day")
            
        return ui.div(ui.h5("Time-series summary"),
                      ui.div(ui.strong("Start: "),
                             data["Timestamp"]
                             .min()
                             .strftime("%d %b %Y %H:%M UTC"),),
                      ui.div(ui.strong("End: "),
                             data["Timestamp"]
                             .max()
                             .strftime("%d %b %Y %H:%M UTC"),),
                      ui.div(ui.strong("Mean: "),
                             f"{values.mean():.2f} µg/m³",),
                      ui.div(ui.strong("Median: "),
                             f"{values.median():.2f} µg/m³",),
                      ui.div(ui.strong("Minimum: "),
                             f"{values.min():.2f} µg/m³",),
                      ui.div(ui.strong("Maximum: "),
                             f"{values.max():.2f} µg/m³",),
                      ui.div(ui.strong("Standard deviation: "),
                             f"{values.std():.2f} µg/m³",),
                      ui.div(ui.strong("Linear trend: "),
                             trend_text,),
                      ui.div(ui.strong(("Readings at or above "
                                        f"{PM25_THRESHOLD:g} "
                                        "µg/m³: ")),
                                        (f"{exceedance_count:,} " 
                                        f"({exceedance_percent:.1f}%)"),),
            class_="summary-box",
        )

    @render_plotly
    def sensor_chart():
        data = selected_data()
        row = selected_row()
        if row is None or data.empty:
            fig = px.line()
            fig.update_layout(annotations=[dict(text=("No data to display"),
                                                x=0.5,
                                                y=0.5,
                                                showarrow=False,)],
                                xaxis_visible=False,
                                yaxis_visible=False,
                                margin=dict(l=10, r=10, t=20, b=10,),)
            return fig

        plot_data = (data[["Timestamp", "Value"]]
                     .copy()
                     .dropna()
                     .sort_values("Timestamp"))

        fig = px.line(plot_data, x="Timestamp", y="Value",
            labels={"Timestamp": "Time", "Value": (f"{input.variable()} ""(µg/m³)"),},
        )

        fig.update_traces(line_color="#1b9e77", 
                          line_width=1.5,
                          name="Observed",
                          showlegend=True,)

        # Permanent dotted threshold line.
        fig.add_hline(y=PM25_THRESHOLD,
                      line_dash="dot",
                      line_color="#c2185b",
                      line_width=2,)

        above_threshold = (plot_data["Value"]>= PM25_THRESHOLD)

        if above_threshold.any():
            time_differences = (plot_data["Timestamp"]
                                .diff()
                                .dropna())

            if time_differences.empty:
                normal_interval = (pd.Timedelta(minutes=30))
            else:
                normal_interval = (time_differences.median())

            threshold_groups = (above_threshold
                                .ne(above_threshold.shift())
                                .cumsum())

            exceeding_data = (plot_data.loc[above_threshold])
            exceeding_groups = (threshold_groups.loc[above_threshold])

            for _, section in (exceeding_data.groupby(exceeding_groups)):
                x0 = (section["Timestamp"].iloc[0] - normal_interval / 2)
                x1 = (section["Timestamp"].iloc[-1] + normal_interval / 2)

                fig.add_vrect(x0=x0, x1=x1,
                              fillcolor="#f48fb1",
                              opacity=0.18,
                              line_width=0,
                              layer="below",)

        selected_trends = (input.trend_lines() or [])

        if "rolling" in selected_trends:
            rolling_data = (plot_data.set_index("Timestamp")["Value"]
                            .rolling("24h", min_periods=2,)
                            .mean())

            fig.add_scatter(x=rolling_data.index, y=rolling_data.values,
                            mode="lines",
                            name=("24-hour rolling mean"),
                            line={"color": "#ff8c00", "width": 2.5,},)

        if ("linear" in selected_trends and len(plot_data) >= 2 and plot_data["Timestamp"].nunique() >= 2):
            elapsed_days = (plot_data["Timestamp"] - plot_data["Timestamp"].iloc[0]).dt.total_seconds() / 86400

            slope, intercept = (np.polyfit(elapsed_days, plot_data["Value"], 1,))

            trend_values = (intercept + slope * elapsed_days)

            fig.add_scatter(x=plot_data["Timestamp"], y=trend_values,
                            mode="lines",
                            name=("Linear trend " f"({slope:+.2f} " "µg/m³/day)"),
                            line={"color": "#6a3d9a", "width": 2, "dash": "dash",},)

        # -------------------------------------------------
        # Temporary Gaussian random-walk demonstration
        # -------------------------------------------------

        if len(plot_data) >= 2:
            time_differences = (plot_data["Timestamp"]
                                .diff()
                                .dropna())
            forecast_interval = (time_differences.median())

            # Protect against unusual gaps in the data.
            if (pd.isna(forecast_interval) or forecast_interval <= pd.Timedelta(0)):
                forecast_interval = pd.Timedelta(hours=1)

            forecast_interval = max(forecast_interval, pd.Timedelta(minutes=1),)
            forecast_interval = min(forecast_interval, pd.Timedelta(hours=6),)
            forecast_steps = max(1, int(pd.Timedelta(hours=DEMO_FORECAST_HOURS) / forecast_interval),)
            observed_differences = (plot_data["Value"]
                                    .diff()
                                    .dropna())
            step_sigma = (observed_differences.std())

            if (pd.isna(step_sigma) or step_sigma <= 0):
                step_sigma = max(plot_data["Value"].std() * 0.05,0.1,)

            # Generate a stable seed for this sensor
            # and final observation time. This prevents
            # the projection changing whenever a checkbox
            # is selected.
            seed_text = (f'{row["sensor_name"]}-' f'{plot_data["Timestamp"].iloc[-1]}')
            seed = int.from_bytes(hashlib.sha256(seed_text.encode("utf-8")).digest()[:8],
                                  byteorder="little",)
            random_generator = (np.random.default_rng(seed))
            random_steps = (random_generator.normal(loc=0.0, scale=step_sigma, size=forecast_steps,))
            last_value = float(plot_data["Value"].iloc[-1])
            forecast_values = (last_value + np.cumsum(random_steps))

            # PM2.5 concentrations cannot be negative.
            forecast_values = np.maximum(forecast_values, 0.0,)
            last_timestamp = (plot_data["Timestamp"].iloc[-1])
            forecast_times = pd.date_range(start=(last_timestamp+ forecast_interval),
                                           periods=forecast_steps,
                                           freq=forecast_interval,)

            # Include the final observed value so the
            # projection joins onto the observed series.
            connected_times = [last_timestamp, *forecast_times,]
            connected_values = [last_value, *forecast_values,]

            fig.add_scatter(x=connected_times, y=connected_values,
                            mode="lines",
                            name=("Demo projection"),
                            line={"color": "#e31a1c", "width": 2,},
                            hovertemplate=("%{x}<br>"
                                           "Demo projection: "
                                           "%{y:.2f} µg/m³"
                                           "<extra></extra>"),)

            # Mark where observed data end and the
            # demonstration projection begins.
            fig.add_vline(x=last_timestamp, 
                         line_color="#555555",
                         line_width=1.5,
                         line_dash="dot",
                         annotation_text=("Demo projection"),
                         annotation_position="top right",)

        fig.update_layout(title=(f'{input.variable()} — ' f'{row["display_name"]}'),
                          hovermode="x unified",
                          margin=dict(l=20, r=10, t=70, b=20,),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, 
                                      xanchor="left", x=0,),)

        return fig


app = App(
    app_ui,
    server,
)