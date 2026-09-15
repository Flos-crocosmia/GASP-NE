from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from geopy.distance import geodesic
import hashlib

from ipyleaflet import CircleMarker, Map, basemaps
from ipywidgets import Layout
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_plotly, render_widget
import plotly.graph_objects as go

from data_sources import (
    get_defra_readings,
    get_sensor_metadata,
    get_uo_readings,
    load_sensor_registry,
)

APP_DIR = Path(__file__).resolve().parent
PM25_THRESHOLD = 10.0
DEMO_FORECAST_HOURS = 24

registry = load_sensor_registry(APP_DIR / "naming.csv")

uo_mon_registry = registry.loc[registry["provider"] == "UO-Mon"]

UO_MON_CHOICES = dict(zip(uo_mon_registry["sensor_name"], uo_mon_registry["display_name"],))

DEFAULT_UO_MON = next(iter(UO_MON_CHOICES), None,)


def closest_sensor(locations_1, locations_2):
    """Find the closest sensor in locations_2 for each sensor in locations_1."""
    results = []

    locations_1 = locations_1.dropna(subset=["sensor_name", "latitude", "longitude"])
    locations_2 = locations_2.dropna(subset=["sensor_name", "latitude", "longitude"])

    for _, row_1 in locations_1.iterrows():
        coords_1 = (float(row_1["latitude"]),
                    float(row_1["longitude"]),)

        for _, row_2 in locations_2.iterrows():
            coords_2 = (float(row_2["latitude"]),
                        float(row_2["longitude"]),)

            results.append({"uo_sensor": row_1["sensor_name"],
                            "reference_sensor": row_2["sensor_name"],
                            "distance_km": geodesic(coords_1, coords_2).km,})

    if not results:
        return pd.DataFrame(columns=["uo_sensor", "reference_sensor", "distance_km"])

    distances = pd.DataFrame(results)

    closest_indices = (distances.groupby("uo_sensor")["distance_km"].idxmin())

    return distances.loc[closest_indices].reset_index(drop=True)


app_ui = ui.page_fillable(

    ui.tags.style(
        """
        .app-header {
            display: flex;
            align-items: center;
            justify-content: space-between;
            flex: 0 0 auto;
            padding: 0.8rem 1.25rem;
            border-bottom: 1px solid #dddddd;
            background-color: #ffffff;
        }

        .app-heading h1 {
            margin: 0;
            color: #1f3b4d;
            font-size: 1.8rem;
            font-weight: 600;
        }

        .app-heading p {
            margin: 0.25rem 0 0 0;
            color: #666666;
            font-size: 0.95rem;
        }

        .app-logos {
            display: flex;
            align-items: center;
            gap: 1.25rem;
            margin-left: 2rem;
        }

        .app-logo {
            width: auto;
            max-width: 170px;
            height: 55px;
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
            color: #666666;
            font-size: 0.9rem;
        }

        .summary-box {
            background-color: #f7f7f7;
            border-radius: 6px;
            padding: 0.75rem;
            margin-bottom: 1rem;
            font-size: 0.9rem;
        }

        .summary-box h5 {
            margin-top: 0;
            margin-bottom: 0.5rem;
        }

        @media (max-width: 800px) {
            .app-header {
                align-items: flex-start;
                flex-direction: column;
                gap: 0.75rem;
            }

            .app-logos {
                margin-left: 0;
            }

            .app-logo {
                max-width: 130px;
                height: 45px;
            }
        }
        """
    ),

    # Visible application header
    ui.tags.header(
        ui.div(ui.h1("GASP-NE Air Quality Application"),
               ui.p("Monitoring and forecasting of PM2.5 across Newcastle upon Tyne."),
               class_="app-heading",),

        ui.div(ui.output_image("ncl_logo",
                               width="170px",
                               height="55px",
                               inline=True,),
               ui.output_image("epsrc_logo",
                               width="170px",
                               height="55px",
                               inline=True,),
                class_="app-logos",),
        class_="app-header",
    ),

    ui.navset_tab(
        ui.nav_panel("Sensor explorer",
            ui.layout_sidebar(
                ui.sidebar(ui.h3(ui.output_text("panel_title")),
                        ui.output_ui("sensor_details"),
                        ui.input_select("period", "Time period",
                                        choices={"24": "Last 24 hours",
                                                    "168": "Last 7 days",
                                                    "720": "Last 30 days",
                                                    "custom": "Custom interval",},
                                            selected="168",),
                            ui.panel_conditional("input.period === 'custom'",
                                                ui.input_date_range("custom_dates", "Custom date range",
                                                                    start=(datetime.now(timezone.utc) - timedelta(days=7)).date(),
                                                                    end=datetime.now(timezone.utc).date(),),),
                            ui.input_select("variable", "Measurement",
                                            choices={"PM2.5": "PM2.5",},
                                            selected="PM2.5",),
                            ui.output_ui("data_status"),
                            ui.output_ui("time_series_summary"),
                            ui.input_checkbox("show_sensor_chart", "Show time-series plot",
                                              value=False,),
                            width=430,
                            open="always",
                            ),
                ui.card(ui.card_body(ui.p("This application collates PM2.5 data from Urban Observatory Monitor sensors, DEFRA "
                                    "AURN sites and locally managed monitoring sites. It provides a cohesive interface for "
                                    "exploring and predicting air quality across the Newcastle upon Tyne region."),
                        output_widget("sensor_map", height='70vh'),),
                    full_screen=True,
                ),
            ),
            value = "explorer"
        ),
        ui.nav_panel("Validation",
            ui.layout_sidebar(
                ui.sidebar(ui.h4("Sensor validation"),
                           ui.input_select("validation_uo_sensor", "UO-Mon sensor",
                                           choices=UO_MON_CHOICES,
                                           selected=DEFAULT_UO_MON,),
                            ui.input_select("validation_period", "Comparison period",
                                            choices={"7": "Past 7 days",
                                                     "30": "Past 30 days",
                                                     "90": "Past 90 days",},
                                            selected="7"),
                            ui.output_ui("validation_pair_details"),
                            ui.output_ui("validation_summary"),
                            width=340,
                            open="always",         
                           ),
                ui.card(ui.card_header(ui.output_text("validation_title")),
                        ui.div(output_widget("validation_chart", height="800px",),
                               style=("width: min(100%, 900px);"
                                     "margin: 0 auto;"),),
                        full_screen=True,
                        ),
            ),
            value="validation"
        ),
        id="main_tabs",
        selected="explorer",
    ),
)


def server(input, output, session):

    @render.image
    def ncl_logo():
        logo_path = (APP_DIR / "ncl_logo.png")
        return {"src": str(logo_path),
                "height": "55px",
                "alt": "Newcastle University logo",}

    @render.image
    def epsrc_logo():
        logo_path = (APP_DIR / "epsrc_logo.jpg")
        return {"src": str(logo_path),
                "height": "55px",
                "alt": "EPSRC logo",}
    
    selected_sensor = reactive.value(None)

    @reactive.calc
    def sensors():
        sensor_df = get_sensor_metadata(registry).copy()
        sensor_df["latitude"] = pd.to_numeric(sensor_df["latitude"],
                                              errors="coerce",)
        sensor_df["longitude"] = pd.to_numeric(sensor_df["longitude"],
                                               errors="coerce",)
        return sensor_df
    
    @render_widget
    def sensor_map():
        sensor_df = sensors()
        map_widget = Map(center=(54.9783, -1.6178), zoom=12,
            basemap=(basemaps.OpenStreetMap.Mapnik),
            scroll_wheel_zoom=True,
            layout=Layout(width="100%", height="100%",),
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

    @reactive.calc
    def validation_pair():
        """Find the selected UO-Mon sensor and its closest reference sensor."""
        selected_name = input.validation_uo_sensor()
        if not selected_name:
            return None

        sensor_df = sensors()
        selected_uo = sensor_df.loc[(sensor_df["sensor_name"] == selected_name) & (sensor_df["provider"] == "UO-Mon")]
        reference_sensors = sensor_df.loc[sensor_df["provider"] == "DEFRA/Local"]
        selected_uo = selected_uo.dropna(subset=["latitude", "longitude"])
        reference_sensors = reference_sensors.dropna(subset=["latitude", "longitude"])

        if selected_uo.empty or reference_sensors.empty:
            return None

        nearest = closest_sensor(selected_uo, reference_sensors,)

        if nearest.empty:
            return None

        nearest_row = nearest.iloc[0]

        reference_match = reference_sensors.loc[reference_sensors["sensor_name"] == nearest_row["reference_sensor"]]

        if reference_match.empty:
            return None

        return {"uo": selected_uo.iloc[0].copy(),
                "reference": reference_match.iloc[0].copy(),
                "distance_km": float(nearest_row["distance_km"]),}


    @reactive.calc
    def validation_data():
        """Download and pair hourly PM2.5 observations."""
        pair = validation_pair()

        if pair is None:
            return pd.DataFrame()

        days = int(input.validation_period())
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=days)
        uo_row = pair["uo"]
        reference_row = pair["reference"]
        uo_data = get_uo_readings(sensor_name=uo_row["sensor_name"],
                                  start=start,
                                  end=end,
                                  variable="PM2.5",)
        reference_data = get_defra_readings(site_code=reference_row["code"],
                                            start=start,
                                            end=end,
                                            variable="PM2.5",
                                            source_type=reference_row["type"],)

        if uo_data.empty or reference_data.empty:
            return pd.DataFrame()

        def prepare_hourly(data, column_name):
            prepared = data[["Timestamp", "Value"]].copy()
            prepared["Timestamp"] = pd.to_datetime(prepared["Timestamp"],
                                                   utc=True,
                                                   errors="coerce",)
            prepared["Value"] = pd.to_numeric(prepared["Value"],
                                              errors="coerce",)

            prepared = prepared.dropna(subset=["Timestamp", "Value"])

            return (prepared.set_index("Timestamp")["Value"]
                    .resample("1h")
                    .mean()
                    .rename(column_name))

        reference_hourly = prepare_hourly(reference_data, "Reference",)
        uo_hourly = prepare_hourly(uo_data, "UO-Mon",)

        return (pd.concat([reference_hourly, uo_hourly], axis=1,)
                .dropna()
                .reset_index())

    @reactive.effect
    @reactive.event(input.show_sensor_chart)
    def show_sensor_chart_modal():
        if not input.show_sensor_chart():
            ui.modal_remove()
            return

        row = selected_row()
        if row is None:
            ui.notification_show("Select a sensor on the map first.", type="warning",)
            ui.update_checkbox("show_sensor_chart", value=False,)
            return

        modal = ui.modal(ui.div(ui.input_checkbox_group("trend_lines", "Plot options",
                                                        choices={"rolling": "24-hour rolling mean",
                                                                 "linear": "Linear trend"},
                                                        selected=[]),
                                style=("padding: 0.25rem 0 0.5rem 0;"),),
                         output_widget("sensor_chart", height="70vh",),
                         title=(f'{input.variable()} time series — 'f'{row["display_name"]}'),
                         size="xl",
                         easy_close=False,
                         footer=ui.input_action_button("close_sensor_chart", "Close",
                                                       class_="btn-secondary",),
                            )
        ui.modal_show(modal)


    @reactive.effect
    @reactive.event(input.close_sensor_chart)
    def dismiss_sensor_chart_modal():
        ui.modal_remove()
        ui.update_checkbox("show_sensor_chart", value=False,)

    @render.text
    def panel_title():
        row = selected_row()
        if row is None:
            return "Select a sensor"
        return row["display_name"]

    @render.text
    def validation_title():
        pair = validation_pair()
        if pair is None:
            return "Sensor validation"
        return (f'{pair["uo"]["display_name"]} vs '
                f'{pair["reference"]["display_name"]}')

    @render.ui
    def sensor_details():
        row = selected_row()
        if row is None:
            return ui.p(
                ("Click a marker to view its details and measurements."),
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
            return ui.p(("No readings were returned for this sensor and period."),
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

    @render.ui
    def validation_pair_details():
        pair = validation_pair()
        if pair is None:
            return ui.p("A validation pair could not be identified.", class_="status-note",)

        return ui.div(ui.div(ui.strong("Selected UO sensor: "),
                             pair["uo"]["display_name"],),
                      ui.div(ui.strong("Closest reference sensor: "),
                             pair["reference"]["display_name"],),
                      ui.div(ui.strong("Reference network: "),
                             pair["reference"]["type"],),
                      ui.div(ui.strong("Distance: "),
                             f'{pair["distance_km"]:.2f} km',),
                    class_="summary-box",)


    @render.ui
    def validation_summary():
        comparison = validation_data()
        if comparison.empty:
            return ui.p("No overlapping PM2.5 readings were available for this pair.", class_="status-note",)

        differences = (comparison["UO-Mon"] - comparison["Reference"])
        correlation = comparison[["Reference", "UO-Mon"]].corr().iloc[0, 1]
        mae = differences.abs().mean()
        rmse = np.sqrt(np.mean(differences ** 2))
        bias = differences.mean()
        correlation_text = (f"{correlation:.3f}"
            if pd.notna(correlation)
            else "Unavailable")

        return ui.div(ui.h5("Validation summary"),
                      ui.div(ui.strong("Paired observations: "),
                             f"{len(comparison):,}",),
                      ui.div(ui.strong("Pearson correlation: "),
                             correlation_text,),
                      ui.div(ui.strong("Mean absolute error: "),
                             f"{mae:.2f} µg/m³",),
                      ui.div(ui.strong("Root mean squared error: "),
                             f"{rmse:.2f} µg/m³",),
                      ui.div(ui.strong("Mean UO − reference bias: "),
                             f"{bias:+.2f} µg/m³",),
                    class_="summary-box",)

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

            # Generate a stable seed for this sensor to prevent the projection changing whenever a checkbox is selected.
            seed_text = (f'{row["sensor_name"]}-' f'{plot_data["Timestamp"].iloc[-1]}')
            seed = int.from_bytes(hashlib.sha256(seed_text.encode("utf-8")).digest()[:8],
                                  byteorder="little",)
            random_generator = (np.random.default_rng(seed))
            number_of_simulations = 1000
            random_steps = (random_generator.normal(loc=0.0, scale=step_sigma, size=(number_of_simulations, forecast_steps,)))
            last_value = float(plot_data["Value"].iloc[-1])

            # PM2.5 concentrations cannot be negative.
            simulated_paths = (last_value + np.cumsum(random_steps, axis=1,))
            simulated_paths = np.maximum(simulated_paths, 0.0,)
            forecast_values = simulated_paths[0]
            # Pointwise 95% prediction interval.
            lower_interval = np.percentile(simulated_paths, 2.5, axis=0,)
            upper_interval = np.percentile(simulated_paths, 97.5, axis=0,)
            last_timestamp = (plot_data["Timestamp"].iloc[-1])
            forecast_times = pd.date_range(start=(last_timestamp+ forecast_interval),
                                           periods=forecast_steps,
                                           freq=forecast_interval,)

            # Include the final observed value so the
            # projection joins onto the observed series.
            connected_times = [last_timestamp, *forecast_times,]
            connected_values = [last_value, *forecast_values,]
            connected_lower = [last_value, *lower_interval,]
            connected_upper = [last_value, *upper_interval,]

            # Invisible lower boundary.
            fig.add_scatter(x=connected_times, y=connected_lower,
                            mode="lines",
                            line={"width": 0,},
                            hoverinfo="skip",
                            showlegend=False,)

            # Upper boundary filled down to the lower boundary.
            fig.add_scatter(x=connected_times, y=connected_upper,
                            mode="lines",
                            line={"width": 0,},
                            fill="tonexty",
                            fillcolor="rgba(227, 26, 28, 0.18)",
                            name="95% demo prediction interval",
                            hovertemplate=("%{x}<br>"
                                           "Upper interval: %{y:.2f} µg/m³"
                                           "<extra></extra>"),)

            fig.add_scatter(x=connected_times, y=connected_values,
                            mode="lines",
                            name=("Demo projection"),
                            line={"color": "#e31a1c", "width": 2,},
                            hovertemplate=("%{x}<br>"
                                           "Demo projection: %{y:.2f} µg/m³"
                                           "<extra></extra>"),)

            # Mark where observed data end and the
            # demonstration projection begins.
            fig.add_vline(x=last_timestamp, 
                         line_color="#555555",
                         line_width=1.5,
                         line_dash="dot",
                         annotation_text=("Demo projection"),
                         annotation_position="top right",)

        fig.update_layout(hovermode="x unified",
                          margin=dict(l=20, r=10, t=70, b=20,),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02, 
                                      xanchor="left", x=0,),)

        return fig

    @render_plotly
    def validation_chart():
        comparison = validation_data()
        pair = validation_pair()

        if comparison.empty or pair is None:
            fig = px.scatter()
            fig.update_layout(annotations=[dict(text="No overlapping readings were available",
                                                x=0.5,
                                                y=0.5,
                                                showarrow=False,)],
                                xaxis_visible=False,
                                yaxis_visible=False,
                                margin=dict(l=30, r=20, t=40, b=30),)
            return fig

        reference_name = pair["reference"]["display_name"]
        uo_name = pair["uo"]["display_name"]
        fig = px.scatter(comparison, x="Reference", y="UO-Mon",
                         hover_data={"Timestamp": True},
                         labels={"Reference": f"{reference_name} PM2.5 (µg/m³)",
                                 "UO-Mon": f"{uo_name} PM2.5 (µg/m³)",},)
        fig.update_traces(marker={"color": "#386cb0",
                                  "size": 8,
                                  "opacity": 0.65,},
                            name="Paired hourly observations",)

        x_values = comparison["Reference"].to_numpy()
        y_values = comparison["UO-Mon"].to_numpy()

        # Add a fitted regression line.
        if len(comparison) >= 2 and np.ptp(x_values) > 0:
            slope, intercept = np.polyfit(x_values, y_values, 1,)
            fitted_x = np.linspace(x_values.min(), x_values.max(),100,)
            fitted_y = intercept + slope * fitted_x
            fig.add_trace(go.Scatter(x=fitted_x, y=fitted_y,
                                     mode="lines",
                                     name="Linear fit",
                                     line={"color": "#e31a1c",
                                           "width": 2,
                                           "dash": "dash",},)
                                           )

        # Add the one-to-one perfect-agreement line.
        overall_min = min(x_values.min(), y_values.min(),)
        overall_max = max(x_values.max(), y_values.max(),)
        fig.add_trace(go.Scatter(x=[overall_min, overall_max], y=[overall_min, overall_max],
                                 mode="lines",
                                 name="Perfect agreement",
                                 line={"color": "#555555",
                                       "width": 1.5,
                                       "dash": "dot",},)
                                       )

        fig.update_layout(margin=dict(l=40, r=20, t=60, b=40),
                          legend=dict(orientation="h", yanchor="bottom",
                                      y=1.02,
                                      xanchor="left",
                                      x=0,),)
        axis_padding = max((overall_max - overall_min) * 0.05, 0.5,)
        axis_min = overall_min - axis_padding
        axis_max = overall_max + axis_padding
        fig.update_xaxes(range=[axis_min, axis_max],)
        fig.update_yaxes(range=[axis_min, axis_max],
                         scaleanchor="x",
                         scaleratio=1,)
        return fig


app = App(app_ui, server,)