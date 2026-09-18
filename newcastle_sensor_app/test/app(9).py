from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
from geopy.distance import geodesic

from ipyleaflet import CircleMarker, Map, basemaps
from ipywidgets import Layout
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_plotly, render_widget
import plotly.graph_objects as go

# File functions/errors imported
from data_sources import (
    get_defra_readings,
    get_sensor_metadata,
    get_uo_readings,
    load_sensor_registry,
)
from kriging import KrigingError, run_kriging_analysis

# ============================================================================
# SINGLE-SENSOR GP INTEGRATION: START (NEW)
# ============================================================================
# Replaces the temporary Gaussian random-walk demonstration.
from GPR_single_sensor import GPForecastError, forecast_single_sensor_gp
# ============================================================================
# SINGLE-SENSOR GP INTEGRATION: END
# ============================================================================

APP_DIR = Path(__file__).resolve().parent
PM25_THRESHOLD = 10.0

# ============================================================================
# SINGLE-SENSOR GP INTEGRATION: START (NEW CONFIGURATION)
# ============================================================================
GP_FORECAST_HOURS = 24
GP_TRAINING_HOURS = 168
# Exact GPR scales cubically, so the live app uses a deliberately bounded fit.
GP_MAX_TRAINING_POINTS = 384
GP_OPTIMISER_ITERATIONS = 100
# ============================================================================
# SINGLE-SENSOR GP INTEGRATION: END
# ============================================================================

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
        ui.nav_panel(
            "Kriging",
            ui.layout_sidebar(
                ui.sidebar(
                    ui.h4("Spatial PM2.5 background"),
                    ui.input_select(
                        "kriging_window",
                        "Averaging window",
                        choices={
                            "24": "Last 24 hours",
                            "168": "Last 7 days",
                            "720": "Last 30 days",
                        },
                        selected="168",
                    ),
                    ui.input_select(
                        "kriging_model",
                        "Variogram model",
                        choices={
                            "exponential": "Exponential",
                            "spherical": "Spherical",
                            "gaussian": "Gaussian",
                        },
                        selected="exponential",
                    ),
                    ui.input_radio_buttons(
                        "kriging_parameter_mode",
                        "Variogram parameters",
                        choices={
                            "auto": "Automatic",
                            "manual": "Manual",
                        },
                        selected="auto",
                    ),
                    ui.panel_conditional(
                        "input.kriging_parameter_mode === 'manual'",
                        ui.input_slider(
                            "kriging_range_km",
                            "Spatial range (km)",
                            min=0.5,
                            max=50,
                            value=10,
                            step=0.5,
                        ),
                        ui.input_slider(
                            "kriging_nugget_fraction",
                            "Nugget fraction",
                            min=0,
                            max=0.8,
                            value=0.10,
                            step=0.05,
                        ),
                    ),
                    ui.input_action_button(
                        "run_kriging",
                        "Load data and run kriging",
                        class_="btn-primary",
                    ),
                    ui.hr(),
                    ui.output_ui("kriging_status"),
                    ui.output_ui("kriging_summary"),
                    ui.p(
                        "Automatic mode chooses the range and nugget using "
                        "leave-one-out prediction error.",
                        class_="status-note",
                    ),
                    width=390,
                    open="always",
                ),
                ui.navset_card_tab(
                    ui.nav_panel(
                        "Background",
                        output_widget("kriging_surface", height="72vh"),
                    ),
                    ui.nav_panel(
                        "Uncertainty",
                        output_widget("kriging_uncertainty", height="72vh"),
                    ),
                    #ui.nav_panel(
                    #    "Variogram",
                    #    output_widget("kriging_variogram", height="72vh"),
                    #),
                    ui.nav_panel(
                        "Covariance matrix",
                        output_widget("kriging_covariance", height="72vh"),
                    ),
                    ui.nav_panel(
                        "Local contributions",
                        output_widget("kriging_local_contributions", height="72vh"),
                    ),
                ),
            ),
            value="kriging",
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
    # Holds the downloaded sensor means so that changing the covariance model or range slider doesn't require reload
    kriging_snapshot_state = reactive.value(None)

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

    # ========================================================================
    # SINGLE-SENSOR GP INTEGRATION: START (NEW REACTIVE FORECAST)
    # ========================================================================
    # Keeping model fitting in its own reactive calculation is important:
    # ticking the rolling-mean or linear-trend boxes redraws the chart but does
    # not refit the GP unless the selected sensor or observation period changes.
    @reactive.calc
    def selected_gp_forecast():
        data = selected_data()
        if data.empty:
            return {
                "forecast": pd.DataFrame(),
                "error": "No observations are available for GP fitting.",
            }

        try:
            forecast = forecast_single_sensor_gp(
                data,
                forecast_hours=GP_FORECAST_HOURS,
                training_hours=GP_TRAINING_HOURS,
                max_training_points=GP_MAX_TRAINING_POINTS,
                optimiser_iterations=GP_OPTIMISER_ITERATIONS,
            )
        except GPForecastError as exc:
            return {"forecast": pd.DataFrame(), "error": str(exc)}
        except Exception as exc:
            # Prevent an unexpected TensorFlow/GPflow failure from removing the
            # observed time series from the popup.
            return {
                "forecast": pd.DataFrame(),
                "error": f"Unexpected GP error: {exc}",
            }

        return {"forecast": forecast, "error": None}
    # ========================================================================
    # SINGLE-SENSOR GP INTEGRATION: END
    # ========================================================================

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
    @reactive.event(input.run_kriging)
    def load_kriging_snapshot():
        """Fetch one common-window PN2.5 average for every available sensor"""
        window_hours=int(input.kriging_window())
        end = datetime.now(timezone.utc)
        start = end - timedelta(hours=window_hours)
        sensor_df = (sensors().loc[lambda frame: frame["provider"].isin(["UO-Mon", "DEFRA/Local"])]
                     .dropna(subset=["latitude", "longitude"])
                     .copy())
        sensor_records = sensor_df.to_dict("records")

        # Gets sensor data for one sensor, and spits out the mean
        def load_one_sensor(row):
            try:
                if str(row["provider"]).startswith("UO-"):
                    readings = get_uo_readings(sensor_name=row["sensor_name"],
                                               start=start,
                                               end=end,
                                               variable="PM2.5")
                else:
                    readings = get_defra_readings(site_code=row["code"],
                                                  start=start,
                                                  end=end,
                                                  variable="PM2.5",
                                                  source_type=row["type"])
            except Exception as exc:
                return None, f'{row["display_name"]}: {exc}'
                
            if readings.empty or "Value" not in readings:
                return None, f'{row["display_name"]}: no PM2.5 readings'

            values = pd.to_numeric(readings["Value"], errors="coerce").dropna()
            if len(values) < 3:
                return None, f'{row["display_name"]}: fewer than 3 valid readings'

            timestamps = pd.to_datetime(readings["Timestamp"], utc=True, errors="coerce",).dropna() 

            result = {"sensor_name": row["sensor_name"],
                      "display_name": row["display_name"],
                      "provider": row["provider"],
                      "code": row["code"],
                      "type": row["type"],
                      "latitude": float(row["latitude"]),
                      "longitude": float(row["longitude"]),
                      "Value": float(values.mean()),
                      "reading_count": int(len(values)),
                      "latest_timestamp": timestamps.max() if not timestamps.empty else pd.NaT,}
            return result, None

        loaded: list[dict] = []
        unavailable: list[str] = []
        # Downloads multiple sensors data for time interval concurrently
        with ui.Progress(min=0, max=max(len(sensor_records), 1)) as progress:
                    progress.set(0, message="Loading PM2.5 sensor averages")
                    worker_count = min(6, max(len(sensor_records), 1))
        
                    with ThreadPoolExecutor(max_workers=worker_count) as executor:
                        futures = {
                            executor.submit(load_one_sensor, row): row
                            for row in sensor_records
                        }
        
                        for completed_count, future in enumerate(
                            as_completed(futures),
                            start=1,
                        ):
                            row = futures[future]
                            try:
                                result, error = future.result()
                            except Exception as exc:
                                result = None
                                error = f'{row["display_name"]}: {exc}'
        
                            if result is not None:
                                loaded.append(result)
                            if error is not None:
                                unavailable.append(error)
        
                            progress.set(
                                completed_count,
                                message="Loading PM2.5 sensor averages",
                                detail=(
                                    f"{completed_count} of {len(sensor_records)} sensors"
                                ),
                            )
        
        snapshot = pd.DataFrame(loaded)
        if not snapshot.empty:
            snapshot = snapshot.sort_values("display_name").reset_index(drop=True)
        # Stores all available, attempted and unavailable sensor mean data for PM2.5 in the time interval
        kriging_snapshot_state.set(
            {
                "data": snapshot,
                "attempted": len(sensor_records),
                "unavailable": unavailable,
                "start": start,
                "end": end,
                "window_hours": window_hours,
            }
        )

    # Run kriging analysis
    @reactive.calc
    def kriging_result():
        state = kriging_snapshot_state.get()
        if state is None:
            return {"analysis": None,
                    "error": "Load sensor data to begin the spatial analysis.",}

        snapshot = state["data"]
        if snapshot.empty:
            return {"analysis": None,
                    "error": "None of the sensors returned valid PM2.5 data.",}

        try:
            analysis = run_kriging_analysis(snapshot,
                                            model=input.kriging_model(),
                                            parameter_mode=input.kriging_parameter_mode(),
                                            range_km=float(input.kriging_range_km()),
                                            nugget_fraction=float(input.kriging_nugget_fraction()),
                                            grid_size=65,
                                            validation_code="NEWC",)
        except (KrigingError, ValueError, np.linalg.LinAlgError) as exc:
            return {"analysis": None, "error": str(exc)}

        return {"analysis": analysis, "error": None} 

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

    # Bunch of error messages
    @render.ui
    def kriging_status():
        state = kriging_snapshot_state.get()
        if state is None:
            return ui.p(
                "Choose an averaging window, then load the sensor data.",
                class_="status-note",
            )

        snapshot = state["data"]
        available = len(snapshot)
        attempted = state["attempted"]
        unavailable_count = len(state["unavailable"])
        window_hours = state["window_hours"]

        status_rows = [ui.div(ui.strong("Usable sensors: "),
                              f"{available} of {attempted}",),
                       ui.div(ui.strong("Averaging window: "),
                              f"{window_hours} hours",),
                       ui.div(ui.strong("Window ended: "),
                              state["end"].strftime("%d %b %Y %H:%M UTC"),),]

        if unavailable_count:
            status_rows.append(ui.div(ui.strong("Unavailable sensors: "),
                                      str(unavailable_count),))

        if 0 < available < 6:
            status_rows.append(ui.p("The surface is based on a very small network and should be "
                                    "treated as exploratory.",
                                    class_="status-note",))

        return ui.div(*status_rows, class_="summary-box")

    # Parameters and validation statistics
    @render.ui
    def kriging_summary():
        kriging_results = kriging_result()
        analysis = kriging_results["analysis"]
        if analysis is None:
            return ui.p(kriging_results["error"], class_="status-note")

        diagnostics_rows = [ui.h5("Kriging diagnostics"),
                            ui.div(ui.strong("Model: "), analysis.model.capitalize()),
                            ui.div(ui.strong("Fitted spatial range: "),
                                   f"{analysis.range_km:.2f} km",),
                            ui.div(ui.strong("Nugget fraction: "),
                                   f"{100 * analysis.nugget_fraction:.0f}%",),
                            ui.div(ui.strong("Leave-one-out RMSE: "),
                                   f"{analysis.loo_rmse:.2f} µg/m³",),
                            ui.div(ui.strong("Leave-one-out MAE: "),
                                   f"{analysis.loo_mae:.2f} µg/m³",),]

        if analysis.validation is None:
            diagnostics_rows.append(ui.p("Newcastle Centre was unavailable, so the NEWC "
                                         "LOO diagnostic could not be calculated.",
                                         class_="status-note",))
        else:
            validation = analysis.validation
            diagnostics_rows.extend([ui.hr(),
                                     ui.h5("NEWC LOO"),
                                     ui.div(ui.strong("Observed mean: "),
                                            f'{validation["observed"]:.2f} µg/m³',),
                                     ui.div(ui.strong("Predicted background: "),
                                            f'{validation["background_prediction"]:.2f} µg/m³',),
                                     ui.div(ui.strong("Local contribution: "),
                                            f'{validation["local_contribution"]:+.2f} µg/m³',),
                                     ui.div(ui.strong("95% prediction interval: "),
                                            (f'{validation["interval_lower"]:.2f} to '
                                             f'{validation["interval_upper"]:.2f} µg/m³'),),])

        return ui.div(*diagnostics_rows, class_="summary-box")

    def empty_plot(message):
        figure = go.Figure()
        figure.update_layout(
            annotations=[
                dict(
                    text=message,
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                )
            ],
            xaxis_visible=False,
            yaxis_visible=False,
            margin=dict(l=20, r=20, t=40, b=20),
        )
        return figure

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

        # ====================================================================
        # SINGLE-SENSOR GP INTEGRATION: START (REPLACES RANDOM WALK)
        # ====================================================================
        gp_result = selected_gp_forecast()
        forecast = gp_result["forecast"]

        if not forecast.empty:
            # Invisible lower boundary for Plotly's filled interval.
            fig.add_scatter(x=forecast["Timestamp"],
                            y=forecast["lower_95"],
                            mode="lines",
                            line={"width": 0,},
                            hoverinfo="skip",
                            showlegend=False,)

            # Fill from the upper boundary down to the lower boundary.
            fig.add_scatter(x=forecast["Timestamp"],
                            y=forecast["upper_95"],
                            mode="lines",
                            line={"width": 0,},
                            fill="tonexty",
                            fillcolor="rgba(227, 26, 28, 0.18)",
                            name="95% GP prediction interval",
                            hovertemplate=("%{x}<br>"
                                           "Upper 95% limit: %{y:.2f} µg/m³"
                                           "<extra></extra>"),)

            fig.add_scatter(x=forecast["Timestamp"],
                            y=forecast["prediction"],
                            mode="lines",
                            name="Single-sensor GP forecast",
                            line={"color": "#e31a1c", "width": 2.5,},
                            hovertemplate=("%{x}<br>"
                                           "GP forecast: %{y:.2f} µg/m³"
                                           "<extra></extra>"),)

            # Separate observed data from genuinely future GP predictions.
            last_timestamp = plot_data["Timestamp"].iloc[-1]
            fig.add_vline(x=last_timestamp,
                          line_color="#555555",
                          line_width=1.5,
                          line_dash="dot",
                          annotation_text="GP forecast",
                          annotation_position="top right",)
        else:
            # The observations and optional trend lines remain usable even if
            # GPflow is unavailable or this sensor has too little data.
            fig.add_annotation(text=("GP forecast unavailable: "
                                     f'{gp_result["error"]}'),
                               x=0.99,
                               y=0.02,
                               xref="paper",
                               yref="paper",
                               xanchor="right",
                               yanchor="bottom",
                               showarrow=False,
                               font={"color": "#a33a3a", "size": 11,},
                               bgcolor="rgba(255,255,255,0.85)",)
        # ====================================================================
        # SINGLE-SENSOR GP INTEGRATION: END
        # ====================================================================

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

        fig.update_layout(height=750,
                          margin=dict(l=40, r=20, t=60, b=40),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                      xanchor="left", x=0,),)
        axis_padding = max((overall_max - overall_min) * 0.05, 0.5,)
        axis_min = overall_min - axis_padding
        axis_max = overall_max + axis_padding
        fig.update_xaxes(range=[axis_min, axis_max],)
        fig.update_yaxes(range=[axis_min, axis_max],
                         scaleanchor="x",
                         scaleratio=1,)
        return fig

    # Estimated regional background
    @render_plotly
    def kriging_surface():
        kriging_results = kriging_result()
        analysis = kriging_results["analysis"]
        if analysis is None:
            return empty_plot(kriging_results["error"])

        stations = analysis.stations
        minimum = min(float(np.min(analysis.prediction)), float(stations["Value"].min()))
        maximum = max(float(np.max(analysis.prediction)), float(stations["Value"].max()))

        figure = go.Figure()
        figure.add_trace(go.Contour(x=analysis.grid_longitude[0, :], y=analysis.grid_latitude[:, 0], z=analysis.prediction,
                                    colorscale="Viridis",
                                    zmin=minimum,
                                    zmax=maximum,
                                    contours={"showlines": False},
                                    colorbar={"title": "PM2.5<br>µg/m³"},
                                    hovertemplate=("Longitude: %{x:.4f}<br>Latitude: %{y:.4f}<br>"
                                                   "Background: %{z:.2f} µg/m³<extra></extra>"),
                                    name="Kriged background",))
        figure.add_trace(go.Scatter(x=stations["longitude"], y=stations["latitude"],
                                    mode="markers",
                                    text=stations["display_name"],
                                    marker={"size": 11, "color": stations["Value"],
                                            "colorscale": "Viridis", "cmin": minimum, "cmax": maximum,
                                            "showscale": False, "line": {"color": "white", "width": 1.5},},
                                    hovertemplate=("<b>%{text}</b><br>Observed mean: "
                                                   "%{marker.color:.2f} µg/m³<extra></extra>"),
                                    name="Sensors",))
        figure.update_layout(title="Ordinary-kriging spatial background",
                             xaxis_title="Longitude", yaxis_title="Latitude",
                             height=750, autosize=True,
                             margin=dict(l=55, r=25, t=65, b=50),
                             hovermode="closest",)
        return figure

    # Kriging standard deviation
    @render_plotly
    def kriging_uncertainty():
        kriging_results = kriging_result()
        analysis = kriging_results["analysis"]
        if analysis is None:
            return empty_plot(kriging_results["error"])

        prediction_sd = np.sqrt(analysis.prediction_variance)
        stations = analysis.stations
        figure = go.Figure(go.Contour(x=analysis.grid_longitude[0, :], y=analysis.grid_latitude[:, 0], z=prediction_sd,
                                      colorscale="Magma", contours={"showlines": False},
                                      colorbar={"title": "Prediction SD<br>µg/m³"},
                                      hovertemplate=("Longitude: %{x:.4f}<br>Latitude: %{y:.4f}<br>"
                                                     "Prediction SD: %{z:.2f} µg/m³<extra></extra>"),))
        figure.add_trace(go.Scatter(x=stations["longitude"], y=stations["latitude"],
                                    mode="markers", text=stations["display_name"],
                                    marker={"size": 9, "color": "white", "line": {"color": "#222222", "width": 1.5},},
                                    hovertemplate="<b>%{text}</b><extra></extra>",
                                    name="Sensors",))
        figure.update_layout(title="Kriging prediction uncertainty",
                             xaxis_title="Longitude", yaxis_title="Latitude",
                             height=750, autosize=True,
                             margin=dict(l=55, r=25, t=65, b=50),)
        return figure

    # Distance-versus-dissimilarity diagnostic variogram
    #@render_plotly
    #def kriging_variogram():
    #    kriging_results = kriging_result()
    #    analysis = kriging_results["analysis"]
    #    if analysis is None:
    #        return empty_plot(kriging_results["error"])
    #
    #    empirical = analysis.empirical_variogram
    #    marker_size = 8 + 1.5 * empirical["pair_count"].to_numpy(dtype=float)
    #    figure = go.Figure()
    #    figure.add_trace(go.Scatter(x=empirical["distance_km"], y=empirical["semivariance"],
    #                                mode="markers",
    #                                customdata=empirical["pair_count"],
    #                                marker={"size": marker_size, "color": "#386cb0", "opacity": 0.75,},
    #                                name="Empirical bins",
    #                                hovertemplate=("Distance: %{x:.2f} km<br>Semivariance: %{y:.3f}<br>"
    #                                               "Pairs: %{customdata}<extra></extra>"),))
    #    figure.add_trace(go.Scatter(x=analysis.theoretical_distance_km, y=analysis.theoretical_semivariance,
    #                                mode="lines", line={"color": "#e31a1c", "width": 2.5},
    #                                name=f"Fitted {analysis.model}",))
    #    figure.add_hline(y=analysis.sill, line_color="#666666", line_dash="dot", annotation_text="Sample variance",)
    #    figure.update_layout(title="Empirical and fitted variogram",
    #                         xaxis_title="Sensor separation (km)", yaxis_title="Semivariance ((µg/m³)²)",
    #                         margin=dict(l=70, r=30, t=70, b=60),
    #                         height=750, autosize=True,
    #                         legend=dict(orientation="h", y=1.02, x=0),)
    #    return figure

    # Sensor covariance heatmap
    @render_plotly
    def kriging_covariance():
        kriging_results = kriging_result()
        analysis = kriging_results["analysis"]
        if analysis is None:
            return empty_plot(kriging_results["error"])

        labels = analysis.stations["display_name"].tolist()
        figure = go.Figure(go.Heatmap(z=analysis.covariance_matrix, x=labels, y=labels,
                                      colorscale="Viridis", colorbar={"title": "Covariance<br>(µg/m³)²"},
                                      hovertemplate=("%{y}<br>%{x}<br>Covariance: %{z:.3f}<extra></extra>"),))
        figure.update_layout(title="Sensor covariance matrix implied by the variogram",
                             margin=dict(l=190, r=35, t=70, b=180),
                             height=750, autosize=True,
                             xaxis={"tickangle": -45},)
        return figure

    # Measured minus background
    @render_plotly
    def kriging_local_contributions():
        kriging_results = kriging_result()
        analysis = kriging_results["analysis"]
        if analysis is None:
            return empty_plot(kriging_results["error"])

        contributions = (analysis.local_contributions
                         .sort_values("local_contribution")
                         .reset_index(drop=True))
        colours = np.where(contributions["local_contribution"] >= 0, "#d95f02", "#386cb0",)
        customdata = np.column_stack([contributions["observed"], contributions["background_prediction"], contributions["prediction_sd"],])

        figure = go.Figure(go.Bar(x=contributions["local_contribution"], y=contributions["display_name"], 
                                  orientation="h", marker_color=colours, customdata=customdata,
                                  hovertemplate=("<b>%{y}</b><br>Local contribution: %{x:+.2f} µg/m³"
                                                 "<br>Observed: %{customdata[0]:.2f} µg/m³"
                                                 "<br>LOO background: %{customdata[1]:.2f} µg/m³"
                                                 "<br>Prediction SD: %{customdata[2]:.2f} µg/m³"
                                                 "<extra></extra>"),))
        figure.add_vline(x=0, line_color="#333333", line_width=1.5)
        figure.update_layout(title=("Local contribution = observed mean − leave-one-out background"),
                             xaxis_title="Estimated local contribution (µg/m³)", yaxis_title=None,
                             height=750, autosize=True,
                             margin=dict(l=210, r=35, t=70, b=60),)
        return figure


app = App(app_ui, server,)
