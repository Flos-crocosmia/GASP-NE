# Ordinary kriging for the GASP-NE app.
# The input is one mean PM2.5 value per sensor. The main function returns:
#    - a spatial background surface;
#    - prediction uncertainty;
#    - the covariance matrix;
#    - an empirical and model variogram;
#    - leave-one-out background estimates and local contributions
#    - a validation result for Newcastle Centre (code ``NEWC``).

from dataclasses import dataclass
import numpy as np
import pandas as pd

MINIMUM_SENSOR_COUNT = 4
SUPPORTED_MODELS = {"exponential", "gaussian", "spherical", "matern"}

# Just to avoid a dictionary ouput
@dataclass
class KrigingAnalysis:
    stations: pd.DataFrame
    grid_longitude: np.ndarray
    grid_latitude: np.ndarray
    prediction: np.ndarray
    prediction_variance: np.ndarray
    covariance_matrix: np.ndarray
    empirical_variogram: pd.DataFrame
    theoretical_distance_km: np.ndarray
    theoretical_semivariance: np.ndarray
    local_contributions: pd.DataFrame
    model: str
    range_km: float
    nugget_fraction: float
    sill: float
    loo_rmse: float
    loo_mae: float
    validation: dict | None


def pairwise_distance(first, second):
    # Euclidean distance between every point
    difference = first[:, None, :] - second[None, :, :]
    return np.sqrt(np.sum(difference**2, axis=2))


def local_coordinates(latitude, longitude):
    # Convert latitude/longitude to approximate local x/y distances in km
    latitude_origin = float(np.mean(latitude))
    longitude_origin = float(np.mean(longitude))
    km_per_degree_lon = 111.320 * np.cos(np.deg2rad(latitude_origin))
    km_per_degree_lat = 110.574
    x = (longitude - longitude_origin) * km_per_degree_lon
    y = (latitude - latitude_origin) * km_per_degree_lat
    return np.column_stack([x, y]), latitude_origin, longitude_origin


def spatial_correlation(distance, model, range_km):
    # Convert distance into a correlation between zero and one
    scaled_distance = np.asarray(distance, dtype=float) / range_km
    if model == "exponential":
        return np.exp(-scaled_distance)
    if model == "gaussian":
        return np.exp(-(scaled_distance**2))
    if model == "spherical":
        correlation = np.zeros_like(scaled_distance)
        inside_range = scaled_distance < 1
        h = scaled_distance[inside_range]
        correlation[inside_range] = 1 - 1.5 * h + 0.5 * h**3
        return correlation
    if model == "matern32":
        correlation = (1 + np.sqrt(3)*scaled_distance)*np.exp(-np.sqrt(3)*scaled_distance)
        return correlation
    


def covariance(distance, model, range_km, sill, nugget_fraction, diagonal=False):
    # Construct covariance from distance.
    # Sill is divided into spatially correlated variance and nugget variance.
    # Nugget is added only when constructing a sensor-by-sensor matrix i.e. square
    nugget = sill * nugget_fraction
    spatial_variance = sill - nugget
    result = spatial_variance * spatial_correlation(distance, model, range_km,)
    if diagonal:
        result = result + nugget * np.eye(result.shape[0])
    return result


def ordinary_kriging(sensor_xy, sensor_values, target_xy, model, range_km, sill, nugget_fraction,):
    # Predict values and variances at target locations
    number_of_sensors = len(sensor_values)
    # Covariance between the observed sensors.
    sensor_covariance = covariance(pairwise_distance(sensor_xy, sensor_xy),
                                   model, range_km, sill, nugget_fraction, diagonal=True,)

    # Very small numerical stabiliser for close or co-located sensors.
    sensor_covariance += max(sill * 1e-9, 1e-12) * np.eye(number_of_sensors)

    # Ordinary kriging system. The final row and column force the weights assigned to the sensors to sum to one.
    system = np.zeros((number_of_sensors + 1, number_of_sensors + 1))
    system[:-1, :-1] = sensor_covariance
    system[:-1, -1] = 1
    system[-1, :-1] = 1

    # Covariance between each sensor and each prediction point.
    target_covariance = covariance(pairwise_distance(sensor_xy, target_xy), 
                                   model, range_km, sill, nugget_fraction,)
    right_hand_side = np.vstack([target_covariance, np.ones(target_xy.shape[0])])

    try:
        solution = np.linalg.solve(system, right_hand_side)
    except np.linalg.LinAlgError:
        solution = np.linalg.lstsq(system, right_hand_side, rcond=None)[0]

    weights = solution[:-1]
    lagrange_multiplier = solution[-1]

    prediction = sensor_values @ weights
    prediction_variance = (sill - np.sum(weights * target_covariance, axis=0) - lagrange_multiplier)

    return prediction, np.maximum(prediction_variance, 0)


def leave_one_out(stations, xy, values, model, range_km, sill, nugget_fraction):
    # Predict each sensor without using that sensor in its own prediction
    rows = []
    for index in range(len(stations)):
        keep = np.arange(len(stations)) != index
        prediction, variance = ordinary_kriging(xy[keep], values[keep], xy[index : index + 1], model, range_km, sill, nugget_fraction,)

        observed = float(values[index])
        background = float(prediction[0])
        prediction_sd = float(np.sqrt(variance[0]))
        station = stations.iloc[index]

        rows.append({"sensor_name": station["sensor_name"],
                     "display_name": station["display_name"],
                     "provider": station["provider"],
                     "code": station.get("code", ""),
                     "latitude": float(station["latitude"]),
                     "longitude": float(station["longitude"]),
                     "observed": observed,
                     "background_prediction": background,
                     "prediction_sd": prediction_sd,
                     "interval_lower": background - 1.96 * prediction_sd,
                     "interval_upper": background + 1.96 * prediction_sd,
                     "local_contribution": observed - background,})

    result = pd.DataFrame(rows)
    result["covered_by_95_interval"] = result["observed"].between(result["interval_lower"], result["interval_upper"],)

    return result




def empirical_variogram(xy, values):
    # Calculate binned semivariance between observed sensor pairs
    distance_matrix = pairwise_distance(xy, xy)
    row, column = np.triu_indices(len(values), k=1)
    distances = distance_matrix[row, column]
    semivariance = 0.5 * (values[row] - values[column]) ** 2

    number_of_bins = min(9, max(4, len(values) // 2))
    edges = np.linspace(0, float(np.max(distances)) + 1e-9, number_of_bins + 1)
    bins = np.digitize(distances, edges[1:-1])

    records = []
    for bin_number in range(number_of_bins):
        selected = bins == bin_number
        if np.any(selected):
            records.append({"distance_km": float(np.mean(distances[selected])),
                            "semivariance": float(np.mean(semivariance[selected])),
                            "pair_count": int(np.sum(selected)),})
            
    return pd.DataFrame(records)


def run_kriging_analysis(sensor_values, *, model="exponential", parameter_mode="auto", range_km=10.0, nugget_fraction=0.10, grid_size=65, validation_code="NEWC",):
    # Run the complete spatial analysis.
    # Automatic mode deliberately uses length scale as the median distance between sensor pairs and the nugget is 10% of the sample variance.

    stations = sensor_values.copy()
    for column in ["latitude", "longitude", "Value"]:
        stations[column] = pd.to_numeric(stations[column], errors="coerce")

    stations = (stations.dropna(subset=["latitude", "longitude", "Value"])
                .drop_duplicates("sensor_name", keep="last")
                .reset_index(drop=True))

    if len(stations) < MINIMUM_SENSOR_COUNT:
        raise ValueError(f"At least {MINIMUM_SENSOR_COUNT} valid sensors are required; "
                         f"only {len(stations)} were available.")

    latitude = stations["latitude"].to_numpy(float)
    longitude = stations["longitude"].to_numpy(float)
    values = stations["Value"].to_numpy(float)
    xy, latitude_origin, longitude_origin = local_coordinates(latitude, longitude)
    stations["x_km"] = xy[:, 0]
    stations["y_km"] = xy[:, 1]

    sill = max(float(np.var(values, ddof=1)), 1e-6)

    if parameter_mode == "auto":
        sensor_distances = pairwise_distance(xy, xy)
        pair_distances = sensor_distances[np.triu_indices(len(xy), k=1)]
        pair_distances = pair_distances[pair_distances > 1e-6]
        if len(pair_distances) == 0:
            raise ValueError("The sensor locations are not distinct.")

        range_km = float(np.median(pair_distances))
        nugget_fraction = 0.10

    range_km = float(range_km)
    nugget_fraction = float(nugget_fraction)

    if range_km <= 0:
        raise ValueError("The spatial length scale must be positive.")
    if not 0 <= nugget_fraction < 1:
        raise ValueError("The nugget fraction must be between 0 and 1.")

    # Leave-one-out results define the local contribution at every sensor.
    local_contributions = leave_one_out(stations, xy, values, model, range_km, sill, nugget_fraction,)
    errors = local_contributions["local_contribution"].to_numpy(float)
    loo_rmse = float(np.sqrt(np.mean(errors**2)))
    loo_mae = float(np.mean(np.abs(errors)))

    # Rectangular grid covering the sensor network.
    padding = max(0.75, 0.08 * max(float(np.ptp(xy[:, 0])), float(np.ptp(xy[:, 1]))))
    grid_x = np.linspace(np.min(xy[:, 0]) - padding, np.max(xy[:, 0]) + padding, grid_size)
    grid_y = np.linspace(np.min(xy[:, 1]) - padding, np.max(xy[:, 1]) + padding, grid_size)
    grid_x_mesh, grid_y_mesh = np.meshgrid(grid_x, grid_y)
    grid_xy = np.column_stack([grid_x_mesh.ravel(), grid_y_mesh.ravel()])

    prediction, prediction_variance = ordinary_kriging(xy, values, grid_xy, model, range_km, sill, nugget_fraction,)

    # Convert the prediction grid back to latitude and longitude for Plotly.
    km_per_degree_lon = 111.320 * np.cos(np.deg2rad(latitude_origin))
    grid_longitude = longitude_origin + grid_x_mesh / km_per_degree_lon
    grid_latitude = latitude_origin + grid_y_mesh / 110.574

    station_distances = pairwise_distance(xy, xy)
    covariance_matrix = covariance(station_distances, model, range_km, sill, nugget_fraction, diagonal=True,)

    observed_variogram = empirical_variogram(xy, values)
    theoretical_distance = np.linspace(0, max(float(np.max(station_distances)), 1.25 * range_km), 200,)
    nugget = sill * nugget_fraction
    theoretical_semivariance = nugget + (sill - nugget) * (1 - spatial_correlation(theoretical_distance, model, range_km))
    theoretical_semivariance[0] = 0

    validation = None
    validation_row = local_contributions.loc[local_contributions["code"].astype(str).str.upper() == validation_code.upper()]
    if not validation_row.empty:
        validation = validation_row.iloc[0].to_dict()

    return KrigingAnalysis(stations=stations,
                           grid_longitude=grid_longitude,
                           grid_latitude=grid_latitude,
                           prediction=prediction.reshape(grid_x_mesh.shape),
                           prediction_variance=prediction_variance.reshape(grid_x_mesh.shape),
                           covariance_matrix=covariance_matrix,
                           empirical_variogram=observed_variogram,
                           theoretical_distance_km=theoretical_distance,
                           theoretical_semivariance=theoretical_semivariance,
                           local_contributions=local_contributions,
                           model=model,
                           range_km=range_km,
                           nugget_fraction=nugget_fraction,
                           sill=sill,
                           loo_rmse=loo_rmse,
                           loo_mae=loo_mae,
                           validation=validation,)
