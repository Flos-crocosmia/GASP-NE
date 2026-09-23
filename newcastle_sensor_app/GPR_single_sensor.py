import numpy as np
import pandas as pd
import gpflow


# SINGLE-SENSOR GP
#
# Changes from the original notebook workflow:
#   - accepts one already-selected sensor dataframe;
#   - uses elapsed time in hours rather than yearly array indices;
#   - uses a Matern-3/2 + fixed daily-periodic kernel;
#   - fits only the most recent training window to keep the app responsive;
#   - performs no Matplotlib plotting, CSV writing or test-data evaluation;
#   - returns future timestamps, posterior means and 95% prediction intervals.


def sampling_interval(timestamps):
    # Choose a regular interval suitable for UO and DEFRA/local data
    # DEFRA/local observations are normally hourly.  Using an hourly grid also
    # avoids fitting an unnecessarily large exact GP to irregular sub-hourly
    # observations.
    differences = timestamps.sort_values().diff().dropna()
    if differences.empty:
        raise GPForecastError("At least two different timestamps are required.")
    median_interval = differences.median()
    if median_interval <= pd.Timedelta(minutes=20):
        return pd.Timedelta(minutes=15), "15min"
    return pd.Timedelta(hours=1), "1h"


def forecast_single_sensor_gp(readings, *, forecast_hours=24, training_hours=168, max_training_points=700, optimiser_iterations=200,):
    # Fit the single-sensor GP and forecast future PM2.5 concentrations.
    # Parameters:
    # readings : df : Must contain Timestamp and Value.  App already supplies data for one selected sensor.
    # forecast_hours : Length of the future forecast.  App uses 24 hours.
    # training_hours : Maximum recent history used for fitting.  App uses seven days.
    # max_training_points : Safety limit for the cubic cost of exact Gaussian-process fitting.
    # optimiser_iterations : Maximum number of SciPy optimisation iterations.
    # Returns:
    # df : Timestamp, prediction, lower_95 and upper_95.

    data = readings[["Timestamp", "Value"]].copy()
    data["Timestamp"] = pd.to_datetime(data["Timestamp"], utc=True, errors="coerce")
    data["Value"] = pd.to_numeric(data["Value"], errors="coerce")
    data = (data.dropna(subset=["Timestamp", "Value"])
            .loc[lambda frame: frame["Value"] >= 0]
            .drop_duplicates("Timestamp", keep="last")
            .sort_values("Timestamp"))

    if len(data) < 12:
        raise RuntimeError("At least 12 valid readings are required for the GP forecast.")

    interval, frequency = sampling_interval(data["Timestamp"])
    regular = (data.set_index("Timestamp")["Value"]
               .resample(frequency)
               .mean()
               .dropna())

    latest_time = regular.index.max()
    training_start = latest_time - pd.Timedelta(hours=float(training_hours))
    training = regular.loc[regular.index >= training_start].iloc[-max_training_points:]

    if len(training) < 12:
        raise RuntimeError("Too few regular observations remain after preparing the GP data.")

    # GP inputs use hours since the start of the fitted window.  PM2.5 is
    # log-transformed and standardised to improve numerical stability.
    origin = training.index[0]
    x_train = ((training.index - origin) / pd.Timedelta(hours=1)).to_numpy(dtype=np.float64).reshape(-1, 1)

    y_log = np.log1p(training.to_numpy(dtype=np.float64))
    y_centre = float(np.mean(y_log))
    y_scale = float(np.std(y_log))
    if not np.isfinite(y_scale) or y_scale < 1e-8:
        y_scale = 1.0
    y_train = ((y_log - y_centre) / y_scale).reshape(-1, 1)

    gpflow.config.set_default_float(np.float64)

    # Adapted directly from the original kernel components:
    # a smooth Matern trend plus a physically interpretable 24-hour cycle.
    trend_kernel = gpflow.kernels.Matern32(variance=0.7, lengthscales=24.0,)
    daily_base = gpflow.kernels.SquaredExponential(variance=0.3, lengthscales=3.0,)
    daily_kernel = gpflow.kernels.Periodic(base_kernel=daily_base, period=24.0,)
    gpflow.utilities.set_trainable(daily_kernel.period, False)

    model = gpflow.models.GPR(data=(x_train, y_train),
                              kernel=trend_kernel + daily_kernel,
                              mean_function=gpflow.mean_functions.Zero(),)
    model.likelihood.variance.assign(0.05)

    try:
        gpflow.optimizers.Scipy().minimize(model.training_loss,
                                           model.trainable_variables,
                                           options={"maxiter": int(optimiser_iterations)},)
    except Exception as exc:
        raise RuntimeError(f"GP optimisation failed: {exc}") from exc

    forecast_steps = max(1, int(np.ceil(pd.Timedelta(hours=float(forecast_hours)) / interval)),)
    forecast_times = pd.date_range(start=latest_time + interval, periods=forecast_steps, freq=frequency,)
    x_forecast = ((forecast_times - origin) / pd.Timedelta(hours=1)).to_numpy(dtype=np.float64).reshape(-1, 1)

    # predict_y includes observation noise, so these are future-observation
    # intervals rather than latent-function confidence intervals.
    predicted_mean, predicted_variance = model.predict_y(x_forecast)
    mean_standardised = predicted_mean.numpy().reshape(-1)
    variance_standardised = predicted_variance.numpy().reshape(-1)

    mean_log = y_centre + y_scale * mean_standardised
    variance_log = np.maximum((y_scale**2) * variance_standardised, 1e-12)
    standard_deviation_log = np.sqrt(variance_log)

    # For a log-normal predictive distribution, this is the expected PM2.5
    # concentration rather than merely the back-transformed latent median.
    prediction = np.expm1(mean_log + 0.5 * variance_log)
    lower_95 = np.expm1(mean_log - 1.96 * standard_deviation_log)
    upper_95 = np.expm1(mean_log + 1.96 * standard_deviation_log)

    result = pd.DataFrame({"Timestamp": forecast_times,
                           "prediction": np.maximum(prediction, 0),
                           "lower_95": np.maximum(lower_95, 0),
                           "upper_95": np.maximum(upper_95, 0),})
    result.attrs.update({"training_points": len(training),
                         "training_start": training.index.min(),
                         "training_end": training.index.max(),
                         "sampling_interval": interval,
                         "kernel": "Matern32 + fixed 24-hour periodic",})
    return result