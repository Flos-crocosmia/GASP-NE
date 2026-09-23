import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import gpflow
from gpflow.utilities import print_summary
import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions

from sklearn.metrics import mean_squared_error, mean_squared_log_error, r2_score
from properscoring import crps_gaussian

def to_float64(X):
    """
    Convert input data to float64.

    GPflow requires input data to be in float64 format.
    This function ensures compatibility and prevents dtype errors.
    """
    return np.asarray(X, dtype=np.float64)

## LOAD DATASETS EXAMPLE ####
# data
# data['Timestamp'] = pd.to_datetime(data['Timestamp'])
# data = data.sort_values('Timestamp')
################

def filter_and_resample_data(sensor_name, data, year):
    """
    Filters PM2.5 data for a specific sensor and year,
    then resamples it to uniform 15-minute intervals.

    Parameters:
    - sensor_name (str): Target sensor ID
    - data (DataFrame): Raw dataset
    - year (int): Target year

    Returns:
    - DataFrame: Cleaned and resampled dataset
    """

    # --- Convert timestamp ---
    data["Timestamp"] = pd.to_datetime(data["Timestamp"], errors="coerce")

    # --- Filter by sensor ---
    sensor_data = data[data["Sensor Name"] == sensor_name].copy()

    # --- Define yearly range ---
    start_date = f"{year}-01-01 00:00:00"
    end_date   = f"{year}-12-31 23:59:59"

    # --- Filter by date ---
    filtered_data = sensor_data.loc[
        (sensor_data["Timestamp"] >= start_date) &
        (sensor_data["Timestamp"] <= end_date)
    ]

    # --- Set index for resampling ---
    filtered_data = filtered_data.set_index("Timestamp")

    # --- Resample to 15-minute intervals (no filling) ---
    # 15-minute resolution balances temporal detail and computational cost
    resampled_data = filtered_data.resample("15T").asfreq()

    # --- Reset index ---
    resampled_data = resampled_data.reset_index()

    return resampled_data

## APPLY FILTERING TO DATASET ################
# sensor_name = "SENSOR_NAME_CHOSEN_IN_APP"    e.g. PER_AIRMON_MONITOR1759150
# data_sensor = filter_and_resample_data(sensor_name, data, year)
########################################

## TIME INDEXING ######################
# start_date = "year-01-01 00:00:00"
# end_date   = "year-12-31 23:59:00"
# time_wholeyear = pd.date_range(start=start_date, end=end_date, freq="15min")
#########################################

## PREPROCESSING AND ROLLING STATS ########################
# --- Ensure datetime format ---
# data_sensor["Timestamp"] = pd.to_datetime(data_sensor["Timestamp"], errors="coerce")
# --- Sort data ---
# data_sensor = data_sensor.sort_values("Timestamp").reset_index(drop=True)
# --- Set index for rolling operations ---
# data_sensor = data_sensor.set_index("Timestamp")
# data_sensor["Rolling Mean"] = data_sensor["Value"].rolling(window=5, min_periods=1, center=True).mean()
# data_sensor["Rolling Std"] = data_sensor["Value"].rolling(window=5, min_periods=1, center=True).std()
# data_sensor["Rolling Median"] = data_sensor["Value"].rolling(window=5, min_periods=1, center=True).median()
# data_sensor["Rolling Mean (1D)"] = data_sensor["Value"].rolling(window=97, min_periods=1, center=True).mean()
# data_sensor["Detrended"] = data_sensor["Value"] - data_sensor["Rolling Mean (1D)"]
# data_sensor["Rolling Detrended"] = data_sensor["Detrended"].rolling(window=5, min_periods=1, center=True).mean()
# data_sensor = data_sensor.reset_index()
# PM25_data = data_sensor['Value']
# Smooth_PM25 = data_sensor['Rolling Mean']

# Evaluation function
def calculate_metrics(y_true_train, y_pred_train, y_true_test, y_pred_test):
    """
    Compute evaluation metrics for training and testing datasets.

    Steps:
    1. Remove NaN values
    2. Compute error-based metrics
    3. Compute bias and agreement metrics

    Returns:
    - Dictionary of evaluation metrics
    """

    # --- Remove NaNs ---
    mask_train = ~ (np.isnan(y_true_train) | np.isnan(y_pred_train))
    mask_test  = ~ (np.isnan(y_true_test)  | np.isnan(y_pred_test))

    y_true_train = y_true_train[mask_train]
    y_pred_train = y_pred_train[mask_train]
    y_true = y_true_test[mask_test]
    y_pred = y_pred_test[mask_test]

    # --- MSE / RMSE ---
    mse_train = mean_squared_error(y_true_train, y_pred_train)
    mse = mean_squared_error(y_true, y_pred)

    rmse_train = np.sqrt(mse_train)
    rmse = np.sqrt(mse)

    # --- RMSLE ---
    rmsle = np.sqrt(mean_squared_log_error(y_true, y_pred))

    # --- Bias metrics ---
    nmb = np.sum(y_pred - y_true) / np.sum(y_true)
    fb  = 2 * (np.mean(y_pred) - np.mean(y_true)) / (np.mean(y_pred) + np.mean(y_true))

    # --- Percentage error ---
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # --- Agreement ---
    y_bar = np.mean(y_true)
    ioa = 1 - (
        np.sum((y_true - y_pred)**2) /
        np.sum((np.abs(y_pred - y_bar) + np.abs(y_true - y_bar))**2)
    )

    # --- R² ---
    r2 = r2_score(y_true, y_pred)

    # --- FAC2 ---
    fac2 = np.mean((y_pred / y_true >= 0.5) & (y_pred / y_true <= 2))

    # --- Relative RMSE ---
    rRMSE = rmse / rmse_train

    return {
        "RMSE_train": rmse_train,
        "RMSE_test": rmse,
        "RMSLE": rmsle,
        "NMB": nmb,
        "MAPE": mape,
        "IOA": ioa,
        "R2": r2,
        "FAC2": fac2,
        "FB": fb,
        "rRMSE": rRMSE
    }

# =========================================================
# 5.1 Utility Functions
# =========================================================

def ensure_numpy_1d(x):
    """Convert TensorFlow / array-like → 1D numpy array."""
    if hasattr(x, "numpy"):
        x = x.numpy()
    return np.ravel(x)

def as1d(a):
    """Return a 1-D numpy array regardless of TF/np shape."""
    if hasattr(a, "numpy"):
        a = a.numpy()
    a = np.asarray(a)
    return np.ravel(a)

def as2d_col(x):
    """Convert to (N,1) shape for GP input."""
    return ensure_numpy_1d(x).reshape(-1, 1)


def sanitize_filename(name):
    """Safe filename."""
    return name.replace("/", "-").replace(":", "-").replace(" ", "_")


def prepare_data(X, Y):
    """Remove NaNs and reshape."""
    mask = ~np.isnan(Y)
    return X[mask][:, None].astype(np.float64), Y[mask][:, None].astype(np.float64)

# Prevent unrealistic long periods that may destabilize the model
def enforce_max_period(kernel, max_period=1440.0):
    for subkernel in kernel.kernels:
        if isinstance(subkernel, gpflow.kernels.Periodic):
            current_period = subkernel.period.numpy()
            if current_period > max_period:
                print(f"⚠️ Period {current_period:.2f} exceeds {max_period}, clipping.")
                subkernel.period.assign(max_period)
# =========================================================
# 5.2 Transformations
# =========================================================

def transform_y(Y, method):
    """Forward transform."""
    if method == "actual":
        return Y
    elif method == "square":
        return np.square(Y)
    elif method == "ln":
        return np.log(Y + 1)
    else:
        raise ValueError("Invalid transform")


def inverse_transform(y, method):
    """Inverse transform."""
    y = ensure_numpy_1d(y)

    if method == "actual":
        return y
    elif method == "square":
        return np.sqrt(np.maximum(y, 0))
    elif method == "ln":
        return np.exp(y) - 1
    else:
        raise ValueError("Invalid transform")


# =========================================================
# 5.3 Confidence Intervals
# =========================================================

def compute_confidence_intervals(mean, variance):
    """Compute 95% and 80% confidence intervals."""
    std_dev = np.sqrt(np.clip(variance, 1e-12, None))
    
    # Ensure mean and std_dev are handled as numpy for consistent slicing
    mean_np = ensure_numpy_1d(mean)
    std_np = ensure_numpy_1d(std_dev)
    
    lower_95, upper_95 = mean_np - 1.96 * std_np, mean_np + 1.96 * std_np
    lower_80, upper_80 = mean_np - 1.28 * std_np, mean_np + 1.28 * std_np
    
    return {
        "95%": (lower_95, upper_95),
        "80%": (lower_80, upper_80),
    }
# =========================================================
# 5.4 Main Function
# =========================================================
def plot_prediction_method(
    model, 
    X, 
    Y,
    Y_sld1_ori, 
    min_x, 
    max_x, 
    t, 
    st_date, 
    method, 
    name, 
    name2, 
    pic_name, 
    kernel_type, 
    save_opt, 
    diagnostics=False
):
    """Generate GP predictions, plot results, and run diagnostics without redundancy.""" 
    gap = (max_x - len(X)) * 15
    delta = 15
    start_idx = len(X)
    
    # 1. Setup Time Ranges
    X_train_raw = np.arange(min_x, (max_x + 1) * 15 - gap, delta)[:, None].astype(np.float64)
    X_test_raw = np.arange(max_x * 15 - gap, max_x * 15, delta)[:, None].astype(np.float64)
    X_full_raw = np.arange(min_x, max_x * 15, delta)[:, None].astype(np.float64)

    # 2. Get Model Predictions
    # predict_y includes likelihood noise, predict_f is for latent function (diagnostics)
    f_mean_train, f_var_train = model.predict_y(X_train_raw)
    y_mean_test, y_cov_test = model.predict_y(X_test_raw)
    # --- Confidence intervals (pointwise/diagonal) ---
    conf_train = compute_confidence_intervals(f_mean_train, f_var_train)
    conf_ytest = compute_confidence_intervals(y_mean_test, y_cov_test)

    # 3. Inverse Transform Function
    def inv_t(val):
        if method == "ln": return np.exp(val) - 1
        if method == "square": return np.sqrt(val) # Assuming input was Y^2
        return val
    # convert to numpy arrays for downstream (1D vectors)
    f_mean_train_np = inv_t(as1d(f_mean_train))
    y_mean_test_np  = inv_t(as1d(y_mean_test))

    # apply inverse transform to CI bounds
    def inv_bounds(bounds):
        lo, hi = bounds
        return inv_t(as1d(lo)), inv_t(as1d(hi))

    conf_train = {"95%": inv_bounds(conf_train["95%"]), "80%": inv_bounds(conf_train["80%"])}
    conf_ytest = {"95%": inv_bounds(conf_ytest["95%"]), "80%": inv_bounds(conf_ytest["80%"])}

    # clip lower bounds at 0 for PM2.5
    for k in ("95%", "80%"):
        lo_t, hi_t = conf_train[k]
        lo_s, hi_s = conf_ytest[k]
        conf_train[k] = (np.clip(lo_t, 0, None), hi_t)
        conf_ytest[k] = (np.clip(lo_s, 0, None), hi_s)

    # --- Prepare arrays for plotting & metrics ---
    X_full_r, X_test_r, X_train_r = map(np.ravel, (X_full_raw, X_test_raw, X_train_raw))
    # Transform everything to original scale for plotting and metrics
    Y_plot = inv_t(Y)
    Y_sld_plot = inv_t(Y_sld1_ori)
    mu_tr_plot = inv_t(ensure_numpy_1d(f_mean_train))
    mu_te_plot = inv_t(ensure_numpy_1d(y_mean_test))
    
    Y_test       = Y_plot[start_idx:]
    Y_sld_mean1h = Y_sld_plot[start_idx:]
    

    # Compute CIs on transformed scale, then inverse transform
    ci_tr = compute_confidence_intervals(f_mean_train, f_var_train)
    ci_te = compute_confidence_intervals(y_mean_test, y_cov_test)
    
    tr_l95, tr_u95 = inv_t(ci_tr["95%"][0]), inv_t(ci_tr["95%"][1])
    tr_l80, tr_u80 = inv_t(ci_tr["80%"][0]), inv_t(ci_tr["80%"][1])
    te_l95, te_u95 = inv_t(ci_te["95%"][0]), inv_t(ci_te["95%"][1])
    te_l80, te_u80 = inv_t(ci_te["80%"][0]), inv_t(ci_te["80%"][1])

    # 4. Single Plotting Block (Removed Plot Repetition)
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(12, 10))
    X_f, X_te, X_tr = map(np.ravel, (X_full_raw, X_test_raw, X_train_raw))

    # Top Plot: Overview
    ax[0].set_title(name, fontsize=20)
    ax[0].plot(X_f, Y_plot, "o", color='red', markersize=4, label='Obs', alpha=0.4)
    ax[0].plot(X_f, Y_sld_plot, "-.", color='purple', label='1h-Smooth')
    ax[0].plot(X_tr, mu_tr_plot, "b-", label='Train Pred')
    ax[0].plot(X_te, mu_te_plot, "c-", label='Forecast', linewidth=2)
    ax[0].fill_between(X_tr, tr_l95, tr_u95, color='orange', alpha=0.2, label='95% CI')
    ax[0].fill_between(X_te, te_l95, te_u95, color='orange', alpha=0.2)
    ax[0].axvline(X_te[0], color='black', linestyle='--')
    ax[0].set_ylabel('PM2.5 (µg/m³)', fontsize=16)
    ax[0].legend(loc='upper left')
    ax[0].grid(True)
    
    # Visualize predictive uncertainty using 80% and 95% confidence intervals
    # Bottom Plot: Zoomed Forecast
    ax[1].set_title(name2, fontsize=20)
    ax[1].plot(X_te, Y_plot[start_idx:], "o", color='red', markersize=6)
    ax[1].plot(X_te, mu_te_plot, "c-", linewidth=3)
    ax[1].fill_between(X_te, te_l95, te_u95, color='orange', alpha=0.2)
    ax[1].fill_between(X_te, te_l80, te_u80, color='blue', alpha=0.2)
    ax[1].set_xlabel('Time (minutes)', fontsize=16)
    ax[1].set_ylabel('PM2.5 (µg/m³)', fontsize=16)
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()

    # 5. Metrics Calculation
    y_test_true = ensure_numpy_1d(Y_plot[start_idx:])
    y_test_pred = mu_te_plot
    # Ensure equal length for mask
    min_len = min(len(y_test_true), len(y_test_pred))
    mask = ~np.isnan(y_test_true[:min_len]) & ~np.isnan(y_test_pred[:min_len])
    
    validation_true = calculate_metrics(Y_plot[:start_idx], mu_tr_plot[:start_idx], y_test_true[:min_len][mask], y_test_pred[:min_len][mask])
    validation_sld1 = calculate_metrics(Y_sld_plot[:start_idx], mu_tr_plot[:start_idx], Y_sld_plot[start_idx:][:min_len][mask], y_test_pred[:min_len][mask])

    # --- Align for deterministic metrics (if function exists) ---
    Y_test_clean       = as1d(Y_test)
    Y_sld_mean1h_clean = as1d(Y_sld_mean1h)
    y_pred_clean       = as1d(y_mean_test_np)

    L = min(len(Y_test_clean), len(Y_sld_mean1h_clean), len(y_pred_clean))
    Y_test_clean       = Y_test_clean[:L]
    Y_sld_mean1h_clean = Y_sld_mean1h_clean[:L]
    y_pred_clean       = y_pred_clean[:L]

    mask = np.isfinite(Y_test_clean) & np.isfinite(Y_sld_mean1h_clean) & np.isfinite(y_pred_clean)
    Y_test_clean       = Y_test_clean[mask]
    Y_sld_mean1h_clean = Y_sld_mean1h_clean[mask]
    y_pred_clean       = y_pred_clean[mask]

    # Train side
    f_mean_train_np_1d = as1d(f_mean_train_np)
    Y_train_clean      = ensure_numpy_1d(Y_plot[:start_idx+1])
    Y_train_mean1h     = ensure_numpy_1d(Y_sld_plot[:start_idx+1])

    m0 = np.isfinite(Y_train_clean) & np.isfinite(Y_train_mean1h) & np.isfinite(f_mean_train_np_1d[:len(Y_train_clean)])
    Y_train_clean   = Y_train_clean[m0]
    Y_train_mean1h  = Y_train_mean1h[m0]
    y_pred_train_cl = f_mean_train_np_1d[:len(m0)][m0] if len(f_mean_train_np_1d) >= len(m0) else f_mean_train_np_1d

    # Deterministic metrics (guard if calculate_metrics missing)
    validation_true, validation_sld1 = None, None
    if 'calculate_metrics' in globals():
        validation_true = calculate_metrics(Y_train_clean, y_pred_train_cl, Y_test_clean, y_pred_clean)
        validation_sld1 = calculate_metrics(Y_train_mean1h, y_pred_train_cl, Y_sld_mean1h_clean, y_pred_clean)
        print(pd.DataFrame({
            "Metric": ["MSE", "RMSE Train", "RMSE Test", "RMSLE", "NMB", "MAPE", "IOA", "R²", "FAC2", "FB", "RMSE Ratio"],
            "True Values": [f"{val:.3f}" for val in validation_true],
            "Smoothed (sld1)": [f"{val:.3f}" for val in validation_sld1]
        }))
    else:
        print("Note: 'calculate_metrics' not found; skipping deterministic metrics table.")
        
# --- Call AQI plotter if available (robust concat) ---
    if 'plot_aqi_uk_values_with_levels' in globals():
        try:
            # 1. Ensure directory exists (Independent of data preparation)
            if save_opt and pic_name:
                out_dir = os.path.dirname(pic_name)
                if out_dir and not os.path.exists(out_dir):
                    os.makedirs(out_dir, exist_ok=True)

            # 2. Prepare data (MUST be outside the directory creation IF block)
            tl80, tu80 = conf_train["80%"]
            tl95, tu95 = conf_train["95%"]
            yl80, yu80 = conf_ytest["80%"]
            yl95, yu95 = conf_ytest["95%"]

            # make sure all are 1D
            tl80, tu80, tl95, tu95 = map(as1d, (tl80, tu80, tl95, tu95))
            yl80, yu80, yl95, yu95 = map(as1d, (yl80, yu80, yl95, yu95))
            
            # Visualize predictive uncertainty using 80% and 95% confidence intervals
            # drop last train point to avoid duplicate stitch
            n_tr_clip = max(len(tl80) - 1, 0)
            f_lower_combined  = np.concatenate((tl80[:n_tr_clip], yl80), axis=0)
            f_upper_combined  = np.concatenate((tu80[:n_tr_clip], yu80), axis=0)
            f_tlower_combined = np.concatenate((tl95[:n_tr_clip], yl95), axis=0)
            f_tupper_combined = np.concatenate((tu95[:n_tr_clip], yu95), axis=0)

            f_mean_combined = np.concatenate((as1d(f_mean_train_np)[:n_tr_clip], as1d(y_mean_test_np)), axis=0)

            # 3. Setup time
            start_time = datetime(2018, 1, 1) + timedelta(days=st_date, hours=t)
            start_time = start_time.replace(minute=0, second=0, microsecond=0)

            # 4. Call Plotter
            plot_aqi_uk_values_with_levels(
                f_mean_combined, Y_plot, Y_sld_plot,
                f_lower_combined, f_upper_combined,
                f_tlower_combined, f_tupper_combined,
                start_time=start_time,
                train_hours=len(X)/4, test_hours=gap/15/4,
                name=name, pic_name=pic_name, kernel_type=kernel_type, save_plot=save_opt
            )
            
        except FileNotFoundError as fnf:
            print(f"❌ Save Failed: The directory for '{pic_name}' does not exist.")
        except Exception as e:
            import traceback
            print(f"⚠️ AQI plot block skipped: {repr(e)}")
            
    # 6. Diagnostics (Fixed: Bastos & O'Hagan implementation)
    if diagnostics:
        print("\n--- Diagnostic Metrics (Bastos & O'Hagan) ---")
        # Use predict_f with full_cov for Mahalanobis calculation
        mu_latent, Sigma_latent = model.predict_f(X_test_raw, full_cov=True)
        mu_latent = ensure_numpy_1d(mu_latent)
        Sigma_m = Sigma_latent[0].numpy() if hasattr(Sigma_latent, "numpy") else Sigma_latent[0]
        
        y_val = y_test_true[:len(mu_latent)]
        idx_finite = np.isfinite(y_val)
        
        if np.any(idx_finite):
            diag_results = _compute_table1_like(
                mu=mu_latent[idx_finite], 
                Sigma=Sigma_m[np.ix_(idx_finite, idx_finite)], 
                y_val=y_val[idx_finite],
                n_train=len(X_tr)
            )
            # Plot Figure 1 panels (Diagnostics)
            _plot_figure1_panels(diag_results["_DI"], diag_results["_mu"], X_te[idx_finite], "GP-Diagnostics")

    return f_mean_train, f_var_train, y_mean_test, y_cov_test, validation_true, validation_sld1

# =========================================================
# 5.5 Main Function
# =========================================================
def plot_kernel_method(kernel, X,Y_train, Y, Y_sld1_ori, min_x, max_x, t,st_date,noise_variance, optimise, TF, mean_func , name, name2, pic_name, kernel_type, save_opt, method):
    """Train GP model and plot kernel predictions."""
    X, Y_train = prepare_data(X.flatten(), Y_train.values.flatten())
    Y_transformed = transform_y(Y, method)
    Y_train_transformed = transform_y(Y_train, method)
    Y_sld1_transformed = transform_y(Y_sld1_ori, method)
    model = gpflow.models.GPR(data=(X, Y_train_transformed), kernel=kernel, mean_function=mean_func)
    model.likelihood.variance.assign(noise_variance)
    gpflow.set_trainable(model.likelihood.variance, TF)

    if optimise:
        gpflow.optimizers.Scipy().minimize(model.training_loss, model.trainable_variables, options={"maxiter": 1000})
#         enforce_max_period(kernel, max_period=1440.0)
# === forced period < 1440 ===
    if "Periodic0" in kernel_type:
        enforce_max_period(kernel, max_period=1440.0)

    gpflow.utilities.print_summary(model, "notebook")

    return plot_prediction_method(model, X, Y_transformed,Y_sld1_transformed, min_x, max_x, t, st_date, method, name, name2, pic_name, kernel_type, save_opt)

# =========================
# 5.6 Random Parameter Generator
# =========================

def sample_lognormal(mean, std, low, high):
    """Sample from truncated log-normal distribution."""
    val = rand_lognormal_truncated(scale=mean, stddev=std, lower=low, upper=high, size=1)[0]
    return float(val)

# Random initialization helps avoid poor local minima during optimization
def rand_lognormal_truncated(scale=1.0, stddev=1.0, lower=None, upper=None, size=1):
    """
    Generate samples from a truncated log-normal distribution.

    This function draws samples from a log-normal distribution and
    filters them to lie within a specified range [lower, upper].

    Parameters
    ----------
    scale : float, optional
        Median (exp(mean_log)) of the log-normal distribution.
    stddev : float, optional
        Standard deviation of the log-space distribution.
    lower : float, optional
        Lower bound of accepted values (inclusive).
    upper : float, optional
        Upper bound of accepted values (inclusive).
    size : int, optional
        Number of valid samples required.

    Returns
    -------
    np.ndarray
        Array of valid samples within [lower, upper].

    Raises
    ------
    ValueError
        If not enough valid samples are found within the maximum attempts.

    Notes
    -----
    - Internally oversamples (default 10,000 samples) to ensure enough valid values.
    - Useful for initializing GP kernel parameters with controlled ranges.
    - Ensures numerical stability by avoiding extreme values.

    Example
    -------
    >>> samples = rand_lognormal_truncated(
    ...     scale=60.0,
    ...     stddev=0.4,
    ...     lower=10.0,
    ...     upper=150.0,
    ...     size=1
    ... )
    >>> print(samples)
    """
    mean_log = np.log(scale)
    dist = lognorm(s=stddev, scale=np.exp(mean_log))
    
    max_attempts = 10000  # Oversample internally

    # Try until we get at least `size` valid samples
    samples = dist.rvs(size=max_attempts)
    valid_samples = samples[(samples >= lower) & (samples <= upper)]
    
    if len(valid_samples) < size:
        raise ValueError(f"Not enough valid samples found! Only {len(valid_samples)} in {max_attempts} trials.")
    
    return valid_samples[:size]

# =========================
# 5.7 Kernel Builder
# =========================
def kernel_opt_new(kernel_type="M32", month=1, opt=False, opt_prior=False):
    dtype = tf.float64

    # --- Generate all parameters randomly ---
    # Sub-daily periodic kernel
    variance_p_subday = rand_lognormal_truncated(scale=1.0, stddev=0.5, lower=0.1, upper=5.0, size=1)[0]
    lengthscale_p_subday = rand_lognormal_truncated(scale=60.0, stddev=0.4, lower=10.0, upper=150.0, size=1)[0]
    period_p_subday = rand_lognormal_truncated(scale=600.0, stddev=0.2, lower=240.0, upper=720.0, size=1)[0]

    # Daily periodic kernel
    variance_p_day = rand_lognormal_truncated(scale=1.0, stddev=0.3, lower=0.1, upper=3.0, size=1)[0]
    lengthscale_p_day = rand_lognormal_truncated(scale=150.0, stddev=0.3, lower=80.0, upper=300.0, size=1)[0]
    
#     period_p_day = rand_lognormal_truncated(scale=600.0, stddev=0.2, lower=600.0, upper=1440.0, size=1)[0]
    period_p_day = rand_lognormal_truncated(
        scale=720.0,       # median around 12h
        stddev=0.6,        # allows more spread
        lower=240.0,       # 4 hours
        upper=1440.0,      # 24 hours
        size=1)[0]

    # Long-term trend (Matern 3/2)
    variance_m32 = rand_lognormal_truncated(scale=1.0, stddev=0.5, lower=0.1, upper=10.0, size=1)[0]
    lengthscale_m32 = rand_lognormal_truncated(scale=300.0, stddev=0.5, lower=60.0, upper=1440.0, size=1)[0]

    # --- Construct kernels ---
    k_trend = gpflow.kernels.Matern32(variance=variance_m32, lengthscales=lengthscale_m32)

    m32_base = gpflow.kernels.Matern32(variance=variance_p_subday, lengthscales=lengthscale_p_subday)
    k_subdaily_m32 = gpflow.kernels.Periodic(base_kernel=m32_base, period=period_p_subday)

    se_sub_base = gpflow.kernels.SquaredExponential(variance=variance_p_subday, lengthscales=lengthscale_p_subday)
    k_subdaily_se = gpflow.kernels.Periodic(base_kernel=se_sub_base, period=period_p_subday)

    se_base = gpflow.kernels.SquaredExponential(variance=variance_p_day, lengthscales=lengthscale_p_day)
    
    k_daily = gpflow.kernels.Periodic(base_kernel=se_base, period=1440.0)
    # Fix daily period to enforce physical interpretability (24-hour cycle)
    gpflow.utilities.set_trainable(k_daily.period, False)

    kernel_dict = {
        "M32": k_trend,
        "P_m32": k_subdaily_m32,
        "P_sub": k_subdaily_se,
        "P_daily": k_daily
    }

    allowed_names = set(kernel_dict.keys())
    safe_locals = {k: kernel_dict[k] for k in allowed_names}

    try:
        base_kernel = eval(kernel_type, {"__builtins__": {}}, safe_locals)
    except Exception as e:
        raise ValueError(f"Invalid kernel_type expression: {kernel_type}\n{e}")

    # --- Set priors if specified ---
    if opt_prior:
        if "P_m32" in kernel_type:
            k_subdaily_m32.base_kernel.lengthscales.prior = tfd.LogNormal(tf.math.log(tf.constant(20.0, dtype=dtype)), scale=0.5)
            k_subdaily_m32.period.prior = tfd.LogNormal(tf.math.log(tf.constant(480.0, dtype=dtype)), scale=0.4)

        if "P_sub" in kernel_type:
            k_subdaily_se.base_kernel.lengthscales.prior = tfd.LogNormal(tf.math.log(tf.constant(20.0, dtype=dtype)), scale=0.5)
            k_subdaily_se.period.prior = tfd.LogNormal(tf.math.log(tf.constant(480.0, dtype=dtype)), scale=0.3)

        if "P_daily" in kernel_type:
            k_daily.base_kernel.variance.prior = tfd.LogNormal(tf.math.log(tf.constant(3.0, dtype=dtype)), scale=0.5)
            k_daily.base_kernel.lengthscales.prior = tfd.LogNormal(tf.math.log(tf.constant(60.0, dtype=dtype)), scale=0.5)

    return base_kernel

# =========================
# 6.2 Utility Functions
# =========================
def get_param_values(param_dict):
    return {k: float(v.numpy()) for k, v in param_dict.items()}

def evaluate_gp_model(model, X, Y_true):
    mean, _ = model.predict_f(X)
    preds = mean.numpy().flatten()
    true = Y_true.flatten()
    return {
        "R2": r2_score(true, preds),
        "RMSE": np.sqrt(mean_squared_error(true, preds)),
        "LML": -model.training_loss().numpy()
    }

def is_valid_value(x):
    return np.all(np.isfinite(x)) and not np.any(np.isnan(x))

# Evaluate model sensitivity to parameter changes (robustness analysis)
def perturb_and_evaluate(best_model, X, Y, perturb_pct=0.1):
    results = []
    param_dict = gpflow.utilities.parameter_dict(best_model)
    original_values = {k: v.numpy().copy() for k, v in param_dict.items()}

    for name, orig_value in original_values.items():
        for factor in [1 - perturb_pct, 1 + perturb_pct]:
            for k, v in original_values.items():
                try:
                    param_dict[k].assign(v)
                except Exception as e:
                    print(f"⚠️ Could not reset {k}: {e}")

            perturbed_value = np.clip(orig_value * factor, 1e-6, 1e3)

            if not is_valid_value(perturbed_value):
                print(f"⚠️ Skip assigning {name}: value={perturbed_value} (invalid)")
                continue

            try:
                param_dict[name].assign(perturbed_value)
                print(f"✅ Perturbed {name}: {orig_value} → {perturbed_value}")
                metrics = evaluate_gp_model(best_model, X, Y)
                metrics.update({
                    "param": name,
                    "perturbed_value": perturbed_value,
                    "perturb_factor": factor
                })
                results.append(metrics)
            except Exception as e:
                print(f"⚠️ Error assigning {name}={perturbed_value}: {e}")

    return pd.DataFrame(results)

tolerance_history = []

# =========================
# 6.3 Convergence Monitoring
# =========================
#   Gradient norm is tracked during optimization to evaluate convergence behavior  
#   and ensure stable parameter estimation
def convergence_callback(model, step, variables, values):
    with tf.GradientTape() as tape:
        loss = model.training_loss()
    grads = tape.gradient(loss, model.trainable_variables)
    grads = [g for g in grads if g is not None]
    grad_norm = np.linalg.norm([tf.norm(g).numpy() for g in grads])
    tolerance_history.append(grad_norm)

    if step % 100 == 0 or step == 999:
        print(f"🌀 Iter {step:3d} | Grad Norm: {grad_norm:.4e}")

# =========================
# 6.4 Main GP Training Function
# =========================
# Multi-start optimization helps avoid local minima in GP hyperparameter estimation
def multi_start_plot_kernel_method(
    kernel_fn,
    X,
    Y_train,
    Y,
    Y_sld1_ori,
    min_x,
    max_x,
    t,
    st_date,
    noise_variance,
    optimise,
    TF,
    name,
    name2,
    pic_name,
    save_opt,
    method,
    num_initializations=5,
    opt_inter=1000,
    mean_type="fixed_constant",
    kernel_type="SE"
    ):
    """
    Run multi-start Gaussian Process training.

    Selects the best model based on Log Marginal Likelihood (LML).

    Returns:
        tuple:
        (gp_predictions, seeds, lmls, best_metrics,
         best_lml, best_init_params, best_opt_params, best_model)
    """

    best_lml = -np.inf
    best_model = None
    gp_predictions = None
    best_seed = None
    best_kernel_init_params = None
    best_metrics = None

    seeds, lmls, r2s, rmses, log_records = [], [], [], [], []

    for i in range(max(1, num_initializations)):
        tf.random.set_seed(i)
        np.random.seed(i)
        print(f"\n🔁 Trial {i+1}/{num_initializations} — Seed: {i}")

        kernel = kernel_fn() if callable(kernel_fn) else kernel_fn

        if mean_type == "zero":
            mean_function = gpflow.mean_functions.Zero()
        elif mean_type == "constant":
            mean_function = gpflow.mean_functions.Constant()
        elif mean_type == "fixed_constant":
            mean_function = gpflow.mean_functions.Constant(c=0.0)
            set_trainable(mean_function.c, False)
        else:
            raise ValueError("mean_type must be 'zero', 'constant', or 'fixed_constant'")

        print("📌 Initial kernel parameters:")
        print(get_param_values(gpflow.utilities.parameter_dict(kernel)))
        print("📌 Initial mean function parameters:")
        print(get_param_values(gpflow.utilities.parameter_dict(mean_function)))

        X_, Y_train_ = prepare_data(X.flatten(), Y_train.values.flatten())
        Y_transformed = transform_y(Y, method)
        Y_train_transformed = transform_y(Y_train_, method)
        Y_sld1_transformed = transform_y(Y_sld1_ori, method)

        model = GPR(data=(X_, Y_train_transformed), kernel=kernel, mean_function=mean_function)
        model.likelihood.variance.assign(noise_variance)
        set_trainable(model.likelihood.variance, TF)

        init_params = get_param_values(gpflow.utilities.parameter_dict(kernel))
        init_params.update(get_param_values(gpflow.utilities.parameter_dict(mean_function)))
        init_params["likelihood.variance"] = float(model.likelihood.variance.numpy())

        tolerance_history.clear()
        opt_result = None

        if optimise:
            opt_result = gpflow.optimizers.Scipy().minimize(
                model.training_loss,
                model.trainable_variables,
                options={'ftol': 1e-9, 'gtol': 1e-6, 'maxiter': opt_inter},
                step_callback=partial(convergence_callback, model)
            )
            print("\n✅ Optimization completed.")
            print(f"📉 Final negative log marginal likelihood (LML): {-opt_result.fun:.6f}")
            print(f"🔁 Converged: {opt_result.success}")
            print(f"📋 Final message: {opt_result.message}")
            print(f"🗪 Final gradient norm (||grad||): {np.linalg.norm(opt_result.jac):.6e}")
            print(f"✅ Final gradient norm: {tolerance_history[-1]:.4e}")
        else:
            print("\n⚠️ Optimization skipped (optimise=False). Using initial parameters only.")

        opt_params = get_param_values(gpflow.utilities.parameter_dict(model.kernel))
        opt_params.update(get_param_values(gpflow.utilities.parameter_dict(model.mean_function)))
        opt_params["likelihood.variance"] = float(model.likelihood.variance.numpy())

        lml = -model.training_loss().numpy()
        metrics = evaluate_gp_model(model, X_, Y_train_transformed)
        r2 = metrics["R2"]
        rmse = metrics["RMSE"]

        for param_name in init_params:
            log_records.append({
                "seed": i,
                "param": param_name,
                "init_value": init_params[param_name],
                "optimized_value": opt_params.get(param_name, None),
                "log_marginal_likelihood": lml,
                "R2": r2,
                "RMSE": rmse
            })

        print(f"📈 Log Marginal Likelihood: {lml:.4f} — R²: {r2:.4f}, RMSE: {rmse:.4f}")
        seeds.append(i)
        lmls.append(lml)
        r2s.append(r2)
        rmses.append(rmse)
        
        # Select model with highest log marginal likelihood (best fit to data)
        #The best model is selected based on the highest Log Marginal Likelihood (LML),
        #which reflects the model's ability to explain the observed data.
        if lml > best_lml:
            best_lml = lml
            best_model = model
            best_seed = i
            best_kernel_init_params = init_params
            best_metrics = metrics
            X_best = X_
            Y_best = Y_transformed
            Y_sld1_best = Y_sld1_transformed
            best_opt_params = opt_params  

        if num_initializations == 1:
            break

    log_df = pd.DataFrame(log_records)
    log_df.to_csv(f"{name}_gp_parameter_log_training_{kernel_type}.csv", index=False)
    print("📄 Saved parameter log to CSV")

    print(f"\n✅ Best model found at seed {best_seed} with log marginal likelihood: {best_lml:.4f}")
    print(f"🌟 R²: {best_metrics['R2']:.4f}, RMSE: {best_metrics['RMSE']:.4f}")
    print("🔍 Best kernel initialization parameters:")
    print(best_kernel_init_params)
    gpflow.utilities.print_summary(best_model)

    gp_predictions = plot_prediction_method(best_model, X_best, Y_best, Y_sld1_best,
                                       min_x, max_x, t, st_date,
                                       method, name, name2,
                                       pic_name, kernel_type, save_opt)


    if num_initializations > 1:
        plt.plot(seeds, r2s, label='R²', marker='o')
        plt.plot(seeds, rmses, label='RMSE', marker='x')
        plt.xlabel('Seed (Trial #)')
        plt.ylabel('Metric Value')
        plt.title('R² and RMSE across Multi-start Trials')
        plt.legend()
        plt.grid(True)
        plt.show()

    plt.plot(seeds, lmls, marker='o')
    plt.xlabel('Seed (Trial #)')
    plt.ylabel('Log Marginal Likelihood')
    plt.title('LML across Multi-start Trials')
    plt.grid(True)
    plt.show()
    return (
        gp_predictions,
        seeds,
        lmls,
        best_metrics,
        best_lml,
        best_kernel_init_params,
        opt_params,  # Optimized parameters of the best model 
        best_model   # Full GP model object if needed for downstream use
    )

# =========================
# 6.5 Utility Function
# =========================

def ensure_numpy_1d(array):
    """
    Convert input array (NumPy / TensorFlow) to a flattened 1D NumPy array.
    """
    if hasattr(array, "numpy"):
        array = array.numpy()
    return np.ravel(array)


# =========================
# 6.6 Main GP Processing Function
# =========================

def process_gp_model(
    train_duration,
    test_duration,
    t_duration,
    start_dates,
    end_dates,
    month,
    PM25_data,
    Smooth_PM25,
    num_int,
    opt_inter,
    method,
    mean_opt,
    set_opt,
    set_prior,
    fig_name,
    kernel_model,
    output_path="gp_results_update"
):
    """
    Run Gaussian Process (GP) training and prediction.

    Returns
    -------
    tuple
        (fmean_train, fvar_train, fmean_test, fvar_test,
         val_true, val_sld1,
         best_metrics, best_lml,
         best_init_params, best_opt_params, best_model)
    """

    from datetime import datetime
    import numpy as np
    import os

    # =========================
    # 1. Initialize placeholders
    # =========================
    fmean_train_ = np.full((train_duration[0],), np.nan)
    fvar_train_  = np.full((train_duration[0],), np.nan)
    fmean_test_  = np.full((test_duration[0],), np.nan)
    fvar_test_   = np.full((test_duration[0],), np.nan)

    val_true_ = np.full((11,), np.nan)
    val_sld1_ = np.full((11,), np.nan)

    best_metrics, best_lml = {}, np.nan
    best_init_params, best_opt_params = {}, {}
    best_model = None

    Day = np.arange(0, train_duration[0] * 15, 15).reshape(-1, 1)

    # =========================
    # 2. Loop over days
    # =========================
    for d, (start_date, end_date) in enumerate(zip(start_dates, end_dates)):
        print(f"🛠️ Processing Day Index: {start_date}")

        for t in t_duration:
            for train in train_duration:
                for p1 in test_duration:

                    k = start_date

                    # =========================
                    # 3. Data Indexing
                    # =========================
                    train_idx = (((k) * 96) + (t * 4)) - train
                    test_idx  = (((k) * 96) + (t * 4)) + p1

                    Y_train = PM25_data[train_idx : train_idx + train]
                    Y       = PM25_data[train_idx : test_idx]
                    Y_sld1  = Smooth_PM25[train_idx : test_idx]

                    # =========================
                    # 4. Kernel + Naming
                    # =========================
                    kernel_fn = lambda: kernel_opt_new(
                        opt_prior=set_prior,
                        kernel_type=kernel_model,
                        month=month
                    )

                    month_names = {1: "Jan", 4: "Apr", 7: "Jul", 10: "Oct"}
                    cur_month = month_names.get(month, "Month")

                    if month == 1:
                        day_val = k + 1
                    elif month == 4:
                        day_val = k - 90 + 1
                    elif month == 7:
                        day_val = k - 181 + 1
                    else:
                        day_val = k - 273 + 1

                    name = f"{day_val}_{cur_month}_tr{int(train*15/60)}h_te{int(p1*15/60)}h_t{t}"
                    name2 = f"{cur_month} {day_val}, Forecast: {int(p1*15/60)}h from {t}:00"

                    # =========================
                    # 5. Output Directory
                    # =========================
                    abs_output_dir = os.path.abspath(output_path)
                    os.makedirs(abs_output_dir, exist_ok=True)

                    full_pic_path = os.path.join(abs_output_dir, f"plot_{name}")

                    # =========================
                    # 6. Run GP
                    # =========================
                    try:
                        results = multi_start_plot_kernel_method(
                            kernel_fn=kernel_fn,
                            X=Day[:train],
                            Y_train=Y_train,
                            Y=Y,
                            Y_sld1_ori=Y_sld1,
                            min_x=0,
                            max_x=train + p1,
                            t=t,
                            st_date=k,
                            noise_variance=0.03,
                            optimise=set_opt,
                            TF=True,
                            name=name,
                            name2=name2,
                            pic_name=full_pic_path,
                            save_opt=True,
                            method=method,
                            num_initializations=num_int,
                            mean_type=mean_opt,
                            kernel_type=kernel_model
                        )

                        (
                            gp_preds,
                            seeds,
                            lmls,
                            b_metrics,
                            b_lml,
                            b_init,
                            b_opt,
                            b_model
                        ) = results

                        (
                            fmean_train_,
                            fvar_train_,
                            fmean_test_,
                            fvar_test_,
                            val_true_,
                            val_sld1_
                        ) = gp_preds

                        best_metrics = b_metrics
                        best_lml = b_lml
                        best_init_params = b_init
                        best_opt_params = b_opt
                        best_model = b_model

                    except Exception as e:
                        print(f"❌ GP failed for {name}: {e}")
                        continue

    return (
        fmean_train_,
        fvar_train_,
        fmean_test_,
        fvar_test_,
        val_true_,
        val_sld1_,
        best_metrics,
        best_lml,
        best_init_params,
        best_opt_params,
        best_model
    )