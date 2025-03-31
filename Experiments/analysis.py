import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.interpolate import interp1d

# Configuration
RESULTS_DIR = "results"
OUTPUT_DIR = "output"
MODELS = ["llm", "nn", "rf"]
OS_LIST = ["ubuntu", "fedora", "debian"]
IMAGE_TYPES = ["base", "cpu-optimized"]
RUNS = 31
IQR_MULTIPLIER = 1.5
TIME_GRID_POINTS = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Color mapping for operating systems and line style mapping for image types.
os_colors = {
    "ubuntu": "blue",
    "fedora": "green",
    "debian": "red"
}
image_linestyles = {
    "base": "solid",
    "cpu-optimized": "dashed"
}

# ----- Simplified Data Loading and Energy Computation -----
def load_data(filepath):
    # Now, each file corresponds to one run, so we simply load it.
    return pd.read_csv(filepath)

def get_cumulative_time_and_power(df):
    """
    Computes cumulative time from the first column (time intervals in ms),
    converts the time to seconds, and extracts power usage from the
    "SYSTEM_POWER (Watts)" column.
    Returns (cumulative_time_in_sec, power_usage).
    """
    cum_time_ms = np.cumsum(df.iloc[:, 0].values)
    # Convert milliseconds to seconds.
    cum_time = cum_time_ms / 1000.0
    power = df["SYSTEM_POWER (Watts)"].values
    return cum_time, power

def compute_energy(filepath):
    # Load data and compute energy consumption for the run.
    df = load_data(filepath)
    # Compute energy as (first_column / 1000) * SYSTEM_POWER (Watts) for all rows, then sum it up.
    df["energy"] = (df.iloc[:, 0] / 1000) * df["SYSTEM_POWER (Watts)"]
    return df["energy"].sum()


# ----- Outlier Detection using IQR -----
def detect_outliers(series, multiplier=IQR_MULTIPLIER):
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR
    return series[(series < lower_bound) | (series > upper_bound)].index.tolist()


# ----- Statistical Test Functions -----
def check_normality(series):
    if len(series) >= 8:
        _, p = stats.normaltest(series)
        return p > 0.05
    return False


def significance_test(g1, g2, normal1, normal2):
    if normal1 and normal2:
        stat, p = stats.ttest_ind(g1, g2, equal_var=False)
        return "Welch’s t-test", stat, p
    else:
        stat, p = stats.mannwhitneyu(g1, g2)
        return "Mann-Whitney U test", stat, p


# ----- Plotting Functions -----
def plot_violin(data, labels, title, filename):
    fig, ax = plt.subplots(figsize=(8, 6))
    parts = ax.violinplot(data, showmeans=False, showmedians=True)
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i % len(colors)])
        pc.set_edgecolor("black")
        pc.set_linewidth(1)
        pc.set_alpha(0.7)
    ax.boxplot(data, medianprops=dict(color="black", linewidth=2))
    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Energy Consumption (J)")
    ax.set_title(title)
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, filename), dpi=300)
    plt.close()


# ----- Time Series Aggregation Functions -----
def aggregate_time_series(model, os_name, image_type, runs=RUNS):
    all_times = []
    all_values = []
    for run in range(runs):
        filename = f"{model}_{os_name}-{image_type}_{run}.csv"
        filepath = os.path.join(RESULTS_DIR, filename)
        if not os.path.exists(filepath):
            continue
        try:
            df = load_data(filepath)
            time_col = df.columns[1]
            times = df[time_col].values
            times = times - times[0]  # align to 0
            # Here, we'll use computed energy as a constant time series since each file is one run.
            energy = compute_energy(filepath)
            values = np.full_like(times, energy, dtype=float)
            all_times.append(times)
            all_values.append(values)
        except Exception as e:
            print(f"Error processing {filepath}: {e}")
    if not all_times:
        return None, None
    min_max_time = min(times[-1] for times in all_times)
    common_grid = np.linspace(0, min_max_time, TIME_GRID_POINTS)
    interpolated_values = []
    for times, values in zip(all_times, all_values):
        try:
            f = interp1d(times, values, kind="linear", bounds_error=False, fill_value="extrapolate")
            interp_vals = f(common_grid)
            interpolated_values.append(interp_vals)
        except Exception as e:
            print("Interpolation error:", e)
    if interpolated_values:
        avg_values = np.mean(interpolated_values, axis=0)
        return common_grid, avg_values
    else:
        return None, None

def compute_average_time_series(model, os_name, image_type, runs=RUNS, grid_points=TIME_GRID_POINTS):
    """
    For a given model, OS, and image_type, this function:
    - Loads each run file.
    - Computes cumulative time (in seconds) and power usage.
    - Computes the mean power of each run and removes outlier runs (using IQR).
    - Interpolates each remaining run's power time series onto a common grid spanning 0 to the longest run.
      Shorter runs are extended by holding the last observed power value.
    - Averages the interpolated curves.
    Returns (common_time, avg_power_usage).
    """
    run_time_series = []
    run_power_series = []
    run_means = []

    for run in range(runs):
        filename = f"{model}_{os_name}-{image_type}_{run}.csv"
        filepath = os.path.join(RESULTS_DIR, filename)
        if not os.path.exists(filepath):
            continue
        try:
            df = load_data(filepath)
            t, power = get_cumulative_time_and_power(df)
            run_time_series.append(t)
            run_power_series.append(power)
            run_means.append(np.mean(power))
        except Exception as e:
            print(f"Error processing {filepath}: {e}")

    if not run_time_series:
        return None, None

    # Remove outlier runs based on mean power (IQR method)
    run_means_series = pd.Series(run_means)
    outlier_indices = detect_outliers(run_means_series)
    filtered_time_series = [run_time_series[i] for i in range(len(run_time_series)) if i not in outlier_indices]
    filtered_power_series = [run_power_series[i] for i in range(len(run_power_series)) if i not in outlier_indices]

    if not filtered_time_series:
        return None, None

    # Use the longest run among the filtered runs to define the common grid
    max_times = [t[-1] for t in filtered_time_series]
    common_max_time = max(max_times)
    common_grid = np.linspace(0, common_max_time, grid_points)

    interpolated_power = []
    for t, power in zip(filtered_time_series, filtered_power_series):
        try:
            # Extend shorter runs by carrying the last value forward.
            interp_func = interp1d(t, power, kind="linear", bounds_error=False, fill_value=(power[0], power[-1]))
            interp_power = interp_func(common_grid)
            interpolated_power.append(interp_power)
        except Exception as e:
            print("Interpolation error:", e)

    if interpolated_power:
        avg_power = np.mean(interpolated_power, axis=0)
        return common_grid, avg_power
    else:
        return None, None


def plot_model_power_time_series(model):
    """
    For the given model, creates one plot with six lines (one per OS–image_type combination).
    OS is distinguished by color and image type by line style. The x-axis is time in seconds and the y-axis
    shows average power usage (W).
    """
    plt.figure(figsize=(10, 7))
    for os_name in OS_LIST:
        for image_type in IMAGE_TYPES:
            t, avg_power = compute_average_time_series(model, os_name, image_type, runs=RUNS,
                                                       grid_points=TIME_GRID_POINTS)
            if t is None or avg_power is None:
                continue
            label = f"{os_name.capitalize()} - {image_type}"
            linestyle = image_linestyles[image_type]
            color = os_colors[os_name]
            plt.plot(t, avg_power, label=label, linestyle=linestyle, color=color)
    plt.xlabel("Time (s)")
    plt.ylabel("Average Power Usage (W)")
    plt.title(f"{model.upper()} - Average Power Usage Over Time")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{model}_power_timeseries.png"), dpi=300)
    plt.close()

# ----- Main Analysis -----
results = {}

for model in MODELS:
    model_data = {}
    print(f"\n=== {model.upper()} ===")

    for image_type in IMAGE_TYPES:
        type_data = {}
        for os_name in OS_LIST:
            energies = []
            for run in range(RUNS):
                filename = f"{model}_{os_name}-{image_type}_{run}.csv"
                filepath = os.path.join(RESULTS_DIR, filename)
                if not os.path.exists(filepath):
                    print(f"Missing: {filepath}")
                    continue
                try:
                    energy = compute_energy(filepath)
                    energies.append(energy)
                except Exception as e:
                    print(f"Error in {filepath}: {e}")
            series = pd.Series(energies)
            outliers = detect_outliers(series)
            series_no_outliers = series.drop(outliers)
            type_data[os_name] = {
                "with_outliers": series,
                "without_outliers": series_no_outliers
            }
            print(
                f"{os_name.capitalize()} ({image_type}) - With Outliers: n={len(series)}, Mean={series.mean():.2f}, Var={np.var(series):.2f}")
            print(
                f"{os_name.capitalize()} ({image_type}) - Without Outliers: n={len(series_no_outliers)}, Mean={series_no_outliers.mean():.2f}, Var={np.var(series_no_outliers):.2f}")
            print(f"Outliers Detected: {len(outliers)}")
        model_data[image_type] = type_data

        # Generate violin plots for energy per image type
        labels = [os.capitalize() for os in OS_LIST]
        plot_violin([model_data[image_type][os]["with_outliers"] for os in OS_LIST],
                    labels,
                    f"{model.upper()} {image_type} - Energy With Outliers",
                    f"{model}_{image_type}_energy_violin_with_outliers.png")
        plot_violin([model_data[image_type][os]["without_outliers"] for os in OS_LIST],
                    labels,
                    f"{model.upper()} {image_type} - Energy Without Outliers",
                    f"{model}_{image_type}_energy_violin_without_outliers.png")

    results[model] = model_data

    # Statistical tests for energy differences between OS (per image type)
    for image_type in IMAGE_TYPES:
        print(f"\nStatistical tests for {model.upper()} - {image_type} Energy (OS differences):")
        data_dict = {os: model_data[image_type][os]["without_outliers"] for os in OS_LIST}
        for i in range(len(OS_LIST)):
            for j in range(i + 1, len(OS_LIST)):
                os1 = OS_LIST[i]
                os2 = OS_LIST[j]
                s1 = data_dict[os1]
                s2 = data_dict[os2]
                if s1.empty or s2.empty:
                    continue
                normal1 = check_normality(s1)
                normal2 = check_normality(s2)
                test_name, stat, p = significance_test(s1, s2, normal1, normal2)
                print(f"{os1.capitalize()} vs {os2.capitalize()} ({image_type} Energy): {test_name} | p = {p:.4f}")

    # Statistical tests for energy differences between image types per OS
    print(f"\nStatistical tests for {model.upper()} - Energy (Image Type differences per OS):")
    for os_name in OS_LIST:
        s_base = model_data["base"][os_name]["without_outliers"]
        s_cpu = model_data["cpu-optimized"][os_name]["without_outliers"]
        if s_base.empty or s_cpu.empty:
            continue
        normal_base = check_normality(s_base)
        normal_cpu = check_normality(s_cpu)
        test_name, stat, p = significance_test(s_base, s_cpu, normal_base, normal_cpu)
        print(f"{os_name.capitalize()} (Base vs CPU-Optimized Energy): {test_name} | p = {p:.4f}")

# Aggregated Time Series Plots for Energy
for model in MODELS:
    plot_model_power_time_series(model)

print("\nAnalysis complete. Results saved in the 'output/' directory.")
