import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Config
RESULTS_DIR = "results"
OUTPUT_DIR = "output"
ALGORITHMS = ["llm", "nn", "rf"]
IMAGES = ["debian", "ubuntu", "fedora"]
RUNS = 31
Z_THRESHOLD = 3.0

os.makedirs(OUTPUT_DIR, exist_ok=True)

def detect_runs(df):
    time_col = df.columns[1]
    df[time_col] = pd.to_numeric(df[time_col], errors="coerce")
    df["time_diff"] = df[time_col].diff()
    df["run_number"] = (df["time_diff"] > 60000).cumsum()
    df.drop(columns=["time_diff"], inplace=True)
    return df

def load_data(filepath):
    df = pd.read_csv(filepath)
    df = detect_runs(df)
    return df

def compute_energy_per_run(df):
    df["energy"] = (df.iloc[:, 0] / 1000) * df["SYSTEM_POWER (Watts)"]
    return df.groupby("run_number")["energy"].sum()

def detect_outliers(energy_series, threshold=Z_THRESHOLD):
    z_scores = (energy_series - energy_series.mean()) / energy_series.std()
    return energy_series[np.abs(z_scores) > threshold].index.tolist()

def check_normality(series):
    if len(series) >= 8:
        _, p = stats.normaltest(series)
        return p > 0.05
    return False

def significance_test(g1, g2, n1, n2):
    if n1 and n2:
        stat, p = stats.ttest_ind(g1, g2, equal_var=False)
        return "Welch’s t-test", stat, p
    else:
        stat, p = stats.mannwhitneyu(g1, g2)
        return "Mann-Whitney U test", stat, p

def plot_violin(data, labels, filename):
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
    ax.set_title("Energy Consumption per Run")
    ax.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/{filename}", dpi=300)
    plt.close()

# Main logic
results = {}

for algo in ALGORITHMS:
    print(f"\n=== {algo.upper()} ===")
    algo_data = {}
    algo_data_no_outliers = {}

    for image in IMAGES:
        energies = []

        for i in range(RUNS):
            filename = f"{algo}_{image}_{i}.csv"
            filepath = os.path.join(RESULTS_DIR, filename)
            if not os.path.exists(filepath):
                print(f"Missing: {filepath}")
                continue

            try:
                df = load_data(filepath)
                energy = compute_energy_per_run(df)
                energies.extend(energy.tolist())
            except Exception as e:
                print(f"Error in {filepath}: {e}")

        series = pd.Series(energies)
        outliers = detect_outliers(series)
        series_no_outliers = series.drop(outliers)

        algo_data[image] = series
        algo_data_no_outliers[image] = series_no_outliers

        print(f"{image.capitalize()} - With Outliers: n={len(series)}, Mean={series.mean():.2f}, Var={np.var(series):.2f}")
        print(f"{image.capitalize()} - Without Outliers: n={len(series_no_outliers)}, Mean={series_no_outliers.mean():.2f}, Var={np.var(series_no_outliers):.2f}")
        print(f"Outliers Detected: {len(outliers)}")

    results[algo] = (algo_data, algo_data_no_outliers)

    # Violin Plots
    labels = [img.capitalize() for img in IMAGES]
    plot_violin([algo_data[img] for img in IMAGES], labels, f"{algo}_violin_with_outliers.png")
    plot_violin([algo_data_no_outliers[img] for img in IMAGES], labels, f"{algo}_violin_without_outliers.png")

    # Statistical Tests
    for i in range(len(IMAGES)):
        for j in range(i + 1, len(IMAGES)):
            img1, img2 = IMAGES[i], IMAGES[j]
            s1 = algo_data_no_outliers[img1]
            s2 = algo_data_no_outliers[img2]

            if s1.empty or s2.empty:
                continue

            normal1 = check_normality(s1)
            normal2 = check_normality(s2)

            test, stat, p = significance_test(s1, s2, normal1, normal2)
            print(f"{img1} vs {img2}: {test} | p = {p:.4f}")

print("\nAnalysis complete. Results saved in the 'output/' directory.")
