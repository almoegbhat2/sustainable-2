# Comparing Energy Consumption of Docker Images for Machine Learning Workloads

## Overview

This repository contains a Python project developed for the **Sustainable Software Engineering** course at **Delft University of Technology**. The project explores whether different Docker images (e.g., base vs. CPU-optimized) have a significant impact on the power usage of typical machine learning pipelines.

## Features

- **Automated ML Runs**: Multiple scripts (`rf.py`, `nn.py`, `llm.py`) launch machine learning experiments in Docker containers.
- **Energy Measurement**: Utilizes **EnergiBridge** to measure and record real-time power consumption, generating CSV logs per experiment.
- **Data Analysis**: Detects outliers, computes summary statistics, and performs significance tests to determine whether image types differ in energy usage.
- **Visualization**: Generates **violin plots** and **time-series graphs** to display energy and power consumption trends.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/ahsmi47/SSE-project2.git
cd SSE-project2
```

### 2. Install Python Dependencies

Make sure you have Python 3.10+ installed. Then install required packages:

```bash
pip install -r requirements.txt
```

Also make sure:

- You have **Docker** installed and running.
- You have **EnergiBridge** installed and accessible from the command line.

## Docker Images

You can pull all required Docker images from Docker Hub:

```bash
docker pull taoufikel/ml-energy-ubuntu-base:latest
docker pull taoufikel/ml-energy-ubuntu-cpu-optimized:latest
docker pull taoufikel/ml-energy-fedora-base:latest
docker pull taoufikel/ml-energy-fedora-cpu-optimized:latest
docker pull taoufikel/ml-energy-debian-base:latest
docker pull taoufikel/ml-energy-debian-cpu-optimized:latest
```

👉 These are available at: [https://hub.docker.com/u/taoufikel](https://hub.docker.com/u/taoufikel)

If you prefer to build them locally, use the Dockerfiles under the `docker/` directory.

## Usage

### 1. Run the Experiment

This script runs 31 experiments for each combination of model and Docker image (random forest, neural net, language model):

```bash
python experiment.py
```

Make sure to set your system-specific EnergiBridge path and repo paths at the top of the script:

```python
energiBridge_path = '/your/path/to/EnergiBridge'
repo = '/your/path/to/this-repo'
BASE_DIR = '/your/path/to/Results/'
```

### 2. Analyze the Results

Once the experiment logs are saved in `results/`, run the analysis script:

```bash
python analysis.py
```

This will:

- Compute energy usage per run
- Remove outliers
- Conduct statistical tests
- Generate violin and time-series plots

Plots and summaries will be saved to the `output/` folder.

## Output

- **CSV Logs**: Each run logs its power usage to a separate `.csv` file in `results/`.
- **Visualizations**: Violin and time-series plots are generated per model in `output/`.
- **Statistics**: Console outputs include mean, variance, and significance tests for energy usage.

## Replication Package

Everything needed to replicate the results is included:

- `experiment.py`: Launches Docker containers and runs models with EnergiBridge logging.
- `analysis.py`: Processes logs, detects outliers, generates plots, and performs statistical tests.
- `rf.py`, `nn.py`, `llm.py`: Machine learning workloads tested across images.
- `results/`: Where energy measurements are logged (generated after running experiments).
- `output/`: Where plots and summary statistics are stored.

## License

This project is **open-source** and **license-free**. Feel free to use, modify, and improve it as needed.

## Authors

Ahmed Driouech, Ahmed Ibrahim, Taoufik el Kadi, Moegiez Bhatti  
**April 4, 2025**
