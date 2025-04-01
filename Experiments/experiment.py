"""
experiment.py

This script automates the execution of energy consumption experiments using Docker containers and EnergiBridge.
It performs multiple runs of machine learning models across different Docker images and logs their power usage.

Usage:
    python experiment.py

Before running:
- Set your paths for EnergiBridge, the repo, and results directory.
- Ensure EnergiBridge is compiled and accessible.
- Ensure all Docker images are available locally or via Docker Hub.
"""

import argparse
import os
import random
import subprocess
import time

# === CONFIGURATION ===
# Path to EnergiBridge binary
energiBridge_path = '/Users/ahmeddriouech/Desktop/EnergiBridge'

# Path to repo where Docker will mount and run ML workloads
repo = '/Users/ahmeddriouech/Desktop/sustainable-2'

# Output path for CSV energy logs
BASE_DIR = '/Users/ahmeddriouech/Desktop/sustainable-2/Experiments/Results/'

def run_experiment(command):
    """
    Executes a shell command to run EnergiBridge with a Docker container.
    Logs output and errors from the subprocess.
    """
    try:
        os.chdir(repo)
        print("Executing command...")
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            shell=True
        )
        stdout, stderr = process.communicate()
        if process.returncode == 0:
            print("Experiment completed successfully")
            if stdout:
                print("Output:", stdout)
            return True
        else:
            print(f"Experiment failed with return code {process.returncode}")
            if stderr:
                print("Error:", stderr)
            return False
    except Exception as e:
        print(f"Error running experiment: {str(e)}")
        return False
    finally:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))

def main():
    """
    Main loop that iterates over ML models and Docker image variants,
    performs multiple experimental runs, and logs results.
    """
    # Dictionary of ML scripts to be executed inside Docker containers
    models = {
        "RandomForest/rf.py": "rf",
        "NeuralNetwork/nn.py": "nn",
        "llm/finetune_llm.py": "llm"
    }

    # Docker image variants (base and CPU-optimized)
    images = [
        "fedora-base", "ubuntu-base", "debian-base",
        "fedora-cpu-optimized", "ubuntu-cpu-optimized", "debian-cpu-optimized"
    ]

    for model_path, model_name in models.items():
        for i in range(31):  # 31 runs per combination
            randomize = images.copy()

            # Shuffle images only after warm-up run
            if i > 0:
                random.shuffle(randomize)

            for image in randomize:
                time.sleep(60)  # Wait 60s between runs to reset baseline
                docker_query = f'docker run -it --rm -v "$(pwd):/app/repo" taoufikel/ml-energy-{image}:latest python /app/repo/{model_path}'
                full_temp_path = f'{BASE_DIR}{model_name}_{image}_{i}.csv'
                command = f'{energiBridge_path}/target/release/energibridge -o {full_temp_path} --summary {docker_query}'
                run_experiment(command)

if __name__ == "__main__":
    main()