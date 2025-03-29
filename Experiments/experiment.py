import argparse
import os
import random
import subprocess
import time

energiBridge_path = '/Users/ahmeddriouech/Desktop/EnergiBridge'
repo = '/Users/ahmeddriouech/Desktop/sustainable-2'
BASE_DIR = '/Users/ahmeddriouech/Desktop/sustainable-2/Experiments/Results/'

def run_experiment(command):
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
    models = {
           "RandomForest/rf.py": "rf",
           "NeuralNetwork/nn.py": "nn",
           "llm/finetune_llm.py": "llm"
              }

    images = ["fedora-base", "ubuntu-base", "debian-base",
             "fedora-cpu-optimized", "ubuntu-cpu-optimized", "debian-cpu-optimized"]

    for model_path, model_name in models.items():
        for i in range(31):
            #First run is warmup
            randomize = images.copy()
            if i > 0:
                random.shuffle(randomize)

            for image in randomize:
                time.sleep(60)
                docker_query = f'docker run -it --rm -v "$(pwd):/app/repo" taoufikel/ml-energy-{image}:latest python /app/repo/{model_path}'
                full_temp_path = f'{BASE_DIR}{model_name}_{image}_{i}.csv'
                command = f'{energiBridge_path}/target/release/energibridge -o {full_temp_path} --summary {docker_query}'
                run_experiment(command)



if __name__ == "__main__":
    main()

