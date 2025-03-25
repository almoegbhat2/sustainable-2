import argparse
import os
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
    parser = argparse.ArgumentParser()

    parser.add_argument("model_name")
    parser.add_argument("model_path")
    args = parser.parse_args()

    operating_systems = ["fedora", "ubuntu", "debian"]

    for i in range(31):
        for os in operating_systems:
            time.sleep(60)
            docker_query = f'docker run -it --rm -v "$(pwd):/app/repo" taoufikel/ml-energy-{os}-base:latest python /app/repo/{args.model_path}'
            full_temp_path = f'{BASE_DIR}{args.model_name}_{os}_{i}.csv'
            command = f'{energiBridge_path}/target/release/energibridge -o {full_temp_path} --summary {docker_query}'
            run_experiment(command)



if __name__ == "__main__":
    main()

