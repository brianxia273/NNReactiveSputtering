# Runner script for full training pipeline for both FCNN and 1D-Conv

import subprocess

print(f"Starting Full Run; DO NOT ADJUST CONFIG WHILE RUNNING.")
subprocess.run(["python", "All-DataGenerate.py"])
subprocess.run(["python", "All-PreTrain.py"])
subprocess.run(["python", "All-MetaTrain.py"])
subprocess.run(["python", "All-FineTune.py"])