# Runner script for full training pipeline for 1D-Conv

import subprocess

print(f"Starting 1D-Conv Full Run; DO NOT ADJUST CONFIG WHILE RUNNING.")
subprocess.run(["python", "All-DataGenerate.py"])
subprocess.run(["python", "1D-Conv-PreTrain.py"])
subprocess.run(["python", "1D-Conv-MetaTrain.py"])
subprocess.run(["python", "1D-Conv-FineTune.py"])