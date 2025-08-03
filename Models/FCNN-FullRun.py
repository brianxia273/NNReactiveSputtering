# Runner script for full training pipeline for FCNN

import subprocess

print(f"Starting FCNN Full Run; DO NOT ADJUST CONFIG WHILE RUNNING.")
subprocess.run(["python", "All-DataGenerate.py"])
subprocess.run(["python", "FCNN-PreTrain.py"])
subprocess.run(["python", "FCNN-MetaTrain.py"])
subprocess.run(["python", "FCNN-FineTune.py"])