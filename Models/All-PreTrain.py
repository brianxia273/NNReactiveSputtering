# Runner script for FCNN and 1D-Conv PreTrain scripts
import subprocess

print(f"Starting PreTrain; DO NOT ADJUST CONFIG P2 WHILE RUNNING")
print("Starting FCNN PreTrain")
subprocess.run(["python", "FCNN-PreTrain.py"])
print("Starting 1D-Conv PreTrain")
subprocess.run(["python", "1D-Conv-PreTrain.py"])
print("Finished all PreTrain!")
