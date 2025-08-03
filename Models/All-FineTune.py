# Runner script for FCNN and 1D-Conv FineTune scripts
import subprocess

print(f"Starting FineTune; DO NOT ADJUST CONFIG P4 WHILE RUNNING")
print("Starting FCNN FineTune")
subprocess.run(["python", "FCNN-FineTune.py"])
print("Starting 1D-Conv FineTune")
subprocess.run(["python", "1D-Conv-FineTune.py"])
print("Finished all FineTune!")
