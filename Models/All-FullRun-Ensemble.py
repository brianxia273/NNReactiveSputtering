# Runner script for full training pipeline for both FCNN and 1D-Conv

import subprocess
import time
import roundConfig

t = time.time()
print(f"Starting Full Run Ensemble; DO NOT ADJUST CONFIG WHILE RUNNING.")
subprocess.run(["python", "All-DataGenerate.py"])

initialRound = curRound = roundConfig.mRound

for _ in range(7):
    subprocess.run(["python", "All-PreTrain.py"])

    subprocess.run(["python", "1D-Conv-MetaTrain.py"])
    subprocess.run(["python", "FCNN-MetaTrain.py"])

    subprocess.run(["python", "All-FineTune.py"])

    curRound += 1
    with open("roundConfig.py", "w") as f:
        f.write(f"mRound: int = {curRound}")

with open("roundConfig.py", "w") as f:
    f.write(f"mRound: int = {initialRound}")

print(f"Elapsed time: {t - time.time()}")