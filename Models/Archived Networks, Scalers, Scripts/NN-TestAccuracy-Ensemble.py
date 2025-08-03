# Testing Fine-Tuned Neural Network Model's accuracy
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
import config
import os
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib

def extractState(file):
    start = file.find("Random_")
    state = ""
    for c in file[start+7:]:
        if c.isdigit():
            state += c
        else:
            break
    if not state:
        raise ValueError(f"State not found: {file}")
    return state

# Select from config
data = config.p5Data
nn = config.p5NN
nnEpoch = config.p5NNEpoch
nnBatch = config.p5NNBatch
augmentedDataCount = config.p5N
output = config.p5Output
outputCols = config.p0OutputCols
ensembleRandomSeed = config.p5EnsembleRandom

# Set np randomSeed
np.random.seed(ensembleRandomSeed)

# Import all NNs
ensemble = "" # Make script to compile models into ensemble folder
ensembleScalers = ""
modelPath = os.path.join("../Fine-Tuned Neural Networks", data, output, nn, ensemble)
scalerPath = os.path.join("../Data Scalers", data, output, nn, ensembleScalers)

models = [f for f in os.listdir(modelPath) if f.endswith('.keras')]
scalers = [f for f in os.listdir(scalerPath) if f.endswith('.pkl')]
stateMap = dict()
for file in models:
    state = extractState(file)
    stateMap[state] = [file]
for file in scalers:
    state = extractState(file)
    if state not in stateMap:
        raise KeyError(f"Missing mlModel for corresponding mlScaler: {file}")
    stateMap[state].append(file)

# Import Real Data CSV file
df = pd.read_csv(os.path.join("../Datasets", data))
x = df.drop(columns=outputCols).values
y = df[output].values  # Selecting output

# Train/Test split
trainSize = int(0.8 * len(x))

# Result map
resultMap = dict()

for state in stateMap:
    pair = stateMap[state]
    mlModel, dataScaler = load_model(os.path.join(modelPath, pair[0])), joblib.load(os.path.join(scalerPath, pair[1]))
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, train_size=trainSize, random_state=state)
    xTestLog = np.log1p(xTest)
    xTestScaled = dataScaler.transform(xTestLog)

    # Check predictions
    yPredict = mlModel.predict(xTestScaled)
    mse = mean_squared_error(yTest, yPredict)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(yTest, yPredict)
    ev = explained_variance_score(yTest, yPredict)
    r2 = r2_score(yTest, yPredict)
    result = {
        "model": mlModel,
        "mse": mse,
        "rmse": rmse,
        "mape": mape,
        "ev": ev,
        "r2": r2
    }
    if state not in resultMap:
        resultMap[state] = result
    else:
        raise KeyError(f"State already exists in resultMap: {state}")

mseValues = [resultMap[s]['mse'] for s in resultMap]
rmseValues = [resultMap[s]['rmse'] for s in resultMap]
mapeValues = [resultMap[s]['mape'] for s in resultMap]
evValues = [resultMap[s]['ev'] for s in resultMap]
r2Values = [resultMap[s]['r2'] for s in resultMap]

mseMean = np.mean(mseValues)
rmseMean = np.mean(rmseValues)
mapeMean = np.mean(mapeValues)
evMean = np.mean(evValues)
r2Mean = np.mean(r2Values)

mseStd = np.std(mseValues)
rmseStd = np.std(rmseValues)
mapeStd = np.std(mapeValues)
evStd = np.std(evValues)
r2Std = np.std(r2Values)

lambdaUncertainty = 0.1
ucb = mseMean + lambdaUncertainty * mseStd

print(f"Model Ensemble Evaluation Metrics:")
print("Average Values")
print(f"MSE: {mseMean}, Upper Confidence Bound: {ucb:.4f}")
print(f"RMSE: {rmseMean}")
print(f"MAPE: {mapeMean}")
print(f"EV: {evMean}")
print(f"R^2: {r2Mean}")

print(f"Max Values")
print(f"MSE: {max(mseValues)}")
print(f"RMSE: {max(rmseValues)}")
print(f"MAPE: {max(mapeValues)}")
print(f"EV: {max(evValues)}")
print(f"R^2: {max(r2Values)}")

print(f"Min Values")
print(f"MSE: {min(mseValues)}")
print(f"RMSE: {min(rmseValues)}")
print(f"MAPE: {min(mapeValues)}")
print(f"EV: {min(evValues)}")
print(f"R^2: {min(r2Values)}")

# Implement plot
