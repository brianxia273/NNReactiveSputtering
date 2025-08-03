# Generating augmented data with SVR regression model, with extrapolation for pre-training
# NOTE: MUST ADJUST SVR HYPERPARAMETERS FOR BEST PERFORMANCE
# NOTE: INTERPOLATED/EXTRAPOLATED DATA RATIO IS APPROXIMATED, NOT A PRECISE RATIO

import numpy as np
import pandas as pd
import os
import config
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib

# Select size, dataset, output, and randomState from config
data = config.p1Data
randomState = config.p1RandomState
model = "SVR"
extrapolationRange = config.p1SvrExtrapolationRange
augmentedDataCount = config.p1N
outputCols = config.p0OutputCols
output = config.p1Output
ensembleRandomSeed = config.p1EnsembleRandom

# Set np randomSeed
np.random.seed(ensembleRandomSeed)

directory = os.path.join("Regression Model Data and Metrics", data, output, model)
os.makedirs(directory, exist_ok=True)
df = pd.read_csv(os.path.join("Datasets", data))
x = df.drop(columns=outputCols).values
y = df[output].values  # Selecting output

# 80% data to train, 20% leave for testing. random_state is set in config
trainSize = int(0.8 * len(x))
xTrain, xTest, yTrain, yTest = train_test_split(x, y, train_size=trainSize, random_state=randomState)

# Scaling data
xTrainLog = np.log1p(xTrain)
xTestLog = np.log1p(xTest)
dataScaler = MinMaxScaler(feature_range=(-1, 1))
xTrainScaled = dataScaler.fit_transform(xTrainLog)
xTestScaled = dataScaler.transform(xTestLog)

# Init SVR model,
svr = SVR(kernel='rbf', C=5000.0, epsilon=0.5, gamma=1)  # ADJUST HYPERPARAMETERS HERE
svr.fit(xTrainScaled, yTrain)

# Interpolation and Extrapolation
xMin = x.min(axis=0)
xMax = x.max(axis=0)
xMin = xMin - extrapolationRange * (xMax - xMin)
xMax = xMax + extrapolationRange * (xMax - xMin)
totalAugmentedX = augmentedDataCount

# Scaling Data
xAugmented = np.random.uniform(xMin, xMax, size=(totalAugmentedX, x.shape[1]))
xAugmentedLog = np.log1p(xAugmented)
xAugmentedScaled = dataScaler.transform(xAugmentedLog)
yAugmented = svr.predict(xAugmentedScaled)

# Saving Augmented Data
inputColNames = df.drop(columns=outputCols).columns.tolist()
allColNames = inputColNames + [output]
dfCSV = pd.DataFrame(np.column_stack((xAugmented, yAugmented)), columns=allColNames)
csvName = f"{model} PreTrain N_{augmentedDataCount} Random_{randomState} Augmented Data.csv"
saveDirectory = os.path.join(directory, csvName)
dfCSV.to_csv(saveDirectory, index=False)

print(f"Finished {csvName}!")
