# Generating augmented data with BRR regression model

import numpy as np
import pandas as pd
import os
import config
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Select size, dataset, output, and randomState from config
data = config.p1Data
randomState = config.p1RandomState
model = "BRR"
augmentedDataCount = config.p1N
outputCols = config.p0OutputCols
output = config.p1Output
ensembleRandomSeed = config.p1EnsembleRandom

# Set np randomSeed
np.random.seed(ensembleRandomSeed)

# Automating file creation
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

# Init BRR model
brr = BayesianRidge()
brr.fit(xTrainScaled, yTrain)

# Interpolation
xMin = x.min(axis=0)
xMax = x.max(axis=0)
totalAugmentedX = augmentedDataCount
xAugmented = np.random.uniform(xMin, xMax, size=(totalAugmentedX, x.shape[1]))
xAugmentedLog = np.log1p(xAugmented)
xAugmentedScaled = dataScaler.transform(xAugmentedLog)
yAugmented = brr.predict(xAugmentedScaled)

inputColNames = df.drop(columns=outputCols).columns.tolist()
allColNames = inputColNames + [output]
dfCSV = pd.DataFrame(np.column_stack((xAugmented, yAugmented)), columns=allColNames)
csvName = f"{model} N_{augmentedDataCount} Random_{randomState} Augmented Data.csv"
saveDirectory = os.path.join(directory, csvName)
dfCSV.to_csv(saveDirectory, index=False)

print(f"Finished {csvName}!")
