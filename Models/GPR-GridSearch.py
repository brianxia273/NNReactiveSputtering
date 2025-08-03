# GridSearch and graph for best hyperparameters of GPR regression model for different randomStates

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
import config
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_percentage_error
import os

# Select size, dataset, output, and randomState from config
data = config.p1Data
randomState = config.p1RandomState
outputCols = config.p0OutputCols
output = config.p1Output
ensembleRandomSeed = config.p1EnsembleRandom

np.random.seed(ensembleRandomSeed)

# Selecting dataset columns
df = pd.read_csv(os.path.join("Datasets", data))
x = df.drop(columns=outputCols).values
y = df[output].values  # Selecting output

# 80% data to train, 20% leave for testing.
trainSize = int(0.8 * len(x))
xTrain, xTest, yTrain, yTest = train_test_split(x, y, train_size=trainSize, random_state=randomState)

# Scaling data
xTrainLog = np.log1p(xTrain)
xTestLog = np.log1p(xTest)
dataScaler = MinMaxScaler(feature_range=(-1, 1))
xTrainScaled = dataScaler.fit_transform(xTrainLog)
xTestScaled = dataScaler.transform(xTestLog)

# GridSearchCV finding optimal hyperparameters, with cross-validation
param_grid = {
    "kernel": [
        ConstantKernel(1.0) * Matern(length_scale=50, nu=1.5),
        # Matern kernel with length_scale=50, nu=1.5 (previously successful)
        ConstantKernel(1.0) * Matern(length_scale=75, nu=1.5),
        ConstantKernel(1.0) * Matern(length_scale=60, nu=1.5),
        ConstantKernel(1.0) * Matern(length_scale=40, nu=1.5),
        ConstantKernel(1.0) * Matern(length_scale=55, nu=3.0),
        ConstantKernel(1.0) * Matern(length_scale=50, nu=0.5),
        ConstantKernel(1.0) * Matern(length_scale=50, nu=2.5),
    ],
    "alpha": [1e-4, 1e-2, 1e-3],
    "n_restarts_optimizer": [5, 10, 20],
    "normalize_y": [True],
    "optimizer": ["fmin_l_bfgs_b"],
}
gridSearch = GridSearchCV(GaussianProcessRegressor(), param_grid, cv=5, scoring='neg_mean_squared_error',
                          n_jobs=-1)  # Optimizing with MSE
gridSearch.fit(xTrainScaled, yTrain)
print("Best GPR Parameters:", gridSearch.best_params_)
bestGPR = gridSearch.best_estimator_
trainScore = bestGPR.score(xTrainScaled, yTrain)
testScore = bestGPR.score(xTestScaled, yTest)
yTrainPredict = bestGPR.predict(xTrainScaled)
trainMSE = mean_squared_error(yTrain, yTrainPredict)
trainRMSE = np.sqrt(trainMSE)
print("Train Set MSE:", trainMSE)
print("Train Set RMSE:", trainRMSE)
print("Train Set Score (R^2):", trainScore)
print("Test Set Score (R^2):", testScore)

# GPR model making predictions
yPredict = bestGPR.predict(xTestScaled)
mseCurrent = mean_squared_error(yTest, yPredict)
rmseCurrent = np.sqrt(mseCurrent)
mapeCurrent = mean_absolute_percentage_error(yTest, yPredict)
evCurrent = explained_variance_score(yTest, yPredict)
currentModelScore = bestGPR.score(xTestScaled, yTest)
print("Current Model Dataset:", data)
print("Random State:", randomState)
print("Current Model MSE:", mseCurrent)
print("Current Model RMSE:", rmseCurrent)
print("Current Model MAPE:", mapeCurrent)
print("Current Model EV:", evCurrent)
print("Current Model R^2:", currentModelScore)

# Plotting data
plt.figure(figsize=(8, 8))
sns.set(style="whitegrid")
sns.scatterplot(x=yTest, y=yPredict, color="blue", s=50, edgecolor='black', alpha=0.75)
min_val = min(min(yTest), min(yPredict))
max_val = max(max(yTest), max(yPredict))
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Perfect Fit (y = x)")
plt.title(f"GPR Model - {data} - {output}", fontsize=16)
plt.xlabel("Measurements", fontsize=14)
plt.ylabel("GPR Predictions", fontsize=14)
plt.legend()
plt.show()
