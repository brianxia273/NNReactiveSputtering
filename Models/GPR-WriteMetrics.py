# Writing metrics for GPR regression model for sizes 5-40.
# NOTE: MUST ADJUST GPR HYPERPARAMETERS FOR BEST PERFORMANCE

import numpy as np
import pandas as pd
from sklearn.gaussian_process.kernels import ConstantKernel, Matern
import os
import config
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, explained_variance_score, mean_absolute_percentage_error

# Select dataset, output, and randomState from config
data = config.p1Data
randomState = config.p1RandomState
model = "GPR"
outputCols = config.p0OutputCols
output = config.p1Output
ensembleRandomSeed = config.p1EnsembleRandom

np.random.seed(ensembleRandomSeed)

# Automating file creation
directory = os.path.join("Regression Model Data and Metrics", data, output, model)
os.makedirs(directory, exist_ok=True)
with open(os.path.join(directory, f"{model} Random_{randomState}_{randomState + 19} Metric Iteration Evaluation.txt"),
          "w") as f:
    f.write("MSE, RMSE, MAPE, EV, and R^2 Metrics\n")
    f.write(f"Current Model Dataset: {data}\n")
    f.write(f"Output Variable: {output}\n")
    f.write("=" * 50 + "\n")
    for rs in range(randomState, randomState + 20):
        df = pd.read_csv(os.path.join("Datasets", data))
        x = df.drop(columns=outputCols).values
        y = df[output].values  # Selecting output

        # 80% data to train, 20% leave for testing. random_state is set in config
        trainSize = int(0.8 * len(x))
        xTrain, xTest, yTrain, yTest = train_test_split(x, y, train_size=trainSize, random_state=rs)

        # Scaling data
        xTrainLog = np.log1p(xTrain)
        xTestLog = np.log1p(xTest)
        dataScaler = MinMaxScaler(feature_range=(-1, 1))
        xTrainScaled = dataScaler.fit_transform(xTrainLog)
        xTestScaled = dataScaler.transform(xTestLog)

        # Init GPR model, ADJUST HYPERPARAMETERS
        gprKernel = ConstantKernel(1.0) * Matern(length_scale=40, nu=1.5)
        gpr = GaussianProcessRegressor(alpha=0.01, kernel=gprKernel, n_restarts_optimizer=10, normalize_y=True,
                                       optimizer="fmin_l_bfgs_b")
        gpr.fit(xTrainScaled, yTrain)

        # Initial predictions
        yPredict = gpr.predict(xTestScaled)
        mseCurrent = mean_squared_error(yTest, yPredict)
        rmseCurrent = np.sqrt(mseCurrent)
        mapeCurrent = mean_absolute_percentage_error(yTest, yPredict)
        evCurrent = explained_variance_score(yTest, yPredict)
        currentModelScore = gpr.score(xTestScaled, yTest)

        # Write metrics
        f.write(f"RandomState: {rs}\n")
        f.write(f"MSE: {mseCurrent}\n")
        f.write(f"RMSE: {rmseCurrent}\n")
        f.write(f"MAPE: {mapeCurrent}\n")
        f.write(f"EV: {evCurrent}\n")
        f.write(f"R^2: {currentModelScore}\n")
        f.write("-" * 50 + "\n")
        print(f"Completed {rs}!")
