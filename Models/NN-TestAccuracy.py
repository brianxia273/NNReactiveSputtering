# Testing Fine-Tuned Neural Network Model's accuracy
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
import config, roundConfig
import os
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib

# Select from config
data = config.p5Data
nn = "FCNN"
nnEpoch = config.p5NNEpoch
nnBatch = config.p5NNBatch
augmentedDataCount = config.p5N
randomState = config.p5RandomState
output = config.p5Output
outputCols = config.p0OutputCols
ensembleRandomSeed = config.p5EnsembleRandom

# Set np randomSeed
np.random.seed(ensembleRandomSeed)

mRound = config.p5Round

# Import NNs
for _ in range(2):
    try:
        mlModelPath = os.path.join("Fine-Tuned Neural Networks", data, output, nn,
                                   f"Fine-Tuned {nn} - N_{augmentedDataCount} Epoch_{nnEpoch} Batch_{nnBatch} Random_{randomState} Round_{mRound}.keras")
        mlModel = load_model(mlModelPath)

        # Import Real Data CSV file
        df = pd.read_csv(os.path.join("Datasets", data))
        x = df.drop(columns=outputCols).values
        y = df[output].values  # Selecting output

        # Train/Test split, use Test for evaluation
        trainSize = int(0.8 * len(x))
        xTrain, xTest, yTrain, yTest = train_test_split(x, y, train_size=trainSize, random_state=randomState)

        # Normalize data, ignore Train
        scalerName = f"Fine-Tuned {nn} - N_{augmentedDataCount} Epoch_{nnEpoch} Batch_{nnBatch} Random_{randomState} DataScaler.pkl"
        dataScaler = joblib.load(os.path.join("Data Scalers", data, output, nn, scalerName))
        xTestLog = np.log1p(xTest)
        xTestScaled = dataScaler.transform(xTestLog)

        # Check predictions
        yPredict = mlModel.predict(xTestScaled)
        mse = mean_squared_error(yTest, yPredict)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(yTest, yPredict)
        ev = explained_variance_score(yTest, yPredict)
        r2 = r2_score(yTest, yPredict)

        # Print accuracy
        print(f"{mlModelPath} Model Evaluation Metrics: ")
        print(f"MSE: {mse}")
        print(f"RMSE: {rmse}")
        print(f"MAPE: {mape}")
        print(f"EV: {ev}")
        print(f"R^2: {r2}")

        plt.figure(figsize=(8, 6))
        plt.scatter(yTest, yPredict, alpha=0.6, edgecolor='k')
        plt.plot([yTest.min(), yTest.max()], [yTest.min(), yTest.max()], 'r--', lw=2)  # ideal line
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("Actual vs Predicted Values")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        nn = "1D-Conv"
    except Exception as e:
        print(e)

