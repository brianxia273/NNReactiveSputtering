# Fine-tuning Neural Network Model after Meta-Learning

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import config, roundConfig
import os
import joblib

# Select from config
nn = "FCNN"
output = config.p4Output
data = config.p4Data
nnEpochs = config.p4NNEpoch
nnBatch = config.p4NNBatch
epochs = config.p4Epochs
batchSize = config.p4BatchSize
learningRate = config.p4LearningRate
augmentedDataCount = config.p4N
randomState = config.p4RandomState
outputCols = config.p0OutputCols
ensembleRandomSeed = config.p4EnsembleRandom

# Set np and tf randomSeed
np.random.seed(ensembleRandomSeed)
tf.random.set_seed(ensembleRandomSeed)

mRound = roundConfig.mRound

# Import NN
mlModelPath = os.path.join("Meta-Trained Neural Networks", data, output, nn,
                           f"Meta-Trained {nn} - N_{augmentedDataCount} Epoch_{nnEpochs} Batch_{nnBatch} Random_{randomState} Round_{mRound}.keras")
mlModel = load_model(mlModelPath)
mlModel.compile(optimizer=Adam(learning_rate=learningRate), loss='mse')

# Import Real Data CSV file
df = pd.read_csv(os.path.join("Datasets", data))
x = df.drop(columns=outputCols).values
y = df[output].values  # Selecting output

# Train/Test split
trainSize = int(0.8 * len(x))
xTrain, xTest, yTrain, yTest = train_test_split(x, y, train_size=trainSize, random_state=randomState)

# Normalize data
dataScaler = MinMaxScaler(feature_range=(-1, 1))
xTrainLog = np.log1p(xTrain)
xTrainScaled = dataScaler.fit_transform(xTrainLog)
xTestLog = np.log1p(xTest)
xTestScaled = dataScaler.transform(xTestLog)

# Save Data Scaler
scalerDirectory = os.path.join("Data Scalers", data, output, nn)
os.makedirs(scalerDirectory, exist_ok=True)
scalerName = f"Fine-Tuned {nn} - N_{augmentedDataCount} Epoch_{epochs} Batch_{batchSize} Random_{randomState} DataScaler.pkl"
joblib.dump(dataScaler, os.path.join(scalerDirectory, scalerName))
print("Saved " + os.path.join(scalerDirectory, scalerName) + "!")

# Fine-Tune NN
history = mlModel.fit(xTrainScaled, yTrain, epochs=epochs, batch_size=batchSize, verbose=1,
                      validation_data=(xTestScaled, yTest))

# Print complete history
# print("Training Loss:", history.history['loss'])

# Save Fine-Tuned NN
directory = os.path.join("Fine-Tuned Neural Networks", data, output, nn)
os.makedirs(directory, exist_ok=True)
modelName = f"Fine-Tuned {nn} - N_{augmentedDataCount} Epoch_{epochs} Batch_{batchSize} Random_{randomState} Round_{mRound}.keras"
mlModel.save(os.path.join(directory, modelName))
print("Saved " + os.path.join(directory, modelName) + "!")
