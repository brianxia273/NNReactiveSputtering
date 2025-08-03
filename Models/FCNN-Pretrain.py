# Pre-training Fully Connected neural network using SVG InterExtra augmented data
# Pre-trained on SVR interpolated and extrapolated data

import numpy as np
import tensorflow as tf
from keras.src.layers import BatchNormalization, Dropout
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.regularizers import l1_l2
import pandas as pd
import config, roundConfig
import os

# Select size, dataset, output, randomState, data, epochs, and batch size from config
randomState = config.p2RandomState
data = config.p2Data
rModel = "SVR"
epochs = config.p2Epochs
batchSize = config.p2BatchSize
learningRate = config.p2LearningRate
augmentedDataCount = config.p2N
output = config.p2Output
ensembleRandomSeed = config.p2EnsembleRandom

# Set np and tf randomSeed
np.random.seed(ensembleRandomSeed)
tf.random.set_seed(ensembleRandomSeed)

# Optional parameters from original study; change/specify as needed
def constructModelLayers(inputDim: int, l1: float = 1e-3, l2: float = 1e-2, drop1: float = 0.1, drop2: float = 0.2,
                         learningRate: float = 0.002):
    model = Sequential()

    # 8 input nodes, into Hidden Layer 1
    model.add(Dense(max(1, inputDim // 2), input_dim=inputDim, kernel_initializer=RandomNormal(),
                    kernel_regularizer=l1_l2(l1=l1, l2=l2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(drop1))
    inputDim //= 2

    # Hidden Layer 2
    model.add(Dense(max(1, inputDim // 2), kernel_initializer=RandomNormal(), kernel_regularizer=l1_l2(l1=l1, l2=l2)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(drop2))
    inputDim //= 2

    # Output Layer
    model.add(Dense(max(1, inputDim // 2), kernel_initializer=RandomNormal()))

    model.compile(optimizer=Adam(learning_rate=learningRate), loss='mean_squared_error')
    return model


# Import Augmented Data CSV file
augDataDirectory = os.path.join("Regression Model Data and Metrics", data, output, rModel,
                                f"{rModel} PreTrain N_{augmentedDataCount} Random_{randomState} Augmented Data.csv")
augmentedData = pd.read_csv(augDataDirectory)
x = augmentedData.iloc[:, :-1].values  # Output always last col in augmented data
y = augmentedData.iloc[:, -1].values
inputDim = augmentedData.shape[1] - 1

# Normalize data
dataScaler = MinMaxScaler(feature_range=(-1, 1))
xLog = np.log1p(x)
xScaled = dataScaler.fit_transform(xLog)

# Construct FCNN Model
fcnnModel = constructModelLayers(inputDim=inputDim, learningRate=learningRate)

# Pre-train FCNN
history = fcnnModel.fit(xScaled, y, epochs=epochs, batch_size=batchSize, verbose=1)

# Print history
# print("Training Loss:", history.history['loss'])

# Save FCNN
directory = os.path.join("Pre-Trained Neural Networks", data, output, "FCNN")
os.makedirs(directory, exist_ok=True)
mRound = roundConfig.mRound
modelName = f"Pre-Trained FCNN - N_{augmentedDataCount} Epoch_{epochs} Batch_{batchSize} Random_{randomState} Round_{mRound}.keras"
fcnnModel.save(os.path.join(directory, modelName))
print("Saved " + os.path.join(directory, modelName) + "!")
