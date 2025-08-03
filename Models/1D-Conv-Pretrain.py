# Pre-training 1-dimensional convolution neural network using SVG InterExtra augmented data
# Pre-trained on SVR interpolated and extrapolated data

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, BatchNormalization, Flatten, Dropout
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.optimizers import Adam
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

def constructModelLayers(inputDim: int, l1: float = 1e-3, l2: float = 1e-2, learningRate: float = 0.002):
    model = Sequential()

    # Layer 1
    model.add(Conv1D(filters=max(1, inputDim // 4), kernel_size=3, padding='same',
                     activation='relu',
                     kernel_initializer='random_normal',
                     kernel_regularizer=l1_l2(l1=l1, l2=l2),
                     input_shape=(inputDim, 1)))
    model.add(BatchNormalization())

    # Layer 2
    model.add(Conv1D(filters=max(1, inputDim // 8), kernel_size=3, padding='same',
                     activation='relu',
                     kernel_initializer='random_normal',
                     kernel_regularizer=l1_l2(l1=l1, l2=l2)))
    model.add(BatchNormalization())

    # Flattening output
    model.add(Flatten())
    model.add(Dense(1, activation='linear',
                    kernel_initializer='random_normal',
                    kernel_regularizer=l1_l2(l1=l1, l2=l2)))
    model.compile(optimizer=Adam(learning_rate=learningRate), loss='mean_squared_error')
    return model


# Import Augmented Data CSV file
augDataDirectory = os.path.join("Regression Model Data and Metrics", data, output, rModel,
                                f"{rModel} PreTrain N_{augmentedDataCount} Random_{randomState} Augmented Data.csv")
augmentedData = pd.read_csv(augDataDirectory)
x = augmentedData.iloc[:, :-1].values
y = augmentedData.iloc[:, -1].values
inputDim = augmentedData.shape[1] - 1

# Normalize data
dataScaler = MinMaxScaler(feature_range=(-1, 1))
xLog = np.log1p(x)
xScaled = dataScaler.fit_transform(xLog)

# Construct 1D-Conv Model
convModel = constructModelLayers(inputDim=inputDim, learningRate=learningRate)

# Pre-train 1D-Conv
history = convModel.fit(xScaled, y, epochs=epochs, batch_size=batchSize, verbose=1)

# Print history
# print("Training Loss:", history.history['loss'])

# Save 1D-Conv
directory = os.path.join("Pre-Trained Neural Networks", data, output, "1D-Conv")
os.makedirs(directory, exist_ok=True)
mRound = roundConfig.mRound
modelName = f"Pre-Trained 1D-Conv - N_{augmentedDataCount} Epoch_{epochs} Batch_{batchSize} Random_{randomState} Round_{mRound}.keras"
convModel.save(os.path.join(directory, modelName))
print("Saved " + os.path.join(directory, modelName) + "!")