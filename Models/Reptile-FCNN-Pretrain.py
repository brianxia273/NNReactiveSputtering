# Pretraining Fully Connected Neural Network Model using SVG augmented data

import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import pandas as pd
import config
import os


# Select size, dataset, output, randomState, data, epochs, and batch size from config
setSize = config.nSize
yIndex = config.nYIndex
randomState = config.nRandomState
data = config.nData
model = "SVR"
epochs = config.epochs
batchSize = config.batchSize
learningRate = config.learningRate

def constructModelLayers():
    model = Sequential()
    model.add(Dense(4, input_dim=8, activation='relu')) # 8 input nodes, into Hidden Layer 1
    model.add(Dense(4, activation='relu')) # Hidden Layer 2
    model.add(Dense(1, activation='linear')) # Output Layer
    model.compile(optimizer=Adam(learning_rate=learningRate), loss='mean_squared_error')
    return model


datasetModels  = "Dataset 1 Models" if "Dataset 1" in data else "Dataset 2 Models"
output = "Film Thickness" if yIndex == -2 else "NTi"

# Import Augmented Data CSV file
augmentedData = pd.read_csv(f"Regression Model Data and Metrics/{datasetModels}/{output}/{model}/{model} Size_{setSize} Random_{randomState} Augmented Data.csv")
x = augmentedData.iloc[:, :-1].values
y = augmentedData.iloc[:, -1].values

# Normalize data
dataScaler = MinMaxScaler(feature_range=(-1, 1))
xLog = np.log1p(x)
xScaled = dataScaler.fit_transform(xLog)

# Construct FCNN Model
fcnnModel = constructModelLayers()

# Pre-train FCNN
history = fcnnModel.fit(xScaled, y, epochs= epochs, batch_size = batchSize, verbose = 1)

# Print history
print("Training Loss:", history.history['loss'])

# Save FCNN
directory = f"Neural Networks/FCNN/{datasetModels}/{output}/{model}/"
os.makedirs(directory, exist_ok=True)
modelName = f"Pretrained NN - {model} Size_{setSize} Epoch_{epochs} Batch_{batchSize}.keras"
fcnnModel.save(directory + modelName)
print("Saved " + directory + modelName + "!")
