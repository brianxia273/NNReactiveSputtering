# Optimization-based first-order meta-learning for neural network models using SVR, BRR, and GPR augmented data

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
import pandas as pd
import config, roundConfig
import os
import time

# Select hyperparameters from config
innerStepSize = config.p3InnerStepSize
metaStepSize = config.p3MetaStepSize  # Same as outer-loop learning rate, interchangeable terms
metaBatchSize = config.p3MetaBatchSize
metaTasks = config.p3MetaTasks
metaEpochs = config.p3MetaEpochs

nnEpoch = config.p3NNEpoch
nnBatch = config.p3NNBatch
nn = "FCNN"

adData = config.p3Data
augmentedDataCount = config.p3N
randomState = config.p3RandomState
output = config.p3Output
ensembleRandomSeed = config.p3EnsembleRandom

# Set np and tf randomSeed
np.random.seed(ensembleRandomSeed)
tf.random.set_seed(ensembleRandomSeed)

seed = config.p3seed
# np.random.seed(seed)
# tf.random.set_seed(seed)

mRound = roundConfig.mRound

if config.p3EnableCPULimit:
    tf.config.threading.set_intra_op_parallelism_threads(config.p3IntraOPThreads)
    tf.config.threading.set_inter_op_parallelism_threads(config.p3InterOPThreads)

# Load and normalize 3 augmented datasets
# SVR
svrDataDirectory = os.path.join("Regression Model Data and Metrics", adData, output, "SVR",
                                f"SVR MetaTrain N_{augmentedDataCount} Random_{randomState} Augmented Data.csv")
svrAugData = pd.read_csv(svrDataDirectory)
svrX = svrAugData.iloc[:, :-1].values
svrY = svrAugData.iloc[:, -1].values  # Always 1 output col in aug data
svrDataScaler = MinMaxScaler(feature_range=(-1, 1))
svrXLog = np.log1p(svrX)
svrXScaled = svrDataScaler.fit_transform(svrXLog)

# BRR
brrDataDirectory = os.path.join("Regression Model Data and Metrics", adData, output, "BRR",
                                f"BRR N_{augmentedDataCount} Random_{randomState} Augmented Data.csv")
brrAugData = pd.read_csv(brrDataDirectory)
brrX = brrAugData.iloc[:, :-1].values
brrY = brrAugData.iloc[:, -1].values
brrDataScaler = MinMaxScaler(feature_range=(-1, 1))
brrXLog = np.log1p(brrX)
brrXScaled = brrDataScaler.fit_transform(brrXLog)

# GPR
gprDataDirectory = os.path.join("Regression Model Data and Metrics", adData, output, "GPR",
                                f"GPR N_{augmentedDataCount} Random_{randomState} Augmented Data.csv")
gprAugData = pd.read_csv(gprDataDirectory)
gprX = gprAugData.iloc[:, :-1].values
gprY = gprAugData.iloc[:, -1].values
gprDataScaler = MinMaxScaler(feature_range=(-1, 1))
gprXLog = np.log1p(gprX)
gprXScaled = gprDataScaler.fit_transform(gprXLog)

# Load pre-trained neural network, set optimizer
nnModelPath = os.path.join("Pre-Trained Neural Networks", adData, output, nn,
                           f"Pre-Trained {nn} - N_{augmentedDataCount} Epoch_{nnEpoch} Batch_{nnBatch} Random_{randomState} Round_{mRound}.keras")
nnModel = load_model(nnModelPath)
optimizer = Adam(learning_rate=innerStepSize)
mse = MeanSquaredError()

trainedModelName = f"Meta-Trained {nn} - N_{augmentedDataCount} Epoch_{nnEpoch} Batch_{nnBatch} Random_{randomState} Round_{mRound}.keras"
print("Training " + trainedModelName)
startTime = time.time()

@tf.function
def innerLoop(xTensor, yTensor):
    # Record MSE computations, to later compute gradients/derivatives
    with tf.GradientTape() as tape:
        # Calculate LMSE of predicted and actual output
        predictions = nnModel(xTensor)
        lmseLoss = mse(yTensor, predictions)
        # Compute gradients/derivatives of MSE equation, tells us direction/magnitude to minimize loss. Multiply by loss function, add onto weight
    gradients = tape.gradient(lmseLoss, nnModel.trainable_weights)
    # Pair gradients and trainable weights, updates weights of NN by adding (innerLearningRate)*(gradients) to weights
    optimizer.apply_gradients(zip(gradients, nnModel.trainable_weights))
    return lmseLoss

# Meta-Iteration Loop
for metaIter in range(metaTasks):
    # Save temporary weights
    oldWeights = nnModel.get_weights()
    # Choose SVR, BRR, or GPR
    datasetChoice = np.random.randint(0, 3)
    xScaled = svrXScaled if datasetChoice == 0 else brrXScaled if datasetChoice == 1 else gprXScaled
    y = svrY if datasetChoice == 0 else brrY if datasetChoice == 1 else gprY
    # Get mini batch
    miniBatchIndices = np.random.choice(len(xScaled), metaBatchSize, replace=False)
    xBatchScaled = xScaled[miniBatchIndices]
    yBatch = y[miniBatchIndices]
    lmseLoss = None
    # Inner Loop Training
    xTensor, yTensor = tf.convert_to_tensor(xBatchScaled, dtype=tf.float32), tf.convert_to_tensor(yBatch, dtype=tf.float32)
    for _ in range(metaEpochs):
        lmseLoss = innerLoop(xTensor, yTensor)
    # Apply Meta-Update
    newWeights = nnModel.get_weights()
    metaUpdatedWeights = [
        old + metaStepSize * (new - old)
        for old, new in zip(oldWeights, newWeights)
    ]
    nnModel.set_weights(metaUpdatedWeights)
    # Logging loss every 100 iterations
    if metaIter % 100 == 0:
        print(f"Meta-iteration {metaIter}: Loss = {tf.reduce_mean(lmseLoss).numpy():.10f}")

# Save trained model
modelDirectory = os.path.join("Meta-Trained Neural Networks", adData, output, nn)
os.makedirs(modelDirectory, exist_ok=True)
nnModel.save(os.path.join(modelDirectory, trainedModelName))
print("Saved " + os.path.join(modelDirectory, trainedModelName) + "!")
endTime = time.time()
print(f"Time Elapsed: {endTime - startTime} seconds")
# Visualize results - to be added
