# Configuration file to define parameters for NN and regression model development
# ================================================================================
# ================================================================================

# Phase 0: Custom Dataset configurations

# Available datasets
p0Datasets: list[str] = ["CritTemp HiPIMS.csv", "CritTemp.csv"]

# Output columns for current dataset
p0OutputCols: list[str] = ["Critical Temperature"]

# ================================================================================
# ================================================================================

# Phase 1: WriteMetrics, DataGenerate, and GridSearch Configuration
# NOTE: NEED TO SELF-MODIFY HYPERPARAMETERS FOR GPR AND SVR IN WriteMetrics AND DataGenerate
p1Data: str = p0Datasets[1]
p1Output: str = p0OutputCols[0]
p1RandomState: int = 56

p1SvrExtrapolationRange: float = 0.03
p1N: int = 25600  # N = Augmented Data Count, {6400, 12800, 25600}

p1EnsembleRandom: int = 0

# ================================================================================
# ================================================================================

# Phase 2: FCNN/1D-Conv PreTrain Configuration
p2Data: str = p0Datasets[1]
p2Output: str = p0OutputCols[0]
p2LearningRate: float = 0.002  # NEED TO DOUBLE-CHECK
p2BatchSize: int = 1028  # {16, 512, 1028}
p2Epochs: int = 1000  # {20, 200, 1000}
p2N: int = 25600  # N = Augmented Data Count, {6400, 12800, 25600}
p2RandomState: int = 56  # Also selects randomState of SVG augmented data

p2EnsembleRandom: int = 0

# ================================================================================
# ================================================================================

# Phase 3: NN MetaTrain Configuration
p3MetaStepSize: float = 0.05  # {0.05, 0.2}
p3MetaEpochs: int = 200  # {5, 10, 100, 200}
p3MetaTasks: int = 2000  # {50, 100, 1000, 2000}
p3MetaBatchSize: int = 20  # {5, 20}
p3InnerStepSize: float = 0.05  # NEED TO DOUBLE-CHECK

# Choosing Pre-Trained NN using its parameters
p3NNEpoch: int = 1000  # {20, 200, 1000}
p3NNBatch: int = 1028  # {16, 512, 1028}

# Selecting Augmented Data parameters
p3Data: str = p0Datasets[1]
p3Output: str = p0OutputCols[0]
p3N: int = 25600  # N {6400, 12800, 25600}

# Choose randomStates of datasets to select from. Must be consistent with SVR, BRR, and GPR randomState
p3RandomState: int = 55

# Seed to control np and tf RNG; change as needed, but currently is not kept track of
p3seed: int = 42

# (Optional) Configuring CPU core usage
p3EnableCPULimit: bool = False  # When True, limits CPU threads based on settings below
p3IntraOPThreads: int = 8
p3InterOPThreads: int = 4

p3EnsembleRandom: int = 0

# ================================================================================
# ================================================================================

# Phase 4: FCNN, 1D-Conv FineTune Configuration

p4LearningRate: float = 0.05  # {0.05, 0.2} - Is same as MetaLearn (?)
p4Epochs: int = 1000  # {5, 10, 100, 200} - Is same as MetaLearn (?)
p4Data: str = p0Datasets[1]
p4Output: str = p0OutputCols[0]
p4BatchSize: int = 1028  # {16, 512, 1028} (?)

# Choosing Meta-Trained NN using its parameters
p4NNEpoch: int = 1000  # {20, 200, 1000}
p4NNBatch: int = 1028  # {16, 512, 1028}
p4N: int = 25600  # N {6400, 12800, 25600}

# Choosing randomState for train/test, must be same as previous phases
p4RandomState: int = 55

p4EnsembleRandom: int = 0

# ================================================================================
# ================================================================================

# Phase 5: Neural Network TestAccuracy Configuration

p5Data: str = p0Datasets[1]
p5Output: str = p0OutputCols[0]

# Choosing Fine-Tuned NN using its parameters
p5N: int = 25600  # N {6400, 12800, 25600}
p5NNEpoch: int = 1000  # {5, 10, 100, 200}
p5NNBatch: int = 1028  # {16, 512, 1028}

# Choosing randomState for same train/test, must be same as previous phases
p5RandomState: int = 55

# Ensemble round number
p5Round: int = 6

p5EnsembleRandom: int = 0

# Round 0 1D-Conv

# ================================================================================
# ================================================================================

# AVAILABLE DATASETS:
# "CritTemp.csv" | ["Critical Temperature"]
# "CritTemp HiPIMS.csv" | ["Critical Temperature"]
