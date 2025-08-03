# Making correlogram of input and output variables
# Use to check if CSV datasets are correctly copied over

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import os

# Selecting dataset
data = "CritTemp.csv"
df = pd.read_csv(os.path.join("Datasets", data))

# Creating correlogram
correlationMatrix = df.corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlationMatrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title(f'Correlogram - {data}')
plt.show()

# AVAILABLE DATASETS:
# CritTemp.csv
# CritTemp HiPIMS.csv
