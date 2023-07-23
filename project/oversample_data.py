# %% Load raw dataset
import pandas as pd

df = pd.read_csv("../data/raw/HIV_train.csv")
df.index = df["index"]
df["HIV_active"].value_counts()
start_index = df.iloc[0]["index"]

# %% Apply oversampling

# Check how many additional samples we need
neg_class = df["HIV_active"].value_counts()[0]
pos_class = df["HIV_active"].value_counts()[1]
multiplier = int(neg_class / pos_class) - 1
print(multiplier)

# Replicate the dataset for the positive class
replicated_pos = [df[df["HIV_active"] == 1]] * multiplier

# Append replicated data
df = pd.concat([df] + replicated_pos, ignore_index=True)
print(df.shape)

# Shuffle dataset
df = df.sample(frac=1).reset_index(drop=True)

# Re-assign index (This is our ID later)
index = range(start_index, start_index + df.shape[0])
df.index = index
df["index"] = df.index
df.head()

# %% Save
df.to_csv("../data/raw/HIV_train_oversampled.csv", index=False)
