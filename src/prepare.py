from sklearn.datasets import load_iris  
import pandas as pd
import os

data = load_iris(as_frame=True)
df = pd.concat([data.data, pd.Series(data.target, name='target')], axis=1)

os.makedirs("data/processed", exist_ok=True)

df.to_csv("data/processed/zameen-updated.csv", index=False)

print("File saved successfully at data/processed/zameen-updated.csv")

