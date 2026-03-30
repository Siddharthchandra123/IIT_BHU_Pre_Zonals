import time
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

start = time.time()
train_df = pd.read_csv("frontend/Training.csv")
X = train_df.drop("prognosis", axis=1)
y = train_df["prognosis"]

model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)
end = time.time()

print(f"Training Time: {end - start:.4f} seconds!")
