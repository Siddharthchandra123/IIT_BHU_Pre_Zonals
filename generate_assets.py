import pandas as pd
import numpy as np
import faiss
import joblib
import os
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier

# Configuration
BASE_DIR = "frontend" # Since we're running from the root

# 1. Load Training Data
print("Loading Training Data from frontend/...")
train_df = pd.read_csv(os.path.join(BASE_DIR, "Training.csv"))

X = train_df.drop("prognosis", axis=1)
y = train_df["prognosis"]

# 2. Train and Save Random Forest Model
print("Training Random Forest...")
model_ml = RandomForestClassifier()
model_ml.fit(X, y)
joblib.dump(model_ml, os.path.join(BASE_DIR, "disease_model.pkl"))
joblib.dump(X.columns.tolist(), os.path.join(BASE_DIR, "feature_columns.pkl"))
print("✅ Disease Model Saved!")

# 3. Load Medical Knowledge Data
print("Loading Medical QA Data...")
df = pd.read_csv(os.path.join(BASE_DIR, "medquad_qa.csv"))
questions = df["Question"].tolist()

# 4. Generate & Save FAISS Index (The Brain)
print("Creating Embeddings (This uses heavy RAM, running on laptop only)...")
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(questions, show_progress_bar=True)
embeddings = np.array(embeddings).astype('float32')

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

faiss.write_index(index, os.path.join(BASE_DIR, "rag_index.faiss"))
print("✅ FAISS Knowledge Index Saved!")

print("\n🚀 BRAIN GENERATION COMPLETE!")
print(f"I have saved the optimized assets into the '{BASE_DIR}' folder.")
