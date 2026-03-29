import pandas as pd
import faiss
import numpy as np
import joblib
import os
from sentence_transformers import SentenceTransformer

# PRODUCTION LITE CONFIGURATION
# We disable the summarizer to save 300MB+ of RAM for Render's 512MB limit.
USE_SUMMARIZER = False

# All CSVs are now in the local 'frontend' folder for Render compatibility
symptoms_df = pd.read_csv("dataset.csv")
desc_df = pd.read_csv("symptom_Description.csv")
prec_df = pd.read_csv("symptom_precaution.csv")
train_df = pd.read_csv("Training.csv")
qa_df = pd.read_csv("medquad_qa.csv")
answers = qa_df["Answer"].tolist()

# Load the Pre-calculated "Brain" if it exists
if os.path.exists("disease_model.pkl") and os.path.exists("feature_columns.pkl"):
    print("✅ Loading Pre-trained Disease Model...")
    model_ml = joblib.load("disease_model.pkl")
    feature_columns = joblib.load("feature_columns.pkl")
else:
    print("⚠️ Pre-trained model not found. Training now (High RAM use!)...")
    X = train_df.drop("prognosis", axis=1)
    y = train_df["prognosis"]
    model_ml = RandomForestClassifier()
    model_ml.fit(X, y)
    feature_columns = X.columns.tolist()

print("Loading Embedding Model (Lite)...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

# Load the Pre-encoded Knowledge Index
if os.path.exists("rag_index.faiss"):
    print("✅ Loading Pre-encoded Knowledge Index...")
    index = faiss.read_index("rag_index.faiss")
else:
    print("⚠️ Knowledge Index not found. Encoding now (High RAM use!)...")
    questions = qa_df["Question"].tolist()
    embeddings = np.array(embed_model.encode(questions, show_progress_bar=True)).astype('float32')
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

def normalize_input(text):
    text = text.lower()
    fixes = {"bodypain": "body pain", "vommit": "vomiting", "weak": "weakness", "blurred vision": "blurry vision"}
    for k,v in fixes.items():
        text = text.replace(k,v)
    return text

def simplify(text):
    # Simple text shortening for memory efficiency
    return " ".join(text.split(". ")[:3])

def get_disease_info(disease):
    desc = desc_df[desc_df["Disease"] == disease]["Description"]
    desc = desc.values[0] if len(desc) else None
    prec = prec_df[prec_df["Disease"] == disease]
    precautions = prec.iloc[0,1:].dropna().tolist() if not prec.empty else []
    return desc, precautions

def predict_rule(symptoms):
    best, score = None, 0
    stop_words = {"hi", "hello", "how", "are", "you", "whats", "up", "buddy", "i", "am", "my", "is", "the", "a", "an"}
    filtered_symptoms = [s for s in symptoms if s not in stop_words and len(s) > 2]
    if not filtered_symptoms: return None
    for _, row in symptoms_df.iterrows():
        text = " ".join(row.dropna().astype(str)).lower()
        matches = sum(1 for s in filtered_symptoms if f" {s} " in f" {text} " or s == text)
        if matches > score:
            score = matches
            best = row["Disease"]
    return best if score > 0 else None

def predict_ml(symptoms):
    input_df = pd.DataFrame([[1 if col in symptoms else 0 for col in feature_columns]], columns=feature_columns)
    probs = model_ml.predict_proba(input_df)[0]
    return model_ml.predict(input_df)[0] if max(probs) > 0.65 else None

def rag_answer(query):
    vec = embed_model.encode([query]).astype('float32')
    D, I = index.search(vec, k=2)
    valid_answers = [answers[I[0][i]] for i in range(len(I[0])) if D[0][i] < 1.3]
    if not valid_answers:
        return "I don't have enough specific medical information to answer that safely. Please consult a doctor."
    return simplify(" ".join(valid_answers))

def ask(user_input):
    user_input = normalize_input(user_input)
    symptoms = user_input.split()
    disease_rule = predict_rule(symptoms)
    disease_ml = predict_ml(symptoms)
    ai_text = rag_answer(user_input)

    response = ""
    if disease_rule:
        desc, precautions = get_disease_info(disease_rule)
        response += f"\n🩺 POSSIBLE CONDITION: {disease_rule}\n"
        if desc: response += f"\nℹ️ ABOUT: {simplify(desc)}\n"
        if precautions:
            response += "\n🏥 CARE ADVICE:\n"
            for p in precautions[:4]: response += f"• {p}\n"

    response += f"\n🧠 AI GUIDANCE:\n{ai_text}\n"
    if any(d in user_input for d in ["chest pain","breathing","bleeding"]):
        response += "\n🚨 EMERGENCY: Seek medical help immediately."

    return response
