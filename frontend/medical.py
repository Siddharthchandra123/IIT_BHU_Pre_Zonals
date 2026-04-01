import pandas as pd
import faiss
import numpy as np
import joblib
import os
from sentence_transformers import SentenceTransformer

# --- CONFIG & PATHS ---
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(DATA_DIR, "disease_model.pkl")
COLS_PATH = os.path.join(DATA_DIR, "feature_columns.pkl")
FAISS_PATH = os.path.join(DATA_DIR, "rag_index.faiss")
QA_CSV = os.path.join(DATA_DIR, "medquad_qa.csv")

# --- INITIALIZATION ---
print("🚀 Initializing Medical AI Core...")

# 1. Load Rule-based Data (Lightweight)
desc_df = pd.read_csv(os.path.join(DATA_DIR, "symptom_Description.csv"))
prec_df = pd.read_csv(os.path.join(DATA_DIR, "symptom_precaution.csv"))
severity_df = pd.read_csv(os.path.join(DATA_DIR, "Symptom-severity.csv"))
symptoms_df = pd.read_csv(os.path.join(DATA_DIR, "dataset.csv"))

# 2. Load ML Model (Pre-trained)
if os.path.exists(MODEL_PATH) and os.path.exists(COLS_PATH):
    print("📦 Loading pre-trained ML model...")
    model_ml = joblib.load(MODEL_PATH)
    ml_columns = joblib.load(COLS_PATH)
else:
    print("⚠️ Pre-trained model not found. Training minimal model (Memory Risk)...")
    train_df = pd.read_csv(os.path.join(DATA_DIR, "Training.csv"))
    X = train_df.drop("prognosis", axis=1)
    y = train_df["prognosis"]
    from sklearn.ensemble import RandomForestClassifier
    model_ml = RandomForestClassifier(n_estimators=10) # Lower trees for memory
    model_ml.fit(X, y)
    ml_columns = X.columns.tolist()

# 3. Load RAG/FAISS Index
print("🔍 Loading Knowledge Base...")
try:
    if os.path.exists(FAISS_PATH):
        index = faiss.read_index(FAISS_PATH)
        # We only need the Answers for RAG, load only that column to save RAM
        qa_df = pd.read_csv(QA_CSV, usecols=["Answer"])
        answers = qa_df["Answer"].tolist()
        print("✅ FAISS index and answers loaded.")
    else:
        print("⚠️ FAISS index not found. RAG will be disabled.")
        index = None
        answers = []
except Exception as e:
    print(f"❌ Error loading knowledge base: {e}")
    index = None
    answers = []

# 4. Lazy Load Embedding Model (Saves RAM on boot)
_embedding_model = None
def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        print("🧠 Loading SentenceTransformer (Lazy Load)...")
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedding_model

# --- CORE LOGIC ---

def simplify_answer(text):
    if not text: return ""
    sentences = text.split(". ")
    return ". ".join(sentences[:3])

def format_output(text):
    parts = text.split(". ")
    bullets = "\n".join(["• " + p.strip() for p in parts if len(p) > 5])
    return bullets

def emergency_check(text):
    danger_words = ["chest pain", "difficulty breathing", "unconscious", "bleeding", "severe pain"]
    for word in danger_words:
        if word in text.lower():
            return "\n🚨 EMERGENCY: Seek immediate medical help."
    return ""

def normalize_input(text):
    text = text.lower()
    replacements = {"bodypain": "body pain", "vommit": "vomiting", "blurred vision": "blurry vision", "weak": "weakness"}
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text

def ask(user_input):
    user_input_clean = normalize_input(user_input)
    
    # 1. ML Prediction
    input_data = pd.DataFrame([[1 if col.lower() in user_input_clean else 0 for col in ml_columns]], columns=ml_columns)
    try:
        probs = model_ml.predict_proba(input_data)[0]
        confidence = max(probs)
        disease_ml = model_ml.predict(input_data)[0] if confidence > 0.65 else "Unknown (Low Confidence)"
    except:
        disease_ml = "Unknown"

    # 2. Rule-based Match
    disease_rule = predict_disease(user_input_clean)

    # 3. RAG Search (FAISS)
    ai_text = ""
    if index and answers:
        try:
            model = get_embedding_model()
            query_vector = model.encode([user_input_clean])
            distances, indices = index.search(query_vector, k=2)
            results = [answers[idx] for dist, idx in zip(distances[0], indices[0]) if dist < 1.2]
            ai_text = " ".join([simplify_answer(r) for r in results])
        except Exception as e:
            print(f"RAG Error: {e}")
            ai_text = "Medical knowledge base temporarily unavailable."

    # --- COMPOSE RESPONSE ---
    response = "\n"
    if disease_rule:
        desc = get_description(disease_rule)
        precautions = get_precautions(disease_rule)
        response += f"🩺 Possible Condition: {disease_rule}\n"
        if desc: response += f"\nAbout:\n{simplify_answer(desc)}\n"
        if precautions:
            response += "\nCare Advice:\n"
            for p in precautions: response += f"• {p}\n"

    response += "\nAdditional Guidance:\n"
    response += format_output(ai_text) if ai_text else "Consult a professional for specific advice."
    response += emergency_check(user_input_clean)
    response += f"\n\n🧠 ML Analysis: {disease_ml}"
    response += "\n👉 Drink ORS or fluids | Rest well | See doctor if symptoms last > 2 days"
    
    return response

# --- SUPPORT FUNCTIONS ---

def predict_disease(user_input):
    filtered_symptoms = [s for s in user_input.split() if len(s) > 2]
    if not filtered_symptoms: return None
    best_match, max_matches = None, 0
    for _, row in symptoms_df.iterrows():
        row_symptoms = " ".join(row.dropna().astype(str)).lower()
        matches = sum(1 for sym in filtered_symptoms if f" {sym} " in f" {row_symptoms} ")
        if matches > max_matches:
            max_matches, best_match = matches, row["Disease"]
    return best_match if max_matches > 0 else None

def get_description(disease):
    result = desc_df[desc_df["Disease"] == disease]
    return result["Description"].values[0] if not result.empty else ""

def get_precautions(disease):
    result = prec_df[prec_df["Disease"] == disease]
    return result.iloc[0, 1:].dropna().tolist() if not result.empty else []

def check_risk(user_input):
    text = user_input.lower()
    return [row["Symptom"].lower() for _, row in severity_df.iterrows() if row["Symptom"].lower() in text and row["weight"] > 5]

print("✅ Medical AI Ready for Production!")
