import pandas as pd
import numpy as np
import joblib
import os

# Lazy-loaded globals
symptoms_df = None
desc_df = None
prec_df = None
train_df = None
qa_df = None
answers = None
model_ml = None
feature_columns = None
tfidf_vectorizer = None
tfidf_matrix = None

data_loaded = False


def load_resources():
    global symptoms_df, desc_df, prec_df, train_df, qa_df
    global model_ml, feature_columns, tfidf_vectorizer, tfidf_matrix, answers, data_loaded

    if data_loaded:
        return

    print("🔥 Loading base AI datasets...")

    # Load datasets
    symptoms_df = pd.read_csv("dataset.csv")
    desc_df = pd.read_csv("symptom_Description.csv")
    prec_df = pd.read_csv("symptom_precaution.csv")
    train_df = pd.read_csv("Training.csv")
    qa_df = pd.read_csv("medquad_qa.csv")
    
    questions = qa_df["Question"].astype(str).tolist()
    answers = qa_df["Answer"].tolist()

    # Load ML model
    print("🧠 Loading ML Model...")
    model_ml = joblib.load("disease_model.pkl")
    feature_columns = joblib.load("feature_columns.pkl")

    print("⚡ Instant-Boot: Compiling TF-IDF Search Engine...")
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    vec = TfidfVectorizer(stop_words='english', max_features=10000)
    mat = vec.fit_transform(questions)

    # assign to globals
    globals()['tfidf_vectorizer'] = vec
    globals()['tfidf_matrix'] = mat

    data_loaded = True
    print("✅ All resources loaded! Server is 100% invincible!")

# ---------------- UTIL FUNCTIONS ---------------- #

def normalize_input(text):
    text = text.lower()
    fixes = {
        "bodypain": "body pain",
        "vommit": "vomiting",
        "weak": "weakness",
        "blurred vision": "blurry vision"
    }
    for k, v in fixes.items():
        text = text.replace(k, v)
    return text


def simplify(text):
    return " ".join(text.split(". ")[:3])


def get_disease_info(disease):
    desc = desc_df[desc_df["Disease"] == disease]["Description"]
    desc = desc.values[0] if len(desc) else None

    prec = prec_df[prec_df["Disease"] == disease]
    precautions = prec.iloc[0, 1:].dropna().tolist() if not prec.empty else []

    return desc, precautions


def predict_rule(symptoms):
    best, score = None, 0
    stop_words = {"hi", "hello", "how", "are", "you", "i", "am", "my", "is", "the", "a", "an"}

    filtered = [s for s in symptoms if s not in stop_words and len(s) > 2]
    if not filtered:
        return None

    for _, row in symptoms_df.iterrows():
        text = " ".join(row.dropna().astype(str)).lower()
        matches = sum(1 for s in filtered if f" {s} " in f" {text} ")

        if matches > score:
            score = matches
            best = row["Disease"]

    return best if score > 0 else None


def predict_ml(user_input):
    ml_text = user_input.replace(" ", "_")
    input_df = pd.DataFrame(
        [[1 if col in ml_text else 0 for col in feature_columns]],
        columns=feature_columns
    )

    probs = model_ml.predict_proba(input_df)[0]
    return model_ml.predict(input_df)[0] if max(probs) > 0.65 else None


def rag_answer(query):
    from sklearn.metrics.pairwise import cosine_similarity
    
    query_vec = tfidf_vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Get top 2 best-matching indices
    top_indices = similarities.argsort()[-2:][::-1]
    
    valid = []
    for i in top_indices:
        if similarities[i] > 0.15: # TF-IDF needs lower threshold
            valid.append(answers[i])
            
    if not valid:
        return "I don't have enough specific medical information to answer that safely. Please consult a doctor."

    return simplify(" ".join(valid))


# ---------------- MAIN FUNCTION ---------------- #

def ask(user_input):
    load_resources()   # 🔥 critical

    user_input = normalize_input(user_input)
    symptoms = user_input.split()

    disease_rule = predict_rule(symptoms)
    disease_ml = predict_ml(user_input)
    ai_text = rag_answer(user_input)

    response = ""

    if disease_rule:
        desc, precautions = get_disease_info(disease_rule)

        response += f"\n🩺 POSSIBLE CONDITION: {disease_rule}\n"

        if desc:
            response += f"\nℹ️ ABOUT: {simplify(desc)}\n"

        if precautions:
            response += "\n🏥 CARE ADVICE:\n"
            for p in precautions[:4]:
                response += f"• {p}\n"

    response += f"\n🧠 AI GUIDANCE:\n{ai_text}\n"

    if any(x in user_input for x in ["chest pain", "breathing", "bleeding"]):
        response += "\n🚨 EMERGENCY: Seek medical help immediately."

    return response