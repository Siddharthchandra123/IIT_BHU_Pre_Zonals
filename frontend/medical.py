import pandas as pd
import faiss
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from transformers import pipeline

# =========================
# PATH CONFIG
# =========================
DATA_PATH = r"C:\Users\Lenovo\Downloads\IIT BHU\dataset"

# =========================
# LOAD DATA
# =========================
symptoms_df = pd.read_csv(f"{DATA_PATH}/Disease symptom prediction/dataset.csv")
desc_df = pd.read_csv(f"{DATA_PATH}/Disease symptom prediction/symptom_Description.csv")
prec_df = pd.read_csv(f"{DATA_PATH}/Disease symptom prediction/symptom_precaution.csv")

train_df = pd.read_csv(f"{DATA_PATH}/disease_prediction/Training.csv")
test_df = pd.read_csv(f"{DATA_PATH}/disease_prediction/Testing.csv")

qa_df = pd.read_csv(r"C:\Users\Lenovo\Downloads\Medical\frontend\medquad_qa.csv")
questions = qa_df["Question"].tolist()
answers = qa_df["Answer"].tolist()

# =========================
# TRAIN ML MODEL
# =========================
X = train_df.drop("prognosis", axis=1)
y = train_df["prognosis"]

model_ml = RandomForestClassifier()
model_ml.fit(X, y)

X_test = test_df.drop("prognosis", axis=1).reindex(columns=X.columns, fill_value=0)
y_test = test_df["prognosis"]

print(f"✅ ML Accuracy: {model_ml.score(X_test, y_test):.2%}")

# =========================
# LOAD EMBEDDING MODEL (RAG)
# =========================
print("Loading embedding model...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = np.array(embed_model.encode(questions, show_progress_bar=True))

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# =========================
# LOAD SUMMARIZER
# =========================
try:
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    print("✅ Summarizer ready")
except:
    summarizer = None
    print("⚠️ Summarizer disabled")

# =========================
# HELPERS
# =========================
def normalize_input(text):
    text = text.lower()
    fixes = {
        "bodypain": "body pain",
        "vommit": "vomiting",
        "weak": "weakness",
        "blurred vision": "blurry vision"
    }
    for k,v in fixes.items():
        text = text.replace(k,v)
    return text

def simplify(text):
    if summarizer:
        try:
            return summarizer(text, max_length=60, min_length=20, do_sample=False)[0]['summary_text']
        except:
            pass
    return ". ".join(text.split(". ")[:2])

def get_disease_info(disease):
    desc = desc_df[desc_df["Disease"] == disease]["Description"]
    desc = desc.values[0] if len(desc) else None

    prec = prec_df[prec_df["Disease"] == disease]
    precautions = prec.iloc[0,1:].dropna().tolist() if not prec.empty else []

    return desc, precautions

def predict_rule(symptoms):
    best, score = None, 0
    for _, row in symptoms_df.iterrows():
        text = " ".join(row.dropna().astype(str)).lower()
        matches = sum(s in text for s in symptoms)
        if matches > score:
            score = matches
            best = row["Disease"]
    return best

def predict_ml(symptoms):
    input_df = pd.DataFrame(
        [[1 if col in symptoms else 0 for col in X.columns]],
        columns=X.columns
    )
    return model_ml.predict(input_df)[0]

def rag_answer(query):
    vec = embed_model.encode([query])
    D, I = index.search(vec, k=2)
    text = " ".join(answers[i] for i in I[0])
    return simplify(text)

def emergency_check(text):
    danger = ["chest pain","breathing","unconscious","bleeding"]
    if any(d in text for d in danger):
        return "\n🚨 EMERGENCY: Seek medical help immediately."
    return ""

# =========================
# MAIN AI FUNCTION
# =========================
def ask(user_input):

    user_input = normalize_input(user_input)
    symptoms = user_input.split()

    disease_rule = predict_rule(symptoms)
    disease_ml = predict_ml(symptoms)
    ai_text = rag_answer(user_input)

    response = ""

    # CONDITION
    if disease_rule:
        desc, precautions = get_disease_info(disease_rule)

        response += f"\n🩺 POSSIBLE CONDITION\n{disease_rule}\n"

        if desc:
            response += f"\nℹ️ WHAT IT MEANS\n{simplify(desc)}\n"

        if precautions:
            response += "\n🏥 CARE ADVICE\n"
            for p in precautions[:4]:
                response += f"• {p}\n"

    # AI Guidance
    response += "\n🧠 AI GUIDANCE\n"
    response += f"{ai_text}\n"

    # Emergency
    response += emergency_check(user_input)

    # MODEL CHECK
    response += "\n🤖 MODEL CHECK\n"
    response += f"ML Prediction: {disease_ml}\n"
    response += f"Dataset Match: {disease_rule}\n"

    if disease_ml == disease_rule:
        response += "✅ High confidence result\n"
    else:
        response += "⚠️ Results differ — consult a doctor\n"

    return response

# =========================
# RUN
# =========================
if __name__ == "__main__":
    while True:
        q = input("\nAsk symptoms (or exit): ")
        if q.lower() == "exit":
            break
        print(ask(q))
    faiss.write_index(index, "rag_index.faiss")
    print("✅ Models saved")
