import pandas as pd
import faiss
import numpy as np
import joblib
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from transformers import pipeline


DATA_PATH = "../dataset"

symptoms_df = pd.read_csv(f"{DATA_PATH}/Disease symptom prediction/dataset.csv")
desc_df = pd.read_csv(f"{DATA_PATH}/Disease symptom prediction/symptom_Description.csv")
prec_df = pd.read_csv(f"{DATA_PATH}/Disease symptom prediction/symptom_precaution.csv")

train_df = pd.read_csv(f"{DATA_PATH}/disease_prediction/Training.csv")
test_df = pd.read_csv(f"{DATA_PATH}/disease_prediction/Testing.csv")

qa_df = pd.read_csv("medquad_qa.csv")
questions = qa_df["Question"].tolist()
answers = qa_df["Answer"].tolist()

X = train_df.drop("prognosis", axis=1)
y = train_df["prognosis"]

model_ml = RandomForestClassifier()
model_ml.fit(X, y)

X_test = test_df.drop("prognosis", axis=1).reindex(columns=X.columns, fill_value=0)
y_test = test_df["prognosis"]

print(f"✅ ML Accuracy: {model_ml.score(X_test, y_test):.2%}")
print("Loading embedding model...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = np.array(embed_model.encode(questions, show_progress_bar=True))

index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

try:
    summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
    print("✅ Summarizer ready")
except:
    summarizer = None
    print("⚠️ Summarizer disabled")

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
    stop_words = {"hi", "hello", "how", "are", "you", "whats", "up", "buddy", "i", "am", "my", "is", "the", "a", "an"}
    
    filtered_symptoms = [s for s in symptoms if s not in stop_words and len(s) > 2]
    
    if not filtered_symptoms:
        return None

    for _, row in symptoms_df.iterrows():
        text = " ".join(row.dropna().astype(str)).lower()
        matches = sum(1 for s in filtered_symptoms if f" {s} " in f" {text} " or s == text or text.startswith(f"{s} ") or text.endswith(f" {s}"))
        
        if matches > score:
            score = matches
            best = row["Disease"]
            
    if score > 0:
        return best
    return None

def predict_ml(symptoms):
    input_df = pd.DataFrame(
        [[1 if col in symptoms else 0 for col in X.columns]],
        columns=X.columns
    )
    probs = model_ml.predict_proba(input_df)[0]
    confidence = max(probs)
    
    if confidence < 0.65:
        return None
    return model_ml.predict(input_df)[0]

def rag_answer(query):
    vec = embed_model.encode([query])
    D, I = index.search(vec, k=2)
    
    valid_answers = [answers[I[0][i]] for i in range(len(I[0])) if D[0][i] < 1.2]
    
    if not valid_answers:
        return "I don't have enough specific medical information to answer that safely. Please consult a doctor."
        
    text = " ".join(valid_answers)
    return simplify(text)

def emergency_check(text):
    danger = ["chest pain","breathing","unconscious","bleeding"]
    if any(d in text for d in danger):
        return "\n🚨 EMERGENCY: Seek medical help immediately."
    return ""

def ask(user_input):

    user_input = normalize_input(user_input)
    symptoms = user_input.split()

    disease_rule = predict_rule(symptoms)
    disease_ml = predict_ml(symptoms)
    ai_text = rag_answer(user_input)

    response = ""

    if disease_rule:
        desc, precautions = get_disease_info(disease_rule)

        response += f"\n🩺 POSSIBLE CONDITION\n{disease_rule}\n"

        if desc:
            response += f"\nℹ️ WHAT IT MEANS\n{simplify(desc)}\n"

        if precautions:
            response += "\n🏥 CARE ADVICE\n"
            for p in precautions[:4]:
                response += f"• {p}\n"

    response += "\n🧠 AI GUIDANCE\n"
    response += f"{ai_text}\n"

    response += emergency_check(user_input)
    response += "\n🤖 MODEL CHECK\n"
    response += f"ML Prediction: {disease_ml if disease_ml else 'Inconclusive (Low Confidence)'}\n"
    response += f"Dataset Match: {disease_rule if disease_rule else 'No Match'}\n"

    if disease_ml and disease_rule and disease_ml == disease_rule:
        response += "✅ High confidence result\n"
    else:
        response += "⚠️ Results differ or are inconclusive — consult a doctor\n"

    return response

if __name__ == "__main__":
    while True:
        q = input("\nAsk symptoms (or exit): ")
        if q.lower() == "exit":
            break
        print(ask(q))
    faiss.write_index(index, "rag_index.faiss")
    print("✅ Models saved")
