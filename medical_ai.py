import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------- OPTIONAL SUMMARIZER ----------
USE_SUMMARIZER = True
symptoms_df = pd.read_csv(r"C:\Users\Lenovo\Downloads\IIT BHU\dataset\Disease symptom prediction\dataset.csv")
desc_df = pd.read_csv(r"C:\Users\Lenovo\Downloads\IIT BHU\dataset\Disease symptom prediction\symptom_Description.csv")
prec_df = pd.read_csv(r"C:\Users\Lenovo\Downloads\IIT BHU\dataset\Disease symptom prediction\symptom_precaution.csv")
severity_df = pd.read_csv(r"C:\Users\Lenovo\Downloads\IIT BHU\dataset\Disease symptom prediction\Symptom-severity.csv")
train_df = pd.read_csv(r"C:\Users\Lenovo\Downloads\IIT BHU\dataset\disease_prediction\Training.csv")
test_df = pd.read_csv(r"C:\Users\Lenovo\Downloads\IIT BHU\dataset\disease_prediction\Testing.csv")


X = train_df.drop("prognosis", axis=1)
y = train_df["prognosis"]
from sklearn.ensemble import RandomForestClassifier

model_ml = RandomForestClassifier()
model_ml.fit(X, y)
def predict_disease_ml(symptoms_list):
    input_data = pd.DataFrame(
        [[1 if col in symptoms_list else 0 for col in X.columns]],
        columns=X.columns
    )
    return model_ml.predict(input_data)[0]


X_test = test_df.drop("prognosis", axis=1)
y_test = test_df["prognosis"]

# 🔥 ALIGN COLUMNS
X_test = X_test.reindex(columns=X.columns, fill_value=0)

accuracy = model_ml.score(X_test, y_test)

print("Accuracy:", accuracy)


print("Accuracy:", accuracy)

if USE_SUMMARIZER:
    try:
        from transformers import pipeline
        summarizer = pipeline(
            "summarization",
            model="sshleifer/distilbart-cnn-12-6"  # lighter & faster
        )
        print("✅ Summarizer loaded")
    except:
        summarizer = None
        print("⚠️ Summarizer not available, using simple shortening")
else:
    summarizer = None


# ---------- SIMPLIFY ANSWER ----------
def simplify_answer(text):

    # limit very long answers
    text = text[:1200]

    if summarizer:
        try:
            summary = summarizer(
                text,
                max_length=60,
                min_length=20,
                do_sample=False
            )[0]['summary_text']
            return summary
        except:
            pass

    # fallback (no AI)
    sentences = text.split(". ")
    return ". ".join(sentences[:3])


# ---------- FORMAT OUTPUT ----------
def format_output(text):
    parts = text.split(". ")
    bullets = "\n".join(["• " + p.strip() for p in parts if len(p) > 5])
    return bullets


# ---------- EMERGENCY CHECK ----------
def emergency_check(text):
    danger_words = [
        "chest pain",
        "difficulty breathing",
        "unconscious",
        "bleeding",
        "severe pain"
    ]
    for word in danger_words:
        if word in text.lower():
            return "\n🚨 EMERGENCY: Seek immediate medical help."
    return ""


# ---------- LOAD DATA ----------
df = pd.read_csv("medquad_qa.csv")

questions = df["Question"].tolist()
answers = df["Answer"].tolist()

# ---------- LOAD EMBEDDING MODEL ----------
print("Loading AI model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

print("Creating embeddings...")
embeddings = model.encode(questions, show_progress_bar=True)
embeddings = np.array(embeddings)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print("✅ Medical AI ready!")

def normalize_for_ml(text):
    return text.replace(" ", "_")

def normalize_input(text):
    text = text.lower()

    replacements = {
        "bodypain": "body pain",
        "vommit": "vomiting",
        "blurred vision": "blurry vision",
        "weak": "weakness"
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    return text

# ---------- ASK FUNCTION ----------
def ask(user_input):

    # -------- AI semantic response --------
    ml_text = normalize_for_ml(user_input)

    input_df = pd.DataFrame(
        [[1 if col in ml_text else 0 for col in X.columns]],
        columns=X.columns
    )

    user_input = normalize_input(user_input)
    query_vector = model.encode([user_input])

    distances, indices = index.search(query_vector, k=5)

    ai_responses = []

    for dist, idx in zip(distances[0], indices[0]):
        text = answers[idx].lower()

        # keep only if symptoms overlap
        if dist < 1.2 and any(sym in text for sym in user_input.split()):
            ai_responses.append(simplify_answer(answers[idx]))


    ai_text = " ".join(ai_responses[:2])


    probs = model_ml.predict_proba(input_df)[0]
    confidence = max(probs)

    disease_ml = model_ml.predict(input_df)[0]

    disease_rule = predict_disease(user_input)

    response = "\n"

    if disease_rule:
        desc = get_description(disease_rule)
        precautions = get_precautions(disease_rule)
        risk = check_risk(user_input)

        response += f"\n🩺 Possible Condition: {disease_rule}\n"

        if desc:
            response += f"\nAbout:\n{simplify_answer(desc)}\n"

        if precautions:
            response += "\nCare Advice:\n"
            for p in precautions:
                response += f"• {p}\n"

        if risk:
            response += "\n⚠️ High-risk symptoms detected. Consider seeing a doctor.\n"

    response += "\nAdditional Guidance:\n"
    response += format_output(ai_text)

    response += symptom_check(user_input)
    response += emergency_check(user_input)
    response += care_advice()

    # -------- Prediction comparison --------
    response += f"\n🧠 ML Prediction: {disease_ml}"
    response += f"\n📚 Dataset Match: {disease_rule}"

    if disease_ml == disease_rule:
        response += "\n✅ High confidence prediction"
    else:
        response += "\n⚠️ Predictions differ — consult a doctor"

    return response

def get_precautions(disease):
    result = prec_df[prec_df["Disease"] == disease]

    if not result.empty:
        precautions = result.iloc[0, 1:].dropna().tolist()
        return precautions

    return []

def symptom_check(text):
    symptoms = text.lower()

    if "pale" in symptoms and "weakness" in symptoms:
        return "\n⚠️ Could indicate anemia or dehydration."

    if "fever" in symptoms and "vomit" in symptoms:
        return "\n⚠️ May be viral infection, food poisoning, or dengue."

    if "nausea" in symptoms and "weakness" in symptoms:
        return "\n⚠️ Could be dehydration or infection."

    return ""
def predict_disease(user_input):
    user_symptoms = user_input.lower().split()

    best_match = None
    max_matches = 0

    for _, row in symptoms_df.iterrows():
        row_symptoms = " ".join(row.dropna().astype(str)).lower()

        matches = sum(sym in row_symptoms for sym in user_symptoms)

        if matches > max_matches:
            max_matches = matches
            best_match = row["Disease"]

    return best_match

def get_description(disease):
    result = desc_df[desc_df["Disease"] == disease]
    if not result.empty:
        return result["Description"].values[0]
def check_risk(user_input):
    text = user_input.lower()
    high_risk = []

    for _, row in severity_df.iterrows():
        symptom = row["Symptom"].lower()
        severity = row["weight"]

        if symptom in text and severity > 5:
            high_risk.append(symptom)

    return high_risk
def health_assistant(user_input):

    disease = predict_disease(user_input)

    if not disease:
        return "Unable to determine condition."

    desc = get_description(disease)
    precautions = get_precautions(disease)
    risk = check_risk(user_input)

    response = f"\nPossible Condition: {disease}\n"

    response += f"\nAbout:\n{desc}\n"

    response += "\nCare Advice:\n"
    for p in precautions:
        response += f"• {p}\n"

    if risk:
        response += "\n⚠️ High-risk symptoms detected. Consider seeing a doctor.\n"

    return response

def care_advice():
    return "\n\n👉 Drink ORS or fluids\n👉 Rest well\n👉 See doctor if symptoms last > 2 days"


# ---------- CHAT LOOP ----------
while True:
    q = input("\nAsk health question (or type exit): ")
    if q.lower() == "exit":
        break

    print("\nAI Doctor:\n", ask(q))

def save_model():
    import joblib
    import faiss

    # save ML model
    joblib.dump(model_ml, "disease_model.pkl")

    # save feature columns
    joblib.dump(X.columns.tolist(), "feature_columns.pkl")

    # save FAISS index
    faiss.write_index(index, "rag_index.faiss")

    print("✅ Models saved!")

save_model()
