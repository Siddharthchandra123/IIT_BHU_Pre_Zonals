# 🏥 Chikitsalya — AI-Powered Rural Telehealth & Clinical Decision Support

## 🌍 Overview
**Chikitsalya** is an AI-driven medical assistance platform designed to improve healthcare accessibility, early diagnosis, and clinical decision support in underserved and rural regions.

The system combines **symptom analysis, medical knowledge retrieval (RAG), risk prediction models, and hospital connectivity** to assist patients and healthcare providers in making timely and informed decisions.

---

## 🎯 Problem Statement
Healthcare access in rural and semi-urban areas faces several challenges:

- ❌ Shortage of qualified doctors
- ❌ Delayed diagnosis & treatment
- ❌ Lack of medical awareness
- ❌ Limited connectivity to nearby hospitals
- ❌ Overburdened healthcare facilities

**Chikitsalya aims to bridge this gap using AI-powered medical assistance.**

---

## 🚀 Key Features

### 🤖 AI Symptom Analysis
- Accepts patient symptoms via chat or form.
- Uses a **Random Forest Classifier** trained on extensive disease-symptom datasets.
- Provides a probability-based confidence score for each prediction.

### 📚 Medical Knowledge RAG System
- **Retrieval-Augmented Generation (RAG)**: Combines pre-trained models with a custom medical knowledge base.
- **FAISS Vector Database**: Enables lightning-fast similarity searches across thousands of medical Q&A pairs (MedQuad dataset).
- **Sentence Transformers**: Uses `all-MiniLM-L6-v2` to create high-dimensional semantic embeddings for precise matching.
- **Reduced Hallucinations**: Direct evidence-based retrieval ensures answers are grounded in verified medical text.

### ⚠️ Risk Prediction & Severity
- ML models assess symptom severity and urgency.
- **High-Risk Detection**: Automatically flags critical symptoms (e.g., chest pain, difficulty breathing) for immediate emergency alerts.

### 🏥 Nearest Hospital Connectivity
- Finds nearby hospitals & healthcare centers.
- Enables quick referrals and location-based triage during emergencies.

### 🌐 Multilingual & Rural-Friendly Interface
- **Google Translator Integration**: Real-time translation between English and regional languages.
- Simple, accessible UI designed for ease of use in low-resource settings.

---

## 🧠 AI & ML Components

### Models Used
- **Random Forest Classifier** — Core disease prediction engine.
- **Sentence Transformers (`all-MiniLM-L6-v2`)** — For semantic similarity in the RAG pipeline.
- **FAISS (Facebook AI Similarity Search)** — Vector indexing and retrieval engine.

---

## 🏗️ System Architecture

### 🔹 Frontend (EJS/Node.js)
- **Express Server**: Handles routing, UI rendering, and proxying requests to the AI service.
- **EJS Templating**: dynamic page rendering for a seamless user experience.

### 🔹 Backend (Python/Flask)
- **Flask API**: Acts as the bridge between the frontend and the AI models.
- **Inference Service**: Runs the ML models and FAISS search in real-time.

---

## 🔄 Workflow
1. **User Input**: Patient describes symptoms in their preferred language.
2. **Translation**: If non-English, query is translated to English for processing.
3. **Symptom Parsing**: NLP engine identifies key symptoms.
4. **Vector Search (RAG)**: FAISS retrieves the most relevant medical guidance from the database.
5. **ML Prediction**: Random Forest model calculates the most likely condition.
6. **Output**: The system displays Possible Conditions, care advice, emergency alerts, and translates it back to the user's language.

---

## 🧰 Technology Stack

| Layer | Technology |
|------|-----------|
| **Frontend** | HTML5, CSS3 (Vanilla), JavaScript, EJS |
| **Backend** | Node.js, Express |
| **AI Services** | Python 3.x, Flask, Flask-CORS |
| **ML Libraries** | scikit-learn, joblib |
| **RAG Pipeline** | FAISS, Sentence Transformers, Transformers |
| **Translation** | deep-translator (Google Translator API) |
| **Data Handling** | Pandas, NumPy |

---

## ⚙️ Installation & Setup (Local Development)

### 1️⃣ Clone Repository
```bash
git clone https://github.com/Siddharthchandra123/IIT_BHU_Pre_Zonals.git
cd IIT_BHU_Pre_Zonals
```

### 2️⃣ Navigate to Project Folder
```bash
cd local_host
```

### 3️⃣ AI Service Setup (Python)
Ensure you have Python installed. It is recommended to use a virtual environment.
```bash
# Install dependencies
pip install -r requirements.txt

# Start the AI Backend
python API.py
```

### 4️⃣ Server Setup (Node.js)
Open a new terminal window:
```bash
cd local_host

# Install dependencies
npm install

# Start the Frontend Server
npm start
```

### 5️⃣ Open in Browser
Visit [http://localhost:3000](http://localhost:3000)

---

## 🛡️ Safety & Reliability

- **Clinical Decision Support**: Designed as a tool for health workers and patients, not a final diagnostic authority.
- **Emergency Protocols**: Hardcoded high-severity triggers for life-threatening symptoms.
- **Verified Data**: Uses the MedQuad dataset for grounding knowledge.

---

## 👨‍💻 Author
**Siddharth Chandra**

---

## ⭐ Support
If you find this project useful, please consider giving it a ⭐ on GitHub!