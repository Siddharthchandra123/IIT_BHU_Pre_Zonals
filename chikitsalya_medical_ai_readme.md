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
- Accepts patient symptoms via chat or form
- Uses NLP to interpret symptoms
- Suggests possible conditions with risk levels

### 📚 Medical Knowledge RAG System
- Retrieves verified medical information
- Reduces hallucination using context-based retrieval
- Provides explainable responses

### ⚠️ Risk Prediction Models
- ML models assess severity & urgency
- Early warning for critical conditions

### 🏥 Nearest Hospital Connectivity
- Finds nearby hospitals & healthcare centers
- Enables quick referrals during emergencies

### 🌐 Multilingual & Rural-Friendly Interface
- Designed for low-literacy & regional language use
- Simple and accessible UI

---

## 🧠 AI & ML Components

### Models Used
- **Random Forest Classifier** — risk prediction & classification
- **Transformer-based NLP Models** — symptom understanding
- **Sentence Transformers** — semantic similarity search
- **FAISS Vector Database** — fast medical knowledge retrieval

---

## 🏗️ System Architecture

### 🔹 Frontend
- HTML, CSS, JavaScript
- EJS templating
- Responsive & lightweight design

### 🔹 Backend
- Node.js / Express (UI & routing)
- Python FastAPI services (AI inference)

### 🔹 AI Layer
- NLP pipeline for symptom parsing
- RAG pipeline for medical knowledge retrieval
- ML model inference engine

### 🔹 Data Layer
- FAISS vector store
- Medical datasets & guidelines
- Patient interaction logs (anonymized)

---

## 🔄 Workflow
1. User inputs symptoms
2. NLP engine interprets symptoms
3. Vector search retrieves relevant medical context
4. ML model assesses risk level
5. System returns:
   - possible conditions
   - severity level
   - recommended next steps
   - nearby hospitals (if needed)

---

## 🧰 Technology Stack

| Layer | Technology |
|------|-----------|
| Frontend | HTML, CSS, JavaScript, EJS |
| Backend | Node.js, Express |
| AI Services | Python, FastAPI |
| ML Libraries | scikit-learn, transformers |
| Embeddings | Sentence Transformers |
| Vector DB | FAISS |
| Data Handling | Pandas, NumPy |

---

## 🛡️ Safety & Reliability

- Uses verified medical knowledge sources
- Risk-level classification instead of definitive diagnosis
- Encourages professional consultation
- Designed to minimize AI hallucinations via RAG

---

## 📈 Use Cases

- Rural telehealth assistance
- Primary health screening
- Emergency triage support
- Community health workers support tool
- Health awareness & early detection

---

## 🔮 Future Enhancements

- Integration with government health systems
- Offline functionality for low-connectivity areas
- Wearable & IoT health monitoring integration
- Voice-based interaction
- Doctor teleconsultation module

---

## ⚙️ Installation & Setup

### 1️⃣ Clone Repository
```bash
git clone https://github.com/yourusername/chikitsalya.git
cd chikitsalya
```

### 2️⃣ Backend Setup
```bash
cd backend
npm install
npm start
```

### 3️⃣ AI Service Setup
```bash
cd ai-service
pip install -r requirements.txt
python app.py
```

### 4️⃣ Open in Browser
```
http://localhost:3000
```

---

## 🤝 Contribution
Contributions are welcome!

1. Fork the repository
2. Create a new branch
3. Commit changes
4. Submit a pull request

---

## ⚠️ Disclaimer
Chikitsalya is a **clinical decision support tool** and not a replacement for professional medical advice, diagnosis, or treatment.

---

## 👨‍💻 Author
**Siddharth Chandra**

---

## ⭐ If you find this pr