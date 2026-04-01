from flask import Flask, request, jsonify
from flask_cors import CORS
from medical import ask
from deep_translator import GoogleTranslator
from vision.image_analyzer import ImageAnalyzer
import os

app = Flask(__name__)
CORS(app)

analyzer = ImageAnalyzer()

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/analyze-image", methods=["POST"])
def analyze_image():

    file = request.files["image"]

    path = os.path.join(UPLOAD_FOLDER, file.filename)

    file.save(path)

    label, score = analyzer.analyze(path)

    return jsonify({
        "prediction": label,
        "confidence": float(score)
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_input = data.get("query")
    target_lang = data.get("lang", "en") # Default to english ('en') if not provided
    
    # 1. Translate User Input from Regional -> English (if necessary)
    if target_lang != "en":
        # 'auto' auto-detects the source language
        user_input = GoogleTranslator(source='auto', target='en').translate(user_input)
    
    # 2. Get AI Prediction in English
    prediction_result = ask(user_input) 
    
    # 3. Translate Prediction back to English -> Regional (if necessary)
    if target_lang != "en":
        prediction_result = GoogleTranslator(source='en', target=target_lang).translate(prediction_result)
        
    return jsonify({"reply": prediction_result})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
