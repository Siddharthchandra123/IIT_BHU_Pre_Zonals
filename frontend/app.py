from flask import Flask, request, jsonify
from flask_cors import CORS
from medical import ask
from deep_translator import GoogleTranslator

app = Flask(__name__)
CORS(app)

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
    app.run(port=5000)
