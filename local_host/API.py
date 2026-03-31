from flask import Flask, request, jsonify
from flask_cors import CORS
from medical import ask
from deep_translator import GoogleTranslator
import os

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_input = data.get("query")
    lang_code = data.get("lang", "en")

    try:
        # Step 1: Translate non-English input to English for the AI
        if lang_code != "en":
            english_input = GoogleTranslator(source=lang_code, target='en').translate(user_input)
        else:
            english_input = user_input
            
        # Step 2: Get AI response natively in English
        ai_response = ask(english_input) 

        # Step 3: Translate the AI response back to the user's language
        if lang_code != "en":
            final_response = GoogleTranslator(source='en', target=lang_code).translate(ai_response)
        else:
            final_response = ai_response

        return jsonify({"reply": final_response})
    
    except Exception as e:
        print("Translation Error:", str(e))
        # Fallback to English if translation fails
        ai_response = ask(user_input)
        return jsonify({"reply": f"Translation Error. Fallback to English:\n\n{ai_response}"})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
