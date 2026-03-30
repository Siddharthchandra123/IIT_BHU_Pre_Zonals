from flask import Flask, request, jsonify
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# ---------------- HEALTH CHECK ---------------- #
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "API is running 🚀"})


# ---------------- MAIN ENDPOINT ---------------- #
@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data or "query" not in data:
            return jsonify({"error": "Missing 'query' in request"}), 400

        user_input = data.get("query")
        lang_code = data.get("lang", "en")

        # Lazy import (VERY IMPORTANT)
        from medical import ask
        from deep_translator import GoogleTranslator

        # Step 1: Translate to English if needed
        if lang_code != "en":
            english_input = GoogleTranslator(
                source=lang_code, target="en"
            ).translate(user_input)
        else:
            english_input = user_input

        # Step 2: Get AI response
        ai_response = ask(english_input)

        # Step 3: Translate back
        if lang_code != "en":
            final_response = GoogleTranslator(
                source="en", target=lang_code
            ).translate(ai_response)
        else:
            final_response = ai_response

        return jsonify({"reply": final_response})

    except Exception as e:
        print("❌ ERROR:", str(e))
        return jsonify({
            "error": "Something went wrong",
            "details": str(e)
        }), 500


# ---------------- LOCAL RUN ---------------- #
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)