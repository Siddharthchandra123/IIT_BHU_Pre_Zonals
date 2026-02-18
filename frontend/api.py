from flask import Flask, request, jsonify
from flask_cors import CORS
from medical import ask

app = Flask(__name__)
CORS(app)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_input = data.get("query")
    response = ask(user_input) # Calls your existing AI logic
    return jsonify({"reply": response})

if __name__ == '__main__':
    app.run(port=5000)