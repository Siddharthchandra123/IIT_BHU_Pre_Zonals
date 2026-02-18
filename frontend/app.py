from flask import Flask, request, jsonify
from flask_cors import CORS
from medical import ask  # Import the 'ask' function from your organized code
app = Flask(__name__)
CORS(app) # This allows the Node.js server to talk to Flask

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    user_input = data.get("query")
    
    # This calls the 'ask' function from your organized code
    prediction_result = ask(user_input) 
    
    return jsonify({"reply": prediction_result})

if __name__ == '__main__':
    # Since it's all on Windows, localhost (127.0.0.1) works perfectly
    app.run(port=5000)