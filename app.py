from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# โหลดโมเดล
model = joblib.load('carbon_footprint_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    electricity_usage = data['electricity_usage']
    prediction = model.predict([[electricity_usage]])
    return jsonify({'carbon_footprint': prediction[0]})

@app.route('/')
def home():
    return "Welcome to Carbon Footprint API!"

if __name__ == '__main__':
    app.run(debug=True)
    