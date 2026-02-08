import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import logging

app = Flask(__name__)
logging.basicConfig(
    filename="api.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load trained model & scaler
model = joblib.load("ev_sales_model.pkl")
scaler = joblib.load("scaler.pkl")
@app.route("/")
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    try:
        if request.method == 'GET':
            return jsonify({"message": "API is running"}), 200
        data = request.get_json(force=True)
        logging.info(f"Received data: {data}")

        features = np.array([
            float(data['record_id']),
            float(data['country']),
            float(data['region']),
            float(data['ev_brand']),
            float(data['vehicle_type']),
            float(data['battery_capacity_kwh']),
            float(data['vehicle_range_km']),
            float(data['charging_time_hours']),
            float(data['charging_stations']),
            float(data['govt_incentives']),
            float(data['avg_ev_price_usd']),
            float(data['year']),
            float(data['fuel_price_index']),
            float(data['co2_regulation']),
            float(data['gdp_growth'])
        ]).reshape(1, -1)

        pred = model.predict(features)
        logging.info(f"Prediction: {pred[0]}")
        return jsonify({"Predicted_EV_Sales": float(pred[0])})
    except Exception as e:
        logging.error(str(e))
        return jsonify({"error": str(e)}), 500



if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)