from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
from flask_cors import CORS

app = Flask(__name__, template_folder='templates', static_folder='static')  # Serve HTML and static files
CORS(app)  # Enable CORS

# Load Models and Preprocessing Objects
clf = joblib.load('classification_model.pkl')
reg = joblib.load('regression_model.pkl')
scaler_class = joblib.load('scaler_class.pkl')
scaler_reg = joblib.load('scaler_reg.pkl')
label_encoders = joblib.load('label_encoders.pkl')

@app.route('/')
def index():
    return render_template('index.html')  # Render the HTML page

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from the frontend
        data = request.json
        season = data['season']
        state = data['state']
        area = float(data['area'])
        rainfall = float(data['rainfall'])
        fertilizer = float(data['fertilizer'])
        pesticide = float(data['pesticide'])

        # Encode categorical inputs
        season_encoded = label_encoders["Season"].transform([season])[0] if season in label_encoders["Season"].classes_ else -1
        state_encoded = label_encoders["State"].transform([state])[0] if state in label_encoders["State"].classes_ else -1

        # Prepare input for classification
        input_class = np.array([[season_encoded, state_encoded, area, rainfall, fertilizer, pesticide]])
        input_class_scaled = scaler_class.transform(input_class)
        predicted_crop_idx = clf.predict(input_class_scaled)[0]
        predicted_crop = label_encoders["Crop"].inverse_transform([predicted_crop_idx])[0]

        # Prepare input for regression
        input_reg = np.array([[predicted_crop_idx, season_encoded, state_encoded, area, rainfall, fertilizer, pesticide]])
        input_reg_scaled = scaler_reg.transform(input_reg)
        predicted_yield = reg.predict(input_reg_scaled)[0]

        # Return the prediction results
        return jsonify({
            "recommended_crop": predicted_crop,
            "predicted_yield": round(predicted_yield, 2),
            "model_accuracy": "N/A"  # You can calculate accuracy separately if needed
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 400  # Return error message if something goes wrong

if __name__ == '__main__':
    app.run(debug=True)