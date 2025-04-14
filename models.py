import joblib

# Save the models and preprocessing objects
joblib.dump(clf, 'classification_model.pkl')
joblib.dump(reg, 'regression_model.pkl')
joblib.dump(scaler_class, 'scaler_class.pkl')
joblib.dump(scaler_reg, 'scaler_reg.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')

# Load the models and preprocessing objects
clf = joblib.load('classification_model.pkl')
reg = joblib.load('regression_model.pkl')
scaler_class = joblib.load('scaler_class.pkl')
scaler_reg = joblib.load('scaler_reg.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Function to handle prediction requests
def predict_crop_yield(season, state, area, rainfall, fertilizer, pesticide):
    season_encoded = label_encoders["Season"].transform([season])[0] if season in label_encoders["Season"].classes_ else -1
    state_encoded = label_encoders["State"].transform([state])[0] if state in label_encoders["State"].classes_ else -1

    input_class = np.array([[season_encoded, state_encoded, area, rainfall, fertilizer, pesticide]])
    input_class_scaled = scaler_class.transform(input_class)
    predicted_crop_idx = clf.predict(input_class_scaled)[0]
    predicted_crop = label_encoders["Crop"].inverse_transform([predicted_crop_idx])[0]
    
    input_reg = np.array([[predicted_crop_idx, season_encoded, state_encoded, area, rainfall, fertilizer, pesticide]])
    input_reg_scaled = scaler_reg.transform(input_reg)
    predicted_yield = reg.predict(input_reg_scaled)[0]
    
    return {
        "recommended_crop": predicted_crop,
        "predicted_yield": round(predicted_yield, 2),
        "model_accuracy": round(accuracy * 100, 2)
    }