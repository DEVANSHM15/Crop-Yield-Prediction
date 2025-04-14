# CropAndYieldPredictionSystem
Welcome to our Crop Yield Prediction tool. Enter your farm details to get accurate yield predictions and crop recommendations based on environmental factors.  Our system uses machine learning models trained on historical agricultural data across India to provide the most accurate predictions.


Overview
This project provides a machine learning-based solution for crop recommendation and yield prediction. It consists of:
A Flask web application (app.py) that serves predictions via API
A Jupyter notebook (SaveModels.ipynb) for training and saving models

Pre-trained models and preprocessing objects
File Descriptions
Core Files
app.py: Flask application that provides:
Web interface for user input
API endpoint (/predict) for crop recommendations and yield predictions
Integration with pre-trained models


SaveModels.ipynb: Jupyter notebook for:
Data preprocessing and cleaning
Model training (Random Forest Classifier and Regressor)
Hyperparameter tuning
Saving trained models and preprocessing objects

Supporting Files
classification_model.pkl: Pre-trained Random Forest Classifier for crop recommendation
regression_model.pkl: Pre-trained Random Forest Regressor for yield prediction
scaler_class.pkl: StandardScaler for classification features
scaler_reg.pkl: StandardScaler for regression features
label_encoders.pkl: Label encoders for categorical variables
crop_yield2.csv: Dataset used for training
templates/index.html: Frontend interface (HTML)
requirements.txt: Python dependencies

Setup Instructions
Prerequisites
Python 3.7+

Installation
Clone the repository
Create and activate a virtual environment:
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

Install dependencies:
pip install -r requirements.txt

Running the Application
Start the Flask server:
python app.py

The application will be available at http://localhost:5000

Using the API
Send a POST request to http://localhost:5000/predict with JSON payload:

{
    "season": "Kharif",
    "state": "Andhra Pradesh",
    "area": 1000,
    "rainfall": 1200,
    "fertilizer": 200,
    "pesticide": 50
}

Model Training
To retrain the models:
Open SaveModels.ipynb in Jupyter Notebook
Run all cells sequentially

The notebook will:
Load and preprocess data
Train classification and regression models
Perform hyperparameter tuning

Save updated models and preprocessing objects

Data Processing Pipeline
Data Cleaning:
Handles missing values with median imputation
Removes outliers using IQR method


Feature Engineering:
Encodes categorical variables (Crop, Season, State)
Scales numerical features using StandardScaler
Handles class imbalance with SMOTE


Model Training:
Random Forest Classifier for crop recommendation
Random Forest Regressor for yield prediction
Hyperparameter tuning using RandomizedSearchCV

Dependencies
Listed in requirements.txt:
Flask
scikit-learn
imbalanced-learn
pandas
numpy
joblib
Flask-CORS

Troubleshooting
If models fail to load, ensure all .pkl files are in the project root
For prediction errors, verify input data matches training data format
Check console logs for detailed error messages

Future Improvements
Add model accuracy metrics to API response
Implement model versioning
Add more comprehensive error handling
Include additional agricultural features
