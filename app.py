from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load models
crop_model = pickle.load(open("crop_model.pkl", "rb"))
fert_model = pickle.load(open("fertilizer_model.pkl", "rb"))

soil_types = ["Clayey", "Loamy", "Sandy", 'black', 'red']

crop_names = [
    "Barley", "Cotton", "Ground Nuts", "Maize", "Millets",
    "Oil seeds", "Paddy", "Pulses", "Sugarcane", "Tobacco", "Wheat"
]

fertilizer_names = [
    "10-26-26", "14-35-14", "17-17-17",
    "20-20", "28-28", "DAP", "Urea"
]

@app.route('/', methods=['GET', 'POST'])
def home():
    crop_result = None
    fert_result = None
    error = None

    if request.method == 'POST':
        try:
            # Extract data from form
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            soil_moisture = float(request.form['soil_moisture'])
            soil_type = request.form['soil_type']
            nitrogen = int(request.form['nitrogen'])
            potassium = int(request.form['potassium'])
            phosphorous = int(request.form['phosphorous'])

            # Encode soil type
            soil_map = {
                "black": 0,
                "clayey": 1,
                "loamy": 2,
                "red": 3,
                "sandy": 4
            }
            
            soil_type = soil_type.lower()
            if soil_type in soil_map:
                soil_encoded = soil_map[soil_type]
            else:
                 # Default to 0 (Black) if not found, or handle error
                soil_encoded = 0 

            # Predict Crop
            X_crop = np.array([[temperature, humidity, soil_moisture,
                                soil_encoded, nitrogen, potassium, phosphorous]])
            crop_pred = int(crop_model.predict(X_crop)[0])
            crop_result = crop_names[crop_pred]

            # Predict Fertilizer
            X_fert = np.array([[temperature, humidity, soil_moisture,
                                soil_encoded, crop_pred,
                                nitrogen, potassium, phosphorous]])
            fert_pred = int(fert_model.predict(X_fert)[0])
            fert_result = fertilizer_names[fert_pred]

        except Exception as e:
            error = str(e)

    return render_template('index.html', crop=crop_result, fertilizer=fert_result, error=error)

if __name__ == '__main__':
    app.run(debug=True)