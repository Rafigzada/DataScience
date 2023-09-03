import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
app = FastAPI()
# Load the trained model
with open("gb_classifier1_model.pkl", "rb") as model_file:
    gbmodel = pickle.load(model_file)
class InputData(BaseModel):

    gender: int
    height: int
    weight: float
    ap_hi: int
    ap_lo: int
    cholesterol: int
    gluc: int
    smoke: int
    alco: int
    active: int
    age_years: float
    bmi: float
    pulse_pressure: int
    age_category_elderly: int
    age_category_middle_aged: int  # Assuming 'age_category_middle-aged' is a column in your dataset
    age_category_young: int
    bmi_category_normal: int
    bmi_category_obese: int
    bmi_category_overweight: int
    bmi_category_underweight: int
    bmi_category_nan: int

    age_years: float  # Assuming '1' is a column in your dataset
    pulse_pressure_1: int
    age_years_pulse_pressure: float

@app.get('/')
def index():
    return {'message': 'Welcome to the Cardiovascular Disease Prediction API'}

@app.get('/predict')
async def predict_gbnote(data: InputData):
    try:
        # Extract the features from the input data
        features = [
           
            data.gender,
            data.height,
            data.weight,
            data.ap_hi,
            data.ap_lo,
            data.cholesterol,
            data.gluc,
            data.smoke,
            data.alco,
            data.active,
            data.age_years,
            data.bmi,
            data.pulse_pressure,
            data.age_category_elderly,
            data.age_category_middle_aged,  # Assuming 'age_category_middle-aged' is a column in your dataset
            data.age_category_young,
            data.bmi_category_normal,
            data.bmi_category_obese,
            data.bmi_category_overweight,
            data.bmi_category_underweight,
            data.bmi_category_nan,
            data.age_years,  # Assuming '1' is a column in your dataset
            data.pulse_pressure_1,
            data.age_years_pulse_pressure
        ]

        # Handle missing or blank values by providing default values (0 in this example)
        features = [0 if feature is None else feature for feature in features]

        # Convert the features to a 2D array for prediction
        input_features_2d = [features]

        # Make predictions using your classifier (replace 'classifier' with your actual model)
        prediction = gbmodel.predict(input_features_2d)

        # Assuming the model returns a probability (between 0 and 1)
        if prediction[0] > 0.5:
            result = "CardioDetect!"
        else:
            result = "You haven't heart disease"

        return {"prediction": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if _name_ == '_main_':
    uvicorn.run(app, host='127.0.0.1', port=8001)