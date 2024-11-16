from fastapi import FastAPI
import joblib
from pydantic import BaseModel

# Load pre-trained model and scaler
model = joblib.load('knn_model.joblib')
scaler = joblib.load('Models/scaler.joblib')

# Initialize FastAPI app
app = FastAPI()

# Define a simple root endpoint that returns a welcome message
@app.get("/")
def root():
    return {"message": "Welcome to Tuwaiq Academy"}

# Define a Pydantic model for input data validation
class InputFeatures(BaseModel):
    age: int
    appearance: int
    goals: int
    minutes_played: int
    Highest_valuated_price_euro: float
    price_category: str

def preprocessing(input_features: InputFeatures):
    # Create a feature dictionary based on the input data
    dict_f = {
        'age': input_features.age,
        'appearance': input_features.appearance,
        'goals': input_features.goals,
        'minutes_played': input_features.minutes_played,
        'Highest_valuated_price_euro': input_features.Highest_valuated_price_euro,
        'price_category_Premium': input_features.price_category == 'Premium',
        'price_category_Mid': input_features.price_category == 'Mid',
        'price_category_Budget': input_features.price_category == 'Budget'
    }

    # Convert dictionary values to a list in the correct order
    features_list = [dict_f[key] for key in sorted(dict_f)]
    
    # Return the list directly without converting to numpy array
    return features_list

@app.get("/predict")
def get_prediction(input_features: InputFeatures):
    # Preprocess and scale input features
    data = preprocessing(input_features)
    scaled_data = scaler.transform([data])  # scale expects a 2D list, so wrap data in another list

    # Make a prediction using the pre-trained model
    y_pred = model.predict(scaled_data)

    # Return prediction result as a JSON response
    return {"pred": y_pred.tolist()[0]}

@app.post("/predict")
async def post_prediction(input_features: InputFeatures):
    # Preprocess and scale input features
    data = preprocessing(input_features)
    scaled_data = scaler.transform([data])  # scale expects a 2D list, so wrap data in another list

    # Make a prediction using the pre-trained model
    y_pred = model.predict(scaled_data)

    # Return prediction result as a JSON response
    return {"pred": y_pred.tolist()[0]}
