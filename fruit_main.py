# fruit_main.py
# FastAPI application for fruit classification using the trained model from fruit.ipynb

import numpy as np
import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

# -------------------------------
# 1. Initialize FastAPI app
# -------------------------------
app = FastAPI(
    title="MARYMAY FRUITS PAGE",
    description="API for fruit classification based on size, shape, weight, price, color, and taste.",
    version="1.0.0"
)

# -------------------------------
# 2. Add CORS middleware (allow all origins for simplicity)
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# 3. Load all saved artifacts
# -------------------------------
# Path to the model directory (same as where fruit.ipynb saved them)
MODEL_DIR = Path("model")

try:
    model = joblib.load(MODEL_DIR / "model.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    target_encoder = joblib.load(MODEL_DIR / "target_encoder.pkl")
    # The following encoders are not strictly needed for prediction,
    # but could be used if you want to accept raw strings (e.g., 'round') instead of integers.
    # For now we keep them for potential future use.
    shape_encoder = joblib.load(MODEL_DIR / "shape_encoder.pkl")
    taste_encoder = joblib.load(MODEL_DIR / "taste_encoder.pkl")
    color_encoder = joblib.load(MODEL_DIR / "color_encoder.pkl")
    print("All artifacts loaded successfully.")
except FileNotFoundError as e:
    raise RuntimeError(f"Missing artifact file: {e}. Ensure the 'model' directory contains all .pkl files from training.")
except Exception as e:
    raise RuntimeError(f"Error loading artifacts: {e}")

# -------------------------------
# 4. Define input data model (Pydantic)
#    All categorical features must be supplied as integers,
#    using the same encoding that was used during training.
#    For convenience, we include example values in the Field description.
# -------------------------------
class FruitFeatures(BaseModel):
    size: float = Field(..., description="Size of the fruit in cm (e.g., 7.5)")
    shape: int = Field(..., description="Shape encoded: 0=long, 1=oval, 2=round")
    weight: float = Field(..., description="Weight in grams (e.g., 180.0)")
    avg_price: float = Field(..., description="Average price in ₹ (e.g., 65.0)")
    color: int = Field(..., description="Color encoded: see mapping below")
    taste: int = Field(..., description="Taste encoded: 0=sour, 1=sweet, 2=tangy")

    class Config:
        schema_extra = {
            "example": {
                "size": 8.5,
                "shape": 1,      # oval
                "weight": 174.6,
                "avg_price": 65.5,
                "color": 2,      # depends on color_encoder classes
                "taste": 1       # sweet
            }
        }

# -------------------------------
# 5. Define API endpoints
# -------------------------------
@app.get("/")
def home():
    """Root endpoint with a welcome message."""
    return {"message": "Welcome to the Fruit Classification API"}

@app.get("/health")
def health_check():
    """Health check endpoint to verify the API is running."""
    return {"status": "healthy", "message": "API is ready to accept requests"}

@app.post("/predict_fruit_name")
def predict_fruit_name(input_data: FruitFeatures):
    """
    Predict the fruit name based on the input features.

    Steps:
    1. Convert input to a 2D numpy array.
    2. Apply the same scaling used during training (only numerical columns).
    3. Run model prediction -> integer class code.
    4. Decode the integer to the actual fruit name using the target encoder.
    5. Return the predicted fruit name.
    """
    try:
        # Prepare feature array in the same order as training:
        # ['size (cm)', 'shape', 'weight (g)', 'avg_price (₹)', 'color', 'taste']
        features = np.array([[
            input_data.size,
            input_data.shape,
            input_data.weight,
            input_data.avg_price,
            input_data.color,
            input_data.taste
        ]])

        # Scale the numerical features (the scaler was fitted on the training set)
        # Note: The scaler expects the same columns: size, weight, avg_price.
        # Here we apply it to the whole row; it will scale all three numerical features.
        # However, the scaler was fitted on DataFrame with columns order: size, weight, avg_price.
        # Since our array has [size, shape, weight, avg_price, color, taste],
        # we need to scale only the numerical ones. The simplest way is to create a copy
        # and scale the appropriate indices.
        # Better: extract numerical part, scale, then reassemble.
        numerical_indices = [0, 2, 3]  # positions of size, weight, avg_price
        numerical_values = features[:, numerical_indices]
        numerical_scaled = scaler.transform(numerical_values)
        features[:, numerical_indices] = numerical_scaled

        # Predict class index
        pred_idx = model.predict(features)[0]

        # Decode index to fruit name
        fruit_name = target_encoder.inverse_transform([pred_idx])[0]

        return {"predicted_fruit": fruit_name, "prediction_code": int(pred_idx)}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# -------------------------------
# 6. Optional: Endpoint that accepts raw string categories (for convenience)
#    This endpoint converts color, shape, taste names to their encoded values
#    using the saved encoders. It's more user-friendly.
# -------------------------------
class FruitFeaturesRaw(BaseModel):
    size: float
    shape_name: str   # 'long', 'oval', 'round'
    weight: float
    avg_price: float
    color_name: str   # e.g., 'red', 'green', 'yellow', etc.
    taste_name: str   # 'sour', 'sweet', 'tangy'

    class Config:
        schema_extra = {
            "example": {
                "size": 8.5,
                "shape_name": "oval",
                "weight": 174.6,
                "avg_price": 65.5,
                "color_name": "green",
                "taste_name": "sweet"
            }
        }

@app.post("/predict_fruit_name_raw")
def predict_fruit_name_raw(input_data: FruitFeaturesRaw):
    """
    Same prediction but accepts human-readable category names.
    The API will encode them using the saved label encoders.
    """
    try:
        # Encode categorical inputs
        try:
            shape_encoded = shape_encoder.transform([input_data.shape_name])[0]
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid shape_name. Allowed: {list(shape_encoder.classes_)}")
        
        try:
            color_encoded = color_encoder.transform([input_data.color_name])[0]
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid color_name. Allowed: {list(color_encoder.classes_)}")
        
        try:
            taste_encoded = taste_encoder.transform([input_data.taste_name])[0]
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid taste_name. Allowed: {list(taste_encoder.classes_)}")

        # Prepare feature array
        features = np.array([[
            input_data.size,
            shape_encoded,
            input_data.weight,
            input_data.avg_price,
            color_encoded,
            taste_encoded
        ]])

        # Scale numerical features
        numerical_indices = [0, 2, 3]
        numerical_values = features[:, numerical_indices]
        numerical_scaled = scaler.transform(numerical_values)
        features[:, numerical_indices] = numerical_scaled

        # Predict
        pred_idx = model.predict(features)[0]
        fruit_name = target_encoder.inverse_transform([pred_idx])[0]

        return {"predicted_fruit": fruit_name, "prediction_code": int(pred_idx)}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# -------------------------------
# To run the API: uvicorn fruit_main:app --reload
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)