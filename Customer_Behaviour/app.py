# app.py - FastAPI service for customer return prediction

import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal, Optional
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import logging

# ------------------------------
# 1. Load pre‑trained model and scaler
# ------------------------------
# The notebook saved:
#   - best_xgb_model.pkl : the tuned XGBoost classifier
#   - scaler.pkl         : MinMaxScaler fitted on the numerical columns
model = joblib.load("best_xgb_model.pkl")
scaler = joblib.load("scaler.pkl")

# ------------------------------
# 2. Define categorical mappings (reconstructed from notebook outputs)
# ------------------------------
# Because the original LabelEncoders were not saved, we rebuild the exact
# mappings that were used during training (including the bug where
# Payment_Method was replaced by Product_Category encoding).

# Product_Category mapping (alphabetical order of unique values)
product_cat_mapping = {
    "Beauty": 0,
    "Books": 1,
    "Electronics": 2,
    "Fashion": 3,
    "Food": 4,
    "Home & Garden": 5,
    "Sports": 6,
    "Toys": 7
}

# Device_Type mapping (alphabetical order)
device_mapping = {
    "Desktop": 0,
    "Mobile": 1,
    "Tablet": 2
}

# City mapping (alphabetical order of the 10 cities present in training)
city_mapping = {
    "Adana": 0,
    "Ankara": 1,
    "Antalya": 2,
    "Bursa": 3,
    "Eskisehir": 4,
    "Gaziantep": 5,
    "Istanbul": 6,
    "Izmir": 7,
    "Kayseri": 8,
    "Konya": 9
}

# Gender one‑hot encoding: [Male, Female, Other]
# We will create three binary columns from the input gender string.

# ------------------------------
# 3. Define the request data model (Pydantic)
# ------------------------------
# All fields match the original DataFrame columns (except the target)
class CustomerData(BaseModel):
    Age: int = Field(..., ge=18, le=75, description="Customer age")
    Gender: Literal["Male", "Female", "Other"] = Field(..., description="Gender")
    City: str = Field(..., description="City name (must be one of the 10 cities)")
    Product_Category: str = Field(..., description="Product category")
    Unit_Price: float = Field(..., gt=0, description="Price per unit")
    Quantity: int = Field(..., ge=1, le=5, description="Number of units purchased")
    Discount_Amount: float = Field(..., ge=0, description="Discount applied")
    Total_Amount: float = Field(..., gt=0, description="Total transaction amount")
    Payment_Method: str = Field(..., description="Payment method")
    Device_Type: str = Field(..., description="Device used for purchase")
    Session_Duration_Minutes: int = Field(..., ge=1, description="Session length in minutes")
    Pages_Viewed: int = Field(..., ge=1, description="Number of pages viewed")
    Delivery_Time_Days: int = Field(..., ge=1, description="Delivery time in days")
    Customer_Rating: int = Field(..., ge=1, le=5, description="Customer rating (1-5)")

# ------------------------------
# 4. Helper function: preprocess input data
# ------------------------------
def preprocess_input(data: CustomerData) -> pd.DataFrame:
    """
    Convert the incoming JSON into a DataFrame and apply the same
    transformations that were applied during training.
    """
    # Convert single record to DataFrame
    df = pd.DataFrame([data.dict()])

    # --- Encode categorical features ---
    # Product_Category
    df["Product_Category"] = df["Product_Category"].map(product_cat_mapping)
    if df["Product_Category"].isnull().any():
        raise ValueError("Invalid Product_Category value")

    # Device_Type
    df["Device_Type"] = df["Device_Type"].map(device_mapping)
    if df["Device_Type"].isnull().any():
        raise ValueError("Invalid Device_Type value")

    # City
    df["City"] = df["City"].map(city_mapping)
    if df["City"].isnull().any():
        raise ValueError("Invalid City value")

    # Payment_Method – Bug replication: use the same encoding as Product_Category
    # The notebook incorrectly replaced Payment_Method with encoded Product_Category.
    df["Payment_Method"] = df["Product_Category"].copy()

    # --- One‑hot encode Gender ---
    # Create three binary columns: Gender_Male, Gender_Female, Gender_Other
    gender_onehot = pd.get_dummies(df["Gender"], prefix="Gender")
    df = pd.concat([df, gender_onehot], axis=1).drop(columns=["Gender"])

    # Ensure all three columns exist (if some gender missing, add zeros)
    for col in ["Gender_Male", "Gender_Female", "Gender_Other"]:
        if col not in df.columns:
            df[col] = 0

    # --- Define the list of numerical columns (as in training) ---
    numerical_cols = [
        "Age", "Unit_Price", "Quantity", "Discount_Amount", "Total_Amount",
        "Session_Duration_Minutes", "Pages_Viewed", "Delivery_Time_Days",
        "Customer_Rating"
    ]

    # Apply MinMaxScaler (fitted during training)
    df[numerical_cols] = scaler.transform(df[numerical_cols])

    # --- Reorder columns to match the training feature order ---
    # The order is taken from the notebook: after all transformations,
    # the columns were (excluding the target):
    feature_order = [
        "Age", "City", "Product_Category", "Unit_Price", "Quantity",
        "Discount_Amount", "Total_Amount", "Payment_Method", "Device_Type",
        "Session_Duration_Minutes", "Pages_Viewed", "Delivery_Time_Days",
        "Customer_Rating", "Gender_Male", "Gender_Female", "Gender_Other"
    ]
    df = df[feature_order]

    return df

# ------------------------------
# 5. FastAPI app initialization
# ------------------------------
app = FastAPI(
    title="Customer Return Prediction API",
    description="Predict whether a customer is a returning buyer based on e‑commerce session data.",
    version="1.0.0"
)

# ------------------------------
# 6. Prediction endpoint
# ------------------------------
@app.post("/predict", response_model=dict)
async def predict(customer: CustomerData):
    """
    Accepts customer data, preprocesses it, and returns a prediction
    (True = returning customer, False = new customer) along with the confidence.
    """
    try:
        # Preprocess the input
        processed_df = preprocess_input(customer)

        # Make prediction (class and probability)
        prediction = model.predict(processed_df)[0]          # bool: True/False
        probability = model.predict_proba(processed_df)[0]  # [prob_false, prob_true]

        # Convert numpy types to Python native for JSON serialization
        return {
            "is_returning_customer": bool(prediction),
            "confidence": float(probability[1] if prediction else probability[0]),
            "probability_false": float(probability[0]),
            "probability_true": float(probability[1])
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logging.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Internal server error")

# ------------------------------
# 7. Health check endpoint (optional)
# ------------------------------
@app.get("/health")
async def health():
    return {"status": "ok"}

# ------------------------------
# 8. Run the app (if executed directly)
# ------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)