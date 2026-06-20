from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    try:
        age = float(request.form["Age"])
        gender = float(request.form["Gender"])
        annual_income = float(request.form["Annual_Income"])
        total_spent = float(request.form["Total_Spent"])
        items_purchased = float(request.form["Items_Purchased"])
        average_rating = float(request.form["Average_Rating"])
        discount_applied = float(request.form["Discount_Applied"])
        days_since_last_purchase = float(request.form["Days_Since_Last_Purchase"])
        satisfaction_level = float(request.form["Satisfaction_Level"])
        product_category = float(request.form["Product_Category"])
        payment_method = float(request.form["Payment_Method"])
        city = float(request.form["City"])
        device_type = float(request.form["Device_Type"])

        data = np.array([[
            age,
            gender,
            annual_income,
            total_spent,
            items_purchased,
            average_rating,
            discount_applied,
            days_since_last_purchase,
            satisfaction_level,
            product_category,
            payment_method,
            city,
            device_type
        ]])

        data = scaler.transform(data)

        prediction = model.predict(data)[0]

        if prediction == 1:
            result = "Returning Customer"
        else:
            result = "Not a Returning Customer"

        return render_template("index.html", prediction_text=result)

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )


if __name__ == "__main__":
    app.run(debug=True)