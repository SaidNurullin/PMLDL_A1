# code/deployment/app/main.py
import streamlit as st
import requests


# Define a function to make predictions from the API
def get_prediction(features):
    # Replace with your API endpoint URL
    api_url = "http://172.18.0.2:8000/predict/"
    data = {"features": features}
    response = requests.post(api_url, json=data)
    if response.status_code == 200:
        return response.json()["prediction"]
    else:
        st.error(f"API request failed with status code: {response.status_code}")
        return None


# Create the Streamlit app
st.title("Catboost Model Prediction App")

# Get input features from the user
st.subheader("Enter your features:")


# Feature 1 (number input)
feature1 = st.number_input("pr_open opening price", value=139.0)

# Feature 2 (number input)
feature2 = st.number_input("pr_high maximum price for the period", value=139.0)

# Feature 3 (number input)
feature3 = st.number_input("pr_low minimum price for the period", value=137.5)

# Feature 4 (number input)
feature4 = st.number_input("pr_close last price for the period", value=137.5)

# Feature 5 (number input)
feature5 = st.number_input("pr_std standard deviation of price", value=0.001655, format="%.6f")

# Feature 6 (number input)
feature6 = st.number_input("vol volume in lots", value=2.0)

# Feature 7 (number input)
feature7 = st.number_input("val volume in rubles", value=2765.0)

# Feature 8 (number input)
feature8 = st.number_input("trades number of trades", value=2.0)

# Feature 9 (number input)
feature9 = st.number_input("pr_vwap weighted average price", value=138.2)

# Feature 10 (number input)
feature10 = st.number_input("pr_change price change for the period, %", value=-1.0791, format="%.4f")

# Feature 11 (number input)
feature11 = st.number_input("trades_b number of purchase trades", value=1.0)

# Feature 12 (number input)
feature12 = st.number_input("trades_s number of sell trades", value=1.0)

# Feature 13 (number input)
feature13 = st.number_input("val_b purchase volume in rubles", value=1390.0)

# Feature 14 (number input)
feature14 = st.number_input("val_s sales volume in rubles", value=1375.0)

# Feature 15 (number input)
feature15 = st.number_input("vol_b purchase volume in lots", value=1.0)

# Feature 16 (number input)
feature16 = st.number_input("vol_s sales volume in lots", value=1.0)

# Feature 17 (number input)
feature17 = st.number_input("disb ratio of purchase and sale volumes", value=0.00, format="%.2f")

# Feature 18 (number input)
feature18 = st.number_input("pr_vwap_b weighted average purchase price", value=139.000000, format="%.6f")

# Feature 19 (number input)
feature19 = st.number_input("pr_vwap_s weighted average sale price", value=137.500000, format="%.6f")

# Feature 20 (number input)
feature20 = st.number_input("year", value=2020)

# Feature 21 (number input)
feature21 = st.number_input("month", value=1)

# Feature 22 (number input)
feature22 = st.number_input("day", value=3)

# Feature 23 (number input)
feature23 = st.number_input("hour", value=10)

# Feature 24 (number input)
feature24 = st.number_input("minute", value=55)

# Feature 25 (number input)
feature25 = st.number_input("secid_encoded instrument code", value=1)

# Make a prediction when the button is clicked
if st.button("Predict"):
    features = [feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10, feature11, feature12, feature13, feature14, feature15, feature16, feature17, feature18, feature19, feature20, feature21, feature22, feature23, feature24, feature25]  
    prediction = get_prediction(features)
    if prediction is not None:
        st.success(f"The predicted value is: {prediction}")