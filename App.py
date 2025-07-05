#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import numpy as np
import pickle

# Load saved model and scaler
with open("best_model.pkl", "rb") as f:
    lr = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)  # ✅ Fixed the missing closing parenthesis

st.title("Customer Gender Prediction App")

# Input fields
age = st.number_input("Age", min_value=10, max_value=100, value=30)
gender = st.selectbox("Gender (0: Female, 1: Male)", [0, 1])
item_purchased = st.selectbox("Item Purchased (Encoded)", range(0, 50))
category = st.selectbox("Category (Encoded)", range(0, 10))
purchase_amount = st.number_input("Purchase Amount (USD)", value=50)
location = st.selectbox("Location (Encoded)", range(0, 50))
size = st.selectbox("Size (Encoded)", range(0, 50))
color = st.selectbox("Color (Encoded)", range(0, 50))
season = st.selectbox("Season (Encoded)", range(0, 50))
review_rating = st.number_input("Review Rating", min_value=0.0, max_value=5.0, step=0.1)
subscription_status = st.selectbox("Subscription Status (0/1)", [0, 1])
shipping_type = st.selectbox("Shipping Type (Encoded)", range(0, 10))
discount_applied = st.selectbox("Discount Applied (0/1)", [0, 1])
promo_code_used = st.selectbox("Promo Code Used (0/1)", [0, 1])
previous_purchases = st.number_input("Previous Purchases", value=0)
payment_method = st.selectbox("Payment Method (Encoded)", range(0, 10))
frequency = st.selectbox("Frequency of Purchases (Encoded)", range(0, 10))

# Button to predict
if st.button("Predict Gender"):
    # Create input array
    X_new = np.array([[age, item_purchased, category, purchase_amount,
                       location, size, color, season, review_rating,
                       subscription_status, shipping_type, discount_applied,
                       promo_code_used, previous_purchases, payment_method, frequency]])
    
    # Scale input
    X_scaled = scaler.transform(X_new)

    # Predict
    y_pred = lr.predict(X_scaled)

    st.success(f"Predicted Gender: {'Male' if y_pred[0] == 1 else 'Female'}")


# In[ ]:




