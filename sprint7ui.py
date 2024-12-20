# -*- coding: utf-8 -*-


import streamlit as st
import numpy as np
import joblib  # To load the scaler
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Load the model
model = load_model('c:\\data\\survival_model.h5')
scaler_resampled = joblib.load('c:/data/scaler_resampled.pkl')  # Path to your saved scaler



# Set up columns
col1, col2 = st.columns([1, 3])  # Adjust column width ratios as needed

# Add image to the first column
with col1:
   st.image("c:\data\logo.jpg", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB", output_format="auto", use_container_width=False)
  # Adjust width to fit nicely

# Add title to the second column
with col2:
    #st.title("Hospital Patient Survival Prediction")
# Custom HTML for colored text
    st.markdown("<h1 style='color:#2E5A88 ;font-size: 35px;'>Hospital Patient Survival Prediction</h1>", unsafe_allow_html=True)
#st.write("Please input the features for the patient:")    
st.markdown("<p style='color: #2E5A88; font-size: 18px;'><b><u>Please input the features for the patient:</u></b></p>", unsafe_allow_html=True)





# User inputs for all features used in the model
apache_2_diagnosis = st.number_input("Apache 2 Diagnosis (Range: 101 - 300)", 101, 300, 180)  # Binary feature, assuming it's 0 or 1
apache_3j_diagnosis = st.number_input("Apache 3J Diagnosis (Range: 0.04 - 502.0)", 0.04, 502.0, 400.0)  # Binary feature, assuming it's 0 or 1
gcs_eyes_apache = st.number_input("GCS Eyes Apache (Range: 1.0 - 4.0)", 1.0, 4.0, 3.12)  # Range based on Glasgow Coma Scale (GCS)
gcs_motor_apache = st.number_input("GCS Motor Apache (Range: 1.0 - 6.0)", 1.0, 6.00, 4.98)  # Range based on Glasgow Coma Scale (GCS)
gcs_verbal_apache = st.number_input("GCS Verbal Apache (Range: 1.0 - 5.0)", 1.0, 5.0, 3.46)  # Range based on Glasgow Coma Scale (GCS)
d1_lactate_max = st.number_input("D1 Lactate Max (Range: 0.4 - 19.8)", 0.4, 19.8, 3.43)  # Max Lactate Value
d1_lactate_min = st.number_input("D1 Lactate Min (Range: 0.4 - 15.1)", 0.4, 15.1, 2.48)  # Min Lactate Value
d1_arterial_ph_max = st.number_input("D1 Arterial pH Max (Range: 6.05 - 7.6)", 6.05, 7.6, 7.38)  # Max pH Value
d1_arterial_ph_min = st.number_input("D1 Arterial pH Min (Range: 6.8 - 7.7 )", 6.8, 7.7, 7.31)  # Min pH Value
d1_pao2fio2ratio_max = st.number_input("D1 PaO2/FIO2 Ratio Max (Range: 54.8 - 830)", 54.8, 830.0, 280.0)  # Max PaO2/FIO2 Ratio

# Prepare the input data in the correct format
input_data = np.array([[apache_2_diagnosis,apache_3j_diagnosis,gcs_eyes_apache, gcs_motor_apache, 
                        gcs_verbal_apache, d1_lactate_max, d1_lactate_min, d1_arterial_ph_max, 
                        d1_arterial_ph_min, d1_pao2fio2ratio_max]])

# Scaling the input data using the same scaler used during model training
#scaler = StandardScaler()
#input_data_scaled = scaler.fit_transform(input_data)  # Apply scaling

# Scale the input data using the loaded scaler
input_data_scaled = scaler_resampled.transform(input_data)  # Use the loaded scaler

# Predict on user input
prediction = model.predict(input_data_scaled)

predict= st.button("Predict")
# Predict on user input
if predict:
    prediction = model.predict(input_data_scaled)
    survival_probability = prediction[0][0]
    
    # Display the results
    st.write(f"Survival Probability: {survival_probability:.2f}")
    if survival_probability > 0.5:
        st.success("The patient is likely to not survive.")
    else:
        st.error("The patient is likely to survive.")

