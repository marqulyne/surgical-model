import streamlit as st
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('surgical_cases_model.joblib')  # Adjust the path as needed

# Set the title of the app
st.title("Surgical Cases Prediction App")

# Create a tab for predictions
tab1, tab2 = st.tabs(["Prediction", "About"])

with tab1:
    st.header("Enter Patient Data for Prediction")

    # Input fields for user to enter the required features (excluding Facility)
    medical = st.number_input("Medical Cases", min_value=0, value=0)
    paediatrics = st.number_input("Pediatric Cases", min_value=0, value=0)
    neonate = st.number_input("Neonate Cases", min_value=0, value=0)
    covid_19 = st.number_input("COVID-19 Cases", min_value=0, value=0)
    radiology = st.number_input("Radiology Cases", min_value=0, value=0)
    ph = st.number_input("Public Health Indicator", min_value=0.0, value=0.0)

    # Button to trigger prediction
    if st.button("Predict Surgical Cases"):
        # Prepare the input data for prediction
        input_data = pd.DataFrame({
            'Medical': [medical],
            'Paediatrics': [paediatrics],
            'Neonate': [neonate],
            'Covid_19': [covid_19],
            'Radiology': [radiology],
            'PH': [ph]
        })

        # Ensure the input data has the same number of features as the model
        input_data_encoded = pd.get_dummies(input_data, drop_first=True)
        input_data_encoded = input_data_encoded.reindex(columns=model.feature_names_in_, fill_value=0)

        # Make the prediction
        prediction = model.predict(input_data_encoded)

        # Check for non-zero prediction
        if prediction[0] > 0:
            st.write(f"Predicted Surgical Cases: {prediction[0]}")
        else:
            st.write("The model predicts zero surgical cases based on the provided data.")

with tab2:
    st.header("About This App")
    st.write("""
    This application predicts the number of surgical cases based on various medical indicators.
    Enter the required patient data, and click 'Predict Surgical Cases' to see the results.
    """)
    st.write("Developed by [Marqulyne, Rita].")

# Run the Streamlit app with the command:
# streamlit run your_script_name.py
