import streamlit as st
import pandas as pd
import joblib

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Medical Appointment No-Show Prediction",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Medical Appointment No-Show Prediction")
st.write("Predict whether a patient will show up for their medical appointment.")

# --------------------------------------------------
# Load Model and Preprocessor
# --------------------------------------------------

model = joblib.load("models/logistic_model.pkl")
preprocessor = joblib.load("models/preprocessor.pkl")

# Load dataset
df = pd.read_csv("data/Medical_appointment_data.csv")

# Convert appointment_date_continuous to numeric
df["appointment_date_continuous"] = pd.to_numeric(
    df["appointment_date_continuous"],
    errors="coerce"
)

# Create separate datetime column
df["appointment_date"] = pd.to_datetime(
    df["appointment_date_continuous"],
    errors="coerce"
)

# Create weekday column
df["appointment_weekday"] = df["appointment_date"].dt.day_name()

st.success("Model and preprocessor loaded successfully.")

# --------------------------------------------------
# Prediction Form
# --------------------------------------------------

st.subheader("Enter Patient Appointment Details")

with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    # -----------------------------
    # Categorical Inputs
    # -----------------------------

    with col1:

        specialty = st.selectbox(
            "Specialty",
            sorted(df["specialty"].dropna().unique())
        )

        gender = st.selectbox(
            "Gender",
            sorted(df["gender"].dropna().unique())
        )

        disability = st.selectbox(
            "Disability",
            sorted(df["disability"].dropna().unique())
        )

        place = st.selectbox(
            "Place",
            sorted(df["place"].dropna().unique())
        )

        appointment_shift = st.selectbox(
            "Appointment Shift",
            sorted(df["appointment_shift"].dropna().unique())
        )

        rain_intensity = st.selectbox(
            "Rain Intensity",
            sorted(df["rain_intensity"].dropna().unique())
        )

        heat_intensity = st.selectbox(
            "Heat Intensity",
            sorted(df["heat_intensity"].dropna().unique())
        )

        appointment_weekday = st.selectbox(
            "Appointment Weekday",
            sorted(df["appointment_weekday"].dropna().unique())
        )

    # -----------------------------
    # Numerical Inputs
    # -----------------------------

    with col2:

        appointment_time = st.number_input(
            "Appointment Time",
            min_value=float(df["appointment_time"].min()),
            max_value=float(df["appointment_time"].max()),
            value=float(df["appointment_time"].median())
        )

        age = st.number_input(
            "Age",
            min_value=float(df["age"].min()),
            max_value=float(df["age"].max()),
            value=float(df["age"].median())
        )

        under_12_years_old = st.number_input(
            "Under 12 Years Old",
            min_value=float(df["under_12_years_old"].min()),
            max_value=float(df["under_12_years_old"].max()),
            value=float(df["under_12_years_old"].median())
        )

        over_60_years_old = st.number_input(
            "Over 60 Years Old",
            min_value=float(df["over_60_years_old"].min()),
            max_value=float(df["over_60_years_old"].max()),
            value=float(df["over_60_years_old"].median())
        )

        patient_needs_companion = st.number_input(
            "Patient Needs Companion",
            min_value=float(df["patient_needs_companion"].min()),
            max_value=float(df["patient_needs_companion"].max()),
            value=float(df["patient_needs_companion"].median())
        )

        average_temp_day = st.number_input(
            "Average Temperature",
            value=float(df["average_temp_day"].median())
        )

        average_rain_day = st.number_input(
            "Average Rain",
            value=float(df["average_rain_day"].median())
        )

        max_temp_day = st.number_input(
            "Maximum Temperature",
            value=float(df["max_temp_day"].median())
        )

        max_rain_day = st.number_input(
            "Maximum Rain",
            value=float(df["max_rain_day"].median())
        )

        rainy_day_before = st.number_input(
            "Rainy Day Before",
            min_value=float(df["rainy_day_before"].min()),
            max_value=float(df["rainy_day_before"].max()),
            value=float(df["rainy_day_before"].median())
        )

        storm_day_before = st.number_input(
            "Storm Day Before",
            min_value=float(df["storm_day_before"].min()),
            max_value=float(df["storm_day_before"].max()),
            value=float(df["storm_day_before"].median())
        )

        appointment_date_continuous = st.number_input(
            "Appointment Date Continuous",
            value=float(df["appointment_date_continuous"].median())
        )

        Hipertension = st.number_input(
            "Hypertension",
            min_value=float(df["Hipertension"].min()),
            max_value=float(df["Hipertension"].max()),
            value=float(df["Hipertension"].median())
        )

        Diabetes = st.number_input(
            "Diabetes",
            min_value=float(df["Diabetes"].min()),
            max_value=float(df["Diabetes"].max()),
            value=float(df["Diabetes"].median())
        )

        Alcoholism = st.number_input(
            "Alcoholism",
            min_value=float(df["Alcoholism"].min()),
            max_value=float(df["Alcoholism"].max()),
            value=float(df["Alcoholism"].median())
        )

        Handcap = st.number_input(
            "Handcap",
            min_value=float(df["Handcap"].min()),
            max_value=float(df["Handcap"].max()),
            value=float(df["Handcap"].median())
        )

        Scholarship = st.number_input(
            "Scholarship",
            min_value=float(df["Scholarship"].min()),
            max_value=float(df["Scholarship"].max()),
            value=float(df["Scholarship"].median())
        )

        SMS_received = st.number_input(
            "SMS Received",
            min_value=float(df["SMS_received"].min()),
            max_value=float(df["SMS_received"].max()),
            value=float(df["SMS_received"].median())
        )

    # --------------------------------------------------
    # Prediction Button
    # --------------------------------------------------

    submitted = st.form_submit_button(
        "🔮 Predict No-Show"
    )

# --------------------------------------------------
# --------------------------------------------------
# Prediction
# --------------------------------------------------

if submitted:

    # Create input DataFrame
    input_data = pd.DataFrame({
        "specialty": [specialty],
        "gender": [gender],
        "disability": [disability],
        "place": [place],
        "appointment_shift": [appointment_shift],
        "rain_intensity": [rain_intensity],
        "heat_intensity": [heat_intensity],
        "appointment_weekday": [appointment_weekday],

        "appointment_time": [appointment_time],
        "age": [age],
        "under_12_years_old": [under_12_years_old],
        "over_60_years_old": [over_60_years_old],
        "patient_needs_companion": [patient_needs_companion],
        "average_temp_day": [average_temp_day],
        "average_rain_day": [average_rain_day],
        "max_temp_day": [max_temp_day],
        "max_rain_day": [max_rain_day],
        "rainy_day_before": [rainy_day_before],
        "storm_day_before": [storm_day_before],
        "appointment_date_continuous": [appointment_date_continuous],
        "Hipertension": [Hipertension],
        "Diabetes": [Diabetes],
        "Alcoholism": [Alcoholism],
        "Handcap": [Handcap],
        "Scholarship": [Scholarship],
        "SMS_received": [SMS_received]
    })

    # Apply the saved preprocessor
    input_processed = preprocessor.transform(input_data)

    # Make prediction
    prediction = model.predict(input_processed)[0]

    # Get prediction probabilities
    probability = model.predict_proba(input_processed)[0]

    # Display result
    st.subheader("Prediction Result")

    if prediction == 1:

        st.error("❌ Patient is predicted to NOT SHOW UP.")

        st.write(
            f"Probability of No-Show: **{probability[1] * 100:.2f}%**"
        )

        st.write(
            f"Probability of Show: **{probability[0] * 100:.2f}%**"
        )

    else:

        st.success("✅ Patient is predicted to SHOW UP.")

        st.write(
            f"Probability of Show: **{probability[0] * 100:.2f}%**"
        )

        st.write(
            f"Probability of No-Show: **{probability[1] * 100:.2f}%**"
        )