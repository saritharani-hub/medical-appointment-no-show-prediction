import streamlit as st
import pandas as pd
import joblib
import os

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Appointment Prediction",
    page_icon="🔮",
    layout="wide"
)

# --------------------------------------------------
# LOAD FINAL TUNED MODEL
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "final_model.pkl"
)

if not os.path.exists(MODEL_PATH):
    st.error("Final tuned model was not found.")
    st.code(MODEL_PATH)
    st.info("Please run hyperparameter_tuning.py first.")
    st.stop()

model = joblib.load(MODEL_PATH)

# --------------------------------------------------
# TITLE
# --------------------------------------------------

st.title("🔮 Medical Appointment No-Show Prediction")

st.write(
    "Predict whether a patient is likely to attend or miss "
    "their scheduled medical appointment."
)

st.divider()

# --------------------------------------------------
# PATIENT INFORMATION
# --------------------------------------------------

st.subheader("👤 Patient Information")

col1, col2, col3 = st.columns(3)

with col1:
    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

with col2:
    age = st.number_input(
        "Age",
        min_value=0,
        max_value=120,
        value=30
    )

with col3:
    disability = st.selectbox(
        "Disability",
        [0, 1],
        format_func=lambda x:
            "No" if x == 0 else "Yes"
    )

# --------------------------------------------------
# MEDICAL INFORMATION
# --------------------------------------------------

st.subheader("🏥 Medical Information")

col1, col2, col3 = st.columns(3)

with col1:
    hypertension = st.selectbox(
        "Hypertension",
        [0, 1],
        format_func=lambda x:
            "No" if x == 0 else "Yes"
    )

with col2:
    diabetes = st.selectbox(
        "Diabetes",
        [0, 1],
        format_func=lambda x:
            "No" if x == 0 else "Yes"
    )

with col3:
    alcoholism = st.selectbox(
        "Alcoholism",
        [0, 1],
        format_func=lambda x:
            "No" if x == 0 else "Yes"
    )

col1, col2, col3 = st.columns(3)

with col1:
    handcap = st.selectbox(
        "Handicap",
        [0, 1, 2, 3, 4],
        index=0
    )

with col2:
    scholarship = st.selectbox(
        "Scholarship",
        [0, 1],
        format_func=lambda x:
            "No" if x == 0 else "Yes"
    )

with col3:
    sms_received = st.selectbox(
        "SMS Received",
        [0, 1],
        format_func=lambda x:
            "No" if x == 0 else "Yes"
    )

# --------------------------------------------------
# APPOINTMENT INFORMATION
# --------------------------------------------------

st.subheader("📅 Appointment Information")

col1, col2, col3 = st.columns(3)

with col1:
    specialty = st.text_input(
        "Specialty",
        value="General"
    )

with col2:
    place = st.text_input(
        "Place",
        value="Hospital"
    )

with col3:
    appointment_shift = st.selectbox(
        "Appointment Shift",
        ["Morning", "Afternoon", "Evening"]
    )

appointment_time = st.time_input(
    "Appointment Time"
)

# --------------------------------------------------
# DATE INFORMATION
# --------------------------------------------------

st.subheader("📆 Appointment Date Information")

col1, col2, col3 = st.columns(3)

with col1:
    appointment_date_continuous = st.number_input(
        "Appointment Date Continuous",
        min_value=0.0,
        value=0.0
    )

with col2:
    appointment_weekday = st.selectbox(
        "Appointment Weekday",
        [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday"
        ]
    )

with col3:
    patient_needs_companion = st.selectbox(
        "Patient Needs Companion",
        [0, 1],
        format_func=lambda x:
            "No" if x == 0 else "Yes"
    )

# --------------------------------------------------
# AGE GROUP INFORMATION
# --------------------------------------------------

st.subheader("👶 Age Group Information")

col1, col2 = st.columns(2)

with col1:
    under_12_years_old = st.selectbox(
        "Under 12 Years Old",
        [0, 1],
        format_func=lambda x:
            "No" if x == 0 else "Yes"
    )

with col2:
    over_60_years_old = st.selectbox(
        "Over 60 Years Old",
        [0, 1],
        format_func=lambda x:
            "No" if x == 0 else "Yes"
    )

# --------------------------------------------------
# WEATHER INFORMATION
# --------------------------------------------------

st.subheader("🌦️ Weather Information")

col1, col2, col3 = st.columns(3)

with col1:
    average_temp_day = st.number_input(
        "Average Temperature",
        value=25.0
    )

with col2:
    average_rain_day = st.number_input(
        "Average Rain",
        min_value=0.0,
        value=0.0
    )

with col3:
    max_temp_day = st.number_input(
        "Maximum Temperature",
        value=30.0
    )

col1, col2, col3 = st.columns(3)

with col1:
    max_rain_day = st.number_input(
        "Maximum Rain",
        min_value=0.0,
        value=0.0
    )

with col2:
    rainy_day_before = st.selectbox(
        "Rainy Day Before",
        [0, 1],
        format_func=lambda x:
            "No" if x == 0 else "Yes"
    )

with col3:
    storm_day_before = st.selectbox(
        "Storm Day Before",
        [0, 1],
        format_func=lambda x:
            "No" if x == 0 else "Yes"
    )

col1, col2 = st.columns(2)

with col1:
    rain_intensity = st.number_input(
        "Rain Intensity",
        min_value=0.0,
        value=0.0
    )

with col2:
    heat_intensity = st.number_input(
        "Heat Intensity",
        min_value=0.0,
        value=0.0
    )

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

st.divider()

if st.button(
    "🔮 Predict Appointment",
    type="primary",
    use_container_width=True
):

    # Convert time to string
    appointment_time_value = (
        appointment_time.strftime("%H:%M:%S")
    )

    # --------------------------------------------------
    # CREATE INPUT DATAFRAME
    # --------------------------------------------------

    input_data = pd.DataFrame({

        "specialty": [specialty],

        "appointment_time": [
            appointment_time_value
        ],

        "gender": [gender],

        "disability": [disability],

        "place": [place],

        "appointment_shift": [
            appointment_shift
        ],

        "age": [age],

        "under_12_years_old": [
            under_12_years_old
        ],

        "over_60_years_old": [
            over_60_years_old
        ],

        "patient_needs_companion": [
            patient_needs_companion
        ],

        "average_temp_day": [
            average_temp_day
        ],

        "average_rain_day": [
            average_rain_day
        ],

        "max_temp_day": [
            max_temp_day
        ],

        "max_rain_day": [
            max_rain_day
        ],

        "rainy_day_before": [
            rainy_day_before
        ],

        "storm_day_before": [
            storm_day_before
        ],

        "rain_intensity": [
            rain_intensity
        ],

        "heat_intensity": [
            heat_intensity
        ],

        "appointment_date_continuous": [
            appointment_date_continuous
        ],

        "Hipertension": [
            hypertension
        ],

        "Diabetes": [
            diabetes
        ],

        "Alcoholism": [
            alcoholism
        ],

        "Handcap": [
            handcap
        ],

        "Scholarship": [
            scholarship
        ],

        "SMS_received": [
            sms_received
        ],

        "appointment_weekday": [
            appointment_weekday
        ]
    })

    # --------------------------------------------------
    # MAKE PREDICTION
    # --------------------------------------------------

    try:

        prediction = model.predict(
            input_data
        )[0]

        probability = model.predict_proba(
            input_data
        )[0]

        no_show_probability = probability[1] * 100

        show_probability = probability[0] * 100

        # --------------------------------------------------
        # DISPLAY RESULT
        # --------------------------------------------------

        st.subheader("📊 Prediction Result")

        if prediction == 1:

            st.error(
                "⚠️ Prediction: Patient is likely to NO-SHOW"
            )

        else:

            st.success(
                "✅ Prediction: Patient is likely to SHOW"
            )

        col1, col2 = st.columns(2)

        with col1:

            st.metric(
                "Show Probability",
                f"{show_probability:.2f}%"
            )

        with col2:

            st.metric(
                "No-Show Probability",
                f"{no_show_probability:.2f}%"
            )

        # --------------------------------------------------
        # PROBABILITY BAR
        # --------------------------------------------------

        st.subheader("📈 Prediction Probability")

        probability_df = pd.DataFrame({
            "Outcome": [
                "Show",
                "No-Show"
            ],
            "Probability": [
                show_probability,
                no_show_probability
            ]
        })

        st.bar_chart(
            probability_df.set_index("Outcome")
        )

    except Exception as e:

        st.error(
            "Prediction failed."
        )

        st.exception(e)

