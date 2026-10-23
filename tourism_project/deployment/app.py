from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# Locate the trained model relative to this script
APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "best_model.joblib"

st.set_page_config(
    page_title="Tourism Package Purchase Predictor",
    page_icon="✈️",
    layout="centered",
)

st.title("Tourism Package Purchase Predictor")
st.write(
    "Enter customer details to predict whether the customer "
    "is likely to purchase the tourism package."
)


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}. "
            "Run the training pipeline and ensure the model "
            "is committed to the repository."
        )
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except Exception as exc:
    st.error(str(exc))
    st.stop()


# Input form
with st.form("customer_input_form"):
    st.subheader("Customer Details")

    age = st.number_input(
        "Age", min_value=18, max_value=100, value=35
    )
    typeofcontact = st.selectbox(
        "Type of Contact",
        ["Company Invited", "Self Enquiry"]
    )  
  
  
    citytier = st.selectbox("City Tier", [1, 2, 3])
    occupation = st.selectbox(
        "Occupation",
        ["Salaried", "Small Business", "Large Business", "Free Lancer"]
    )

    gender = st.selectbox("Gender", ["Male", "Female"])
    persons_visiting = st.number_input(
        "Number of Persons Visiting",
        min_value=1, max_value=20, value=2
    )
    property_star = st.selectbox(
        "Preferred Property Star", [3, 4, 5]
    )
    marital_status = st.selectbox(
        "Marital Status",
        ["Married", "Single", "Divorced", "Unmarried"],
    )
    number_of_trips = st.number_input(
        "Number of Trips", min_value=0, max_value=100, value=3
    )
    passport = st.selectbox("Has Passport?", [0, 1])
    own_car = st.selectbox("Owns a Car?", [0, 1])
    children_visiting = st.number_input(
        "Number of Children Visiting",
        min_value=0, max_value=10, value=0
    )
    designation = st.selectbox(
        "Designation",
        ["Executive", "Manager", "Senior Manager", "AVP", "VP"],
    )
    monthly_income = st.number_input(
        "Monthly Income", min_value=0, value=25000, step=1000
    )
    pitch_satisfaction = st.selectbox(
        "Pitch Satisfaction Score", [1, 2, 3, 4, 5]
    )
    product_pitched = st.selectbox(
        "Product Pitched",
        ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"],
    )
    followups = st.number_input(
        "Number of Follow-ups", min_value=0, max_value=20, value=3
    )
    pitch_duration = st.number_input(
        "Duration of Pitch (minutes)",
        min_value=0.0, max_value=120.0, value=10.0
    )

    submitted = st.form_submit_button("Predict Purchase")




if submitted:
    try:
        # Build customer input using the same feature names as training.
        input_data = pd.DataFrame([{
            "Age": age,
            "TypeofContact": typeofcontact,
            "CityTier": citytier,
            "Occupation": occupation,
            "Gender": gender,
            "NumberOfPersonVisiting": persons_visiting,
            "PreferredPropertyStar": property_star,
            "MaritalStatus": marital_status,
            "NumberOfTrips": number_of_trips,
            "Passport": passport,
            "OwnCar": own_car,
            "NumberOfChildrenVisiting": children_visiting,
            "Designation": designation,
            "MonthlyIncome": monthly_income,
            "PitchSatisfactionScore": pitch_satisfaction,
            "ProductPitched": product_pitched,
            "NumberOfFollowups": followups,
            "DurationOfPitch": pitch_duration,
        }])

        # Check that the loaded model expects these features.
        expected_columns = getattr(model, "feature_names_in_", None)

        if expected_columns is not None:
            expected_columns = list(expected_columns)

            missing = set(expected_columns) - set(input_data.columns)
            extra = set(input_data.columns) - set(expected_columns)

            if missing or extra:
                st.error(
                    f"Feature mismatch. Missing: {sorted(missing)}; "
                    f"unexpected: {sorted(extra)}"
                )
                st.stop()

            input_data = input_data[expected_columns]

        # Predict the class and purchase probability.
        prediction = int(model.predict(input_data)[0])
        classes = list(model.classes_)

        if 1 not in classes:
            raise ValueError("Purchase class ProdTaken=1 was not found.")

        purchase_index = classes.index(1)
        purchase_probability = float(
            model.predict_proba(input_data)[0][purchase_index]
        )

        st.subheader("Prediction Result")

        if prediction == 1:
            st.success("The model predicts purchase.")
        else:
            st.info("The model predicts no purchase.")

        st.metric(
            "Purchase Probability",
            f"{purchase_probability * 100:.2f}%"
        )

        st.caption(
            "This is the probability estimated by the loaded model. "
            "It is not a guarantee that the customer will purchase."
        )

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
