import os
os.environ['STREAMLIT_SERVER_FILE_WATCHER_TYPE'] = 'none'

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration
st.set_page_config(
    page_title="AMI 3-Month Mortality Prediction",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Define the feature list (must match the order used during model training)
FEATURES = [
    'Oral_Furosemide', 'Metoprolol', 'Norepinephrine', 'Age', 'LOS',
    'SOFA', 'CCI', 'Lactate', 'Total_CO2', 'BUN', 'PT', 'Phosphate',
    'IMV_hours', 'Output_urine'
]

# Provide reasonable input ranges and units for each feature to guide the user
FEATURE_RANGES = {
    'Oral_Furosemide': (0, 500, 'mg/day'),
    'Metoprolol': (0, 400, 'mg/day'),
    'Norepinephrine': (0, 100, 'mcg/min'),
    'Age': (18, 100, 'years'),
    'LOS': (0, 100, 'days'),
    'SOFA': (0, 24, 'points'),
    'CCI': (0, 40, 'points'),
    'Lactate': (0, 30, 'mmol/L'),
    'Total_CO2': (0, 50, 'mmol/L'),
    'BUN': (0, 200, 'mg/dL'),
    'PT': (0, 120, 'seconds'),
    'Phosphate': (0, 20, 'mg/dL'),
    'IMV_hours': (0, 1000, 'hours'),
    'Output_urine': (0, 5000, 'mL/24h')
}

@st.cache_resource
def load_model():
    """Load the model (cached to avoid reloading)."""
    try:
        model = joblib.load('final_RF_model_selected_features.pkl')
        # Ensure the model has a predict_proba method
        if not hasattr(model, 'predict_proba'):
            st.error("The loaded model does not have a predict_proba method. Please check the model file.")
            return None
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

# Load the model
model = load_model()
if model is None:
    st.stop()

# Title and description
st.title("3-Month Mortality Risk Prediction for AMI Patients")
st.markdown("A random forest model developed on the MIMIC-AMI database to predict 3‑month mortality after admission for acute myocardial infarction.")
st.markdown("---")

# Layout: left column for input, right column for results (prediction probability)
col_input, col_result = st.columns([2, 2])

with col_input:
    st.header("📋 Patient Information Input")
    with st.form("input_form"):
        # Split the 14 features into two columns
        input_col1, input_col2 = st.columns(2)
        input_values = {}

        # Left column (first 7 features)
        with input_col1:
            for feat in FEATURES[:7]:
                min_val, max_val, unit = FEATURE_RANGES[feat]
                input_values[feat] = st.number_input(
                    f"{feat} ({unit})",
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float((min_val + max_val) / 2),  # default value = mid‑range
                    step=0.1,
                    key=f"in_{feat}"
                )

        # Right column (remaining 7 features)
        with input_col2:
            for feat in FEATURES[7:]:
                min_val, max_val, unit = FEATURE_RANGES[feat]
                input_values[feat] = st.number_input(
                    f"{feat} ({unit})",
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float((min_val + max_val) / 2),
                    step=0.1,
                    key=f"in_{feat}"
                )

        submitted = st.form_submit_button("🚀 Predict Mortality Risk", use_container_width=True)

# Right column for results
with col_result:
    st.header("📊 Prediction Results")

    if submitted:
        # Convert input to a DataFrame with the correct feature order
        input_df = pd.DataFrame([input_values])[FEATURES]

        try:
            # Predict the probability of death (assuming binary classification, take the positive class)
            prob = model.predict_proba(input_df)[0][1]
            death_prob = prob * 100

            st.metric(
                label="3‑Month Mortality Probability",
                value=f"{death_prob:.2f}%",
                delta=None
            )

        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")
    else:
        st.info("Please enter patient information on the left and click the prediction button.")

# Footer information
st.markdown("---")
st.header("📝 About This Tool")
st.markdown(f"""
**Model Description**  
- Developed on the MIMIC-AMI database (acute myocardial infarction patients) using a random forest algorithm.  
- Input features: {len(FEATURES)} variables: {', '.join(FEATURES)}.  
- Output: 3‑month mortality probability (0–100%).  

**Important Notes**  
- This tool is intended for clinical research support only and should not replace professional medical judgment.  
- Predictions are based on historical data; actual risk may vary due to individual differences.  
""")
