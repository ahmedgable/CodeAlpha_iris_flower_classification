import os
import joblib
import numpy as np
import streamlit as st

st.set_page_config(page_title="Iris Flower Classification", page_icon="🌸", layout="centered")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_assets():
    model_path = os.path.join(BASE_DIR, 'svm_model.pkl')
    scaler_path = os.path.join(BASE_DIR, 'scaler.pkl')
    encoder_path = os.path.join(BASE_DIR, 'species_encoder.pkl')
    
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    encoder = joblib.load(encoder_path)
    return model, scaler, encoder

try:
    model, scaler, encoder = load_assets()
except Exception as e:
    st.error(f"Error loading saved files: {e}")
    st.stop()

# ---------------------------------------------------------
# 2. User Interface
# ---------------------------------------------------------
st.title("🌸 Iris Flower Classification App")
st.write("Enter the flower measurements below to predict its species using the **SVM** model:")

st.divider()

col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, value=5.1, step=0.1)
    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, value=3.5, step=0.1)

with col2:
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, value=1.4, step=0.1)
    petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, value=0.2, step=0.1)

st.divider()

# ---------------------------------------------------------
# 3. Prediction & Output Display
# ---------------------------------------------------------
if st.button("🔮 Predict Species", use_container_width=True):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Scale inputs and perform inference
    input_scaled = scaler.transform(input_data)
    prediction_numeric = model.predict(input_scaled)
    probabilities = model.predict_proba(input_scaled)[0]
    
    # Decode numerical label to original species name
    species_name = encoder.inverse_transform(prediction_numeric)[0]
    
    st.success(f"Predicted Species: **{species_name}**")
    
    st.write("### Prediction Confidence:")
    for class_index, class_name in enumerate(encoder.classes_):
        prob = probabilities[class_index] * 100
        st.write(f"- **{class_name}**: {prob:.2f}%")
        st.progress(int(prob))