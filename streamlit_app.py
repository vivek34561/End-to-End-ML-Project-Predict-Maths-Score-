





import streamlit as st
from src.mlproject.predict_pipelines import PredictPipeline

st.set_page_config(page_title="📊 Predict Math Score")
st.title("🎓 Student Math Score Predictor")

# Input form
gender = st.selectbox("Gender", ["female", "male"])
race = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
parent_edu = st.selectbox("Parental Level of Education", [
    "some high school", "high school", "some college",
    "associate's degree", "bachelor's degree", "master's degree"
])
lunch = st.selectbox("Lunch Type", ["standard", "free/reduced"])
prep_course = st.selectbox("Test Preparation Course", ["none", "completed"])
reading_score = st.number_input("Reading Score", min_value=0, max_value=100, value=90)
writing_score = st.number_input("Writing Score", min_value=0, max_value=100, value=93)

input_data = {
    "gender": gender,
    "race_ethnicity": race,
    "parental_level_of_education": parent_edu,
    "lunch": lunch,
    "test_preparation_course": prep_course,
    "reading_score": reading_score,
    "writing_score": writing_score
}

if st.button("🎯 Predict Math Score"):
    pipeline = PredictPipeline()
    result = pipeline.predict(input_data)
    st.success(f"📈 Predicted Math Score: {result:.2f}")
