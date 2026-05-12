import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# -----------------------------------
# DATA DEFAULTS
# -----------------------------------
try:
    _df = pd.read_csv("stress.csv")
    DEFAULT_FEATURE_MEANS = _df.drop(columns=["stress_level"]).mean().to_dict()
except Exception:
    DEFAULT_FEATURE_MEANS = {
        "anxiety_level": 11.0636,
        "self_esteem": 17.7773,
        "mental_health_history": 0.4927,
        "depression": 12.5555,
        "headache": 2.5082,
        "blood_pressure": 2.1818,
        "sleep_quality": 2.66,
        "breathing_problem": 2.7536,
        "noise_level": 2.6491,
        "living_conditions": 2.5182,
        "safety": 2.7373,
        "basic_needs": 2.7727,
        "academic_performance": 2.7727,
        "study_load": 2.6218,
        "teacher_student_relationship": 2.6482,
        "future_career_concerns": 2.6491,
        "social_support": 1.8818,
        "peer_pressure": 2.7345,
        "extracurricular_activities": 2.7673,
        "bullying": 2.6173,
    }

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="Student Stress Analysis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------
# LOAD MODELS
# -----------------------------------

rf_model = joblib.load("models/rf.pkl")
lasso_model = joblib.load("models/lasso.pkl")
kmeans = joblib.load("models/kmeans.pkl")
pca = joblib.load("models/pca.pkl")
scaler = joblib.load("models/scaler.pkl")
# -----------------------------------
# SIDEBAR
# -----------------------------------

st.sidebar.title("🎓 Student Stress Analysis")

page = st.sidebar.radio("Navigation",["Survey", "Dashboard", "EDA Explorer", "About Models"])


def slider_with_help(question, label, min_value, max_value, default, left_description, right_description, key):
    st.markdown(f"**{question}**")
    col_slider, col_help = st.columns([4, 2])
    with col_slider:
        value = st.slider(label, min_value, max_value, default, key=key)
    with col_help:
        st.markdown(f"**Scale meaning**\n- {min_value}: {left_description}\n- {max_value}: {right_description}")
    return value

# ===================================
# SURVEY PAGE
# ===================================

if page == "Survey":

    st.title("📋 Student Stress Survey")

    with st.expander("Need help with the sliders?"):
        st.write("Each slider asks a question about your current academic stress factors. Use the scale descriptions to understand what the low and high ends mean.")
        st.write("For rating sliders, lower values usually indicate better stress outcomes for negative factors (like anxiety), while higher values indicate better outcomes for positive factors (like sleep quality).")

    sleep_quality = slider_with_help(
        "How would you rate your sleep quality?",
        "Sleep Quality", 0, 5,3,
        "Very poor sleep or frequent restlessness",
        "Consistent, restful sleep",
        "sleep_quality"
    )

    study_load = slider_with_help(
        "How heavy is your academic workload right now?",
        "Study Load", 0,5,2,
        "Light workload or few deadlines",
        "Very heavy workload or many deadlines",
        "study_load"
    )

    extracurricular_activities = slider_with_help(
        "How often do you participate in extracurricular activities?",
        "Extracurricular Activities",0,5,2,
        "Rarely or never",
        "Very regularly",
        "extracurricular_activities"
    )

    social_support = slider_with_help(
        "How strong is your current social support network?",
        "Social Support", 0,5,2,
        "Very limited support from friends/family",
        "Strong support and reliable connections",
        "social_support"
    )

    anxiety_level = slider_with_help(
        "How intense is your anxiety about academic life?",
        "Anxiety Level", 0,21,11,
        "No or minimal anxiety",
        "Very high anxiety",
        "anxiety_level"
    )

    depression = slider_with_help(
        "How would you rate your current mood and depressive symptoms?",
        "Depression Level",0,27,12,
        "No or minimal depressive symptoms",
        "Severe depressive symptoms",
        "depression"
    )

    future_career_concerns = slider_with_help(
        "How worried are you about your future career path?",
        "Future Career Concerns",0,5,2,
        "Not worried about career plans",
        "Very worried about career direction",
        "future_career_concerns"
    )

    academic_performance = slider_with_help(
        "How would you rate your current academic performance?",
        "Academic Performance", 0,5,2,
        "Struggling academically",
        "Performing very well academically",
        "academic_performance"
    )

    # CREATE DATAFRAME

    full_input = {
        "anxiety_level": anxiety_level,
        "self_esteem": DEFAULT_FEATURE_MEANS["self_esteem"],
        "mental_health_history": DEFAULT_FEATURE_MEANS["mental_health_history"],
        "depression": depression,
        "headache": DEFAULT_FEATURE_MEANS["headache"],
        "blood_pressure": DEFAULT_FEATURE_MEANS["blood_pressure"],
        "sleep_quality": sleep_quality,
        "breathing_problem": DEFAULT_FEATURE_MEANS["breathing_problem"],
        "noise_level": DEFAULT_FEATURE_MEANS["noise_level"],
        "living_conditions": DEFAULT_FEATURE_MEANS["living_conditions"],
        "safety": DEFAULT_FEATURE_MEANS["safety"],
        "basic_needs": DEFAULT_FEATURE_MEANS["basic_needs"],
        "academic_performance": academic_performance,
        "study_load": study_load,
        "teacher_student_relationship": DEFAULT_FEATURE_MEANS["teacher_student_relationship"],
        "future_career_concerns": future_career_concerns,
        "social_support": social_support,
        "peer_pressure": DEFAULT_FEATURE_MEANS["peer_pressure"],
        "extracurricular_activities": extracurricular_activities,
        "bullying": DEFAULT_FEATURE_MEANS["bullying"],
    }

    user_df = pd.DataFrame([full_input])

    st.info("Some predictor values are filled with dataset averages so the model can generate a complete stress profile.")

    # ANALYZE BUTTON

    if st.button("Analyze Student"):
        scaled = scaler.transform(user_df)
        rf_prediction = rf_model.predict(scaled)[0]
        lasso_prediction = lasso_model.predict(scaled)[0]
        pca_data = pca.transform(scaled)
        cluster = kmeans.predict(pca_data)[0]

        # SAVE RESULTS
        st.session_state["user_df"] = user_df
        st.session_state["rf_prediction"] = rf_prediction
        st.session_state["lasso_prediction"] = lasso_prediction
        st.session_state["cluster"] = cluster
        st.session_state["pca_data"] = pca_data
        st.success("Analysis complete! Open Dashboard." )

# ===================================
# DASHBOARD PAGE
# ===================================

elif page == "Dashboard":

    st.title("📊 Student Dashboard")

    if "rf_prediction" not in st.session_state:
        st.warning("Please complete survey first." )
        st.stop()

    # LOAD SESSION DATA

    user_df = st.session_state["user_df"]
    rf_prediction = st.session_state["rf_prediction"]
    lasso_prediction = st.session_state["lasso_prediction"]
    cluster = st.session_state["cluster"]

    # METRICS

    col1, col2, col3 = st.columns(3)
    col1.metric("RF Prediction",rf_prediction)
    col2.metric("Lasso Score",round(lasso_prediction, 2)    )
    col3.metric("Cluster",cluster )

    # CLUSTER LABELS

    cluster_names = {0: "Balanced Students",1: "Burnout Risk",2: "Sleep-Deprived Achievers",3: "Low Engagement Students" }
    st.subheader(f"Student Type: {cluster_names.get(cluster, 'Unknown')}" )

    # SAFE GAUGE VALUE

    gauge_value = max(0,min(float(lasso_prediction) * 10, 100))

    fig = go.Figure(go.Indicator(mode="gauge+number",value=gauge_value, title={'text': "Predicted Stress Level"},gauge={'axis': {'range': [0, 100]}}))

    st.plotly_chart(fig,use_container_width=True)

    # ADVICE ENGINE

    st.header("🧠 Personalized Recommendations")

    advice = []

    # USE user_df VALUES

    if user_df["sleep_quality"][0] < 3:
        advice.append("Improve sleep quality with a consistent sleep schedule.")

    if user_df["anxiety_level"][0] > 15:
        advice.append("High anxiety detected. Consider mindfulness or counseling.")

    if user_df["extracurricular_activities"][0] < 2:
        advice.append("Increasing extracurricular involvement may improve well-being.")

    if user_df["social_support"][0] < 2:
        advice.append("Strengthen your support network through friends or organizations.")

    if user_df["depression"][0] > 15:
        advice.append("Elevated depression indicators detected. Consider support resources.")

    if user_df["future_career_concerns"][0] > 3:
        advice.append("Career uncertainty detected. Career planning workshops may help.")

    if user_df["academic_performance"][0] < 2:
        advice.append("Academic support resources may improve performance.")

    if user_df["study_load"][0] > 3:
        advice.append("Heavy study load detected. Balance productivity with recovery.")

    if len(advice) == 0:
        st.success("Your wellness indicators appear balanced.")

    else:
        for item in advice:
            st.info(item)

# ===================================
# EDA PAGE
# ===================================

elif page == "EDA Explorer":

    st.title("📈 Exploratory Data Analysis")
    st.markdown("""
    This section explores the relationships, distributions,
    and behavioral trends found within the student stress dataset.

    The visualizations below help identify patterns associated
    with academic pressure, emotional wellness, and lifestyle habits.
    """)

    df = pd.read_csv("stress.csv")
    st.subheader("Dataset Preview")
    st.markdown("""
    The table below displays a sample of the dataset used
    to train the machine learning models.

    Each row represents an individual student, while
    each column represents a behavioral, emotional,
    or academic feature collected during the study.
    """)

    st.dataframe(df.head())
    st.subheader("Feature Correlation Heatmap")
    st.markdown("""
    The correlation heatmap visualizes relationships
    between variables in the dataset.

    Positive correlations indicate that two variables
    increase together, while negative correlations
    indicate inverse relationships.

    Strong correlations may reveal major contributors
    to academic stress and emotional wellness.
    """)
    st.image("images/correlation_heatmap.png")
    st.subheader("Stress Distribution")
    st.markdown("""
    This visualization shows how stress levels are
    distributed across the dataset.

    It helps identify whether stress levels are balanced,
    concentrated in specific ranges, or skewed toward
    higher or lower stress categories.
    """)
    st.image("images/stress_distribution.png")
    st.subheader("PCA Cluster Visualization")
    st.markdown("""
    Principal Component Analysis (PCA) reduces the dataset
    into lower dimensions while preserving important patterns.

    The scatterplot below visualizes how students group
    together based on similarities in behavioral,
    emotional, and academic characteristics.

    Different clusters may represent distinct stress
    profiles or student wellness patterns.
    """)

    st.image("images/labeled_clusters.png")
    st.subheader("Interactive Feature Distribution")

    st.markdown("""
    Select a feature below to explore how values
    are distributed across the student population.

    This helps identify trends, concentration ranges,
    and variability within individual features.
    """)
    feature = st.selectbox("Select Feature",df.columns)
    fig = px.histogram(df,x=feature)
    st.plotly_chart(fig)

# ===================================
# ABOUT MODELS PAGE
# ===================================

elif page == "About Models":
    st.title("🧠 Machine Learning Pipeline")
    st.markdown("""
    ### Models Used
    - Random Forest Classifier
    - Lasso Regression
    - KMeans Clustering
    - Principal Component Analysis (PCA)

    ### Pipeline

    1. Data Cleaning and Preprocessing
    2. Feature Scaling
    3. PCA Dimensionality Reduction
    4. Clustering Analysis
    5. Stress Prediction
    6. Recommendation Engine
    """)