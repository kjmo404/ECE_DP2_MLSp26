import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Student Wellness Analytics",layout="wide",initial_sidebar_state="expanded")
rf_model = joblib.load("models/rf.pkl")
lasso_model = joblib.load("models/lasso.pkl")
kmeans = joblib.load("models/kmeans.pkl")
pca = joblib.load("models/pca.pkl")
scaler = joblib.load("models/scaler.pkl")
st.sidebar.title("🎓 Student Wellness Analytics")

page = st.sidebar.radio( "Navigation",["Survey","Dashboard","EDA Explorer","About Models"])

if page == "Survey":

    st.title("📋 Academic Wellness Survey")
    sleep = st.slider("Sleep Hours", 0, 12, 7)

study = st.slider("Study Hours Per Day", 0, 15, 4)

exercise = st.slider("Exercise Frequency", 0, 14, 3)

social = st.slider("Social Activity", 0, 10, 5)

anxiety = st.slider("Anxiety Level", 1, 10, 5)

depression = st.slider("Depression Level", 1, 10, 4)

financial = st.slider("Financial Stress", 1, 10, 5)

attendance = st.slider("Attendance Percentage", 0, 100, 85)

user_df = pd.DataFrame({
    "sleep_hours": [sleep],
    "study_hours": [study],
    "exercise_frequency": [exercise],
    "social_activity": [social],
    "anxiety_level": [anxiety],
    "depression_level": [depression],
    "financial_stress": [financial],
    "attendance": [attendance]
})
if st.button("Analyze Student"):

    scaled = scaler.transform(user_df)

    rf_prediction = rf_model.predict(scaled)[0]

    lasso_prediction = lasso_model.predict(scaled)[0]

    pca_data = pca.transform(scaled)

    cluster = kmeans.predict(pca_data)[0]

    st.session_state["user_df"] = user_df

    st.session_state["rf_prediction"] = rf_prediction

    st.session_state["lasso_prediction"] = lasso_prediction

    st.session_state["cluster"] = cluster

    st.session_state["pca_data"] = pca_data

    st.success("Analysis complete! Open Dashboard.")

if page == "Dashboard":

    st.title("📊 Student Dashboard")
    if "rf_prediction" not in st.session_state:
        st.warning("Please complete survey first.")
        st.stop()
    user_df = st.session_state["user_df"]
    rf_prediction = st.session_state["rf_prediction"]
    lasso_prediction = st.session_state["lasso_prediction"]
    cluster = st.session_state["cluster"]

    col1, col2, col3 = st.columns(3)

col1.metric("RF Prediction", rf_prediction)
col2.metric("Lasso Score", round(lasso_prediction,2))
col3.metric("Cluster", cluster)

cluster_names = {
    0: "Balanced Students",
    1: "Burnout Risk",
    2: "Sleep-Deprived Achievers",
    3: "Low Engagement Students"
}

st.subheader(f"Student Type: {cluster_names[cluster]}")
fig = go.Figure(go.Indicator(
    mode="gauge+number",
    value=lasso_prediction * 10,
    title={'text': "Predicted Stress Level"},
    gauge={'axis': {'range': [0, 100]}}
))

st.plotly_chart(fig, use_container_width=True)

st.header("🧠 Personalized Recommendations")

advice = []
if sleep < 6:

    advice.append(
        "Increase sleep consistency and reduce late-night studying."
    )

if anxiety > 7:

    advice.append(
        "Consider stress management techniques and workload balancing."
    )

if exercise < 2:

    advice.append(
        "Adding light weekly exercise may improve mental wellness."
    )

if financial > 7:

    advice.append(
        "Financial stress appears elevated. Consider budgeting resources or campus support."
    )

if study > 10:

    advice.append(
        "High study duration may increase burnout risk. Use Pomodoro scheduling."
    )

    for item in advice:

    st.info(item)


    if page == "EDA Explorer":

    st.title("📈 Exploratory Data Analysis")

    df = pd.read_csv("student_stress.csv")

    st.subheader("Dataset Preview")

st.dataframe(df.head())
st.subheader("Feature Correlation Heatmap")

st.image("eda/correlation_heatmap.png")

st.subheader("Stress Distribution")

st.image("eda/stress_distribution.png")

st.subheader("PCA Cluster Visualization")

st.image("eda/pca_clusters.png")

st.subheader("PCA Cluster Visualization")

st.image("eda/pca_clusters.png")

feature = st.selectbox(
    "Select Feature",
    df.columns
)

fig = px.histogram(df, x=feature)

st.plotly_chart(fig)

feature = st.selectbox(
    "Select Feature",
    df.columns
)

fig = px.histogram(df, x=feature)

st.plotly_chart(fig)

if page == "About Models":

    st.title("🧠 Machine Learning Pipeline")

    st.markdown("""
    ### Models Used

    - Random Forest Regressor
    - Lasso Regression
    - KMeans Clustering
    - Principal Component Analysis (PCA)

    ### Pipeline

    1. Data Cleaning
    2. Feature Scaling
    3. PCA Dimensionality Reduction
    4. Clustering Analysis
    5. Stress Prediction
    6. Recommendation System
    """)