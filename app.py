import streamlit as st
import pandas as pd
import numpy as np
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load your model and data preprocessing here
# (or load a saved model)

# Sample input form
st.title("Titanic Survival Prediction & Explanation")

Pclass = st.selectbox("Pclass", options=[1,2,3], index=2)
Sex = st.selectbox("Sex", options=["male","female"], index=0)
Age = st.slider("Age", 0, 100, 30)
Fare = st.slider("Fare", 0.0, 512.0, 32.0)
SibSp = st.number_input("SibSp", 0, 10, 0)
Parch = st.number_input("Parch", 0, 10, 0)
Embarked = st.selectbox("Embarked", options=["S", "C", "Q"], index=0)

# Encode inputs
sex_encoded = 0 if Sex=="male" else 1
embarked_encoded = {"S":0, "C":1, "Q":2}[Embarked]

input_features = np.array([[Pclass, sex_encoded, Age, Fare, SibSp, Parch, embarked_encoded]])

# Model prediction
# Load the trained model (make sure rf_model.pkl is in the same folder as app.py)
model = joblib.load('rf_model.pkl')
X_train = joblib.load('X_train.pkl')

# model.fit(...)  # load your trained model or retrain here

prediction_proba = model.predict_proba(input_features)[0,1]
st.write(f"Predicted Survival Probability: {prediction_proba:.2f}")

# SHAP explainability
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(input_features)

shap.initjs()
st.subheader("SHAP Explanation")
import streamlit.components.v1 as components
force_plot_html = shap.force_plot(
    explainer.expected_value[1], 
    shap_values[0,:,1], 
    input_features[0], 
    matplotlib=False
)
components.html(force_plot_html.html(), height=300)

# LIME explanation
explainer_lime = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=["Not Survived", "Survived"],
    mode="classification")
exp = explainer_lime.explain_instance(input_features[0], model.predict_proba, num_features=7)

st.subheader("LIME Explanation")
st.write(exp.as_list())
