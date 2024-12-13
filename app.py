#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
# Load the model
model = joblib.load('XGBoost.pkl')

# Define feature names
feature_names = ["Age", "Sex", "Cardiac troponin", "Cystatin C", "Post-percutaneous coronary intervention ", "TSH", 
                 "Allergy history", "Hyperlipidemia", "Diabetes mellitus","COPD", "Antiplatelet drugs", "Atrial fibrillation", 
                 "Alcohol consumption history","Beta-blockers","Diuretics","ACEI/ARB", "Vasodilators", "Renal insufficiency","Smoking history","Statins"]
# Streamlit user interface
st.title("Heart Disease Predictor")

# age: numerical input
年龄 = st.number_input("Age:", min_value=1, max_value=120, value=50)

# sex: categorical selection
性别 = st.selectbox("Sex (0=Male, 1=Female):", options=[0, 1], format_func=lambda x: 'Female (1)' if x == 0 else 'Male (0)')

# cp: categorical selection
肌钙蛋白 = st.number_input("Cardiac troponin in mg/ml(cTn):", min_value=1, max_value=457, value=63)

# trestbps: numerical input
胱抑素C = st.number_input("Cystatin C in mg/L(Cys-C):", min_value=1.0, max_value=2.4, value=1.5)

# chol: numerical input
PCI术后 = st.selectbox("Post-percutaneous coronary intervention(0=No, 1=Yes):", 
                     options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# fbs: categorical selection
促甲状腺素 = st.number_input("Thyroid-stimulating hormone in mg/dl (TSH):", min_value=1.4, max_value=7.8, value=3.4)

# restecg: categorical selection
过敏史 = st.selectbox("Allergy history(0=No, 1=Yes):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# thalach: numerical input
高脂血症 = st.selectbox("Hyperlipidemia (0=No, 1=Yes):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# exang: categorical selection
糖尿病 = st.selectbox("Diabetes mellitus (0=No, 1=Yes):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# oldpeak: numerical input
COPD = st.selectbox("COPD(0=No, 1=Yes):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# slope: categorical selection
抗血小板药物 = st.selectbox("Antiplatelet drugs(0=No, 1=Yes) :", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# ca: numerical input
房颤 = st.selectbox("Atrial fibrillation(0=No, 1=Yes):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# thal: categorical selection
饮酒史 = st.selectbox("Alcohol consumption history(0=No, 1=Yes):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')


β受体阻滞剂 = st.selectbox("Beta-blockers(0=No, 1=Yes):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
利尿剂 = st.selectbox("Diuretics(0=No, 1=Yes):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
ACEI_ARB = st.selectbox("ACEI/ARB(0=No, 1=Yes):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
扩血管药物 = st.selectbox("Vasodilators(0=No, 1=Yes):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
肾功能不全 = st.selectbox("Renal insufficiency(0=No, 1=Yes):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
吸烟史 = st.selectbox("Smoking history(0=No, 1=Yes):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
他汀类药物 = st.selectbox("Statins(0=No, 1=Yes):", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# Process inputs and make predictions
feature_values = ['年龄','性别','肌钙蛋白','胱抑素C','PCI术后','促甲状腺素','过敏史','高脂血症','糖尿病','COPD','抗血小板药物','房颤',
                  '饮酒史','β受体阻滞剂','利尿剂','ACEI/ARB','扩血管药物','肾功能不全','吸烟史',
                  '他汀类药物']
features = np.array([feature_values])

 # 按钮触发预测
if st.button("Predict"):
    # 预测结果
    predicted_class = model.predict(features)[0]    
    predicted_proba = model.predict_proba(features)[0]
    
    
    # 显示每个标签的预测结果

    st.write("### Predicted Classes")
    st.write(f"Heart Failure: {'Yes' if predicted_class[0] == 1 else 'No'}")
    st.write(f"Heart Attack: {'Yes' if predicted_class[1] == 1 else 'No'}")
    st.write(f"Stroke: {'Yes' if predicted_class[2] == 1 else 'No'}")
    st.write(f"Death Risk: {'Yes' if predicted_class[3] == 1 else 'No'}")
    
    st.write("### Predicted Probabilities")
    st.write(f"Heart Failure Probability: {predicted_proba[0]:.2f}")
    st.write(f"Heart Attack Probability: {predicted_proba[1]:.2f}")
    st.write(f"Stroke Probability: {predicted_proba[2]:.2f}")
    st.write(f"Death Risk Probability: {predicted_proba[3]:.2f}")
    st.write(advice)
    
    # Calculate SHAP values and display force plot    
    explainer = shap.TreeExplainer(model)    
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))
    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)    
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")

