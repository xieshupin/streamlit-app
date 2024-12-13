#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载模型
model = joblib.load('XGBoost.pkl')

# 英文特征和标签
feature_names = ["Age", "Sex", "Cardiac Troponin", "Cystatin C", "Post-PCI", "Thyroid-Stimulating Hormone", 
                 "Allergy History", "Hyperlipidemia", "Diabetes Mellitus", "COPD", "Antiplatelet Drugs", 
                 "Atrial Fibrillation", "Alcohol Consumption History", "Beta-blockers", "Diuretics", "ACEI/ARB", 
                 "Vasodilators", "Renal Insufficiency", "Smoking History", "Statins"]

label_names = ["Heart Failure", "Myocardial Infarction", "Stroke", "Death"]

# Streamlit 用户界面
st.title("Risk Prediction System for Coronary Heart Disease Combined with Hypertension")

# 用户输入特征值
Age = st.number_input("Age:", min_value=1, max_value=120, value=50)
Sex = st.selectbox("Sex:", options=[0, 1], format_func=lambda x: 'Female (0)' if x == 0 else 'Male (1)')
Cardiac Troponin = st.number_input("Cardiac Troponin:", min_value=1, max_value=457, value=63)
Cystatin C = st.number_input("Cystatin C:", min_value=1.0, max_value=2.4, value=1.5)
Post-PCI = st.selectbox("Post-PCI:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
Thyroid-Stimulating Hormone = st.number_input("Thyroid-Stimulating Hormone:", min_value=1.4, max_value=7.8, value=3.4)
Allergy History = st.selectbox("Allergy History:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
Hyperlipidemia = st.selectbox("Hyperlipidemia:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
Diabetes Mellitus = st.selectbox("Diabetes Mellitus:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
COPD = st.selectbox("COPD:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
Antiplatelet Drugs = st.selectbox("Antiplatelet Drugs:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
Atrial Fibrillation = st.selectbox("Atrial Fibrillation:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
Alcohol Consumption History = st.selectbox("Alcohol Consumption History:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
Beta-blockers = st.selectbox("Beta-blockers:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
Diuretics = st.selectbox("Diuretics:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
ACEI/ARB = st.selectbox("ACEI/ARB:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
Vasodilators = st.selectbox("Vasodilators:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
Renal Insufficiency = st.selectbox("Renal Insufficiency:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
Smoking History = st.selectbox("Smoking History:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
Statins = st.selectbox("Statins:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# 将用户输入的数据转换为模型可用的格式
feature_values = [Age, Sex, Cardiac Troponin, Cystatin C, Post-PCI, Thyroid-Stimulating Hormone, 
                 Allergy History, Hyperlipidemia, Diabetes Mellitus, COPD, Antiplatelet Drugs, 
                 Atrial Fibrillation, Alcohol Consumption History, Beta-blockers, Diuretics, ACEI/ARB, 
                 Vasodilators, Renal Insufficiency, Smoking History, Statins]
feature_values = np.array([feature_values])

# 按钮触发预测
if st.button("Predict"):
    # 模型预测
    predicted_class = model.predict(feature_values)
    predicted_proba = model.predict_proba(feature_values)

    # 显示预测结果
    st.subheader("Prediction Results:")
    for i, english_label in enumerate(label_names):
        st.write(f"{english_label}: {'Yes' if predicted_class[0] == i else 'No'} (Probability: {predicted_proba[0][i]:.2f})")
   
    # SHAP值可视化
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame(feature_values, columns=feature_names))
    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame(feature_values, columns=feature_names), matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")

