#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import streamlit as st
import joblib
import numpy as np
import pandas as pd
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt

# 加载模型
model = joblib.load('XGBoost.pkl')

# 英文特征和标签
feature_names = ["Age", "Sex", "Cardiac Troponin", "Cystatin C", "Post-PCI", "Thyroid-Stimulating Hormone", 
                 "Allergy History", "Hyperlipidemia", "Diabetes Mellitus", "COPD", "Antiplatelet Drugs", 
                 "Atrial Fibrillation", "Alcohol Consumption History", "Beta-blockers", "Diuretics", "ACEI/ARB", 
                 "Vasodilators", "Renal Insufficiency", "Smoking History", "Statins"]

label_names = ["Heart Failure", "Myocardial Infarction", "Stroke", "Death"]
label =["HeartFailure", "MyocardialInfarction", "Stroke", "Death"]

# Streamlit 用户界面
st.title("Risk Prediction System for Coronary Heart Disease Combined with Hypertension")

# 用户输入特征值
Age = st.number_input("Age:", min_value=1, max_value=120, value=50)
Sex = st.selectbox("Sex:", options=[0, 1], format_func=lambda x: 'Female (0)' if x == 0 else 'Male (1)')
CardiacTroponin = st.number_input("Cardiac Troponin:", min_value=1, max_value=457, value=63)
CystatinC = st.number_input("Cystatin C:", min_value=1.0, max_value=2.4, value=1.5)
PostPCI = st.selectbox("Post-PCI:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
ThyroidStimulatingHormone = st.number_input("Thyroid-Stimulating Hormone:", min_value=1.4, max_value=7.8, value=3.4)
AllergyHistory = st.selectbox("Allergy History:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
Hyperlipidemia = st.selectbox("Hyperlipidemia:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
DiabetesMellitus = st.selectbox("Diabetes Mellitus:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
COPD = st.selectbox("COPD:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
AntiplateletDrugs = st.selectbox("Antiplatelet Drugs:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
AtrialFibrillation = st.selectbox("Atrial Fibrillation:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
AlcoholConsumptionHistory = st.selectbox("Alcohol Consumption History:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
Betablockers = st.selectbox("Beta-blockers:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
Diuretics = st.selectbox("Diuretics:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
ACEIARB = st.selectbox("ACEI/ARB:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
Vasodilators = st.selectbox("Vasodilators:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
RenalInsufficiency = st.selectbox("Renal Insufficiency:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
SmokingHistory = st.selectbox("Smoking History:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
Statins = st.selectbox("Statins:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# 将用户输入的数据转换为模型可用的格式
feature_values = [Age, Sex, CardiacTroponin, CystatinC, PostPCI, ThyroidStimulatingHormone, 
                 AllergyHistory, Hyperlipidemia, DiabetesMellitus, COPD, AntiplateletDrugs, 
                 AtrialFibrillation, AlcoholConsumptionHistory, Betablockers, Diuretics, ACEIARB, 
                 Vasodilators, RenalInsufficiency, SmokingHistory, Statins]
feature_values = np.array([feature_values])

# 按钮触发预测
if st.button("Predict"):
    # 模型预测
    predicted_class = model.predict(feature_values)
    predicted_proba = model.predict_proba(feature_values)

    # 显示预测结果
    st.subheader("Prediction Results:")
    for i, label in enumerate(label_names):
        st.write(f"{label}: {'Yes' if predicted_class[0] == i else 'No'} (Probability: {predicted_proba[0][i]:.2f})")

    # 使用 LIME 解释模型
    explainer = LimeTabularExplainer(
        training_data=np.random.random((100, len(feature_names))),  # 假设随机生成训练数据
        feature_names=feature_names,
        class_names=label_names,
        mode='classification'
    )

    explanation = explainer.explain_instance(
        data_row=feature_values[0], 
        predict_fn=model.predict_proba
    )

    # 显示 LIME 可视化结果
    explanation_fig = explanation.as_pyplot_figure()
    st.pyplot(explanation_fig)

