#!/usr/bin/env python
# coding: utf-8

# In[363]:


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
    predicted_class = model.predict(feature_values)[0]
    predicted_proba = model.predict_proba(feature_values)
    
    # 显示预测结果
    st.subheader("Prediction Results:")
    st.write("Risk Prediction:")
    st.write(f"- Heart Failure: {'Yes' if predicted_class[0] == 1 else 'No'} (Probability: {predicted_proba[0][0][1]})")
    st.write(f"- Myocardial Infarction: {'Yes' if predicted_class[1] == 1 else 'No'} (Probability: {predicted_proba[1][0][1]})")
    st.write(f"- Stroke: {'Yes' if predicted_class[2] == 1 else 'No'} (Probability: {predicted_proba[2][0][1]})")
    st.write(f"- Death: {'Yes' if predicted_class[3] == 1 else 'No'} (Probability: {predicted_proba[3][0][1]})")
    
    
    # 生成每个分类器的 LIME 解释
    classifiers = model.estimators_
    explainer = LimeTabularExplainer(feature_values, mode="classification",feature_names=feature_names)
    lime_explanations = []
    for i, classifier in enumerate(classifiers):
        explanation = explainer.explain_instance(feature_values[0], classifier.predict_proba)
        lime_explanations.append(explanation)
    
    # 可视化四个分类器的 LIME 解释实例
    for i, explanation in enumerate(lime_explanations):
        st.subheader(f"Explanation for Classifier {i + 1} ({['Heart Failure', 'Myocardial Infarction', 'Stroke', 'Death'][i]})")
        
        # 将 LIME 解释结果转为 matplotlib 图形并显示
        fig = explanation.as_pyplot_figure()  # LIME 返回的解释结果转换为 matplotlib 图形
        st.pyplot(fig)  # 在 Streamlit 中显示图形
    
    


# In[ ]:




