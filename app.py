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

# 定义特征名（中文）
feature_names_chinese = ["年龄", "性别", "肌钙蛋白", "胱抑素C", "PCI术后", "促甲状腺素", "过敏史", "高脂血症", 
                         "糖尿病", "COPD", "抗血小板药物", "房颤", "饮酒史", "β受体阻滞剂", "利尿剂", 
                         "ACEI/ARB", "扩血管药物", "肾功能不全", "吸烟史", "他汀类药物"]

# 定义英文变量名（用于界面展示）
feature_names_english = ["Age", "Sex", "Cardiac Troponin", "Cystatin C", "Post-PCI", "Thyroid-Stimulating Hormone (TSH)", 
                         "Allergy History", "Hyperlipidemia", "Diabetes Mellitus", "COPD", "Antiplatelet Drugs", 
                         "Atrial Fibrillation", "Alcohol Consumption History", "Beta-blockers", "Diuretics", "ACEI/ARB", 
                         "Vasodilators", "Renal Insufficiency", "Smoking History", "Statins"]

# 映射中文到英文
feature_mapping = dict(zip(feature_names_chinese, feature_names_english))

# 定义标签名（中文）
label_names_chinese = ["心力衰竭", "心脏病发作", "中风", "死亡"]

# 定义标签名（英文）
label_names_english = ["Heart Failure", "Heart Attack", "Stroke", "Death"]

# 映射标签名（中文到英文）
label_mapping = dict(zip(label_names_chinese, label_names_english))


# Streamlit用户界面
st.title("Risk Prediction System for Coronary Heart Disease Combined with Hypertension")

# 用户输入特征值
年龄 = st.number_input(feature_mapping["年龄"] + ":", min_value=1, max_value=120, value=50)
性别 = st.selectbox(feature_mapping["性别"] + ":", options=[0, 1], format_func=lambda x: 'Female (0)' if x == 0 else 'Male (1)')
肌钙蛋白 = st.number_input(feature_mapping["肌钙蛋白"] + ":", min_value=1, max_value=457, value=63)
胱抑素C = st.number_input(feature_mapping["胱抑素C"] + ":", min_value=1.0, max_value=2.4, value=1.5)
PCI术后 = st.selectbox(feature_mapping["PCI术后"] + ":", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
促甲状腺素 = st.number_input(feature_mapping["促甲状腺素"] + ":", min_value=1.4, max_value=7.8, value=3.4)
过敏史 = st.selectbox(feature_mapping["过敏史"] + ":", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
高脂血症 = st.selectbox(feature_mapping["高脂血症"] + ":", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
糖尿病 = st.selectbox(feature_mapping["糖尿病"] + ":", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
COPD = st.selectbox(feature_mapping["COPD"] + ":", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
抗血小板药物 = st.selectbox(feature_mapping["抗血小板药物"] + ":", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
房颤 = st.selectbox(feature_mapping["房颤"] + ":", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
饮酒史 = st.selectbox(feature_mapping["饮酒史"] + ":", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
β受体阻滞剂 = st.selectbox(feature_mapping["β受体阻滞剂"] + ":", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
利尿剂 = st.selectbox(feature_mapping["利尿剂"] + ":", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
ACEI_ARB = st.selectbox(feature_mapping["ACEI/ARB"] + ":", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
扩血管药物 = st.selectbox(feature_mapping["扩血管药物"] + ":", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
肾功能不全 = st.selectbox(feature_mapping["肾功能不全"] + ":", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
吸烟史 = st.selectbox(feature_mapping["吸烟史"] + ":", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
他汀类药物 = st.selectbox(feature_mapping["他汀类药物"] + ":", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# 将用户输入的数据转换为模型可用的格式（中文变量名）
feature_values = np.array([[年龄, 性别, 肌钙蛋白, 胱抑素C, PCI术后, 促甲状腺素, 过敏史, 高脂血症, 
                            糖尿病, COPD, 抗血小板药物, 房颤, 饮酒史, β受体阻滞剂, 利尿剂, 
                            ACEIARB, 扩血管药物, 肾功能不全, 吸烟史, 他汀类药物]])

# 按钮触发预测
if st.button("Predict"):
    # 模型预测
    predicted_class = model.predict(feature_values)
    predicted_proba = model.predict_proba(feature_values)


    # 显示预测结果
    st.subheader("Prediction Results:")
    for i, chinese_label in enumerate(label_names_chinese):
        english_label = label_mapping[chinese_label]  # 映射到英文标签
        st.write(f"{english_label}: {'Yes' if predicted_class[0] == i else 'No'} (Probability: {predicted_proba[0][i]:.2f})")
   
    # SHAP值可视化
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame(feature_values, columns=feature_names_chinese))
    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame(feature_values, columns=feature_names_chinese), matplotlib=True)
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")

