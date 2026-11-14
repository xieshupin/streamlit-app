import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import shap
import re

from ecc_inference import load_ecc, ecc_predict_proba

st.set_page_config(page_title="Cardiovascular Risk Predictor (ECC)", layout="wide")

# ============================================================
# 1. 精准字典
# ============================================================
FEATURE_MAP = {
    # --- 基础信息 ---
    'age': ('continuous', 'Age', 65.0, 1.0, 'Years'),
    'gender_M': ('binary', 'Gender', 0, 1.0, ''),
    'omr_bmi': ('continuous', 'BMI', 24.0, 1.0, 'kg/m²'),
    
    # --- 生命体征 ---
    'first_sbp': ('continuous', 'Systolic BP', 120.0, 1.0, 'mmHg'),
    'first_dbp': ('continuous', 'Diastolic BP', 75.0, 1.0, 'mmHg'),
    'first_heart_rate': ('continuous', 'Heart Rate', 75.0, 1.0, 'bpm'),
    'map_calculated': ('continuous', 'Mean Arterial Pressure', 90.0, 1.0, 'mmHg'),
    
    # --- 合并症与用药史 (全为二元变量) ---
    'peripheral_vascular_disease': ('binary', 'Peripheral Vascular Disease', 0, 1.0, ''),
    'chronic_pulmonary_disease': ('binary', 'Chronic Pulmonary Disease', 0, 1.0, ''),
    'diabetes_without_cc': ('binary', 'Diabetes (Without Complications)', 0, 1.0, ''),
    'diabetes_with_cc': ('binary', 'Diabetes (With Complications)', 0, 1.0, ''),
    'renal_disease': ('binary', 'Renal Disease', 0, 1.0, ''),
    'smoking_flag': ('binary', 'Smoking History', 0, 1.0, ''),
    'afib_flag': ('binary', 'Atrial Fibrillation', 0, 1.0, ''),
    'gerd_flag': ('binary', 'GERD', 0, 1.0, ''),
    'osa_flag': ('binary', 'Obstructive Sleep Apnea', 0, 1.0, ''),
    'hyperlipidemia_flag': ('binary', 'Hyperlipidemia', 0, 1.0, ''),
    'thrombocytopenia_flag': ('binary', 'Thrombocytopenia', 0, 1.0, ''),
    'gout_flag': ('binary', 'Gout', 0, 1.0, ''),
    'anemia_flag': ('binary', 'Anemia', 0, 1.0, ''),
    'hypothyroidism_flag': ('binary', 'Hypothyroidism', 0, 1.0, ''),
    'insulin_flag': ('binary', 'Insulin Use', 0, 1.0, ''),
    
    'preadmit_antiplatelet': ('binary', 'Pre-admit Antiplatelet', 0, 1.0, ''),
    'preadmit_statin': ('binary', 'Pre-admit Statin', 0, 1.0, ''),
    'preadmit_bb': ('binary', 'Pre-admit Beta-Blocker', 0, 1.0, ''),
    'preadmit_acei_arb': ('binary', 'Pre-admit ACEI/ARB', 0, 1.0, ''),
    'preadmit_diuretic': ('binary', 'Pre-admit Diuretic', 0, 1.0, ''),
    'preadmit_anticoagulant': ('binary', 'Pre-admit Anticoagulant', 0, 1.0, ''),
    'preadmit_vasodilator': ('binary', 'Pre-admit Vasodilator', 0, 1.0, ''),
    
    'pci_flag': ('binary', 'Prior PCI', 0, 1.0, ''),
    'cabg': ('binary', 'Prior CABG', 0, 1.0, ''),
    'Angina': ('binary', 'Angina', 0, 1.0, ''),
    'mitral_valve': ('binary', 'Mitral Valve Disease', 0, 1.0, ''),
    'aortic_valve': ('binary', 'Aortic Valve Disease', 0, 1.0, ''),
    'AKI': ('binary', 'Acute Kidney Injury', 0, 1.0, ''),
    'carotid_stenosis': ('binary', 'Carotid Stenosis', 0, 1.0, ''),
    
    # 已彻底删除 htn_group, degree_avb, ckd_stage1, htn_etiology_ckd
    
    # --- 实验室指标 (带单位转换) ---
    'Creatinine_50912': ('continuous', 'Creatinine', 1.0, 88.4, 'μmol/L', 10.0, 2000.0, 1.0),
    'Urea Nitrogen_51006': ('continuous', 'Urea (BUN)', 15.0, 0.357, 'mmol/L', 1.0, 100.0, 0.1),
    'Glucose_50931': ('continuous', 'Glucose', 100.0, 0.0555, 'mmol/L', 1.0, 50.0, 0.1),
    'Calcium, Total_50893': ('continuous', 'Total Calcium', 9.0, 0.25, 'mmol/L', 0.5, 5.0, 0.01),
    'Magnesium_50960': ('continuous', 'Magnesium', 2.0, 0.411, 'mmol/L', 0.1, 5.0, 0.01),
    'Phosphate_50970': ('continuous', 'Phosphate', 3.5, 0.3229, 'mmol/L', 0.1, 5.0, 0.01),
    'Potassium_50971': ('continuous', 'Potassium', 4.0, 1.0, 'mmol/L', 1.0, 10.0, 0.1),
    'Sodium_50983': ('continuous', 'Sodium', 140.0, 1.0, 'mmol/L', 100.0, 200.0, 1.0),
    'Chloride_50902': ('continuous', 'Chloride', 100.0, 1.0, 'mmol/L', 50.0, 150.0, 1.0),
    
    'Bicarbonate_50882': ('continuous', 'Bicarbonate', 24.0, 1.0, 'mEq/L', 5.0, 50.0, 1.0),
    'Anion Gap_50868': ('continuous', 'Anion Gap', 12.0, 1.0, 'mEq/L', 1.0, 50.0, 1.0),
    'Calculated Total CO2_50804': ('continuous', 'Total CO2', 24.0, 1.0, 'mEq/L', 5.0, 50.0, 1.0),
    
    'Hemoglobin_51222': ('continuous', 'Hemoglobin', 13.5, 10.0, 'g/L', 30.0, 250.0, 1.0), 
    'Hematocrit_51221': ('continuous', 'Hematocrit', 40.0, 1.0, '%', 10.0, 70.0, 0.1),
    'White Blood Cells_51301': ('continuous', 'White Blood Cells', 8.0, 1.0, 'x10^9/L', 0.1, 100.0, 0.1),
    'Red Blood Cells_51279': ('continuous', 'Red Blood Cells', 4.5, 1.0, 'x10^12/L', 1.0, 10.0, 0.1),
    'Platelet Count_51265': ('continuous', 'Platelet Count', 250.0, 1.0, 'x10^9/L', 1.0, 1000.0, 1.0),
    'RDW_51277': ('continuous', 'RDW', 13.0, 1.0, '%', 5.0, 30.0, 0.1),
    'MCV_51250': ('continuous', 'MCV', 88.0, 1.0, 'fL', 40.0, 150.0, 1.0),
    'MCH_51248': ('continuous', 'MCH', 29.0, 1.0, 'pg', 10.0, 50.0, 0.1),
    'MCHC_51249': ('continuous', 'MCHC', 33.0, 10.0, 'g/L', 200.0, 400.0, 1.0), 
    
    'INR(PT)_51237': ('continuous', 'INR (PT)', 1.0, 1.0, '', 0.5, 10.0, 0.1),
    'PT_51274': ('continuous', 'PT', 12.0, 1.0, 'seconds', 5.0, 100.0, 0.1),
    'PTT_51275': ('continuous', 'PTT', 30.0, 1.0, 'seconds', 10.0, 200.0, 1.0)
}

CORE_FEATURES = [
    # 核心人口学与体征
    'age', 'gender_M', 'omr_bmi', 'first_sbp', 'first_heart_rate', 'map_calculated',
    # 血液学与凝血 (SHAP高频出现)
    'Hemoglobin_51222', 'RDW_51277', 'Platelet Count_51265', 'PTT_51275', 'INR(PT)_51237',
    # 代谢与生化 (SHAP高频出现)
    'Creatinine_50912', 'Urea Nitrogen_51006', 'Glucose_50931', 'Magnesium_50960', 'Phosphate_50970', 
    'Calcium, Total_50893', 'Anion Gap_50868', 'Calculated Total CO2_50804',
    # 既往病史与用药 (SHAP高频出现)
    'cabg', 'pci_flag', 'Angina', 'peripheral_vascular_disease', 'carotid_stenosis', 'aortic_valve',
     'preadmit_antiplatelet', 'preadmit_statin'
]
# ============================================================
# ============================================================
# 2. 加载模型
# ============================================================
@st.cache_resource
def load_models():
    return load_ecc('models_ecc', n_chains=10, n_labels=4)

feature_names, label_cols, label_names, imputer, chains = load_models()

DISPLAY_NAMES = {
    'mi_readmission':     'Myocardial Infarction',
    'hf_readmission':     'Heart Failure',
    'stroke_readmission': 'Stroke',
    'death_1year':        '1-Year Death',
}
display_labels = [DISPLAY_NAMES.get(c, c) for c in label_cols]

# ============================================================
# 3. 页面头部
# ============================================================

st.title("❤️ Cardiovascular Risk Prediction (ECC Model)")

with st.expander("📋 Model Information & Disclaimer", expanded=True):
    # 1. 使用 HTML 卡片排版模型版本信息
    st.markdown(
        f"""
        <div style="background-color: #f8f9fa; padding: 15px; border-radius: 8px; border-left: 5px solid #2E86AB; margin-bottom: 20px;">
            <h4 style="margin-top: 0; color: #2E86AB;">Model Version</h4>
            <ul style="margin-bottom: 0;">
                <li><b>Version:</b> v2.0-ECC</li>
                <li><b>Release Date:</b> 2026-11-13</li>
                <li><b>Architecture:</b> Ensemble of Classifier Chains (ECC) + CatBoost</li>
            </ul>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # 2. 使用 HTML 卡片排版免责声明（红色警告框）
    st.markdown(
        """
        <div style="background-color: #fff3cd; padding: 15px; border-radius: 8px; border-left: 5px solid #ffc107;">
            <h4 style="margin-top: 0; color: #856404;">⚠️ Important Disclaimer</h4>
            <p style="color: #856404; font-weight: bold; margin-bottom: 10px;">
                This tool is for clinical research purposes only and should NOT be used as a standalone clinical decision-making tool.
            </p>
            <ol style="color: #856404; margin-bottom: 0; padding-left: 20px;">
                <li>Predictions are <b>supplementary</b> and <b>cannot replace</b> physician judgment;</li>
                <li>All results should be interpreted in conjunction with clinical presentation, laboratory findings, and imaging studies;</li>
                <li>This model was developed using retrospective cohort data and may have selection bias and unmeasured confounders;</li>
                <li>Final treatment decisions must be made by qualified clinicians based on complete clinical information.</li>
            </ol>
            <p style="color: #856404; font-style: italic; margin-top: 10px; margin-bottom: 0; font-size: 0.9em;">
                By using this tool, you acknowledge that you have read and understood this disclaimer.
            </p>
        </div>
        """, 
        unsafe_allow_html=True
    )
# ============================================================
# 4. 特征输入渲染逻辑
# ============================================================
input_values = {}

def render_exact_input(feat):
    if feat not in FEATURE_MAP:
        return 0.0
        
    config = FEATURE_MAP[feat]
    v_type, label, default, factor, unit = config[0], config[1], config[2], config[3], config[4]
    
    if v_type == 'binary':
        if 'gender' in feat:
            return st.selectbox(label, [0, 1], format_func=lambda x: "Female" if x == 0 else "Male", key=f"in_{feat}")
        return st.selectbox(label, [0, 1], format_func=lambda x: "No" if x == 0 else "Yes", key=f"in_{feat}")
        
    elif v_type == 'continuous':
        display_default = default * factor
        label_with_unit = f"{label} ({unit})" if unit else label
        
        if len(config) >= 8:
            min_val, max_val, step_val = config[5], config[6], config[7]
        else:
            min_val, max_val, step_val = 0.0, 1000.0, 1.0 if factor > 10 else 0.1
        
        val_display = st.number_input(
            label_with_unit, 
            min_value=float(min_val), 
            max_value=float(max_val), 
            value=float(round(display_default, 2)), 
            step=float(step_val),
            key=f"in_{feat}"
        )
        return val_display / factor
    return 0.0

st.subheader("🩺 Core Clinical Features (From SHAP Analysis)")
core_cols = st.columns(3)
core_count = 0
for feat in CORE_FEATURES:
    if feat in feature_names:
        with core_cols[core_count % 3]:
            input_values[feat] = render_exact_input(feat)
        core_count += 1

with st.expander("🔧 Advanced Clinical Features (Click to expand)", expanded=False):
    adv_cols = st.columns(3)
    adv_feats = [f for f in feature_names if f not in CORE_FEATURES and f in FEATURE_MAP]
    for i, feat in enumerate(adv_feats):
        with adv_cols[i % 3]:
            input_values[feat] = render_exact_input(feat)

# ============================================================
# 5. 预测按钮与结局选择（表单区，完美解决重绘 Bug）
# ============================================================
st.markdown("---")

with st.form("prediction_form"):
    st.subheader("🔬 Individual Risk Explanation Setting")
    shap_label_idx = st.selectbox(
        "Select Outcome to Explain", 
        options=list(range(len(label_cols))), 
        format_func=lambda i: display_labels[i], 
        index=0,
        key="main_outcome_select"  
    )
    submit_btn = st.form_submit_button("🔍 Predict Risk", type="primary")

# ============================================================
# 6. 预测输出与 SHAP 画图（只有点按钮才会执行）
# ============================================================
if submit_btn:
    full_input = np.zeros(len(feature_names))
    for i, feat in enumerate(feature_names):
        full_input[i] = input_values.get(feat, 0.0)

    input_df  = pd.DataFrame([full_input], columns=feature_names)
    input_imp = imputer.transform(input_df)
    
    proba_matrix = ecc_predict_proba(input_imp, chains)
    predictions  = proba_matrix[0].tolist()

    st.markdown("---")
    st.subheader("📊 Prediction Results: Event Risk Probabilities")
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    res_cols = st.columns(len(label_cols))
    for i, (name, prob, color) in enumerate(zip(display_labels, predictions, colors)):
        with res_cols[i]:
            st.markdown(f"""
            <div style="background-color: {color}10; border-radius: 10px; padding: 15px; text-align: center; border: 1px solid {color}40; height: 100%;">
                <div style="font-size: 14px; font-weight: bold; color: {color};">{name}</div>
                <div style="font-size: 32px; font-weight: bold; margin: 10px 0;">{prob:.2%}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("📈 Risk Visualization")
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=display_labels, y=predictions, text=[f"{p:.2%}" for p in predictions], 
        textposition='outside', marker_color=colors, name='Predicted Risk'
    ))
    fig.update_layout(yaxis_title="Predicted Probability", yaxis_range=[0, 1], yaxis_tickformat='.0%', height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # ============================================================
    # 7. SHAP 解释（彻底根除 other features）
    # ============================================================
    st.markdown("---")
    st.subheader("🔬 Individual Risk Explanation (SHAP)")

    chain0 = chains[0]
    order0 = chain0['order']
    pos = order0.index(shap_label_idx) 

    X_aug = np.asarray(input_imp).copy()
    for j in range(pos):
        p_prev = chain0['cats'][j].predict_proba(X_aug)[:, 1].reshape(-1, 1)
        X_aug = np.column_stack([X_aug, p_prev])

    clean_names = [FEATURE_MAP[f][1] if f in FEATURE_MAP else f for f in feature_names]
    feat_names_shap = clean_names + [f"Predicted {DISPLAY_NAMES.get(label_cols[order0[j]], label_cols[order0[j]])}" for j in range(pos)]

    explainer  = shap.TreeExplainer(chain0['cats'][pos])
    shap_vals  = explainer.shap_values(X_aug)

    # 👇 核心修改：在画图前，手动找出绝对值最大的前 15 个特征索引
    # 取单条样本的 SHAP 值
    single_shap = shap_vals[0] 
    # 按绝对值大小排序，提取前 15 名的索引
    top_indices = np.argsort(np.abs(single_shap))[-15:]
    
    # 过滤出这 15 个特征的数据、SHAP 值和名字
    filtered_shap_vals = single_shap[top_indices]
    filtered_data = X_aug[0][top_indices]
    filtered_names = [feat_names_shap[i] for i in top_indices]

    # 使用过滤后的数据画图，这样 SHAP 就不知道还有剩下的特征了
    fig_shap, ax = plt.subplots(figsize=(10, 5))
    shap.waterfall_plot(
        shap.Explanation(
            values=filtered_shap_vals, 
            base_values=explainer.expected_value, 
            data=filtered_data, 
            feature_names=filtered_names
        ),
        max_display=15, 
        show=False
    )
    
    # 之前加的裁剪 Y 轴和清理前缀依然保留（为了安全起见）
    ax.set_yticklabels([label.get_text().split(' = ')[-1] for label in ax.get_yticklabels()])

    # 概率转换显示
    for text in ax.texts:
        text_str = text.get_text()
        if text_str.startswith('f(x)'):
            val = float(text_str.split('=')[1].strip())
            prob = 1 / (1 + np.exp(-val))
            text.set_text(f"Pred Risk: {prob:.1%}")
        elif text_str.startswith('E[f(X)]'):
            val = float(text_str.split('=')[1].strip())
            prob = 1 / (1 + np.exp(-val))
            text.set_text(f"Base Risk: {prob:.1%}")

    ax.set_title(f"Risk Factors for {display_labels[shap_label_idx]}", fontsize=12)
    plt.tight_layout()
    st.pyplot(fig_shap)
