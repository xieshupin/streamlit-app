# ecc_inference.py
import numpy as np
import joblib
from catboost import CatBoostClassifier


def load_ecc(models_dir='models_ecc', n_chains=10, n_labels=4):
    """加载 ECC 部署产物：10 条链 × 4 个 CatBoost + 10 个 order + imputer + 元信息。"""
    feature_names = joblib.load(f'{models_dir}/feature_names.pkl')
    label_cols    = joblib.load(f'{models_dir}/label_cols.pkl')
    label_names   = joblib.load(f'{models_dir}/label_names.pkl')
    imputer       = joblib.load(f'{models_dir}/imputer.pkl')

    chains = []
    for c in range(n_chains):
        order = list(joblib.load(f'{models_dir}/chain{c}_order.pkl'))
        cats = []
        for pos in range(n_labels):
            cat = CatBoostClassifier()
            cat.load_model(f'{models_dir}/chain{c}_pos{pos}.cbm')
            cats.append(cat)
        chains.append({'order': order, 'cats': cats})

    return feature_names, label_cols, label_names, imputer, chains


def ecc_predict_proba(X_imp, chains, n_labels=4):
    """ECC 链式推理：按每条链的 order 串行预测，前面标签的概率拼到特征末尾，10 条链取平均。"""
    all_probas = []
    for chain in chains:
        order, cats = chain['order'], chain['cats']
        proba = np.zeros((X_imp.shape[0], n_labels))
        X_aug = np.asarray(X_imp).copy()
        for pos, label_idx in enumerate(order):
            p = cats[pos].predict_proba(X_aug)[:, 1]
            proba[:, label_idx] = p
            X_aug = np.column_stack([X_aug, p])
        all_probas.append(proba)
    return np.mean(all_probas, axis=0)