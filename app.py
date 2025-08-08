import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from PIL import Image

from sklearn.preprocessing import StandardScaler, PolynomialFeatures

st.set_page_config(page_title='Protein RMSD Prediction', layout='wide')
st.title('Protein RMSD Prediction App')


# Map internal feature names to display names
feature_display_names = {
    'F1': 'Total surface area (F1)',
    'F2': 'Non polar exposed area (F2)',
    'F3': 'Fractional area of exposed non polar residue (F3)',
    'F4': 'Fractional area of exposed non polar part of residue (F4)',
    'F5': 'Molecular mass weighted exposed area (F5)',
    'F6': 'Average deviation from standard exposed area of residue (F6)',
    'F7': 'Euclidian distance (F7)',
    'F8': 'Secondary structure penalty (F8)',
    'F9': 'Spacial Distribution constraints (N,K Value) (F9)'
}

default_values = {
    'F1': 5.0,   
    'F2': 10.0,
    'F3': 3.5,
    'F4': 7.0,
    'F5': 1.2,
    'F6': 0.0,
    'F7': 2.0,
    'F8': 4.5,
    'F9': 6.0
}

# Load model and feature names
def load_model():
    model_path = 'models/RandomForest_best_model.pkl'
    if os.path.exists(model_path):
        return joblib.load(model_path)
    st.error('Model file not found!')
    return None

def get_feature_names():
    # Use the original features for input
    return ['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8', 'F9']

model = load_model()
feature_names = get_feature_names()

# st.sidebar.header('Input Features')
# user_input = {}
# for feat in feature_names:
#     if feat.startswith('F'):
#         user_input[feat] = st.sidebar.number_input(f'{feat}', value=0.0, format='%f')


st.sidebar.header('Input Features')
user_input = {}
for feat in feature_names:  # ['F1','F2',...]
    display_name = feature_display_names.get(feat, feat)
    default_val = default_values.get(feat, 0.0)  # fallback to 0 if not defined
    user_input[feat] = st.sidebar.number_input(display_name, value=default_val, format='%f')



if st.sidebar.button('Predict RMSD'):
    # Prepare input for model (polynomial features not handled here)
    input_df = pd.DataFrame([user_input])
    # Use the same scaling as in feature_engineering.py
    train_df = pd.read_csv('data/protein_clean.csv')
    X_train = train_df[feature_names]
    scaler = StandardScaler()
    poly = PolynomialFeatures(degree=2, include_bias=False)
    scaler.fit(X_train)
    X_scaled = scaler.transform(input_df)
    X_poly = poly.fit(X_train).transform(X_scaled)
    # Predict
    if model:
        pred = model.predict(X_poly)
        st.success(f'Predicted RMSD: {pred[0]:.3f}')
        # SHAP explainability (for tree-based models)
        # if hasattr(model, 'feature_importances_'):
        #     st.subheader('Prediction Explainability (SHAP)')
        #     # Fit explainer on a small sample for speed
        #     background = poly.transform(scaler.transform(X_train.sample(100, random_state=42)))
        #     explainer = shap.TreeExplainer(model, background)
        #     shap_values = explainer.shap_values(X_poly)
        #     # Show force plot
        #     shap.initjs()
        #     st_shap = st.components.v1.html if hasattr(st, 'components') else st.markdown
        #     force_html = shap.force_plot(explainer.expected_value, shap_values[0], input_df, feature_names=poly.get_feature_names_out(feature_names), matplotlib=False, show=False)
        #     st_shap(shap.getjs(force_html), height=300)
        # else:
        #     st.info('SHAP explainability is only available for tree-based models.')

st.header('Dashboard')
eda_dir = 'outputs/eda_visuals'
if os.path.exists(eda_dir):
    eda_imgs = [f for f in os.listdir(eda_dir) if f.endswith('.png')]
    for img in sorted(eda_imgs):
        st.subheader(img.replace('_', ' ').replace('.png', '').title())
        st.image(os.path.join(eda_dir, img), use_container_width=True)

st.header('Model Performance')
perf_path = 'outputs/model_performance.csv'
if os.path.exists(perf_path):
    perf_df = pd.read_csv(perf_path, index_col=0)
    st.dataframe(perf_df)

st.header('Feature Importances')
fi_path = 'outputs/feature_importance.csv'
if os.path.exists(fi_path):
    fi_df = pd.read_csv(fi_path)
    st.dataframe(fi_df.head(20))
