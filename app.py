import pickle

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Wine Quality Predictor",
    layout="wide",
)


@st.cache_resource
def load_artifacts():
    with open("wine_model.pkl", "rb") as model_file:
        loaded_model = pickle.load(model_file)

    with open("scaler.pkl", "rb") as scaler_file:
        loaded_scaler = pickle.load(scaler_file)

    return loaded_model, loaded_scaler


model, scaler = load_artifacts()


st.markdown(
    """
    <style>
        :root {
            --wine: #7f1234;
            --wine-dark: #3d0b1f;
            --rose: #f8e7ec;
            --ink: #20161a;
            --muted: #6d5f64;
            --line: rgba(127, 18, 52, 0.16);
        }

        .stApp {
            background:
                radial-gradient(circle at top left, rgba(127, 18, 52, 0.12), transparent 32rem),
                linear-gradient(135deg, #fff8f5 0%, #fbf4ef 45%, #f7eef2 100%);
            color: var(--ink);
        }

        .block-container {
            max-width: 1120px;
            padding-top: 2.25rem;
            padding-bottom: 2.5rem;
        }

        .hero {
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 2rem;
            background:
                linear-gradient(130deg, rgba(61, 11, 31, 0.94), rgba(127, 18, 52, 0.88)),
                url("https://images.unsplash.com/photo-1510812431401-41d2bd2722f3?auto=format&fit=crop&w=1400&q=80");
            background-size: cover;
            background-position: center;
            color: white;
            box-shadow: 0 24px 60px rgba(61, 11, 31, 0.16);
            margin-bottom: 1.25rem;
        }

        .eyebrow {
            color: #ffd6df;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.6rem;
        }

        .hero h1 {
            font-size: clamp(2.15rem, 4vw, 3.7rem);
            line-height: 1.02;
            margin: 0;
            letter-spacing: 0;
        }

        .hero p {
            max-width: 680px;
            margin: 0.85rem 0 0;
            color: rgba(255, 255, 255, 0.82);
            font-size: 1.04rem;
        }

        .panel {
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 1.25rem;
            background: rgba(255, 255, 255, 0.82);
            box-shadow: 0 16px 36px rgba(61, 11, 31, 0.08);
            height: 100%;
        }

        .panel h2 {
            font-size: 1.05rem;
            margin: 0 0 0.35rem;
            color: var(--wine-dark);
        }

        .panel p {
            color: var(--muted);
            margin-top: 0;
            font-size: 0.94rem;
        }

        div[data-testid="stSlider"] {
            padding: 0.25rem 0 0.65rem;
        }

        div[data-testid="stSlider"] label {
            color: var(--ink);
            font-weight: 650;
        }

        .stButton > button {
            width: 100%;
            border: 0;
            border-radius: 8px;
            background: linear-gradient(135deg, var(--wine), #b42355);
            color: white;
            font-weight: 800;
            padding: 0.8rem 1rem;
            box-shadow: 0 14px 30px rgba(127, 18, 52, 0.22);
        }

        .stButton > button:hover {
            color: white;
            background: linear-gradient(135deg, #681029, #a91f4d);
        }

        div[data-testid="stMetric"] {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.76);
            padding: 0.9rem 1rem;
        }

        div[data-testid="stMetricLabel"] {
            color: var(--muted);
        }

        div[data-testid="stMetricLabel"] p {
            color: var(--muted);
        }

        div[data-testid="stMetricValue"] {
            color: var(--wine-dark);
        }

        div[data-testid="stMetricValue"] div {
            color: var(--wine-dark);
        }

        .result {
            border: 1px solid rgba(127, 18, 52, 0.2);
            border-radius: 8px;
            padding: 1rem 1.1rem;
            background: var(--rose);
            color: var(--wine-dark);
            font-weight: 800;
            font-size: 1.15rem;
            margin-top: 1rem;
        }

        .note {
            color: var(--muted);
            font-size: 0.88rem;
            margin-top: 0.75rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


st.markdown(
    """
    <section class="hero">
        <div class="eyebrow">Chemical profile analysis</div>
        <h1>Wine Quality Predictor</h1>
        <p>Fine-tune the main chemistry signals and estimate the quality score with a trained machine learning model.</p>
    </section>
    """,
    unsafe_allow_html=True,
)


left_col, right_col = st.columns([1.25, 0.85], gap="large")

with left_col:
    st.markdown(
        """
        <div class="panel">
            <h2>Chemical Inputs</h2>
            <p>Adjust the values below to model how the wine profile affects predicted quality.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    alcohol = st.slider("Alcohol Percentage (%)", min_value=8.0, max_value=15.0, value=10.5, step=0.1)
    volatile_acidity = st.slider(
        "Volatile Acidity (g/dm3 of acetic acid)",
        min_value=0.1,
        max_value=1.6,
        value=0.5,
        step=0.01,
    )
    sulphates = st.slider(
        "Sulphates (g/dm3 potassium sulphate)",
        min_value=0.3,
        max_value=2.0,
        value=0.6,
        step=0.01,
    )
    ph = st.slider("pH Level", min_value=2.7, max_value=4.0, value=3.3, step=0.01)

with right_col:
    st.markdown(
        """
        <div class="panel">
            <h2>Current Profile</h2>
            <p>A quick snapshot of the selected chemical balance.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_col_1, metric_col_2 = st.columns(2)
    metric_col_1.metric("Alcohol", f"{alcohol:.1f}%")
    metric_col_2.metric("pH", f"{ph:.2f}")
    metric_col_1.metric("Volatile acidity", f"{volatile_acidity:.2f}")
    metric_col_2.metric("Sulphates", f"{sulphates:.2f}")

    predict = st.button("Predict Quality Score", type="primary")
    st.markdown(
        '<div class="note">Unadjusted features use typical baseline values from the training profile.</div>',
        unsafe_allow_html=True,
    )


input_data = {
    "fixed acidity": 8.3,
    "volatile acidity": volatile_acidity,
    "citric acid": 0.27,
    "residual sugar": 2.5,
    "chlorides": 0.08,
    "free sulfur dioxide": 15.0,
    "total sulfur dioxide": 46.0,
    "density": 0.996,
    "pH": ph,
    "sulphates": sulphates,
    "alcohol": alcohol,
}

features_df = pd.DataFrame([input_data])

if predict:
    scaled_features = scaler.transform(features_df)
    prediction = model.predict(scaled_features)[0]
    st.markdown(
        f'<div class="result">Predicted Quality Score: {prediction:.2f} / 10</div>',
        unsafe_allow_html=True,
    )
