# Wine Quality Predictor

A Streamlit web app that predicts wine quality from key chemical profile inputs using a trained machine learning model.

Live app: [wine-quality-predictor12.streamlit.app](https://wine-quality-predictor12.streamlit.app/)

## Overview

This project provides an interactive interface for estimating wine quality. Users can adjust chemistry signals such as alcohol percentage, volatile acidity, sulphates, and pH, then generate a predicted quality score from a saved model.

## Features

- Interactive sliders for wine chemical properties
- Current profile summary cards
- Machine learning prediction using saved model artifacts
- Clean Streamlit UI designed for quick experimentation
- Ready for deployment on Streamlit Community Cloud

## Project Structure

```text
.
├── app.py              # Streamlit application
├── requirements.txt    # Python dependencies for deployment
├── scaler.pkl          # Saved feature scaler
├── wine_model.pkl      # Trained prediction model
└── week1.ipynb         # Notebook used during model development
```

## Tech Stack

- Python
- Streamlit
- Pandas
- Scikit-learn

## Run Locally

Clone the repository and install the dependencies:

```bash
pip install -r requirements.txt
```

Start the Streamlit app:

```bash
streamlit run app.py
```

## Deployment

The app is deployed on Streamlit Community Cloud.

For deployment, use:

```text
Main file path: app.py
```

## Model Inputs

The app exposes the following adjustable inputs:

- Alcohol percentage
- Volatile acidity
- Sulphates
- pH level

Other model features use typical baseline values from the training profile.
