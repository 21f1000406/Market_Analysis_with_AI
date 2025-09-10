# config.py
import streamlit as st

# Stock configuration
STOCK_SYMBOL = "RELIANCE.NS"  # You can change this to any Indian stock
STOCK_NAME = "Reliance Industries"

# Model parameters
PREDICTION_DAYS = 60
FUTURE_DAYS = 5

# Streamlit page config
PAGE_CONFIG = {
    "page_title": "AI Market Trend Analysis",
    "page_icon": "📈",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Chart colors
COLORS = {
    "price": "#1f77b4",
    "prediction": "#ff7f0e",
    "confidence": "#2ca02c",
    "volume": "#d62728"
}
