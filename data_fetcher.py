# data_fetcher.py
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import ta
import streamlit as st

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_stock_data(symbol, period="2y"):
    """Fetch stock data from Yahoo Finance"""
    try:
        stock = yf.Ticker(symbol)
        data = stock.history(period=period)
        
        if data.empty:
            st.error(f"No data found for symbol {symbol}")
            return None
            
        return data
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def add_technical_indicators(data):
    """Add technical indicators to the dataframe"""
    # Moving Averages
    data['SMA_20'] = ta.trend.sma_indicator(data['Close'], window=20)
    data['EMA_20'] = ta.trend.ema_indicator(data['Close'], window=20)
    data['SMA_50'] = ta.trend.sma_indicator(data['Close'], window=50)
    
    # RSI
    data['RSI'] = ta.momentum.rsi(data['Close'], window=14)
    
    # MACD
    macd = ta.trend.MACD(data['Close'])
    data['MACD'] = macd.macd()
    data['MACD_signal'] = macd.macd_signal()
    data['MACD_histogram'] = macd.macd_diff()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(data['Close'])
    data['BB_upper'] = bollinger.bollinger_hband()
    data['BB_lower'] = bollinger.bollinger_lband()
    data['BB_middle'] = bollinger.bollinger_mavg()
    
    return data

def get_stock_info(symbol):
    """Get basic stock information"""
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        return {
            'name': info.get('longName', 'N/A'),
            'sector': info.get('sector', 'N/A'),
            'market_cap': info.get('marketCap', 'N/A'),
            'pe_ratio': info.get('trailingPE', 'N/A'),
            'dividend_yield': info.get('dividendYield', 'N/A')
        }
    except:
        return {}
