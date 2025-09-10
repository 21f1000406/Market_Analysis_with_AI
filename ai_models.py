# ai_models.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import streamlit as st

class StockPredictor:
    def __init__(self, prediction_days=60):
        self.prediction_days = prediction_days
        self.scaler = MinMaxScaler()
        self.model = None
        self.model_type = None
        
    def prepare_data(self, data, target_column='Close'):
        """Prepare data for training"""
        # Scale the data
        scaled_data = self.scaler.fit_transform(data[[target_column]])
        
        # Create sequences
        x_train, y_train = [], []
        
        for i in range(self.prediction_days, len(scaled_data)):
            x_train.append(scaled_data[i-self.prediction_days:i, 0])
            y_train.append(scaled_data[i, 0])
            
        return np.array(x_train), np.array(y_train)
    
    def create_lstm_model(self, input_shape):
        """Create LSTM model"""
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def train_lstm(self, data, epochs=50):
        """Train LSTM model"""
        x_train, y_train = self.prepare_data(data)
        
        if len(x_train) == 0:
            st.error("Insufficient data for training")
            return False
            
        # Reshape for LSTM
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
        
        # Create and train model
        self.model = self.create_lstm_model((x_train.shape[1], 1))
        
        with st.spinner('Training LSTM model...'):
            history = self.model.fit(
                x_train, y_train,
                epochs=epochs,
                batch_size=32,
                verbose=0,
                validation_split=0.2
            )
        
        self.model_type = 'LSTM'
        return True
    
    def train_random_forest(self, data):
        """Train Random Forest model"""
        # Prepare features with technical indicators
        features = ['Open', 'High', 'Low', 'Volume', 'SMA_20', 'RSI', 'MACD']
        
        # Remove rows with NaN values
        clean_data = data.dropna()
        
        if len(clean_data) < 100:
            st.error("Insufficient clean data for training")
            return False
            
        X = clean_data[features].values
        y = clean_data['Close'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        with st.spinner('Training Random Forest model...'):
            self.model.fit(X_scaled, y)
            
        self.model_type = 'RandomForest'
        return True
    
    def predict(self, data, future_days=5):
        """Make predictions"""
        if self.model is None:
            return None, None
            
        if self.model_type == 'LSTM':
            return self._predict_lstm(data, future_days)
        elif self.model_type == 'RandomForest':
            return self._predict_rf(data, future_days)
    
    def _predict_lstm(self, data, future_days):
        """LSTM predictions"""
        # Get last prediction_days of data
        last_data = data['Close'].values[-self.prediction_days:]
        scaled_data = self.scaler.transform(last_data.reshape(-1, 1))
        
        predictions = []
        current_batch = scaled_data.reshape(1, self.prediction_days, 1)
        
        for i in range(future_days):
            pred = self.model.predict(current_batch)[0]
            predictions.append(pred[0])
            
            # Update batch for next prediction
            current_batch = np.append(current_batch[:, 1:, :], 
                                    [[pred]], axis=1)
        
        # Inverse transform predictions
        predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        
        return predictions.flatten(), None
    
    def _predict_rf(self, data, future_days):
        """Random Forest predictions"""
        features = ['Open', 'High', 'Low', 'Volume', 'SMA_20', 'RSI', 'MACD']
        
        # Get latest features
        latest_features = data[features].iloc[-1:].values
        latest_scaled = self.scaler.transform(latest_features)
        
        predictions = []
        for i in range(future_days):
            pred = self.model.predict(latest_scaled)[0]
            predictions.append(pred)
        
        return predictions, None


# Add this to your existing ai_models.py

class EnhancedStockPredictor(StockPredictor):
    def __init__(self, prediction_days=60):
        super().__init__(prediction_days)
        self.sentiment_analyzer = SentimentAnalyzer()
        
    def prepare_data_with_sentiment(self, stock_data, sentiment_data):
        """Prepare data including sentiment features"""
        # Merge stock data with sentiment
        if not sentiment_data.empty:
            # Convert dates for merging
            sentiment_data['date'] = pd.to_datetime(sentiment_data['date']).dt.date
            stock_data_reset = stock_data.reset_index()
            stock_data_reset['Date'] = stock_data_reset['Date'].dt.date
            
            # Merge datasets
            merged_data = stock_data_reset.merge(
                sentiment_data, 
                left_on='Date', 
                right_on='date', 
                how='left'
            )
            
            # Fill missing sentiment values
            merged_data[['avg_sentiment', 'sentiment_volatility', 'news_count']] = \
                merged_data[['avg_sentiment', 'sentiment_volatility', 'news_count']].fillna(0)
                
            return merged_data.set_index('Date')
        else:
            # Add dummy sentiment columns if no data available
            stock_data['avg_sentiment'] = 0
            stock_data['sentiment_volatility'] = 0
            stock_data['news_count'] = 0
            return stock_data
    
    def train_enhanced_model(self, stock_data, sentiment_data, model_type='RandomForest'):
        """Train model with sentiment features"""
        # Prepare enhanced dataset
        enhanced_data = self.prepare_data_with_sentiment(stock_data, sentiment_data)
        
        # Enhanced features including sentiment
        features = [
            'Open', 'High', 'Low', 'Volume', 'SMA_20', 'RSI', 'MACD',
            'avg_sentiment', 'sentiment_volatility', 'news_count'
        ]
        
        # Remove rows with NaN values
        clean_data = enhanced_data.dropna()
        
        if len(clean_data) < 100:
            st.error("Insufficient data for training enhanced model")
            return False
        
        X = clean_data[features].values
        y = clean_data['Close'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        if model_type == 'RandomForest':
            from sklearn.ensemble import RandomForestRegressor
            self.model = RandomForestRegressor(n_estimators=150, random_state=42)
        else:
            from sklearn.ensemble import GradientBoostingRegressor
            self.model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        
        with st.spinner(f'Training enhanced {model_type} model with sentiment...'):
            self.model.fit(X_scaled, y)
        
        self.model_type = f'Enhanced_{model_type}'
        
        # Get feature importance
        self.feature_importance = dict(zip(features, self.model.feature_importances_))
        
        return True
    
    def get_feature_importance(self):
        """Get feature importance including sentiment features"""
        if hasattr(self, 'feature_importance'):
            return self.feature_importance
        return {}
