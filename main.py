# main.py - Updated with Sentiment Analysis Integration
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from config import *
from data_fetcher import fetch_stock_data, add_technical_indicators, get_stock_info
from ai_models import StockPredictor
from sentiment_analyzer import SentimentAnalyzer

# Page configuration
st.set_page_config(**PAGE_CONFIG)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .sentiment-positive {
        color: #28a745;
        font-weight: bold;
    }
    .sentiment-negative {
        color: #dc3545;
        font-weight: bold;
    }
    .sentiment-neutral {
        color: #6c757d;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<h1 class="main-header">📈 AI Market Trend Analysis</h1>', unsafe_allow_html=True)
st.markdown(f"<h3 style='text-align: center; color: #666;'>Analyzing: {STOCK_NAME} ({STOCK_SYMBOL})</h3>", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.header("🔧 Configuration Panel")

# Stock selection (for future expansion)
selected_stock = st.sidebar.selectbox(
    "Select Stock",
    [STOCK_SYMBOL],  # Currently only one stock, can be expanded
    index=0
)

# Model selection
model_type = st.sidebar.selectbox(
    "Select AI Model",
    ["LSTM", "Random Forest", "Enhanced Random Forest (with Sentiment)"],
    index=2
)

# Prediction parameters
future_days = st.sidebar.slider(
    "Prediction Days",
    min_value=1,
    max_value=30,
    value=5,
    help="Number of days to predict into the future"
)

prediction_days = st.sidebar.slider(
    "Historical Window",
    min_value=30,
    max_value=120,
    value=60,
    help="Number of historical days to use for prediction"
)

# Data period
data_period = st.sidebar.selectbox(
    "Historical Data Period",
    ["6mo", "1y", "2y", "5y"],
    index=2,
    help="Amount of historical data to load"
)

# Advanced settings
with st.sidebar.expander("⚙️ Advanced Settings"):
    show_technical_details = st.checkbox("Show Technical Details", value=True)
    show_sentiment_analysis = st.checkbox("Show Sentiment Analysis", value=True)
    enable_alerts = st.checkbox("Enable Price Alerts", value=False)
    
    if enable_alerts:
        alert_threshold = st.slider("Alert Threshold (%)", 1, 10, 5)

# Main content with tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🤖 AI Predictions", "📰 Sentiment Analysis", "📈 Technical Analysis"])

# Load and process data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_and_process_data(symbol, period):
    """Load and process stock data"""
    data = fetch_stock_data(symbol, period)
    if data is not None:
        data = add_technical_indicators(data)
    return data

# Initialize session state for predictions
if 'predictions_made' not in st.session_state:
    st.session_state.predictions_made = False
    st.session_state.last_predictions = None
    st.session_state.model_performance = None

# Load data
with st.spinner('🔄 Loading stock data...'):
    stock_data = load_and_process_data(selected_stock, data_period)

if stock_data is not None:
    # Get stock info
    stock_info = get_stock_info(selected_stock)
    
    # Tab 1: Overview
    with tab1:
        st.subheader("📊 Market Overview")
        
        # Key metrics
        current_price = stock_data['Close'][-1]
        previous_price = stock_data['Close'][-2]
        daily_change = current_price - previous_price
        daily_change_pct = (daily_change / previous_price) * 100
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Current Price", 
                f"₹{current_price:.2f}",
                f"{daily_change:.2f} ({daily_change_pct:.2f}%)"
            )
        with col2:
            volume_change = ((stock_data['Volume'][-1] - stock_data['Volume'][-2]) / stock_data['Volume'][-2]) * 100
            st.metric(
                "Volume", 
                f"{stock_data['Volume'][-1]:,.0f}",
                f"{volume_change:.1f}%"
            )
        with col3:
            st.metric("52W High", f"₹{stock_data['High'].max():.2f}")
        with col4:
            st.metric("52W Low", f"₹{stock_data['Low'].min():.2f}")
        with col5:
            volatility = stock_data['Close'].pct_change().std() * np.sqrt(252) * 100
            st.metric("Volatility (Annual)", f"{volatility:.1f}%")
        
        # Company info
        if stock_info:
            st.subheader("🏢 Company Information")
            info_col1, info_col2, info_col3 = st.columns(3)
            
            with info_col1:
                st.write(f"**Sector:** {stock_info.get('sector', 'N/A')}")
                st.write(f"**Market Cap:** {stock_info.get('market_cap', 'N/A')}")
            with info_col2:
                st.write(f"**P/E Ratio:** {stock_info.get('pe_ratio', 'N/A')}")
                st.write(f"**Dividend Yield:** {stock_info.get('dividend_yield', 'N/A')}")
            with info_col3:
                # Risk assessment
                if volatility < 20:
                    risk_level = "🟢 Low Risk"
                elif volatility < 40:
                    risk_level = "🟡 Medium Risk"
                else:
                    risk_level = "🔴 High Risk"
                st.write(f"**Risk Level:** {risk_level}")
        
        # Price chart with volume
        fig_overview = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Price Movement', 'Volume'),
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3]
        )
        
        # Candlestick chart
        fig_overview.add_trace(
            go.Candlestick(
                x=stock_data.index[-100:],  # Last 100 days
                open=stock_data['Open'][-100:],
                high=stock_data['High'][-100:],
                low=stock_data['Low'][-100:],
                close=stock_data['Close'][-100:],
                name="Price",
                increasing_line_color='green',
                decreasing_line_color='red'
            ), row=1, col=1
        )
        
        # Volume
        colors = ['green' if close >= open else 'red' 
                 for close, open in zip(stock_data['Close'][-100:], stock_data['Open'][-100:])]
        
        fig_overview.add_trace(
            go.Bar(
                x=stock_data.index[-100:],
                y=stock_data['Volume'][-100:],
                name='Volume',
                marker_color=colors,
                opacity=0.6
            ), row=2, col=1
        )
        
        fig_overview.update_layout(
            title="Recent Price and Volume Trend",
            xaxis_rangeslider_visible=False,
            height=600,
            showlegend=False
        )
        
        st.plotly_chart(fig_overview, use_container_width=True)
    
    # Tab 2: AI Predictions
    with tab2:
        st.subheader("🤖 AI-Powered Predictions")
        
        # Model training and prediction
        col1, col2 = st.columns([2, 1])
        
        with col1:
            if st.button("🚀 Generate Predictions", type="primary", use_container_width=True):
                # Initialize predictor
                if "Enhanced" in model_type:
                    # Enhanced model with sentiment
                    from sentiment_analyzer import EnhancedStockPredictor
                    predictor = EnhancedStockPredictor(prediction_days=prediction_days)
                    
                    # Get sentiment data
                    sentiment_analyzer = SentimentAnalyzer()
                    with st.spinner('Analyzing market sentiment...'):
                        news_sentiment = sentiment_analyzer.get_news_sentiment(STOCK_NAME, days_back=7)
                        daily_sentiment = sentiment_analyzer.aggregate_daily_sentiment(news_sentiment)
                    
                    # Train enhanced model
                    success = predictor.train_enhanced_model(stock_data, daily_sentiment, 'RandomForest')
                    model_name = "Enhanced Random Forest with Sentiment"
                    
                else:
                    # Standard models
                    predictor = StockPredictor(prediction_days=prediction_days)
                    
                    if model_type == "LSTM":
                        success = predictor.train_lstm(stock_data, epochs=30)
                    else:
                        success = predictor.train_random_forest(stock_data)
                    
                    model_name = model_type
                
                if success:
                    # Make predictions
                    predictions, confidence = predictor.predict(stock_data, future_days)
                    
                    if predictions is not None:
                        # Store predictions in session state
                        st.session_state.predictions_made = True
                        st.session_state.last_predictions = predictions
                        st.session_state.model_type = model_name
                        
                        # Create future dates (excluding weekends for stock market)
                        future_dates = []
                        current_date = stock_data.index[-1]
                        days_added = 0
                        
                        while days_added < future_days:
                            current_date += timedelta(days=1)
                            # Skip weekends
                            if current_date.weekday() < 5:  # Monday=0, Friday=4
                                future_dates.append(current_date)
                                days_added += 1
                        
                        st.success(f"✅ {model_name} predictions generated successfully!")
                        
                        # Predictions visualization
                        fig_pred = go.Figure()
                        
                        # Historical prices
                        fig_pred.add_trace(
                            go.Scatter(
                                x=stock_data.index[-50:],
                                y=stock_data['Close'][-50:],
                                mode='lines',
                                name='Historical Price',
                                line=dict(color='blue', width=2)
                            )
                        )
                        
                        # Predictions
                        fig_pred.add_trace(
                            go.Scatter(
                                x=future_dates,
                                y=predictions,
                                mode='lines+markers',
                                name='Predictions',
                                line=dict(color='red', width=3, dash='dot'),
                                marker=dict(size=8, color='red')
                            )
                        )
                        
                        # Connect last historical point to first prediction
                        fig_pred.add_trace(
                            go.Scatter(
                                x=[stock_data.index[-1], future_dates[0]],
                                y=[stock_data['Close'][-1], predictions[0]],
                                mode='lines',
                                name='Transition',
                                line=dict(color='orange', width=2, dash='dash'),
                                showlegend=False
                            )
                        )
                        
                        fig_pred.update_layout(
                            title=f"Stock Price Predictions - {model_name}",
                            xaxis_title="Date",
                            yaxis_title="Price (₹)",
                            height=500,
                            hovermode='x unified'
                        )
                        
                        st.plotly_chart(fig_pred, use_container_width=True)
                        
                        # Predictions table
                        st.subheader("📋 Detailed Predictions")
                        pred_df = pd.DataFrame({
                            'Date': future_dates,
                            'Predicted Price': [f"₹{p:.2f}" for p in predictions],
                            'Change from Current': [f"{((p - current_price) / current_price * 100):+.2f}%" for p in predictions],
                            'Model': [model_name] * len(predictions)
                        })
                        st.dataframe(pred_df, use_container_width=True)
                        
                        # Analysis summary
                        st.subheader("🎯 Prediction Analysis")
                        avg_prediction = np.mean(predictions)
                        trend_direction = "📈 Bullish" if avg_prediction > current_price else "📉 Bearish"
                        confidence_level = "High" if abs(avg_prediction - current_price) / current_price > 0.05 else "Moderate"
                        
                        analysis_col1, analysis_col2, analysis_col3, analysis_col4 = st.columns(4)
                        
                        with analysis_col1:
                            st.metric("Trend Direction", trend_direction)
                        with analysis_col2:
                            st.metric("Avg Predicted Price", f"₹{avg_prediction:.2f}")
                        with analysis_col3:
                            expected_change = ((avg_prediction - current_price) / current_price) * 100
                            st.metric("Expected Change", f"{expected_change:+.2f}%")
                        with analysis_col4:
                            st.metric("Confidence Level", confidence_level)
                        
                        # Feature importance (for enhanced models)
                        if hasattr(predictor, 'get_feature_importance'):
                            feature_importance = predictor.get_feature_importance()
                            if feature_importance:
                                st.subheader("🎯 Feature Importance")
                                importance_df = pd.DataFrame(
                                    list(feature_importance.items()), 
                                    columns=['Feature', 'Importance']
                                ).sort_values('Importance', ascending=True)
                                
                                fig_importance = go.Figure(go.Bar(
                                    x=importance_df['Importance'],
                                    y=importance_df['Feature'],
                                    orientation='h',
                                    marker_color='lightblue'
                                ))
                                
                                fig_importance.update_layout(
                                    title="Model Feature Importance",
                                    xaxis_title="Importance Score",
                                    height=400
                                )
                                
                                st.plotly_chart(fig_importance, use_container_width=True)
                
        with col2:
            st.info("**Model Information**")
            if model_type == "LSTM":
                st.write("🧠 **LSTM Neural Network**")
                st.write("- Deep learning model")
                st.write("- Good for sequential data")
                st.write("- Captures long-term patterns")
            elif model_type == "Random Forest":
                st.write("🌳 **Random Forest**")
                st.write("- Ensemble method")
                st.write("- Uses multiple decision trees")
                st.write("- Good feature importance")
            else:
                st.write("🚀 **Enhanced Random Forest**")
                st.write("- Includes sentiment analysis")
                st.write("- Market news integration")
                st.write("- Higher accuracy potential")
            
            # Display last prediction summary if available
            if st.session_state.predictions_made:
                st.success("✅ Model Ready")
                st.write(f"**Last Model:** {st.session_state.get('model_type', 'Unknown')}")
                if st.session_state.last_predictions is not None:
                    next_day_pred = st.session_state.last_predictions[0]
                    change = ((next_day_pred - current_price) / current_price) * 100
                    st.write(f"**Next Day:** ₹{next_day_pred:.2f} ({change:+.1f}%)")
    
    # Tab 3: Sentiment Analysis
    with tab3:
        if show_sentiment_analysis:
            st.subheader("📰 Market Sentiment Analysis")
            
            # Initialize sentiment analyzer
            sentiment_analyzer = SentimentAnalyzer()
            
            # News sentiment section
            with st.expander("📈 News Sentiment Analysis", expanded=True):
                with st.spinner('Fetching and analyzing news sentiment...'):
                    news_sentiment = sentiment_analyzer.get_news_sentiment(STOCK_NAME, days_back=7)
                
                if not news_sentiment.empty:
                    # Sentiment overview
                    avg_sentiment = news_sentiment['sentiment_score'].mean()
                    
                    sentiment_col1, sentiment_col2, sentiment_col3, sentiment_col4 = st.columns(4)
                    
                    with sentiment_col1:
                        sentiment_label = "Positive" if avg_sentiment > 0.1 else "Negative" if avg_sentiment < -0.1 else "Neutral"
                        sentiment_color = "sentiment-positive" if avg_sentiment > 0.1 else "sentiment-negative" if avg_sentiment < -0.1 else "sentiment-neutral"
                        st.markdown(f'<p class="{sentiment_color}">Overall Sentiment: {sentiment_label}</p>', unsafe_allow_html=True)
                        st.metric("Sentiment Score", f"{avg_sentiment:.3f}")
                    
                    with sentiment_col2:
                        positive_news = len(news_sentiment[news_sentiment['sentiment_score'] > 0.1])
                        st.metric("Positive News", positive_news)
                    
                    with sentiment_col3:
                        negative_news = len(news_sentiment[news_sentiment['sentiment_score'] < -0.1])
                        st.metric("Negative News", negative_news)
                    
                    with sentiment_col4:
                        total_news = len(news_sentiment)
                        st.metric("Total Articles", total_news)
                    
                    # Sentiment trend chart
                    if len(news_sentiment) > 1:
                        fig_sentiment = go.Figure()
                        
                        fig_sentiment.add_trace(
                            go.Scatter(
                                x=news_sentiment['date'],
                                y=news_sentiment['sentiment_score'],
                                mode='lines+markers',
                                name='Sentiment Score',
                                line=dict(color='purple', width=2),
                                marker=dict(size=6)
                            )
                        )
                        
                        fig_sentiment.add_hline(y=0, line_dash="dash", line_color="gray")
                        fig_sentiment.add_hline(y=0.1, line_dash="dot", line_color="green", annotation_text="Positive Threshold")
                        fig_sentiment.add_hline(y=-0.1, line_dash="dot", line_color="red", annotation_text="Negative Threshold")
                        
                        fig_sentiment.update_layout(
                            title="News Sentiment Trend",
                            xaxis_title="Date",
                            yaxis_title="Sentiment Score",
                            height=400
                        )
                        
                        st.plotly_chart(fig_sentiment, use_container_width=True)
                    
                    # Recent news with sentiment
                    st.subheader("Recent News Headlines")
                    for idx, row in news_sentiment.head(5).iterrows():
                        sentiment_emoji = "🟢" if row['sentiment_score'] > 0.1 else "🔴" if row['sentiment_score'] < -0.1 else "🟡"
                        st.write(f"{sentiment_emoji} **{row['title']}** (Score: {row['sentiment_score']:.3f})")
                
                else:
                    st.warning("Unable to fetch news sentiment data. Using mock data for demonstration.")
            
            # Social media sentiment (mock data)
            with st.expander("📱 Social Media Sentiment", expanded=False):
                social_sentiment = sentiment_analyzer.get_social_sentiment(selected_stock)
                
                social_col1, social_col2, social_col3 = st.columns(3)
                
                with social_col1:
                    st.metric("Social Sentiment", f"{social_sentiment['sentiment_score']:.3f}")
                with social_col2:
                    st.metric("Discussion Volume", social_sentiment['volume'])
                with social_col3:
                    st.metric("Bullish %", f"{social_sentiment['bullish_percentage']:.1f}%")
                
                # Social sentiment gauge
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = social_sentiment['sentiment_score'] * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Social Media Sentiment"},
                    gauge = {
                        'axis': {'range': [-100, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [-100, -20], 'color': "lightcoral"},
                            {'range': [-20, 20], 'color': "lightgray"},
                            {'range': [20, 100], 'color': "lightgreen"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': social_sentiment['sentiment_score'] * 100
                        }
                    }
                ))
                
                fig_gauge.update_layout(height=300)
                st.plotly_chart(fig_gauge, use_container_width=True)
        
        else:
            st.info("Sentiment analysis is disabled. Enable it in the sidebar to see market sentiment insights.")
    
    # Tab 4: Technical Analysis
    with tab4:
        if show_technical_details:
            st.subheader("📈 Technical Analysis Dashboard")
            
            # Technical indicators overview
            latest_data = stock_data.iloc[-1]
            
            # RSI Analysis
            rsi_col1, rsi_col2 = st.columns(2)
            
            with rsi_col1:
                st.subheader("📊 RSI Analysis")
                rsi_value = latest_data['RSI']
                
                if rsi_value > 70:
                    rsi_signal = "🔴 Overbought - Consider Selling"
                    rsi_color = "red"
                elif rsi_value < 30:
                    rsi_signal = "🟢 Oversold - Consider Buying"
                    rsi_color = "green"
                else:
                    rsi_signal = "🟡 Neutral - Hold Position"
                    rsi_color = "orange"
                
                st.metric("Current RSI", f"{rsi_value:.2f}", rsi_signal)
                
                # RSI Chart
                fig_rsi = go.Figure()
                fig_rsi.add_trace(
                    go.Scatter(
                        x=stock_data.index[-50:],
                        y=stock_data['RSI'][-50:],
                        mode='lines',
                        name='RSI',
                        line=dict(color=rsi_color, width=2)
                    )
                )
                
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                fig_rsi.add_hline(y=50, line_dash="dot", line_color="gray", annotation_text="Neutral")
                
                fig_rsi.update_layout(
                    title="RSI Trend (50 days)",
                    yaxis_title="RSI Value",
                    height=300
                )
                
                st.plotly_chart(fig_rsi, use_container_width=True)
            
            with rsi_col2:
                st.subheader("📈 MACD Analysis")
                macd_value = latest_data['MACD']
                macd_signal = latest_data['MACD_signal']
                
                if macd_value > macd_signal:
                    macd_trend = "🟢 Bullish Signal"
                    macd_color = "green"
                else:
                    macd_trend = "🔴 Bearish Signal"
                    macd_color = "red"
                
                st.metric("MACD Signal", macd_trend)
                st.write(f"MACD: {macd_value:.4f}")
                st.write(f"Signal: {macd_signal:.4f}")
                
                # MACD Chart
                fig_macd = go.Figure()
                
                fig_macd.add_trace(
                    go.Scatter(
                        x=stock_data.index[-50:],
                        y=stock_data['MACD'][-50:],
                        mode='lines',
                        name='MACD',
                        line=dict(color='blue', width=2)
                    )
                )
                
                fig_macd.add_trace(
                    go.Scatter(
                        x=stock_data.index[-50:],
                        y=stock_data['MACD_signal'][-50:],
                        mode='lines',
                        name='Signal',
                        line=dict(color='red', width=2)
                    )
                )
                
                fig_macd.update_layout(
                    title="MACD Trend (50 days)",
                    yaxis_title="MACD Value",
                    height=300
                )
                
                st.plotly_chart(fig_macd, use_container_width=True)
            
            # Technical indicators summary table
            st.subheader("📋 Technical Indicators Summary")
            
            indicators_data = {
                'Indicator': ['RSI (14)', 'SMA (20)', 'SMA (50)', 'EMA (20)', 'MACD', 'Bollinger Upper', 'Bollinger Lower'],
                'Current Value': [
                    f"{latest_data['RSI']:.2f}",
                    f"₹{latest_data['SMA_20']:.2f}",
                    f"₹{latest_data['SMA_50']:.2f}",
                    f"₹{latest_data['EMA_20']:.2f}",
                    f"{latest_data['MACD']:.4f}",
                    f"₹{latest_data['BB_upper']:.2f}",
                    f"₹{latest_data['BB_lower']:.2f}"
                ],
                'Signal': [
                    "Overbought" if latest_data['RSI'] > 70 else "Oversold" if latest_data['RSI'] < 30 else "Neutral",
                    "Bullish" if current_price > latest_data['SMA_20'] else "Bearish",
                    "Bullish" if current_price > latest_data['SMA_50'] else "Bearish",
                    "Bullish" if current_price > latest_data['EMA_20'] else "Bearish",
                    "Bullish" if latest_data['MACD'] > latest_data['MACD_signal'] else "Bearish",
                    "Resistance" if current_price < latest_data['BB_upper'] else "Breakout",
                    "Support" if current_price > latest_data['BB_lower'] else "Breakdown"
                ],
                'Strength': [
                    "Strong" if abs(latest_data['RSI'] - 50) > 20 else "Moderate",
                    "Strong" if abs(current_price - latest_data['SMA_20'])/current_price > 0.02 else "Weak",
                    "Strong" if abs(current_price - latest_data['SMA_50'])/current_price > 0.05 else "Weak",
                    "Strong" if abs(current_price - latest_data['EMA_20'])/current_price > 0.02 else "Weak",
                    "Strong" if abs(latest_data['MACD'] - latest_data['MACD_signal']) > 0.5 else "Weak",
                    "Strong",
                    "Strong"
                ]
            }
            
            indicators_df = pd.DataFrame(indicators_data)
            st.dataframe(indicators_df, use_container_width=True)
            
            # Trading recommendations
            st.subheader("💡 Trading Recommendations")
            
            # Calculate overall signal
            bullish_signals = 0
            bearish_signals = 0
            
            for signal in indicators_data['Signal']:
                if 'Bullish' in signal or signal == 'Oversold':
                    bullish_signals += 1
                elif 'Bearish' in signal or signal == 'Overbought':
                    bearish_signals += 1
            
            if bullish_signals > bearish_signals:
                overall_recommendation = "🟢 **BUY** - Bullish signals dominate"
                rec_color = "green"
            elif bearish_signals > bullish_signals:
                overall_recommendation = "🔴 **SELL** - Bearish signals dominate"
                rec_color = "red"
            else:
                overall_recommendation = "🟡 **HOLD** - Mixed signals"
                rec_color = "orange"
            
            st.markdown(f'<h3 style="color: {rec_color};">{overall_recommendation}</h3>', unsafe_allow_html=True)
            
            # Recommendation details
            rec_col1, rec_col2, rec_col3 = st.columns(3)
            
            with rec_col1:
                st.metric("Bullish Signals", bullish_signals)
            with rec_col2:
                st.metric("Bearish Signals", bearish_signals)
            with rec_col3:
                st.metric("Neutral Signals", len(indicators_data['Signal']) - bullish_signals - bearish_signals)
        
        else:
            st.info("Technical analysis details are disabled. Enable them in the sidebar for detailed technical indicators.")

else:
    st.error("❌ Failed to load stock data. Please check your internet connection and try again.")
    st.info("**Troubleshooting Tips:**")
    st.write("1. Check your internet connection")
    st.write("2. Verify the stock symbol is correct")
    st.write("3. Try refreshing the page")
    st.write("4. Check if Yahoo Finance is accessible")

# Footer with disclaimer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p><strong>⚠️ Disclaimer:</strong> This application is for educational and research purposes only. 
    The predictions and analysis provided should not be considered as financial advice. 
    Always consult with a qualified financial advisor before making investment decisions.</p>
    <p><em>Developed with ❤️ using Streamlit • Data provided by Yahoo Finance</em></p>
</div>
""", unsafe_allow_html=True)

# Debug information (only in development)
if st.sidebar.checkbox("🐛 Debug Info", value=False):
    st.sidebar.subheader("Debug Information")
    st.sidebar.write(f"Stock Symbol: {selected_stock}")
    st.sidebar.write(f"Data Period: {data_period}")
    st.sidebar.write(f"Model Type: {model_type}")
    st.sidebar.write(f"Prediction Days: {future_days}")
    if stock_data is not None:
        st.sidebar.write(f"Data Points: {len(stock_data)}")
        st.sidebar.write(f"Date Range: {stock_data.index[0].strftime('%Y-%m-%d')} to {stock_data.index[-1].strftime('%Y-%m-%d')}")
