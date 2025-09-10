# sentiment_analyzer.py
import pandas as pd
import numpy as np
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from bs4 import BeautifulSoup
import streamlit as st
from datetime import datetime, timedelta
import re

class SentimentAnalyzer:
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        
    def get_news_sentiment(self, company_name, days_back=7):
        """Get news sentiment for a company"""
        try:
            # Google News search
            query = f"{company_name} stock market news"
            news_data = self._scrape_google_news(query, days_back)
            
            if not news_data:
                return pd.DataFrame()
                
            # Analyze sentiment for each article
            sentiments = []
            for article in news_data:
                sentiment_scores = self._analyze_text_sentiment(article['title'] + " " + article['snippet'])
                sentiments.append({
                    'date': article['date'],
                    'title': article['title'],
                    'sentiment_score': sentiment_scores['compound'],
                    'positive': sentiment_scores['pos'],
                    'negative': sentiment_scores['neg'],
                    'neutral': sentiment_scores['neu']
                })
            
            return pd.DataFrame(sentiments)
            
        except Exception as e:
            st.warning(f"Could not fetch news sentiment: {str(e)}")
            return pd.DataFrame()
    
    def _scrape_google_news(self, query, days_back):
        """Scrape Google News for recent articles"""
        try:
            # This is a simplified version - in production, use proper news APIs
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            }
            
            # Mock news data for demonstration
            # In production, replace with actual Google News API or NewsAPI
            mock_news = [
                {
                    'date': datetime.now() - timedelta(days=1),
                    'title': 'Company reports strong quarterly earnings',
                    'snippet': 'The company exceeded expectations with robust growth in key segments'
                },
                {
                    'date': datetime.now() - timedelta(days=2),
                    'title': 'Market volatility affects stock prices',
                    'snippet': 'Economic uncertainties create challenges for investors'
                },
                {
                    'date': datetime.now() - timedelta(days=3),
                    'title': 'New product launch drives investor confidence',
                    'snippet': 'Innovation and market expansion boost company prospects'
                }
            ]
            
            return mock_news[:days_back]
            
        except:
            return []
    
    def _analyze_text_sentiment(self, text):
        """Analyze sentiment of text using VADER"""
        # Clean text
        text = re.sub(r'http\S+', '', text)  # Remove URLs
        text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
        
        # VADER sentiment analysis
        vader_scores = self.vader_analyzer.polarity_scores(text)
        
        return vader_scores
    
    def get_social_sentiment(self, stock_symbol):
        """Get social media sentiment (simplified mock implementation)"""
        # In production, integrate with Twitter API or Reddit API
        # This is a mock implementation for demonstration
        mock_social_data = {
            'sentiment_score': np.random.uniform(-0.5, 0.5),
            'volume': np.random.randint(100, 1000),
            'bullish_percentage': np.random.uniform(40, 80),
            'bearish_percentage': np.random.uniform(20, 60)
        }
        
        return mock_social_data
    
    def aggregate_daily_sentiment(self, sentiment_df):
        """Aggregate sentiment scores by day"""
        if sentiment_df.empty:
            return pd.DataFrame()
            
        daily_sentiment = sentiment_df.groupby('date').agg({
            'sentiment_score': ['mean', 'std', 'count'],
            'positive': 'mean',
            'negative': 'mean',
            'neutral': 'mean'
        }).reset_index()
        
        # Flatten column names
        daily_sentiment.columns = [
            'date', 'avg_sentiment', 'sentiment_volatility', 'news_count',
            'avg_positive', 'avg_negative', 'avg_neutral'
        ]
        
        return daily_sentiment
