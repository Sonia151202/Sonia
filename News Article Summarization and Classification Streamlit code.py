# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 15:38:40 2024

@author: hp
"""

import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import nltk
from nltk.corpus import stopwords
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import joblib
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Load pre-trained models and vectorizers
random_forest = joblib.load("random_forest_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Initialize the Sentiment Analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to scrape articles and create a dataframe
def scrape_articles():
    base_url = "https://www.thestar.com.my/news/latest/"
    responses = []
    headlines_list = []
    urls_list = []
    for page_num in range(1, 6):  # Scraping only the first 5 pages for demonstration
        url = f"{base_url}?pgno={page_num}#latest"
        response = requests.get(url)
        if response.status_code == 200:
            responses.append(response.content)

    for response in responses:
        soup = BeautifulSoup(response, "html.parser")
        headlines = soup.find_all("h2", class_="f18")
        for headline in headlines:
            headlines_list.append(headline.get_text(strip=True))
            link = headline.find("a")
            if link and link.get("href"):
                urls_list.append(link.get("href"))
            else:
                urls_list.append(None)

    # Create a DataFrame
    df = pd.DataFrame({
        "Headline": headlines_list,
        "URL": urls_list
    })
    return df
# Function to get article content
def get_article_content(url):
    try:
        article_response = requests.get(url)
        if article_response.status_code == 200:
            article_soup = BeautifulSoup(article_response.content, "html.parser")
            article_content = article_soup.find("div", class_="row content-holder story-wrapper")
            if article_content:
                paragraphs = article_content.find_all("p")
                article_text = " ".join([para.get_text(strip=True) for para in paragraphs])
                return article_text
            else:
                return None
        else:
            return None
    except Exception as e:
        return None
    # Function to clean and preprocess the article content
def clean_text(text):
    if not text:
        return None
    # Remove non-alphabetical characters and extra spaces
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text    
# Function to perform sentiment analysis using VADER
def get_sentiment(text):
    sentiment_score = analyzer.polarity_scores(text)
    compound_score = sentiment_score['compound']
    if compound_score > 0.05:
        return 'Positive'
    elif compound_score >= -0.05:
        return 'Neutral'
    else:
        return 'Negative'

# Function to summarize text using GPT-2
def summarize_with_gpt2(text):
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
    input_text = "Summarize: " + text
    inputs = gpt2_tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True, padding=True)
    summary_ids = gpt2_model.generate(inputs["input_ids"], attention_mask=inputs["attention_mask"], max_new_tokens=50, num_return_sequences=1, early_stopping=True)
    summary = gpt2_tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

# Streamlit UI
st.title("News Article Summarizer and Sentiment Analyzer")

# Button to scrape articles
if st.button("Scrape Latest Articles"):
    df = scrape_articles()
    st.write("Articles scraped successfully!")
    st.dataframe(df)

    # Add article content to DataFrame
    df["Article Content"] = df["URL"].apply(lambda url: get_article_content(url) if url else None)
    df["Cleaned Content"] = df["Article Content"].apply(clean_text)

    # Display content in the DataFrame
    st.write("Articles with cleaned content:")
    st.dataframe(df[['Headline', 'Cleaned Content']])

    # Summarize articles using GPT-2
    df['GPT-2 Summary'] = df['Cleaned Content'].apply(summarize_with_gpt2)
    st.write("Summarized Articles:")
    st.dataframe(df[['Headline', 'GPT-2 Summary']])

    # Sentiment Analysis
    df['Sentiment'] = df['Cleaned Content'].apply(get_sentiment)
    st.write("Sentiment Analysis Results:")
    st.dataframe(df[['Headline', 'Sentiment']])

    # Sentiment Classification using Random Forest
    X = vectorizer.transform(df['Cleaned Content'])
    df['Predicted Sentiment'] = random_forest.predict(X)
    st.write("Predicted Sentiments (Random Forest):")
    st.dataframe(df[['Headline', 'Predicted Sentiment']])

    # Displaying sentiment counts
    sentiment_counts = df['Predicted Sentiment'].value_counts()
    st.write("Sentiment Counts:")
    st.bar_chart(sentiment_counts)

# To run the app: 
# Streamlit will automatically recognize the script and run the app when the command below is executed in the terminal:
# streamlit run streamlit_app.py
