import os
import random
import streamlit as st
import sys
import nltk
import requests
import pickle
import sklearn
from io import BytesIO
from nltk.corpus import sentiwordnet as swn
from nltk import pos_tag, word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('sentiwordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

def loadCNN():
    # Raw URLs for the files on GitHub
    articles_url = 'https://raw.githubusercontent.com/Timothevtl/TD2/main/Desktop/data/CNNArticles'
    abstracts_url = 'https://raw.githubusercontent.com/Timothevtl/TD2/main/Desktop/data/CNNGold'

    # Download and load articles
    response = requests.get(articles_url)
    articles = pickle.load(BytesIO(response.content))

    # Clean articles
    articlesCl = []  
    for article in articles:
        articlesCl.append(article.replace("”", "").rstrip("\n"))
    articles = articlesCl

    # Download and load abstracts
    response = requests.get(abstracts_url)
    abstracts = pickle.load(BytesIO(response.content))

    # Clean abstracts
    articlesCl = []  
    for article in abstracts:
        articlesCl.append(article.replace("”", "").rstrip("\n"))
    abstracts = articlesCl

    return articles, abstracts

# Load the data
articles, abstracts = loadCNN()

def classify_review(review):
    scores = sia.polarity_scores(review)
    compound_score = scores['compound']

    # Use compound score for overall sentiment
    if compound_score >= 0.05:
        return "Good", compound_score
    elif compound_score <= -0.05:
        return "Bad", compound_score
    else:
        return "Neutral", compound_score

def movie_review_page():
    st.title("Movie Review Sentiment Analysis")
    user_input = st.text_area("Enter your movie review here:")
    if st.button("Analyze"):
        if user_input:
            sentiment, score = classify_review(user_input)
            st.write(f"Sentiment: {sentiment}")
            st.write("Sentiment Score:", score)
            # Visual indicator
            if sentiment == "Good":
                st.progress(min(score, 1.0))
            elif sentiment == "Bad":
                st.progress(-min(score, 1.0))
            else:
                st.progress(0.5)
        else:
            st.write("Please enter a movie review.")

def information_retrieval_page():
    st.markdown("<h1 style='text-align: center;'>CNN Information Retrieval System</h1>", unsafe_allow_html=True)

    # Create TF-IDF model
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(articles)

    user_query = st.text_area("Enter a summary to search for related document:", height=100)
    if st.button("Search"):
        if user_query:
            query_vec = vectorizer.transform([user_query])
            cosine_similarities = linear_kernel(query_vec, tfidf_matrix).flatten()

            top_index = cosine_similarities.argsort()[-1]
            similarity_score = cosine_similarities[top_index]
            
            st.markdown(f"<p style='font-weight:bold;'>Top matching document ID: {top_index}\nSimilarity Score: {similarity_score:.4f}</p>", unsafe_allow_html=True)
            with st.expander("Show Top Document Text"):
                st.text(articles[top_index])
            st.markdown("<h2 style='color: green;'>Summary of the Top Matching Document:</h2>", unsafe_allow_html=True)
            st.text(abstracts[top_index])
        else:
            st.write("Please enter a summary.")





def main():
    st.sidebar.title("Navigation")
    options = ["Movie Review Analysis", "Information Retrieval System"]
    selection = st.sidebar.selectbox("Choose a page", options)

    if selection == "Movie Review Analysis":
        movie_review_page()
    if selection == "Information Retrieval System":
        information_retrieval_page()

if __name__ == "__main__":
    main()
