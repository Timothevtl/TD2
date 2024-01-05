import os
import random
import streamlit as st
import nltk
from nltk.corpus import sentiwordnet as swn
from nltk import pos_tag, word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('sentiwordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('vader_lexicon')

def identify_adverbs(tagged_sentence):
    adverbs = [word for word, tag in tagged_sentence if tag.startswith('RB')]
    return adverbs

def get_sentiment_scores(adverb):
    synsets = list(swn.senti_synsets(adverb))
    if synsets:
        pos_score = sum([synset.pos_score() for synset in synsets]) / len(synsets)
        neg_score = sum([synset.neg_score() for synset in synsets]) / len(synsets)
        obj_score = sum([synset.obj_score() for synset in synsets]) / len(synsets)
        return pos_score, neg_score, obj_score
    else:
        return 0.0, 0.0, 1.0

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

# Placeholder for other functionalities
def other_functionality_page():
    st.title("Other Functionality")
    st.write("This page is under construction.")

def main():
    st.sidebar.title("Navigation")
    options = ["Movie Review Analysis", "Other Functionality"]
    selection = st.sidebar.selectbox("Choose a page", options)

    if selection == "Movie Review Analysis":
        movie_review_page()
    elif selection == "Other Functionality":
        other_functionality_page()

if __name__ == "__main__":
    main()
