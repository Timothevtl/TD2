import os
import random
import streamlit as st
import nltk
from nltk.corpus import sentiwordnet as swn
from nltk import pos_tag, word_tokenize

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('sentiwordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

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
    tokens = word_tokenize(review)
    tagged_sentence = pos_tag(tokens)
    adverbs = identify_adverbs(tagged_sentence)
    total_pos, total_neg = 0.0, 0.0

    for adverb in adverbs:
        pos_score, neg_score, _ = get_sentiment_scores(adverb)
        total_pos += pos_score
        total_neg += neg_score

    threshold = 0.1
    if total_pos > total_neg + threshold:
        return "Good"
    else:
        return "Bad"

def analyze_movie_reviews(reviews):
    for review in reviews:
        classification = classify_review(review)
        print(f"Review: {review}")
        print(f"Classification: {classification}\n")

def movie_review_page():
    st.title("Movie Review Sentiment Analysis")
    user_input = st.text_area("Enter your movie review here:")
    if st.button("Analyze"):
        if user_input:
            sentiment = classify_review(user_input)
            if sentiment == "Good":
                st.write("Sentiment: Good")
            elif sentiment == "Bad":
                st.write("Sentiment: Bad")
            else:
                st.write("Sentiment: Hard to say... either good or bad")
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
