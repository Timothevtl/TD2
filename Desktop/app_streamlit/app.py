import os
import random
import streamlit as st
import nltk
from nltk.corpus import sentiwordnet as swn
from nltk import pos_tag, word_tokenize

# Import NLTK and download the "punkt" tokenizer resource
nltk.download('punkt')
nltk.download('sentiwordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')


# List all the files in the positive and negative directories
def file_declaration(): 
    positive_files = [os.path.join(positive_dir, filename) for filename in os.listdir(positive_dir)]
    negative_files = [os.path.join(negative_dir, filename) for filename in os.listdir(negative_dir)]
    return positive_files, negative_files

def identify_adverbs(tagged_sentence):
    adverbs = [word for word, tag in tagged_sentence if tag.startswith('RB')]
    return adverbs

# Function to get sentiment scores for a word
def get_sentiment_scores(adverb):
    synsets = list(swn.senti_synsets(adverb))
    if synsets:
        pos_score = sum([synset.pos_score() for synset in synsets]) / len(synsets)
        neg_score = sum([synset.neg_score() for synset in synsets]) / len(synsets)
        obj_score = sum([synset.obj_score() for synset in synsets]) / len(synsets)
        return pos_score, neg_score, obj_score
    else:
        return 0.0, 0.0, 1.0 

# Function to classify a review
def classify_review(review):
    tokens = word_tokenize(review)
    tagged_sentence = pos_tag(tokens)
    
    total_pos, total_neg, total_obj = 0.0, 0.0, 0.0

    for word, tag in tagged_sentence:
        if tag.startswith('RB') or tag.startswith('JJ') or tag.startswith('VB'):
            pos_score, neg_score, obj_score = get_sentiment_scores(word)
            total_pos += pos_score
            total_neg += neg_score
            total_obj += obj_score

    # Define a threshold to classify as positive, negative, or neutral
    threshold = 0.1
    if total_pos > total_neg + threshold:
        return "Positive"
    elif total_neg > total_pos + threshold:
        return "Negative"
    else:
        return "Neutral"
        
def analyze_movie_reviews(reviews):
    for review in reviews:
        classification = classify_review(review)
        print(f"Review: {review}")
        print(f"Classification: {classification}\n")
        

# Define the Streamlit app
def main():
    st.title("Movie Review Sentiment Analysis")
    user_input = st.text_area("Enter your movie review here:")
    # Button to analyze a random review
    if st.button("Analyze"):
        if user_input:
            sentiment = classify_review(user_input)
            if sentiment == "Positive":
                st.write("Sentiment: Positive")
            elif sentiment == "Negative":
                st.write("Sentiment: Negative")
            else:
                st.write("Sentiment: Neutral")
        else:
            st.write("Please enter a movie review.")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("An error occured, try another option")
