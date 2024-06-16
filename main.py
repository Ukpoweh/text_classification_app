#importing packages
import streamlit as st
import numpy as np
import pandas as pd
import pickle
import time
import keras
import json
import re
import string
import nltk
from nltk.corpus import stopwords
nltk.download('punkt')
nltk.download('stopwords')
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, model_from_json
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout, LSTM
from keras.utils import to_categorical

#setting page layout
st.set_page_config(
    page_title="Text Classifier",
    layout="centered",
)

#loading models
maxlen=100
max_words = 10000

with open('convet_model.json', 'r') as json_file:
    convet_model_json = json_file.read()

convet_model = model_from_json(convet_model_json)

with open('rnn_model.json', 'r') as json_file:
    rnn_model_json = json_file.read()

rnn_model = model_from_json(rnn_model_json)

tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))

convet_model.load_weights('convet_model_weights.h5')
rnn_model.load_weights('rnn_model_weights.h5')

#main function
def main():
    st.title("Gender Based Violence Classification App")

    st.write("""
                Gender-based violence, or GBV, is an ongoing and ever-resent scourge around the world, and is particularly prevalent in developing and least-developed countries. Gender-based violence also increased in many parts of the world during the COVID-19 pandemic.

One of the greatest challenges in combating GBV is the ‘culture of silence’, where victims of violence are scared, ashamed or intimidated to discuss their experiences with others and often do not report their experiences to authorities.

Another challenge faced by victims is achieving justice for their abusers. Some may not be aware of support systems, or not know where and how to report the perpetrators.

Victims may find and safety sharing their experiences online (as evidenced by the #MeToo movement), allowing them to get more support in an anonymous and/or safe way.

This app classifies texts about GBV into one of five categories: sexual violence, emotional violence, harmful traditional practices, physical violence and economic violence.
    
To use the app, you can either manually enter the text or upload the document containing the text;
             """)

    st.write("Do you want to enter the text manually or upload a file?")
    def user_input():
        #for user input
        input = ""

        if st.checkbox("Enter text manually"):
            text = st.text_input("Enter the text you want to classify")
            input = text

        elif st.checkbox("Upload a file"):

            uploaded_file = st.file_uploader("Choose a file", type=['docx', 'pdf', 'txt'])

            if uploaded_file is not None:
                try:
                    text = uploaded_file.getvalue().decode("utf-8")
                    input = text
                    st.success("File successfully uploaded!")
                except:
                    st.warning("The document you are trying to upload is not pure text")
        else:
            st.write("Choose one of the options")
        return input



    def clean_text(text):
        # Remove punctuation and numbers
        text = re.sub(r'[0-9{}]+'.format(re.escape(string.punctuation)), '', text)
        # Convert text to lowercase
        text = text.lower()
        # Tokenization (using nltk)
        tokens = nltk.word_tokenize(text)
        # Remove stopwords
        tokens = [word for word in tokens if word not in stopwords.words('english')]
        # Join the tokens back into a cleaned text
        cleaned_text = ' '.join(tokens)
        return cleaned_text
    
    def preprocess_text(input_text):
        # Tokenize the text
        test_sequences = tokenizer.texts_to_sequences([input_text])
        return test_sequences
    
    def predict(model, text):
        pred = model.predict(text)
        prob_list = []
        for prob in pred[0]:
            prob_list.append(prob * 100)
        results_df = pd.DataFrame({"Category": ["Harmful Traditional Practice", "Physical Violence", 
        "Economic Violence", "Emotional violence", "Sexual violence"], "Probability (%)": prob_list})

        return results_df

    input = user_input()
    try:
        cleaned_text = clean_text(input)
        test_sequences = preprocess_text(cleaned_text)
        test_data = pad_sequences(test_sequences, maxlen=maxlen)
        prediction_convet = predict(convet_model, test_data)
        prediction_rnn = predict(rnn_model, test_data)
        if st.button("Use the convet model"):
            with st.spinner("Fetching classifications... "):
                time.sleep(5)
            st.success("Here's the breakdown of the claasification of the text")
            st.dataframe(prediction_convet, hide_index=True, use_container_width=True)
        if st.button("Use the rnn model"):
            with st.spinner("Fetching classifications... "):
                time.sleep(5)
            st.success("Here's the breakdown of the claasification of the text")
            st.dataframe(prediction_rnn, hide_index=True, use_container_width=True)
    except:
        st.warning("You have not entered any input")


    

if __name__ == "__main__":
    main()





