import streamlit as st
import numpy as np
import pandas as pd
import pickle
import time
import keras
import json
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential, model_from_json
from keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from keras.utils import to_categorical

st.set_page_config(
    page_title="Text Classifier",
    layout="centered",
)
maxlen=100
max_words = 10000

with open('model.json', 'r') as json_file:
    loaded_model_json = json_file.read()

model = model_from_json(loaded_model_json)

tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))


model.load_weights('model_weights.h5')

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

    input = []

    if st.checkbox("Enter text manually"):
        text = st.text_input("Enter the text you want to classify")
        input.append(text)

    elif st.checkbox("Upload a file"):

        uploaded_file = st.file_uploader("Choose a file", type=['docx', 'pdf', 'txt'])

        if uploaded_file is not None:
            st.success("File successfully uploaded!")
            text = uploaded_file.getvalue().decode("utf-8")
            input.append(text)


    
    def preprocess_text(input_text):
    # Tokenize the text

        test_sequences = tokenizer.texts_to_sequences(input_text)
        return test_sequences


    try:
        test_sequences = preprocess_text(input)
        test_data = pad_sequences(test_sequences, maxlen=maxlen)
        pred = model.predict(test_data)
        real_pred = np.argmax(pred, axis=1)[0]
        prediction = ""
        if real_pred == 0:
            prediction = 'Harmful Traditional Practice'
        elif  real_pred == 1:
            prediction = 'Physical Violence'
        elif real_pred == 2:
            prediction = 'Economic Violence'
        elif  real_pred == 3:
            prediction = 'Emotional Violence'
        elif real_pred == 4:
            prediction = 'Sexual Violence'
    except:
        st.write("Choose one of the options")


    if st.button("Classify the text"):
        with st.spinner("Fetching classifications... "):
            time.sleep(5)
        st.success(f"The predicted class is {prediction}")
         



if __name__ == "__main__":
    main()





