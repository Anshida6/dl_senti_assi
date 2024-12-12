import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import os

# Load the trained mod
# Define paths for model and tokenizer files
model_path = 'sentiment_model.h5'  # Update with correct path to your model

# Check if the model file exists
if not os.path.exists(model_path):
    st.error(f"Model file not found at {model_path}")
else:
    # Load the trained model
    model = tf.keras.models.load_model(model_path)

    # Create a simple tokenizer if you don't have the original tokenizer
    # This is just an example. Replace it with your actual preprocessing logic.
    tokenizer = Tokenizer(num_words=10000)

# Define max length for padding (same as during training)
max_length = 100  # Set this to the max length used during model training

def predict_sentiment(sentence):
    """
    Predict the sentiment of the input sentence.
    """
    tokenizer.fit_on_texts([sentence])  # Fit tokenizer on the input (if you don't have a tokenizer)
    tokenized_sentence = tokenizer.texts_to_sequences([sentence])  # Tokenizing the sentence
    padded_sentence = pad_sequences(tokenized_sentence, maxlen=max_length)
    # Predict the sentiment class (0, 1, or 2)
    prediction = model.predict(padded_sentence)
    predicted_class = np.argmax(prediction, axis=1)  # Get the class with the highest probability

    # Map the predicted class to the sentiment label
    label_map = {0: 'Positive', 1: 'Neutral', 2: 'Negative'}
    return label_map[predicted_class[0]]

# Streamlit UI
st.title("Sentiment Analysis Web App")
st.write("Enter a sentence to predict the sentiment (Positive, Neutral, or Negative).")

# User input
sentence = st.text_input("Enter a sentence:")

if sentence:
    # Predict sentiment when the user submits the sentence
    predicted_sentiment = predict_sentiment(sentence)
    st.write(f"Predicted Sentiment: {predicted_sentiment}")
