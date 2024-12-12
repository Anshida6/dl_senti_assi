# -*- coding: utf-8 -*-
"""Untitled16.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1_Rsp1D4rzTE_aUvHoXDT_ZT8pzF-mXOC
"""

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Step 1: Load and Merge Datasets
print("----- Downloading and Merging Datasets -----")
url_set1 = '/content/drive/MyDrive/deep_learning/Set-I.csv'
url_set2 = '/content/drive/MyDrive/deep_learning/Set-II.csv'
set1 = pd.read_csv(url_set1)
set2 = pd.read_csv(url_set2)
dataset = pd.concat([set1, set2], ignore_index=True)
dataset.head(10)

# Step 2: Data Preprocessing
print("----- Preprocessing Data -----")

# Map sentiment labels to numerical values
dataset['label'] = dataset['label'].map({'Positive': 0, 'Neutral': 1, 'Negative': 2})  # Multi-class mapping

# Extract features (tweets) and labels
X = dataset['Tweets'].values  # Feature: Tweets
y = dataset['label'].values   # Target: Labels (0, 1, 2)


# Verify label conversion
print("Mapped labels:")
print(dataset['label'].values)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenization and Padding
print("----- Tokenizing and Padding -----")
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(X_train)
encoded_train = tokenizer.texts_to_sequences(X_train)
encoded_test = tokenizer.texts_to_sequences(X_test)
max_length = 40

padded_train = tf.keras.preprocessing.sequence.pad_sequences(encoded_train, maxlen=max_length, padding='post')
padded_test = tf.keras.preprocessing.sequence.pad_sequences(encoded_test, maxlen=max_length, padding='post')
print(padded_train[0:2])

vocab_size = len(tokenizer.word_index) + 1
print(vocab_size)

from tensorflow.keras.utils import to_categorical
y = to_categorical(y, num_classes=3)

# Step 3: Define the RNN Model for Multi-Class Classification
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(input_dim=vocab_size, output_dim=24, input_length=max_length),
    tf.keras.layers.SimpleRNN(24, return_sequences=False),  # RNN layer with 24 units
    tf.keras.layers.Dense(64, activation='relu'),           # Dense layer with 64 units
    tf.keras.layers.Dropout(0.7),                           # Dropout for regularization
    tf.keras.layers.Dense(32, activation='relu'),           # Dense layer with 32 units
    tf.keras.layers.Dropout(0.7),                           # Dropout for regularization
    tf.keras.layers.Dense(3, activation='softmax')          # Output layer for 3 classes
])

model.compile(optimizer="adam", loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()
early_stop = EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True)

history = model.fit(
    x=padded_train,                 # Preprocessed and padded input sequences
    y=y,                        # One-hot encoded labels
    epochs=100,                 # Number of epochs
    validation_split=0.2,       # Split 20% of data for validation
    callbacks=[early_stop],     # Early stopping
    batch_size=32,              # Batch size for training
    verbose=1                   # Display training progress
)

#Step 5: Save the Model
os.makedirs("models", exist_ok=True)
model.save("/content/models/sentiment_model.h5")

from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def predict_sentiment(sentence):
    # Preprocess the sentence (e.g., tokenization, padding)
    # For example, assuming you have a tokenizer and max_length:
    tokenized_sentence = tokenizer.texts_to_sequences([sentence])  # Tokenizing the sentence
    padded_sentence = pad_sequences(tokenized_sentence, maxlen=max_length)  # Padding the sentence

    # Predict the sentiment class (0, 1, or 2)
    prediction = model.predict(padded_sentence)
    predicted_class = np.argmax(prediction, axis=1)  # Get the class with the highest probability

    # Map the predicted class back to the original label
    label_map = {0: 'Positive', 1: 'Neutral', 2: 'Negative'}
    return label_map[predicted_class[0]]

# Example prediction:
sentence = "I love this product!"
predicted_sentiment = predict_sentiment(sentence)
print(f"Predicted sentiment: {predicted_sentiment}")