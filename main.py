import pickle
import numpy as np
import streamlit as st

# Load the models and vectorizers
with open('nb_model.pkl', 'rb') as f:
    nb_model = pickle.load(f)

with open('logistic_model.pkl', 'rb') as f:
    log_model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Load GloVe embeddings from the pickle file
with open('glove_embeddings.pkl', 'rb') as f:
    glove_embeddings = pickle.load(f)

# Convert text to GloVe embeddings
def text_to_glove_embeddings(text, glove_embeddings):
    words = text.split()
    embeddings = [glove_embeddings[word] if word in glove_embeddings else np.zeros(100) for word in words]
    return np.mean(embeddings, axis=0) if embeddings else np.zeros(100)  # Avoid empty input

# Streamlit UI
st.title("📩 Spam Classifier App")
st.write("Enter a message below to classify it as Spam or Ham.")

# Model selection
model_choice = st.radio("Choose a Model:", ["Naïve Bayes (TF-IDF)", "Logistic Regression (GloVe)"])

# Text input
user_input = st.text_area("Enter message:")

# Prediction button
if st.button("Classify"):
    if user_input.strip():
        if model_choice == "Naïve Bayes (TF-IDF)":
            text_vector = vectorizer.transform([user_input])  # TF-IDF for Naïve Bayes
            prediction = nb_model.predict(text_vector)
            probabilities = nb_model.predict_proba(text_vector)
            spam_prob = round(probabilities[0][1] * 100, 2)
            ham_prob = round(probabilities[0][0] * 100, 2)
            label = "Spam" if prediction[0] == 1 else "Ham"
            st.write(f"### **Prediction: {label}**")
            st.write(f"📊 **Spam Probability:** {spam_prob}%")
            st.write(f"📊 **Ham Probability:** {ham_prob}%")

        else:  # Logistic Regression (GloVe)
            text_embedding = text_to_glove_embeddings(user_input, glove_embeddings)
            text_embedding = text_embedding.reshape(1, -1)  # Reshape to (1, 100)
            prediction = log_model.predict(text_embedding)
            label = "Spam" if prediction[0] == 1 else "Ham"
            st.write(f"### **Prediction: {label}**")

    else:
        st.warning("⚠️ Please enter a message to classify!")
