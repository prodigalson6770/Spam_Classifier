import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify

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
    embeddings = []
    for word in words:
        if word in glove_embeddings:
            embeddings.append(glove_embeddings[word])
        else:
            # If word is not in GloVe, use a zero vector
            embeddings.append(np.zeros(100))  # Assuming 100-dimensional GloVe vectors
    return np.mean(embeddings, axis=0)  # Average of word embeddings

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')  # This serves your HTML page

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get("text", "")
    model_choice = data.get("model", "naive_bayes")

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Naive Bayes Prediction with TF-IDF Vectorizer
    if model_choice == "naive_bayes":
        text_vector = vectorizer.transform([text])  # TF-IDF for Naive Bayes
        prediction = nb_model.predict(text_vector)
        # Get probabilities
        probabilities = nb_model.predict_proba(text_vector)
        spam_prob = probabilities[0][1]  # Probability for 'spam' (class 1)
        ham_prob = probabilities[0][0]   # Probability for 'ham' (class 0)

        # Prepare the result with probability
        result = "Spam" if prediction[0] == 1 else "Ham"
        return jsonify({
            "prediction": result,
            "spam_prob": round(spam_prob * 100, 2),  # Percentage
            "ham_prob": round(ham_prob * 100, 2)    # Percentage
        })

    # Logistic Regression Prediction with GloVe Embeddings
    elif model_choice == "logistic_regression":
        text_embedding = text_to_glove_embeddings(text, glove_embeddings)
        text_embedding = text_embedding.reshape(1, -1)  # Reshape to (1, 100)
        prediction = log_model.predict(text_embedding)

        result = "Spam" if prediction[0] == 1 else "Ham"
        return jsonify({"prediction": result})

    else:
        return jsonify({"error": "Invalid model choice"}), 400


if __name__ == '__main__':
    app.run(debug=False)
