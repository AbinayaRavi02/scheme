from flask import Flask, request, render_template_string
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load models and encoders
with open("naive_bayes_model.pkl", "rb") as f:
    nb_model = pickle.load(f)

with open("svm_model.pkl", "rb") as f:
    svm_model = pickle.load(f)

with open("le_occupation.pkl", "rb") as f:
    le_occupation = pickle.load(f)

with open("le_location.pkl", "rb") as f:
    le_location = pickle.load(f)

with open("le_scheme.pkl", "rb") as f:
    le_scheme = pickle.load(f)

@app.route("/")
def home():
    with open('index.html', 'r') as f:
        return render_template_string(f.read())

@app.route("/predict", methods=['POST'])
def predict():
    if request.method == 'POST':
        data = request.form

        # Extract input
        age = int(data['Age'])
        salary = float(data['Salary'])
        occupation = le_occupation.transform([data['Occupation']])[0]
        location = le_location.transform([data['Location']])[0]

        # Input vector for Naive Bayes
        input_vector = np.array([[age, salary, occupation, location]])

        # Get top 5 predicted scheme indices from Naive Bayes
        probs = nb_model.predict_proba(input_vector)
        top_5_indices = probs.argsort()[0][-5:][::-1]  # descending order

        # Prepare input for SVM
        svm_inputs = []
        for scheme_idx in top_5_indices:
            svm_inputs.append([age, salary, occupation, location, scheme_idx])

        svm_inputs = np.array(svm_inputs)
        svm_probs = svm_model.predict_proba(svm_inputs)[:, 1]  # probability that it's the correct scheme

        best_index = np.argmax(svm_probs)
        final_scheme_idx = top_5_indices[best_index]

        # Decode scheme names
        top_5_schemes = le_scheme.inverse_transform(top_5_indices)
        final_scheme = le_scheme.inverse_transform([final_scheme_idx])[0]

        with open('index.html', 'r') as f:
            html = f.read()

        return render_template_string(
            html,
            prediction_nb=', '.join(top_5_schemes),
            prediction_svm=final_scheme
        )
    return "Invalid request method.", 405

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
