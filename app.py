from flask import Flask, request, render_template_string
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load models and preprocessors
with open("naive_bayes .pkl", "rb") as f:
    nb_model = pickle.load(f)

with open("svm_model (2).pkl", "rb") as f:
    svm_model = pickle.load(f)

with open("preprocessor_nb.pkl", "rb") as f:
    preprocessor_nb = pickle.load(f)

with open("preprocessor_svm.pkl", "rb") as f:
    preprocessor_svm = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

@app.route("/")
def home():
    with open('index.html', 'r') as f:
        return render_template_string(f.read())

@app.route("/predict", methods=['POST'])
def predict():
    data = request.form
    df = pd.DataFrame([{
        'Category': data['Category'],
        'Education': data['Education'],
        'Employment': data['Employment'],
        'Marital_Status': data['Marital_Status'],
        'Area': data['Area'],
        'Disability': data['Disability'],
        'Income': data['Income'],
        'Age': data['Age'],
        'Gender': data['Gender']
    }])

    # Preprocess input for NB
    input_nb = preprocessor_nb.transform(df)

    # Get probability predictions for each scheme
    nb_probs = nb_model.predict_proba(input_nb)[0]

    # Get top 5 scheme indices
    top_5_indices = np.argsort(nb_probs)[-5:][::-1]
    top_5_schemes = label_encoder.inverse_transform(top_5_indices)

    # Prepare input for SVM - use same input, but include only top 5 labels
    df_top5 = df.copy()
    df_top5 = pd.concat([df_top5]*5, ignore_index=True)
    df_top5["Scheme"] = label_encoder.inverse_transform(top_5_indices)

    # Add the label-encoded scheme column
    df_top5["Scheme"] = label_encoder.transform(df_top5["Scheme"])

    # Preprocess for SVM
    input_svm = preprocessor_svm.transform(df_top5)

    # Predict with SVM on each of the top 5 scheme inputs
    svm_preds = svm_model.predict(input_svm)

    # Pick the scheme (among top 5) with highest confidence (or first one marked 1)
    final_scheme_index = np.argmax(svm_preds)
    final_scheme = label_encoder.inverse_transform([df_top5.iloc[final_scheme_index]["Scheme"]])[0]

    # Return predictions
    with open('index.html', 'r') as f:
        html = f.read()
    return render_template_string(
        html,
        prediction_nb=", ".join(top_5_schemes),
        prediction_svm=final_scheme
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
