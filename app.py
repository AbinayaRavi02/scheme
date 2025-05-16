from flask import Flask, request, render_template_string, jsonify
import pickle
import pandas as pd
import os
from difflib import get_close_matches

app = Flask(__name__)

# =============================
# Load models and encoders
# =============================
try:
    with open("naive_bayes.pkl", "rb") as f:
        nb_model = pickle.load(f)
    with open("svm_model.pkl", "rb") as f:
        svm_model = pickle.load(f)
    with open("preprocessor_nb.pkl", "rb") as f:
        preprocessor_nb = pickle.load(f)
    with open("preprocessor_svm.pkl", "rb") as f:
        preprocessor_svm = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
except Exception as e:
    print("Model or preprocessor load failed:", e)

# =============================
# Load Excel data for chatbot
# =============================
excel_path = r"E:\New folder\Additional_Scheme_Required_Documents.xlsx"
if os.path.exists(excel_path):
    df = pd.read_excel(excel_path)
    scheme_data = dict(zip(df["Scheme Name"], df["Required Documents"]))
else:
    print(f"❌ File not found: {excel_path}")
    scheme_data = {}

# =============================
# Chatbot Knowledge Base
# =============================
greetings = ["hi", "hello", "hey", "good morning", "good evening", "namaste"]
conversation_dict = {
    "hi": "👋hi",
    "hello": "👋 Hello! How can I assist you today?",
    "hey": "👋 Hey there!",
    "how are you": "😊 I'm doing well, thank you! How can I help?",
    "your name": "🤖 I'm your Government Scheme Assistant.",
    "what is your name": "🤖 I'm your Government Scheme Assistant.",
    "thank you": "🙏 You're welcome! Ask anything related to government schemes.",
    "thanks": "🙏 Happy to help!",
    "what can you do": "💡 I can provide document requirements for Indian government schemes. Try typing a scheme name.",
    "help": "💡 You can ask about any government scheme or type 'list' to see available ones.",
    "who made you": "🛠️ I was developed to assist with government scheme information.",
    "who are you": "🤖 I'm a bot built to guide you with Indian government schemes.",
    "bye": "👋 Goodbye! Feel free to return anytime.",
    "goodbye": "👋 Take care! Have a great day!"
}
document_procedures = {
    "aadhar card": "Visit your nearest Aadhaar Seva Kendra or apply online via UIDAI portal (https://uidai.gov.in).",
    "income certificate": "Apply at your local tehsildar office or through the state’s online portal with ID and address proof.",
    "domicile certificate": "Submit proof of residence (rent agreement, utility bill, etc.) at your district office or CSC.",
    "caste certificate": "Visit your local SDM office or apply online on your state government portal with community proof.",
    "bpl certificate": "Contact your Gram Panchayat or municipal office with income proof and ration card details.",
    "bank passbook": "Visit your bank branch with ID proof and request a new or duplicate passbook.",
    "job card": "Apply through your Gram Panchayat under MGNREGA. Provide address and ID proof.",
    "10th marksheet": "Contact your school or state board office. You may need school ID or registration number."
}


# =============================
# Route: Homepage with form
# =============================
@app.route("/")
def home():
    with open('index.html', 'r') as f:
        return render_template_string(f.read())

# =============================
# Route: Predict Schemes
# =============================
@app.route("/predict", methods=['POST'])
def predict():
    try:
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

        input_nb = preprocessor_nb.transform(df)
        input_svm = preprocessor_svm.transform(df)

        nb_pred = label_encoder.inverse_transform(nb_model.predict(input_nb))[0]
        svm_pred = label_encoder.inverse_transform(svm_model.predict(input_svm))[0]

        with open('index.html', 'r') as f:
            html = f.read()
        return render_template_string(
            html,
            prediction_nb=nb_pred,
            prediction_svm=svm_pred
        )
    except Exception as e:
        return f"An error occurred: {str(e)}", 500

# =============================
# Route: Chatbot UI
# =============================
@app.route("/chat")
def chat():
    return render_template_string("""
    <!doctype html>
    <html>
    <head>
        <title>Gov Scheme Chatbot</title>
        <style>
            body { font-family: Arial; background: #f0f0f0; padding: 20px; }
            .chatbox { max-width: 700px; margin: auto; background: white; padding: 20px; border-radius: 10px; }
            .messages { min-height: 300px; }
            .message { margin: 10px 0; }
            .user { text-align: right; color: #2c3e50; }
            .bot { text-align: left; color: #2c7a7b; }
            .input-area { display: flex; margin-top: 20px; }
            input[type=text] { flex: 1; padding: 10px; border: 1px solid #ccc; border-radius: 5px; }
            button { margin-left: 10px; padding: 10px 20px; background: #2c7a7b; color: white; border: none; border-radius: 5px; }
        </style>
    </head>
    <body>
        <div class="chatbox">
            <h2>🤖 Real-Time Government Scheme Chatbot</h2>
            <div class="messages" id="messages"></div>
            <div class="input-area">
                <input type="text" id="userInput" placeholder="Type your question..." autofocus autocomplete="off">
                <button onclick="sendMessage()">Send</button>
            </div>
        </div>
        <script>
            async function sendMessage() {
                const input = document.getElementById("userInput");
                const text = input.value.trim();
                if (!text) return;

                const msgBox = document.getElementById("messages");
                msgBox.innerHTML += `<div class='message user'>${text}</div>`;
                input.value = '';

                const res = await fetch("/api/chat", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ message: text })
                });

                const data = await res.json();
                msgBox.innerHTML += `<div class='message bot'>${data.reply}</div>`;
                msgBox.scrollTop = msgBox.scrollHeight;
            }

            document.getElementById("userInput").addEventListener("keypress", function(e) {
                if (e.key === "Enter") sendMessage();
            });
        </script>
    </body>
    </html>
    """)

# =============================
# Route: Chatbot API Logic
# =============================
@app.route("/api/chat", methods=["POST"])
def api_chat():
    user_msg = request.json.get("message", "").strip().lower()
    if not user_msg:
        return jsonify(reply="❗ Please enter something.")

    for key, reply in conversation_dict.items():
        if key in user_msg:
            return jsonify(reply=reply)

    for doc_name in document_procedures:
        if doc_name in user_msg:
            return jsonify(reply=f"📄 Procedure to get <strong>{doc_name.title()}</strong>:<br>{document_procedures[doc_name]}")

    if user_msg == "list":
        schemes = list(scheme_data.keys())
        return jsonify(reply="📋 Available schemes:<br>" + "<br>".join(f"- {s}" for s in schemes))

    matches = get_close_matches(user_msg, scheme_data.keys(), n=1, cutoff=0.4)
    if matches:
        scheme = matches[0]
        documents = scheme_data[scheme].split(', ')
        return jsonify(reply=f"✅ Documents required for <strong>{scheme}</strong>:<br>" + "<br>".join(f"- {doc}" for doc in documents))

    return jsonify(reply="🤖 I'm not sure how to respond. Try saying 'list' or ask about a scheme like 'documents for CLSS'.")

# =============================
# Run App
# =============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
