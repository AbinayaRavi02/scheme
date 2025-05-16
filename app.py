from flask import Flask, request, render_template, jsonify, redirect, session, flash
import pickle
import pandas as pd
from difflib import get_close_matches

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# MySQL config
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = ''
app.config['MYSQL_DB'] = 'gov_scheme_db'



# Load models
with open("naive_bayes .pkl", "rb") as f:
    nb_model = pickle.load(f)
with open("svm_model .pkl", "rb") as f:
    svm_model = pickle.load(f)
with open("preprocessor_nb .pkl", "rb") as f:
    preprocessor_nb = pickle.load(f)
with open("preprocessor_svm .pkl", "rb") as f:
    preprocessor_svm = pickle.load(f)
with open("label_encoder .pkl", "rb") as f:
    label_encoder = pickle.load(f)



# Chatbot responses
conversation_dict = {
    "hi": "👋 Hi there!",
    "hello": "👋 Hello! How can I assist you today?",
    "how are you": "😊 I'm doing great, thanks for asking!",
    "your name": "🤖 I'm your Government Scheme Assistant.",
    "thank you": "🙏 You're welcome!",
    "thanks": "🙏 Glad to help!",
    "help": "💡 Ask me about government schemes or documents required.",
    "bye": "👋 Bye! Have a nice day."
}
 
document_procedures = {
    "aadhar card": "Visit Aadhaar Seva Kendra or UIDAI portal.",
    "income certificate": "Apply at tehsildar or online portal.",
    "domicile certificate": "Submit residence proof at district office.",
    "caste certificate": "Apply online or visit SDM office.",
    "bpl certificate": "Apply with ration card and income proof.",
    "bank passbook": "Request at your bank with ID.",
    "job card": "Apply under MGNREGA at Panchayat.",
    "10th marksheet": "Contact school or board office."
}

def get_documents_for_scheme(query):
    matches = get_close_matches(query, scheme_data.keys(), n=1, cutoff=0.4)
    if matches:
        scheme = matches[0]
        return scheme, scheme_data[scheme].split(', ')
    return None, None

@app.route("/register", methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        password = bcrypt.generate_password_hash(request.form['password']).decode('utf-8')

        cur = mysql.connection.cursor()
        cur.execute("INSERT INTO users (name, email, password) VALUES (%s, %s, %s)", (name, email, password))
        mysql.connection.commit()
        cur.close()

        flash("Registered successfully!")
        return redirect('/login')
    return render_template("register.html")

@app.route("/login", methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password_candidate = request.form['password']

        cur = mysql.connection.cursor()
        result = cur.execute("SELECT * FROM users WHERE email = %s", [email])
        if result > 0:
            data = cur.fetchone()
            if bcrypt.check_password_hash(data[3], password_candidate):
                session['logged_in'] = True
                session['email'] = email
                flash("Login successful.")
                return redirect("/")
            else:
                flash("Wrong password.")
        else:
            flash("User not found.")
        cur.close()
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("You are logged out.")
    return redirect("/login")

@app.route("/")
def home():
    if 'logged_in' not in session:
        return redirect('/login')
    with open("index.html", "r") as f:
        return render_template(f.read())

@app.route("/predict", methods=["POST"])
def predict():
    data = request.form
    df = pd.DataFrame([{
        'Category': data['Category'], 'Education': data['Education'], 'Employment': data['Employment'],
        'Marital_Status': data['Marital_Status'], 'Area': data['Area'], 'Disability': data['Disability'],
        'Income': data['Income'], 'Age': data['Age'], 'Gender': data['Gender']
    }])
    input_nb = preprocessor_nb.transform(df)
    input_svm = preprocessor_svm.transform(df)
    nb_pred = label_encoder.inverse_transform(nb_model.predict(input_nb))[0]
    svm_pred = label_encoder.inverse_transform(svm_model.predict(input_svm))[0]
    with open("index.html", "r") as f:
        html = f.read()
    return render_template(html, prediction_nb=nb_pred, prediction_svm=svm_pred)

@app.route("/api/chat", methods=["POST"])
def api_chat():
    user_msg = request.json.get("message", "").strip().lower()
    if not user_msg:
        return jsonify(reply="❗ Please type something.")
    for key, reply in conversation_dict.items():
        if key in user_msg:
            return jsonify(reply=reply)
    for doc in document_procedures:
        if doc in user_msg:
            return jsonify(reply=f"📄 {doc.title()}: {document_procedures[doc]}")
    if user_msg == "list":
        return jsonify(reply="📋 " + "<br>".join(f"- {k}" for k in scheme_data))
    scheme, docs = get_documents_for_scheme(user_msg)
    if scheme:
        return jsonify(reply=f"✅ {scheme} requires:<br>" + "<br>".join(f"- {d}" for d in docs))
    return jsonify(reply="🤖 I didn't understand. Try asking 'documents for PMAY' or type 'list'.")

if __name__ == "__main__":
    app.run(debug=True)
