from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

with open('model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('features.pkl', 'rb') as f:
    feature_columns = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        age = int(request.form['Age'])
        smoking_status = request.form['Smoking Status']
        health_score = int(request.form['Health Score'])
        location = request.form['Location']
        dependents = int(request.form['Number of Dependents'])

        raw_input = pd.DataFrame([{
            'Age': age,
            'Gender': 'Male',
            'Smoking Status': smoking_status,
            'Health Score': health_score,
            'Location': location,
            'Number of Dependents': dependents
        }])

        input_encoded = pd.get_dummies(raw_input, drop_first=True)  # Use drop_first=True to match training
        for col in feature_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0
        input_encoded = input_encoded[feature_columns]

        prediction = model.predict(input_encoded)[0]
        prediction = round(prediction, 2)

        return render_template('index.html', prediction=prediction)

    except Exception as e:
        return f"❌ Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)