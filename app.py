from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input values from form
        year = int(request.form['year'])
        km_driven = int(request.form['km_driven'])
        fuel = int(request.form['fuel'])
        seller_type = int(request.form['seller_type'])
        transmission = int(request.form['transmission'])
        owner = int(request.form['owner'])
        mileage = float(request.form['mileage'])
        engine = float(request.form['engine'])
        max_power = float(request.form['max_power'])
        seats = float(request.form['seats'])

        # Create feature array
        features = np.array([[year, km_driven, fuel, seller_type,
                              transmission, owner, mileage,
                              engine, max_power, seats]])

        # Predict price
        prediction = model.predict(features)[0]

        # Return formatted result
        return render_template(
            'index.html',
            prediction_text=f"💰 Estimated Price: ₹ {int(prediction):,}"
        )

    except Exception as e:
        return render_template(
            'index.html',
            prediction_text="⚠️ Invalid Input! Please check values."
        )

if __name__ == "__main__":
    app.run(debug=True)