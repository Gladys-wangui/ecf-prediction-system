from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

# Load the pre-trained machine learning model
model = joblib.load('trained_model.joblib')

# Render the homepage with the input form
@app.route('/')
def home():
    return render_template('index.html')

# Handle form submission and make predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        temperature = float(request.form['temperature'])
        swollen_lymph_nodes = int(request.form.get('swollen_lymph_nodes', 0))
        loss_of_appetite = int(request.form.get('loss_of_appetite', 0))
        weakness = int(request.form.get('weakness', 0))
        coughing = int(request.form.get('coughing', 0))
        rapid_breathing = int(request.form.get('rapid_breathing', 0))
        nasal_discharge = int(request.form.get('nasal_discharge', 0))
        anaemia = int(request.form.get('anaemia', 0))
        
        # Create a list of feature values
        features = [
            temperature,
            swollen_lymph_nodes,
            loss_of_appetite,
            weakness,
            coughing,
            rapid_breathing,
            nasal_discharge,
            anaemia,
        ]

        # Make predictions using the trained model
        prediction = model.predict([features])

        # Assuming your model returns binary predictions (0 or 1)
        result = "Positive" if prediction[0] == 1 else "Negative"

        return jsonify({'result': result})

        # Pass the result to the template
        return render_template('result.html', result=result)

    except ValueError as e:
        # Print the error message for debugging
        return jsonify({'error': str(e)})

        return render_template('error.html', message='Invalid input. Please enter valid values for input features.')

if __name__ == '__main__':
    app.run(debug=True)
