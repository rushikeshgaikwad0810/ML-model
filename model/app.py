from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('model.pkl')

# Read the CSV to get unique occupations
df = pd.read_csv('Mall_Customers_100_with_Occupation.csv')
occupations = df['Occupation'].unique().tolist()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    error_message = None

    if request.method == 'POST':
        try:
            # Get form data
            age = float(request.form['age'])
            income = float(request.form['income'])
            gender = request.form['gender']
            occupation = request.form['occupation']

            # Validate input data
            if age <= 0 or income <= 0:
                raise ValueError("Age and Income must be positive numbers")

            # Prepare the input data in the correct format (as a DataFrame)
            input_data = pd.DataFrame([[age, income, gender, occupation]],
                                      columns=['Age', 'Annual Income (k$)', 'Gender', 'Occupation'])

            # Make the prediction using the model
            prediction = model.predict(input_data)[0]

        except ValueError as e:
            error_message = str(e)
        except Exception as e:
            error_message = "An error occurred during prediction. Please try again."

    # Render the template with the prediction result or error message
    return render_template('index.html', prediction=prediction, error_message=error_message, occupations=occupations)

if __name__ == '__main__':
    app.run(debug=True)
