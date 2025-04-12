from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load('model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None

    if request.method == 'POST':
        # Get form data
        age = float(request.form['age'])
        income = float(request.form['income'])
        gender = request.form['gender']
        occupation = request.form['occupation']

        # Prepare the input data in the correct format (as a DataFrame)
        input_data = pd.DataFrame([[age, income, gender, occupation]],
                                  columns=['Age', 'Annual Income (k$)', 'Gender', 'Occupation'])

        # Make the prediction using the model
        prediction = model.predict(input_data)[0]

    # Render the template with the prediction result
    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
