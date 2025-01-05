import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

# Step 1: Read the iris.csv file
data = pd.read_csv('D:/python projects/new pwc1/iris.csv')

# Step 2: Prepare the data
X = data.drop('species', axis=1)  # Features
y = data['species']  # Target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Create and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'D:/python projects/new pwc1/model.pkl')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the features from the form
    sepal_length = float(request.form['sepal_length'])
    sepal_width = float(request.form['sepal_width'])
    petal_length = float(request.form['petal_length'])
    petal_width = float(request.form['petal_width'])

    # Prepare the input for prediction
    features = [[sepal_length, sepal_width, petal_length, petal_width]]

    # Load the saved model
    model = joblib.load('model.pkl')

    # Make a prediction
    prediction = model.predict(features)

    return render_template('index.html', prediction=prediction[0])

if __name__ == '__main__':
    app.run(debug=True)
