import numpy as np
from flask import Flask, request, jsonify, render_template
from joblib import load


app = Flask(__name__)
model = load(open('model.joblib', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = "setosa"
    if prediction == 1:
        output = "versicolor"
    elif prediction == 2:
        output = "virginica"

    return render_template('index.html', prediction_text='This iris is a {}.'.format(output))

    return render_template('index.html', prediction_text='Sales should be {}.'.format(output))


if __name__ == "__main__":
    app.run(debug=True)