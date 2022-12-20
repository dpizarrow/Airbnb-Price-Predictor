import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
model = pickle.load(open('xgb_model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 0)

    print(output)

    # noches disponibles
    nights = request.form.get("availability_30")

    # monto ganado al mes
    month = nights * output

    return render_template('index.html', prediction_text=output, nights=nights, month=month)

if __name__ == "__main__":
    app.run(debug=True, port=5000)