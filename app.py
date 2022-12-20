import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)
model = pickle.load(open('xgb_model.pkl', 'rb'))

def chilean_currency(amount):
    currency = "${:,.1f}".format(amount)
    main_currency, fractional_currency = currency.split('.')
    new_main_currency = main_currency.replace(',', '.')
    currency = new_main_currency

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

    output = round(prediction[0], 2)

    print(output)

    # noches disponibles
    nights = int(request.form.get("availability_30"))

    # monto ganado al mes
    month = nights * int(output)

    return render_template('index.html', prediction_text=output, nights=nights, month=month)

if __name__ == "__main__":
    app.run(debug=True, port=5000)