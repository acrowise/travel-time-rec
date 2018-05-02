import sys
import pickle
from flask import Flask, render_template, request, jsonify, Response
import pandas as pd

app = Flask(__name__)

#model = pickle.load(open('linreg.p', 'rb'))




@app.route('/', methods = ['GET'])
def home():
    return '<h1>Welcome to TravelX!</h1>'


@app.route('/recommender', methods = ['GET'])
def recommender():
    return render_template('recommender.html')


@app.route('/inference', methods = ['POST'])
def inference():
    # m = pikle.load('model.1')
    # m.predict([3,  4])
    req = request.get_json()
    print('req:', req)
    # call model and do predictions and send back result
    return jsonify({"kammy": "hello"})
    # city_type, gender, age, style = req['CityType'], req['gender'], req['age'], req['style']
    # prediction = model.predict([[city_type, gender, age, style]])
    # return jsonify({'city_type':city_type, 'gender': gender, 'age': age, 'style': style, 'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 3333, debug = True)
