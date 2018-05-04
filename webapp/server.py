import sys
import pickle
from flask import Flask, render_template, request, jsonify, Response
import pandas as pd
import os.path
sys.path.append('/Users/kammy/Desktop/galvanize/travel-time-rec/src')
import main_model_no_spark

model = pickle.load(open('../src/samp.p', 'rb'))


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
    #model.predict([3,  4])
    req = request.get_json()
    city_id, user_id = req['city'], req['user']
    prediction = model.predict(city_id, user_id)
    print(prediction)
    # call model and do predictions and send back result

    return jsonify({'city': city_id, 'user': user_id,
                    'predictions': {'pred1' : prediction[0], 'pred2' :prediction[1]}})



    #return jsonify({"kammy": "hello"})
    # city_type, gender, age, style = req['CityType'], req['gender'], req['age'], req['style']
    # prediction = model.predict([[city_type, gender, age, style]])
    # return jsonify({'city_type':city_type, 'gender': gender, 'age': age, 'style': style, 'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(host = '0.0.0.0', port = 3333, debug = True)
