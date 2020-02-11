from flask import Flask,render_template,request
from sklearn.externals import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

app=Flask(__name__)
ml_model=open('Random_Forest.pkl','rb')
Random_Forest_Model=joblib.load(ml_model)
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods = ['POST','GET'])
def predict():
    if request.method=='POST':
        try:
            to_predict_list = request.form.to_dict()
            to_predict_list = list(to_predict_list.values())
            to_predict_list = list(map(float, to_predict_list))
            to_predict_list_arr=np.array(to_predict_list)
            to_predict_list_arr=to_predict_list_arr.reshape(1,-1)
            predictions=Random_Forest_Model.predict(to_predict_list_arr)

        except ValueError:
            return "Please enter valid values"



        return render_template('predict.html',Predictions=predictions[0])


if __name__=="__main__":
    app.run(debug=True,host='127.0.0.1', port=5000)
