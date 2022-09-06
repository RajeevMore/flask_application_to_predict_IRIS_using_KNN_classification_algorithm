from flask import Flask,app,jsonify,url_for, render_template, request
import pandas,pickle
import numpy as np
import matplotlib

app = Flask(__name__)

app.route("/")
def fun1():
    return render_template("info.html")

app.route("/predict_iris", methods = ["post"])
def fun2():
    value1 = float(request.form['value1'])
    value2 = float(request.form['value2'])
    value3 = float(request.form['value3'])
    value4 = float(request.form['value4'])
    knn_prediction = pickle.load(open('knn_model.pkl','rb'))
    #x_new = np.array([[5.1, 3.5, 1.4, 0.2]])
    x_new = np.array([[value1, value2, value3, value4]])
    prediction = knn_prediction.predict(x_new)
    #print("Prediction: {}".format(prediction)) 
    #print("Predicted target name:{}".format(iris_dataset["target_names"][prediction]))
    return "Prediction: {}".format(prediction) 

if __name__ == "__main__" :
    app.run(debug=True)