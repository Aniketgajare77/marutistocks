from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import warnings
import pickle


app = Flask(__name__)

def predictiondata(input_data):
    se = pickle.load(open('scaling.pkl', 'rb'))
    ra_reg = pickle.load(open('marutimodel.pkl', 'rb'))
    X = se.transform(input_data)
    ans = ra_reg.predict(X)[0]
    return ans


@app.route('/')
def displayform():

    return render_template('home.html')


@app.route('/inputs', methods=['POST'])
def getinput():
    Last = float(request.form['Last'])
    VWAP = float(request.form['VWAP'])
    High = float(request.form['High'])
    Low = float(request.form['Low'])
    Open = float(request.form['Open'])
    PrevClose = float(request.form['Prev Close'])
    year = float(request.form['year'])
    Turnover = float(request.form['Turnover'])

    input_data = pd.DataFrame(data=[[Last, VWAP, High, Low, Open, PrevClose, year, Turnover]],
                              columns=['Last', 'VWAP', 'High', 'Low', 'Open', 'Prev Close', 'year', 'Turnover'])

    ans = predictiondata(input_data)

    return render_template('display.html', data=ans)






if __name__ == "__main__":
    app.run(debug=True)