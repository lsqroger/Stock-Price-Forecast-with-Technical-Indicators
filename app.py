from flask import Flask, app, render_template, request
import numpy as np
import pickle
import pandas as pd
import lightgbm as lgb
from ta import add_all_ta_features
import yfinance as yf
import datetime as dt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, auc, classification_report, confusion_matrix
from skopt import BayesSearchCV
import time
import warnings
import matplotlib.pyplot as plt
import seaborn as sns  
from model_pipeline import price_forecast

random_seed = 2021
warnings.filterwarnings("ignore")

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def ml_fcst():
    req_type = request.method
    if req_type == 'GET':
        return render_template('index.html')
    else:
        forecast_date = request.form['date']
        forecast_stock = request.form['ticker']
        prediction = price_forecast(forecast_date, forecast_stock)
        return render_template('index.html', value=prediction)

app.run(port=8080)
