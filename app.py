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
        
        if prediction['error'] != 'error':
            val_accuracy = round(accuracy_score(prediction['val_true'], prediction['val_class']), 2)
            val_f1 = round(f1_score(prediction['val_true'], prediction['val_class']), 2)
            val_auc = round(roc_auc_score(prediction['val_true'], prediction['val_pred']), 2)
            prediction['val_accuracy']=val_accuracy
            prediction['val_f1']=val_f1
            prediction['val_auc']=val_auc
            
            cm = confusion_matrix(prediction['val_true'], prediction['val_class'])
            ax= plt.subplot()
            sns.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation
            # labels, title and ticks
            ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
            ax.set_title('Confusion Matrix'); 
            ax.xaxis.set_ticklabels(['0-Down', '1-Up']); ax.yaxis.set_ticklabels(['0-Down', '1-Up']);
            ax.figure.savefig('static/cm.png')
            prediction['cm_link'] = 'static/cm.png'
            
            fpr, tpr, threshold = roc_curve(prediction['val_true'], prediction['val_pred'])
            roc_auc = auc(fpr, tpr)
            plt.figure(figsize=(12, 8))
            plt.title('Receiver Operating Characteristic')
            plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
            plt.legend(loc = 'lower right')
            plt.plot([0, 1], [0, 1],'r--')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.ylabel('True Positive Rate')
            plt.xlabel('False Positive Rate')
            plt.savefig('static/roc_figure.png')
            prediction['figure_link'] = 'static/roc_figure.png'
            


            
        return render_template('index.html', value=prediction)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
