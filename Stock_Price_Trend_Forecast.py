#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import lightgbm as lgb
import ta
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

random_seed = 2021
warnings.filterwarnings("ignore")



msft = yf.Ticker("MSFT")
history = msft.history(period='Max')
meta_cols = history.columns
history.tail()


ta_df = add_all_ta_features(df=history, open='Open', high='High', low='Low',close='Close',volume='Volume', fillna=True)
feature_cols = [x for x in ta_df.columns if x not in meta_cols]


toSelect = True
lag_length = 7
history_win = 2
forecast_date = "2021-09-03"


### create lagged dataset
df_lagged = ta_df.copy()
df_lagged[feature_cols] = df_lagged[feature_cols].apply(lambda x: x.shift(lag_length))
df_lagged['Close_prev'] = df_lagged['Close'].shift(lag_length)

### Include only data in the specified time window
start_date = pd.to_datetime(forecast_date)-pd.DateOffset(years=history_win)
start_date = start_date.strftime("%Y-%m-%d")

df_lagged.reset_index(drop=False, inplace=True)
df_lagged = df_lagged.loc[(df_lagged.Date >= start_date)&(df_lagged.Date <= forecast_date)]
temp_rows = df_lagged.shape[0]
df_lagged.dropna(axis=0, inplace=True)
print(f"number of NA records is {temp_rows - df_lagged.shape[0]}")

### Create target class labels
df_lagged['label'] = np.where(df_lagged['Close'] > df_lagged['Close_prev'], 1,0)   # 1 = Price increase, 0 = Price decrease

### Separate test data
test_data = df_lagged.loc[df_lagged.Date == forecast_date]


# ## 2. Modelling

### Split train validation/test set
X_train, X_test, y_train, y_test = train_test_split(df_lagged[feature_cols], df_lagged['label'], test_size=0.2, 
                                                    stratify=df_lagged['label'], random_state=random_seed)

# #### Feature Selection (optional)
### Feature selection if needed
### Simple training
model_lgb = lgb.LGBMClassifier(random_state=random_seed, n_jobs=-1)
model_lgb.fit(X_train, y_train)
test_preds = model_lgb.predict(X_test)
test_proba = model_lgb.predict_proba(X_test)[:, 1]
test_auc = roc_auc_score(y_test, test_proba)
test_accuracy = accuracy_score(y_test, test_preds)
print(f"Without feature selection: validation AUC is {test_auc} and accuracy is {test_accuracy}")

if toSelect:    
    df_features_impt = pd.DataFrame({"feature":feature_cols,
                                     "scores":model_lgb.feature_importances_}).sort_values("scores", ascending=False)
    
    best_auc = 0
    best_accuracy = 0
    best_features = None
    start_time = time.time()
    for ittr in range(len(df_features_impt.feature)):
        current_set = df_features_impt.feature[:ittr+1]
        model_lgb = lgb.LGBMClassifier(random_state=random_seed, n_jobs=-1)
        model_lgb.fit(X_train[current_set], y_train)
        test_preds = model_lgb.predict(X_test[current_set])
        test_proba = model_lgb.predict_proba(X_test[current_set])[:, 1]
        test_auc = roc_auc_score(y_test, test_proba)
        test_accuracy = accuracy_score(y_test, test_preds)
        if test_auc > best_auc and (test_auc-best_auc) > 0.001:
            best_auc = test_auc
            best_accuracy = test_accuracy
            best_features = current_set.copy()

    print(f"Number of features selected is {len(best_features)} with validation AUC: {best_auc} and accuracy: {best_accuracy}")
    feature_cols = best_features
    end_time = time.time()
    dur = round((end_time-start_time)/60,2)
    print(f"Feature selection duration: {dur} mins")


# #### Hyperparameter Tuning

params_grid = {
    "learning_rate": (0.01, 0.3),
    "n_estimators": (50, 500),
    "max_depth": (2, 7),
    "subsample": (0.8, 1),
    "feature_fraction": (0.8, 1),
    "lambda_l1": (0, 1),
    "lambda_l2": (0, 1) 
}

model_lgb = lgb.LGBMClassifier(random_state=random_seed, n_jobs=-1)

start_time = time.time()
bayes_gs = BayesSearchCV(model_lgb, params_grid, scoring='roc_auc', cv=5, n_iter=50, n_jobs=-1, random_state=random_seed)
bayes_gs.fit(X_train[feature_cols], y_train)
end_time = time.time()
print(f"Hyperparamter Tuning time: {round((end_time-start_time)/60, 2)} mins")

best_estimator = bayes_gs.best_estimator_
best_params = bayes_gs.best_params_

validation_preds = best_estimator.predict(X_test[feature_cols])
validation_proba = best_estimator.predict_proba(X_test[feature_cols])

print(classification_report(y_test,validation_preds))

cm = confusion_matrix(y_test,validation_preds)
ax= plt.subplot()
sns.heatmap(cm, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation
# labels, title and ticks
ax.set_xlabel('Predicted labels');ax.set_ylabel('True labels'); 
ax.set_title('Confusion Matrix'); 
ax.xaxis.set_ticklabels(['0-Down', '1-Up']); ax.yaxis.set_ticklabels(['0-Down', '1-Up']);



fpr, tpr, threshold = roc_curve(y_test, validation_proba[:, 1])
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
plt.show()
print(f"AUC score is {round(roc_auc_score(y_test, validation_proba[:, 1]), 5)}")


### re-fit to traing and validation data combined
X_train_val = pd.concat([X_train[feature_cols], X_test[feature_cols]], axis=0)
y_train_val = pd.concat([y_train, y_test], axis=0)

final_model = best_estimator
final_model.fit(X_train_val, y_train_val)


# ## 3. Prediction

oos_pred = final_model.predict(test_data[feature_cols])
print(f"True test label is {test_data.label.values[0]} and model prediction is {oos_pred[0]}")





