from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score , accuracy_score , f1_score , recall_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score

data = pd.read_csv('heart.csv')


df_tk=data
def lower_limit(data):
  col = data.columns
  data_lower_limit = []
  for i in col:
    df = data[i]
    x = (1*df[0] + 1*df[1] + 2*df[2] + 3*df[3] + 4*df[4]) / sum(df)
    data_lower_limit.append(x)
  return(data_lower_limit)
  
def middle_value(data):
  col = data.columns
  data_middle_value = []
  for i in col:
    df = data[i]
    x = (1*df[0] + 2*df[1] + 3*df[2] + 4*df[3] + 5*df[4]) / sum(df)
    data_middle_value.append(x)
  return(data_middle_value)

def upper_limit(data):
  col = data.columns
  data_upper_limit = []
  for i in col:
    df = data[i]
    x = (2*df[0] + 3*df[1] + 4*df[2] + 5*df[3] + 5*df[4]) / sum(df)
    data_upper_limit.append(x)
  return(data_upper_limit)

def fuzzi_work(data):
  clower_limit = lower_limit(data)
  dmiddle_value = middle_value(data)
  eupper_limit = upper_limit(data)
  df = pd.DataFrame(list(zip(clower_limit, dmiddle_value, eupper_limit)),columns =['batas bawah', 'nilai tengah', 'batas atas'])
  return(df)

fuzz_tk = fuzzi_work(df_tk)

X = df.drop(['target'],axis =1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y,stratify = y)

regressor = GradientBoostingClassifier(
    max_depth=2,
    n_estimators=3,
    learning_rate=1.0
)
regressor.fit(X_train, y_train)

errors = [mean_squared_error(y_test,y_pred) for y_pred in regressor.staged_predict(X_test)]
best_n_estimators = np.argmin(errors)

best_regressor = GradientBoostingClassifier(
    max_depth =2 ,
    n_estimators=best_n_estimators,
    learning_rate = 0.5
)
best_regressor.fit(X_train,y_train)


pred = best_regressor.predict(X_test)
