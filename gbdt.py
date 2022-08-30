from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score

df = pd.read_csv('heart.csv')

X = df.drop(['target'],axis =1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y)

regressor = GradientBoostingRegressor(
    max_depth=2,
    n_estimators=3,
    learning_rate=1.0
)
regressor.fit(X_train, y_train)

errors = [mean_squared_error(y_test,y_pred) for y_pred in regressor.staged_predict(X_test)]
best_n_estimators = np.argmin(errors)

best_regressor = GradientBoostingRegressor(
    max_depth =2 ,
    n_estimators=best_n_estimators,
    learning_rate = 1.0
)
best_regressor.fit(X_train,y_train)

y_pred = best_regressor.predict(X_test)
mean_absolute_error(y_test,y_pred)

scores = cross_val_score(GradientBoostingRegressor(),X,y)
print("The score of cross_validiation_score : ")
print(scores.mean())
