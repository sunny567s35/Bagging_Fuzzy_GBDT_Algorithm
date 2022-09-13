from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import precision_score , accuracy_score , f1_score , recall_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import BaggingClassifier

df = pd.read_csv('heart.csv')

X = df.drop(['target'],axis =1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y,stratify = y, random_state =10)

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


bag_model = BaggingClassifier(
    base_estimator= best_regressor,
    n_estimators=100,
    max_samples = 0.8,
    bootstrap=True,
    oob_score = True,
    random_state = 0
)
bag_model.fit(X_train,y_train)
bag_model.oob_score_
bag_model.score(X_test,y_test)

pred = bag_model.predict(X_test)

scores = cross_val_score(bag_model,X,y,cv= 5)
print("The score of cross_validiation_score :")
print(" %.3f " % scores.mean())

print("mean abolute error : ",end = " ")
print("% .3f" % mean_absolute_error(y_test,pred))

print("Precision : ",end=" ")
print(" %.3f " % precision_score(y_test,pred))

print("Accuracy :",end=" ")
print(" %.3f " % accuracy_score(y_test,pred))

print("recall score :",end=" ")
print(" %.3f " % recall_score(y_test,pred))

print("f1_score :",end=" ")
print(" %.3f " % f1_score(y_test,pred))

