import numpy as np
import  pandas as pd
import sklearn as sk
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df = pd.read_csv("heart.csv")


X = df.drop(['target'],axis =1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y,stratify = y,random_state =0)

model = DecisionTreeClassifier()
model.fit(X_train,y_train)
pred = model.predict(X_test) 
scores = cross_val_score(DecisionTreeClassifier(),X,y,cv= 5)
print("The score of cross_validiation_score :")
print(scores.mean())
