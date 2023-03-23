#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

#prepare dataset
df = pd.read_csv("dataset.csv")
cv = CountVectorizer()
X = cv.fit_transform(df.Message)
y = df.Category
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#hyperparametrs for algorithm
parameters = {"alpha": [0.2,1,2,5,10], "fit_prior": [True, False]}
grid = GridSearchCV(MultinomialNB(), param_grid=parameters)
grid.fit(X_train,y_train)
alpha, fit_prior = grid.best_params_['alpha'], grid.best_params_['fit_prior']

#train
model = MultinomialNB(alpha = alpha)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

#just results
print(f'Accuracy: {round(accuracy_score(y_test,y_pred),3)}\n')
print(classification_report(y_test,y_pred))

import pickle

#paste wayes, where you will save model and cv in 44-48 rows
model_filename = "your dir" + "model.pkl"
cv_filename = "your dir" + "cv.pkl"


with open(model_filename, 'wb') as file:
    pickle.dump(model, file)

with open(cv_filename, 'wb') as file:
    pickle.dump(cv, file) 


def display_result(df, number=1):
    for i in range(number):
        msg = df['Message'].iloc[i]
        label = df["Category"].iloc[i]
        msg_vec = cv.transform([msg])
        pred_label = model.predict(msg_vec)
        print(f"Message: {msg}\nReal category: {label}\nPredicted category:{pred_label[0]}")
        print("\n")    

message = "Free entry to the competition for winning the FA Cup final on May 21, 2023. Send an FA message to 871232131 to get a T&C question for the application 08452810075718"        

with open(model_filename, 'rb') as modelLoaded, open(cv_filename, 'rb') as cvLoaded: 
    model = pickle.load(modelLoaded)
    cv = pickle.load(cvLoaded)
    msg_vec = cv.transform([message])
    predict = model.predict(msg_vec)
    print(f"\nPredicted category:{predict[0]}")
    print("\n")



