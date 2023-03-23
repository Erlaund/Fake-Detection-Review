import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

#prepare dataset 
#paste way to csv file
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