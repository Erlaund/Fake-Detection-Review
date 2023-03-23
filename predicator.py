# -*- coding: utf-8 -*-

import pickle 
import sys
import json
import sklearn.naive_bayes 
import os

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics._pairwise_distances_reduction import _datasets_pair
from sklearn.metrics._pairwise_distances_reduction import _middle_term_computer

model = "C:/xampp/htdocs/TanyaGit/model.pkl"
cv = "C:/xampp/htdocs/TanyaGit/cv.pkl"
#message = json.load(sys.argv[1])
#message = str(sys.argv[1])
#message = "Free entry to the competition for winning the FA Cup final on May 21, 2023. Send an FA message to 871232131 to get a T&C question for the application 08452810075718"        

wayToFile = sys.argv[1]
print(wayToFile, type(wayToFile))
print("Важный тестовы принт: ", os.path.join(os.path.dirname(sys.executable), model), os.path.join(os.path.dirname(sys.executable), cv) )

model = os.path.join(os.path.dirname(sys.executable), model)
cv = os.path.join(os.path.dirname(sys.executable), cv)

with open(model, 'rb') as modelLoaded, open(cv, 'rb') as cvLoaded, open(wayToFile, 'r', encoding='utf-8') as f: 
    model = pickle.load(modelLoaded)
    cv = pickle.load(cvLoaded)
    message = json.loads(f.read()).get("message")
    msg_vec = cv.transform([message])
    predict = model.predict(msg_vec)
    print(predict[0])


