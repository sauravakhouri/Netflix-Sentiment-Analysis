
import joblib
from flask import Flask, request, render_template

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

app=Flask(__name__)

# Loading the model and tfidf vectorizer ffrom the disk
vector=joblib.load('netflix_vector.pkl')
model=joblib.load('netflix_model.pkl')

def Classification(review):
    vectorized=vector.transform([review])
    my_pred=model.predict(vectorized)

    if my_pred==1:
        return (['Review is Positive'])
    else:
        return (['Review is Negative'])



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/Classify',methods=['POST'])
def Classify():
    review=request.form['Review']
    result=Classification(review)

    return render_template('index.html',classify_text=result)


if __name__=="__main__":
    app.run(debug=True)