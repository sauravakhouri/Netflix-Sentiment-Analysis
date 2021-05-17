# Netflix-Sentiment-Analysis
An end to end project implementing Netflix Sentiment Analysis to classify a Review is Positive or Negative.

Project Link: https://netflix-sentiment-bysaurav.herokuapp.com/

The project uses Natural Language Processing(NLP) along with Multinomial Naive Bayes to classify the sentiments

The basic process followed are:

-Text analytics using NLTK library

-Creating a target field for positive and negative reviews

-Data Cleaning(reviews) by removing usernames, punctuation, stopwords, lemmatization

-Vectorization to convert text data into vectors

-Multinomial Naive Bayes is chosen over SVM as the results from the former were better

-Model is evaluated using Recall and F1-Score.


The final model is deployed on Heroku Platform on the link provided above.

