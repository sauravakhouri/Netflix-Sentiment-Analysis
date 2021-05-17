# import libraries

import pandas as pd
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from nltk.stem import WordNetLemmatizer
import re
from sklearn.naive_bayes import MultinomialNB
import joblib

sw=stopwords.words('english')

# read positive data
pos_rev=pd.read_csv('pos.txt',sep='\n',header=None,encoding='latin=1')
#adding a target column
pos_rev['mood']=1.0
#changing the column name
pos_rev=pos_rev.rename(columns={0:'review'})


# read negative data
neg_rev=pd.read_csv('negative.txt',sep='\n',header=None,encoding='latin=1')
neg_rev['mood']=0.0
neg_rev=neg_rev.rename(columns={0:'review'})

# cleaning data
# 1)lower
# 2)remove spaces
# 3)punctuation
# 4)stopwords
# 5)lemmatize

pos_rev.loc[:,'review']=pos_rev.loc[:,'review'].apply(lambda x:x.lower())
pos_rev.loc[:,'review']=pos_rev.loc[:,'review'].apply(lambda x:re.sub(r"@\S+","",x))
pos_rev.loc[:,'review']=pos_rev.loc[:,'review'].apply(lambda x:x.translate(str.maketrans(dict.fromkeys(string.punctuation))))
pos_rev.loc[:,'review']=pos_rev.loc[:,'review'].apply(lambda x:" ".join([word for word in x.split() if word not in (sw)]))

neg_rev.loc[:,'review']=neg_rev.loc[:,'review'].apply(lambda x:x.lower())
neg_rev.loc[:,'review']=neg_rev.loc[:,'review'].apply(lambda x:re.sub(r"@\S+","",x))  
neg_rev.loc[:,'review']=neg_rev.loc[:,'review'].apply(lambda x:x.translate(str.maketrans(dict.fromkeys(string.punctuation))))
neg_rev.loc[:,'review']=neg_rev.loc[:,'review'].apply(lambda x:" ".join([word for word in x.split() if word not in (sw)]))

# combining into one DataFrame
con_rev=pd.concat([pos_rev,neg_rev],axis=0).reset_index()

#word tokenize
token= con_rev['review'].apply(lambda x: nltk.word_tokenize(x))

#lemmatization 
token=token.apply(lambda x:[WordNetLemmatizer().lemmatize(word,pos='v') for word in x])

# after lemmatization rejoining the tokenized words into a sentence
for i in range(len(token)):
    token[i]=" ".join(token[i])

# updating the review in the df
con_rev['review']=token

#splitting X and y variables
X=con_rev['review']
y=con_rev['mood']

# train test split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=101)

vectorizer=TfidfVectorizer()
train_vectors=vectorizer.fit_transform(X_train)
test_vectors=vectorizer.transform(X_test)

MNB=MultinomialNB()
MNB.fit(train_vectors,y_train)
MNB_pred=MNB.predict(test_vectors)

model_file_name='netflix_model.pkl'
vectorizer_filename='netflix_vector.pkl'
joblib.dump(MNB,model_file_name)
joblib.dump(vectorizer,vectorizer_filename)


