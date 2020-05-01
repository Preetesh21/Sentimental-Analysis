# -*- coding: utf-8 -*-
"""Sentimental_Analysis.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1Qi8zvS5nv7pFnjRdMbjOi0g6f1clDuoA
"""

!unzip '/content/drive/My Drive/amazon-music-reviews.zip'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

Musical_Instruments_5.json
 Musical_instruments_reviews.csv

df=pd.read_csv('/content/drive/My Drive/Musical_instruments_reviews.csv')
df.head()

df.isnull().sum()

df.describe().T

df.reviewText.fillna("",inplace = True)
df=df.drop(['reviewerID','asin','reviewerName','helpful','unixReviewTime','reviewTime'],1)

df['text'] = df['reviewText'] + ' ' + df['summary']
# del df['reviewText']
# del df['summary']
df.head()

plt.figure(figsize = (20,20)) # Text Reviews with Poor Ratings
wc = WordCloud(min_font_size = 3,  max_words = 2000 , width = 1600 , height = 800).generate(" ".join(df.text))
plt.imshow(wc,interpolation = 'bilinear')

sns.countplot(df.overall).set_title('movie reviews')

!wget http://nlp.stanford.edu/data/glove.6B.zip

!unzip glove*.zip

glove_path='/content/drive/My Drive/glove.6B.100d.txt'

EMBEDDING_DIM = 100

embeddings_index = {}
f = open(glove_path)
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

#print('Found %s unique tokens.' % len(word_index))
print('Total %s word vectors.' % len(embeddings_index))

import keras
from keras import backend as K
from keras.engine.topology import Layer
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import Sequential, Model, load_model
from keras.callbacks import ModelCheckpoint
from keras.initializers import Constant
from keras.layers.merge import add

from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.utils import np_utils
from keras import regularizers
from keras.regularizers import l2

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import sent_tokenize,word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
import os

df.head()

sw=[]
sw=df['text']
sw

stop_words = set(stopwords.words('english'))
ws=[]
for example_sent in sw:
  word_tokens = word_tokenize(example_sent) 
  filtered_sentence = [w for w in word_tokens if not w in stop_words]
  ws.append(filtered_sentence)

ws[1]

for i in range(len(ws)):
  ws[i]= [word for word in ws[i] if word.isalpha()]

lemma=WordNetLemmatizer()
for i in range(len(ws)):
  for j in range(len(ws[i])):
    ws[i][j]=lemma.lemmatize(ws[i][j])

# tokenizing
df['text']=ws
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df.text)
X = tokenizer.texts_to_sequences(df.text)
df['words'] = X

# delete some empty and short data
df['word_length'] = df.words.apply(lambda i: len(i))
df = df[df.word_length >= 5]

df.head()

maxlen = 50
X = list(sequence.pad_sequences(df.words, maxlen=maxlen))

word_index = tokenizer.word_index

embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index)+1,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=maxlen,
                            trainable=False)

X = np.array(X)
Y = np_utils.to_categorical(list(df.overall))

seed = 29
x_train, x_val, y_train, y_val = train_test_split(X, Y, test_size=0.2, random_state=seed)

inp = Input(shape=(maxlen,), dtype='int32')
x = embedding_layer(inp)
x = Bidirectional(LSTM(128, return_sequences=True, name="BiLSTM-1",recurrent_regularizer=l2(0.01)))(x)
x = Dropout(0.5, name="Dropout-1")(x)
x = Bidirectional(LSTM(128, name="BiLSTM-2",recurrent_regularizer=l2(0.01)))(x)
x = Dropout(0.5, name="Dropout-2")(x)
outp = Dense(6, activation='softmax', name="FC-layer")(x)
model = Model(inp, outp)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.summary()

model_history = model.fit(x_train, y_train, batch_size=128, epochs=15, validation_data=(x_val, y_val))

acc = model_history.history['acc']
val_acc = model_history.history['val_acc']
loss = model_history.history['loss']
val_loss = model_history.history['val_loss']
epochs = range(1, len(acc) + 1)
val_acc1=max(val_acc)

plt.title('Training and validation accuracy')
plt.plot(epochs, acc, 'red', label='Training acc')
plt.plot(epochs, val_acc, 'blue', label='Validation acc')
plt.legend()

plt.figure()
plt.title('Training and validation loss')
plt.plot(epochs, loss, 'red', label='Training loss')
plt.plot(epochs, val_loss, 'blue', label='Validation loss')
plt.legend()
plt.show()

data=pd.read_csv('/content/drive/My Drive/Musical_instruments_reviews.csv')

data.text.fillna("",inplace = True)
data['text'] = data['reviewText'] + ' ' + data['summary']

x_train,x_test,y_train,y_test = train_test_split(data.text,data.overall,test_size = 0.2 , random_state = 0)

cv=CountVectorizer(min_df=0,max_df=1,binary=False,ngram_range=(1,3))
#transformed train reviews
cv_train_reviews=cv.fit_transform(x_train)
#transformed test reviews
cv_test_reviews=cv.transform(x_test)

print('BOW_cv_train:',cv_train_reviews.shape)
print('BOW_cv_test:',cv_test_reviews.shape)

tv=TfidfVectorizer(min_df=0,max_df=1,use_idf=True,ngram_range=(1,3))
#transformed train reviews
tv_train_reviews=tv.fit_transform(x_train)
#transformed test reviews
tv_test_reviews=tv.transform(x_test)
print('Tfidf_train:',tv_train_reviews.shape)
print('Tfidf_test:',tv_test_reviews.shape)

tv_train_reviews

cv_train_reviews

lr=LogisticRegression(penalty='l2',max_iter=500,C=1,random_state=0)
#Fitting the model for Bag of words
lr_bow=lr.fit(cv_train_reviews,y_train)
print(lr_bow)
#Fitting the model for tfidf features
lr_tfidf=lr.fit(tv_train_reviews,y_train)
print(lr_tfidf)

#Predicting the model for bag of words
lr_bow_predict=lr.predict(cv_test_reviews)
##Predicting the model for tfidf features
lr_tfidf_predict=lr.predict(tv_test_reviews)
#Accuracy score for bag of words
lr_bow_score=accuracy_score(y_test,lr_bow_predict)
print("lr_bow_score :",lr_bow_score)
#Accuracy score for tfidf features
lr_tfidf_score=accuracy_score(y_test,lr_tfidf_predict)
print("lr_tfidf_score :",lr_tfidf_score)

#training the model
mnb=MultinomialNB()
#fitting the nb for bag of words
mnb_bow=mnb.fit(cv_train_reviews,y_train)
print(mnb_bow)
#fitting the nb for tfidf features
mnb_tfidf=mnb.fit(tv_train_reviews,y_train)
print(mnb_tfidf)

#Predicting the model for bag of words
mnb_bow_predict=mnb.predict(cv_test_reviews)
#Predicting the model for tfidf features
mnb_tfidf_predict=mnb.predict(tv_test_reviews)
#Accuracy score for bag of words
mnb_bow_score=accuracy_score(y_test,mnb_bow_predict)
print("mnb_bow_score :",mnb_bow_score)
#Accuracy score for tfidf features
mnb_tfidf_score=accuracy_score(y_test,mnb_tfidf_predict)
print("mnb_tfidf_score :",mnb_tfidf_score)

model1 = Sequential()
model1.add(Dense(units = 75 , activation = 'relu' , input_dim = cv_train_reviews.shape[1]))
model1.add(Dense(units = 50 , activation = 'relu'))
model1.add(Dense(units = 25 , activation = 'relu'))
model1.add(Dense(units = 10 , activation = 'relu')) 
model1.add(Dense(units = 1 , activation = 'sigmoid'))
model1.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'])

model1.summary()

model1.train_on_batch(cv_train_reviews,y_train)

cnn_test_accuracy=model1.test_on_batch(cv_test_reviews,y_test)

cnn_test_accuracy

acc=dict({'LSTM':val_acc1,'CNN':cnn_test_accuracy[0],'LR-TF_IDF':lr_tfidf_score,'LR-BOW':lr_bow_score,'MNB-TF_IDF':mnb_tfidf_score,'MNB-BOW':mnb_bow_score})
acc

fig = plt.figure(figsize=(20,5))

#  subplot #1
fig.add_subplot(1,3,1)
sns.barplot(x=[*acc],y=list(acc.values())).set_title('Validation_accuracy')

#  subplot #2
fig.add_subplot(1,3,2)
sns.lineplot(x=[*acc],y=list(acc.values())).set_title('Validation_accuracy')

#  subplot #3
fig.add_subplot(1,3,3)
sns.boxplot(x=[*acc],y=list(acc.values())).set_title('Validation_accuracy')

