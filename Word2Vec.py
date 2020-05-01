#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import re
import nltk
import matplotlib.pyplot as plt


# In[2]:


from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize,word_tokenize
from nltk.corpus import state_union
from nltk.stem import WordNetLemmatizer


# In[3]:


lemma=WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


# In[148]:


text1=state_union.raw('2005-GWBush.txt')


# In[149]:


text1


# In[6]:


words_token=word_tokenize(text)


# In[10]:


words_token=[w.lower() for w in words_token]
words_token=[w for w in words_token if w.isalpha()]
words_token=[w for w in words_token if not w in stop_words]
words_token=[lemma.lemmatize(w) for w in words_token]


# In[230]:


from wordcloud import WordCloud
plt.figure(figsize = (20,20)) # Text Reviews with Poor Ratings
wc = WordCloud(min_font_size = 3,  max_words = 2000 , width = 1600 , height = 800).generate(" ".join(words_token))
plt.imshow(wc,interpolation = 'bilinear')


# In[18]:


import keras
from keras.preprocessing import text
from keras.utils import np_utils
from keras.preprocessing import sequence
from keras.layers import *
import keras.backend as K
from keras.models import Sequential


# In[60]:


tokenizer = text.Tokenizer()
tokenizer.fit_on_texts(words_token)
word2id = tokenizer.word_index

# build vocabulary of unique words
word2id['PAD'] = 0
id2word = {v:k for k, v in word2id.items()}
wids=[word2id[w] for w in words_token]

vocab_size = len(word2id)
embed_size = 100
window_size = 2 # context window size

print('Vocabulary Size:', vocab_size)
print('Vocabulary Sample:', list(word2id.items())[:10])


# In[96]:


def generates(corpus,window_size,vocab_size):
    context_length = window_size*2
    
    for index, word in enumerate(corpus):
        context_words = []
        label_word   = []            
        start = index - window_size
        end = index + window_size + 1
        context_words.append([corpus[i] 
                             for i in range(start, end)
                             if 0 <= i < len(corpus) 
                                 and i != index])
        label_word.append(word)

        x = sequence.pad_sequences(context_words, maxlen=context_length)
        y = np_utils.to_categorical(label_word, vocab_size)
        #print(context_words)
        yield (x, y)


# In[97]:


i = 0
for x, y in generates(corpus=wids, window_size=window_size, vocab_size=vocab_size):
    if 0 not in x[0]:
        print('Context (X):', [id2word[w] for w in x[0]], '-> Target (Y):', id2word[np.argwhere(y[0])[0][0]])
    
        if i == 10:
            break
        i += 1


# In[98]:


cbow = Sequential()
cbow.add(Embedding(input_dim=vocab_size, output_dim=embed_size, input_length=window_size*2))
cbow.add(Lambda(lambda x: K.mean(x, axis=1), output_shape=(embed_size,)))
cbow.add(Dense(vocab_size, activation='softmax'))

cbow.compile(loss='categorical_crossentropy', optimizer='rmsprop')
print(cbow.summary())


# In[99]:


for epoch in range(1, 6):
    loss = 0.
    i = 0
    for x, y in generates(corpus=wids, window_size=window_size, vocab_size=vocab_size):
        i += 1
        loss += cbow.train_on_batch(x, y)
        if i % 1000 == 0:
            print('Processed {} (context, word) pairs'.format(i))

    print('Epoch:', epoch, '\tLoss:', loss)
    print()


# In[106]:


weights = cbow.get_weights()[0]
weights = weights[1:]
print(weights.shape)

pd.DataFrame(weights, index=list(id2word.values())[1:]).head()


# In[116]:


from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=0, n_iter=10000, perplexity=3)
np.set_printoptions(suppress=True)
T = tsne.fit_transform(weights)
labels = list(id2word.values())

plt.figure(figsize=(24, 14))
plt.scatter(T[:80, 0], T[:80, 1], c='steelblue', edgecolors='k')
for label, x, y in zip(labels, T[:80, 0], T[:, 1]):
    plt.annotate(label, xy=(x+1, y+1), xytext=(0, 0), textcoords='offset points')


# In[192]:


from gensim.models import word2vec

x=[]
x.append([w for w in words_token])

# Set values for various parameters
feature_size = 100    # Word vector dimensionality  
window_context = 5          # Context window size                                                                                    
min_word_count = 1   # Minimum word count                        
sample = 1e-3   # Downsample setting for frequent words

w2v_model = word2vec.Word2Vec(x, size=feature_size, 
                          window=window_context, min_count=min_word_count,
                          sample=sample, iter=50)


# In[193]:


w2v_model.similarity('terror','terrorist')


# In[218]:


model2_skip = gensim.models.Word2Vec(x, min_count = 1, size = 100, 
                                             window = 5,iter=50, sg = 1)


# In[219]:


model2_skip.similarity('terror','terrorist')


# In[224]:


words = w2v_model.wv.index2word
wvs = w2v_model.wv[words]

tsne = TSNE(n_components=2, random_state=0, n_iter=10000, perplexity=2)
np.set_printoptions(suppress=True)
T = tsne.fit_transform(wvs)
labels = words

plt.figure(figsize=(24, 14))
plt.scatter(T[:40, 0], T[:40, 1], c='orange', edgecolors='r')
for label, x, y in zip(labels, T[:40, 0], T[:40, 1]):
    plt.annotate(label, xy=(x+1, y+1), xytext=(0, 0), textcoords='offset points')


# In[222]:


words =model2_skip.wv.index2word
wvs = model2_skip.wv[words]

tsne = TSNE(n_components=2, random_state=0, n_iter=5000, perplexity=2)
np.set_printoptions(suppress=True)
T = tsne.fit_transform(wvs)
labels = words

plt.figure(figsize=(24, 14))
plt.scatter(T[:40, 0], T[:40, 1], c='orange', edgecolors='r')
for label, x, y in zip(labels, T[:40, 0], T[:40, 1]):
    plt.annotate(label, xy=(x+1, y+1), xytext=(0, 0), textcoords='offset points')


# In[ ]:




