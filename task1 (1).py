#!/usr/bin/env python
# coding: utf-8

# ## Task 1: Introduction
# 
# Welcome to **Sentiment Analysis with Keras and TensorFlow**.
# 
# ![Sentiment Analysis](images/basic_sentiment_analysis.png)
# 
# 
# ## Task 2: The IMDB Reviews Dataset
# ____
# Note: If you are starting the notebook from this task, you can run cells from all previous tasks in the kernel by going to the top menu and then selecting **Kernel > Restart and Run All**
# ____

# In[1]:


from tensorflow.python.keras.datasets import imdb
(x_train, y_train) , (x_test, y_test) = imdb.load_data(num_words=10000)


# In[2]:


print(x_train[0])


# In[3]:


print(y_train[0])


# In[4]:


class_names = ['Negative', 'Positive']


# In[5]:


word_index = imdb.get_word_index()
print(word_index['hello'])


# ## Task 3: Decoding the Reviews
# ___
# Note: If you are starting the notebook from this task, you can run cells from all previous tasks in the kernel by going to the top menu and then selecting **Kernel > Restart and Run All**
# ___
# 

# In[6]:


reverse_word_index= dict((value, key) for key, value in word_index.items())
def decode(review):
    text = ''
    for i in review:
        text += reverse_word_index[i]
        text += ' '
    return text


# In[7]:


decode(x_train[0])


# In[8]:


def show_len():
    print('Len of 1st training examples: ', len(x_train[0]))
    print('Len of 2nd training examples: ', len(x_train[1]))
    print('Len of 1st test examples: ', len(x_test[0]))
    print('Len of 2nd test examples: ', len(x_test[1]))
    
show_len()


# 
# ## Task 4: Padding the Examples
# ___
# Note: If you are starting the notebook from this task, you can run cells from all previous tasks in the kernel by going to the top menu and then selecting **Kernel > Restart and Run All**
# ___
# 

# In[9]:


word_index['the']


# In[10]:


from tensorflow.python.keras.preprocessing.sequence import pad_sequences
x_train = pad_sequences(x_train, value = word_index['the'], padding = 'post', maxlen =256)
x_test = pad_sequences(x_test, value = word_index['the'], padding = 'post', maxlen =256)


# In[11]:


show_len()


# In[12]:


decode(x_train[0])


# ## Task 5: Word Embeddings
# ___
# Note: If you are starting the notebook from this task, you can run cells from all previous tasks in the kernel by going to the top menu and then selecting **Kernel > Restart and Run All**
# ___
# Word Embeddings:
# 
# ![Word Embeddings](images/word_embeddings.png)
# 
# Feature Vectors:
# 
# ![Learned Embeddings](images/embeddings.png)
# 
# 
# ## Task 6: Creating and Training the Model
# ___
# Note: If you are starting the notebook from this task, you can run cells from all previous tasks in the kernel by going to the top menu and then selecting **Kernel > Restart and Run All**
# ___

# In[13]:


from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Embedding, GlobalAveragePooling1D

model = Sequential([
    Embedding(10000, 16),
    GlobalAveragePooling1D(),
    Dense(16, activation='relu'),
    Dense(1, activation = 'sigmoid')
])

model.compile(
loss = 'binary_crossentropy',
optimizer = 'adam',
metrics = ['accuracy']
)

model.summary()


# In[14]:


from tensorflow.python.keras.callbacks import LambdaCallback 
simple_log = LambdaCallback(on_epoch_end = lambda e, l: print(e, end='.'))

E = 20 

h = model.fit(
    x_train, y_train,
    validation_split = 0.2,
    epochs = E,
    callbacks = [simple_log],
    verbose =  False
    
)


# ## Task 7: Predictions and Evaluation
# ___
# Note: If you are starting the notebook from this task, you can run cells from all previous tasks in the kernel by going to the top menu and then selecting **Kernel > Restart and Run All**
# ___
# 

# In[19]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

plt.plot(range(E), h.history['acc'], label = "Training")
plt.plot(range(E), h.history['val_acc'], label = "Validation")

plt.legend()
plt.show()


# In[21]:


loss, acc = model.evaluate(x_test, y_test)
print('Test set accuracy: ', acc *100 )


# 

# In[23]:


import numpy as np
p = model.predict(np.expand_dims(x_test[0], axis=0))
print(class_names[np.argmax(p[0])])


# In[24]:


decode(x_test[0])


# In[ ]:




