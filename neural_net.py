import tensorflow as tf
import pandas  as pd
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
# Here I've proposed a Neural Network that uses the Bidirectional LSTM cells for the processing of our feature vectors .

# HYPERPARAMS
vocab_size = 10000
maxlen = 150
embedding_dim = 20

df = pd.read_csv("H:/study_lake/Fake_News/Fake_News_Detection/train.csv")
sentence , label = df.loc[:,"Statement"].values , df.loc[:,"Label"].values

for each in label:
    if each=="TRUE":
        each =1
        continue
    each=0
print(label) 
# Prepairing the tokenizer 
tokenizer = Tokenizer(num_words=vocab_size,oov_token = "<OOV>")
tokenizer.fit_on_texts(sentence)
sequences = tokenizer.texts_to_sequences(sentence)

# Padding the data
pad_sequences = pad_sequences(sequences,maxlen,padding="pre")


x_train,x_test,y_train,y_test = train_test_split(pad_sequences,label,test_size=0.2,stratify=label)

# Prepairing the Model

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size,embedding_dim,input_length=maxlen),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,return_sequences=True)),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(32,activation="relu"),
    tf.keras.layers.Dense(64,activation="relu"),
    tf.keras.layers.Dense(1,activation="sigmoid")
])

model.compile(loss="binary_crossentropy",optimizer=tf.keras.optimizers.RMSprop(lr=0.01,momentum=0.5,decay=0.2),metrics=["accuracy"])

history = model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=20,verbose=1)

model.save("Fake_news.h5")