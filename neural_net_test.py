from keras.models import load_model
from keras.preprocessing.text import Tokenizer
clf = load_model("Fake_news.h5")
# Getting the predictions out of it.
sentence = ["America is on its knees nowadays"]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentence)
sequence = tokenizer.texts_to_sequences(sentence)
prediction = model.predict(sequence)
print(prediction)
# Output : 0 (False)