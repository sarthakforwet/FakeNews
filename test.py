import pickle
import pytesseract
import numpy as np
from PIL import Image

clf = pickle.load(open("FakeNewsDetection.pkl","rb"))
test_sentence = ["Various shops and schools are closed due to corona-virus"]
label = {False:"FakeNews",True:"CorrectNews"}
output = clf.predict(test_sentence)[0]
proba = clf.predict_proba(test_sentence)[0]
max_proba = proba[np.argmax(proba)]
print(f"\n\t This news is : {label[output]}")
print(f"\n\t Level of Truth: {round(max_proba,4)*100}")

# Lets try it for Image
img = Image.open("testImg_model.jpeg")
sentence = pytesseract.image_to_string(img)
test_sentence = ""
for each in sentence:
    if each == "\n":
        each = " "
    test_sentence+=each

output = clf.predict([test_sentence])[0]
proba = clf.predict_proba([test_sentence])[0]
max_proba = proba[np.argmax(proba)]

print(f"\n\t This news is:{label[output]}")
print(f"\n\t Level of Truth: {round(max_proba,4)*100}")
