import pickle
import pytesseract
import numpy as np
from PIL import Image
import argparse

label = {True:"CorrectNews",False:"FakeNews"}

# Loading our Model.
clf = pickle.load(open("FakeNewsDetection.pkl","rb"))
 
def calculate_result(test_sentence):
    output = clf.predict(test_sentence)[0]
    proba = clf.predict_proba(test_sentence)[0]
    max_proba = proba[np.argmax(proba)]
    print(f"\n\t This news is : {label[output]}")
    print(f"\n\t Level of Truth: {round(max_proba,4)*100}")

def preprocess_image(sentence):
    '''
    Note Here I have only taken into account for the newline character to be present in the image 
    if you get any other character necessary to be removed please feel free to raise a PR for that.
    '''
    test_sentence = ""
    for each in sentence:
        if each == "\n":
            each = " "
        test_sentence+=each
    return test_sentence

# Parse Argument for the image.
parser = argparse.ArgumentParser()
parser.add_argument("--image",help="The image which is to be tested")
parser.add_argument("--text",help="The sentence or group of sentences which are to be tested")
parser.add_argument("analyze",help="Analyze the model",default=True)
args = parser.parse_args()

if args.analyze=="True":
    test_sentence = ["America will soon be on its kmees"]
    calculate_result(test_sentence)
    
    # Lets try it for Image
    img = Image.open("testImg_model.jpeg")
    sentence = pytesseract.image_to_string(img)
    test_sentence = preprocess_image(sentence)
    calculate_result([test_sentence])

if args.image:
    sentence = pytesseract.image_to_string(args.image)
    test_sentence = preprocess_image(sentence)
    calculate_result([test_sentence])

if args.text:
    calculate_result([args.text])
