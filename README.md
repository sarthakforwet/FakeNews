# FakeNews
Model to detect Fake News

Basically this model is designed to classify a news to be fake or not by training it with examples of correct and fake news. It is a kind of Binary classification task. 

This model has been trained over a csv file having `Sentence` and `label` as two columns , one as feature and other is label.

After training I've saved the model in a pickle file named <a href ="https://drive.google.com/drive/u/0/folders/1A_fyCbB1JcUVkybIDRVPjO8EjxOgL8Kr">FakeNewsDetection.pkl</a>.


To run this model follow the steps below:
 
  1 Clone the Repo : git clone <a>https://github.com/sarthakforwet/FakeNews.git</a><br>
  2 Go to the FakeNews directory<br>
  3 open terminal and type `python test.py True/False --image (path to your image file if exist)`<br> 
  4 run it !

 - The Argument in step 3 `True/False` is basically if you wanna analyze the model or not. 
Note : It is not necessary that you pass image argument ,if you won't pass an argument then it would prompt for it.

The above command was for dealing with images. However if you want to run it on text input then you should change command at step 3 to 
`python test.py --text (The Statement you wanna analysize)`  

Here `FakeNews.ipynb` shows how the model was trained and whats the stats of it.

There is also a Model for the same problem using `Neural Nets(LSTM cells)` . <br> This model achieved an accuracy of `56.15%` which can be improved using preprocessing as i have done in the `Fake_News.ipynb` file . If you wish you can make a PR for that .

Future Work:- Implementation of [this](https://arxiv.org/pdf/1805.08751.pdf) research paper.
