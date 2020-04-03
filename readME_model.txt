Basically the software requirements for this project is :
	> Python 3.6.5 or higher
	> Tesseract-OCR

Note : For tesseract ocr - 

 o For tesseract to be rendered in Python script a module named `pytesseract ` must be installed. And if you have newly installed pytesseract then you must configure the path for your installed tesseract-ocr.

o The tesseract-ocr is by-default installed inside C://ProgramFiles/Tesseract-OCR . You have to find "tesseract.exe" file inside that folder and copy its path.

o If you are using VS-Code then you can simply write -
	import pytesseract
   Just press Ctrl+LeftMouseClick over pytesseract and pytesseract.py file will be opened. Otherwise just go to the site-packages folder inside the installed Python directory and search for pytesseract directory.

o After this find tesseract_cmd variable inside the file and replace it with your copied path. Remember to change the direction of slashes( \ to /) .  


I've also added a Model architecture using LSTM cells (Neural Networks) . This model achieved an accuracy of 56.15% which is not so good but that I achieved using raw sentences for processing and not done any preprocessing . As I preprocessed the data in the `Fake_news.ipynb` , in similar way data can be preprocessed here also to achieve higher accuracy. 