# Project Description for "zircons_recognition"

This project contains PYTHON codes for the following paper. Please consider citing the paper if you find it of any help.
1. Please start with main file: "main.py", training models and predicting zircons types of Jack Hills zircons.
2. All other (function) files in "models.py" for SVM,WSVM,TSVM and RF are called in the function train().
3. Other figures including fig 1,fig 2 and fig 3 in the paper shall be easy to reproduce with the provided function file "plot_func.py".
4. Models include SVM,WSVM,TSVM and RF. TSVM is selected as our working model for its best performance.

# Running the project in terminal for Ubuntu or command-line interface for windows
Firstly, clone this project.
```
git clone https://github.com/jiangjilian/zircons_recognition.git
```
Then, to install dependent library for our project,run
```
pip install -r requirements.txt
```
And then, run the main.py to obtain prediction results and our figures.
```
python main.py
```
  
