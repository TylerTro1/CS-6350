In each of the files, there is a kaggle.py file. I have provided the required code to run it, you just need to change the data loading to wherever your train, test and eval.anon.data files are located for you. 
For me, they were 'D:/CS 6350/Data' and they're likely elsewhere for you. Once you do that, everything else should be primed and ready to run. I have included my models and my cross_validation code so that it runs
just like how I did it. I also include the imports I used to test the XGBoost and use the pre-processing techniques that I did. 


You may execute the code to see some of my testing, but it may take a while to run as cross validation and other such testing is performed (Pipeline and GridSearchCV).