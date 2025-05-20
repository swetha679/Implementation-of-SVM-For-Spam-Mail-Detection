# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM for Spam Mail Detection.

## Equipment Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary Python packages using import statements.
2. Read the given CSV file using the read_csv() method and print the number of contents to be displayed using df.head().
3. Split the dataset using train_test_split.
4. Calculate Y_Pred and accuracy.
5. Print all the outputs.
6. End the Program.

## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: B R SWETHA NIVASINI
RegisterNumber: 212224040345
*/
import chardet
file='spam.csv'
with open (file,'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result
import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')
data.head()
```
![image](https://github.com/user-attachments/assets/cb03e943-02fc-4f36-a81c-ad3e5dead0e9)


```
data.info()
```
![image](https://github.com/user-attachments/assets/145af677-8d97-456c-959f-c752490dbff7)


```
data.isnull().sum()
```
![image](https://github.com/user-attachments/assets/14ca0be9-88b7-406d-9400-b6ecb7ce7f81)


```
x=data["v1"].values
y=data["v2"].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)
from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred
```
![image](https://github.com/user-attachments/assets/8cdfabfc-6e41-47c5-b376-afa3770cbac8)


```
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
![image](https://github.com/user-attachments/assets/97159524-0d1c-4422-a7fd-28b3ad3cd2c7)








## Result:
Thus, the program to implement the SVM for Spam Mail Detection is written and verified using Python programming.
